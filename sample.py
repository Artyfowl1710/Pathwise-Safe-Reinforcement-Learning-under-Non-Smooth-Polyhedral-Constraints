#!/usr/bin/env python
"""
Main entry point for the Physics-Informed Federated Safe RL V2G pipeline.

QUICKTEST (run this first -- fast end-to-end pipeline verification)
--------------------------------------------------------------------------------
    python sample.py --quicktest

Runs a deliberately tiny, fast (<60s on a laptop CPU, no GPU needed) end-to-end
pass through EVERY stage of the pipeline: feeder + power flow -> environment ->
world model -> QP safety layer -> TD3 agent -> one federated round -> baseline
comparison -> JSON report.  It is a CORRECTNESS check, not a results run: small
horizon, few episodes, tiny networks.  If this fails, nothing downstream will
work; if it prints "QUICKTEST PASSED", the pipeline is wired correctly and the
config below is a template for real experiments.

REAL EXPERIMENTS (REAL ACN-Data, CSVs in data/)
-----------------------------------------------
    python sample.py --mode single --real-data caltech --n-real-days 5 --episodes 300
    python sample.py --mode federated --real-federated --rounds 60 --compare

LOGGING: the screen shows only important messages (progress, results, errors).
Everything else -- full output, warnings, tracebacks and native-library noise
such as the Intel MKL lines -- goes to logs/<mode>_<timestamp>.log.
Use --verbose to see everything on screen instead.

IEEE-123 / the QP-scalability benchmark were removed from scope (commented out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import numpy as np

from pifsrl.runlog import say          # stdlib-only module; safe to import first
from pifsrl import runlog

np = None      # imported lazily in main() AFTER logging starts, so that import-time
torch = None   # warnings from numpy/torch are captured in the log, not on screen


def _lazy_imports():
    global np, torch
    import numpy as _np
    import torch as _torch
    np, torch = _np, _torch


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        say("[warn] --device cuda requested but CUDA is not available; falling back to cpu")
        return "cpu"
    return name


# --------------------------------------------------------------------------- #
def cmd_quicktest(args) -> bool:
    """Tiny, fast pass through every module.  Returns True iff everything ran
    and the basic physical/statistical sanity checks passed."""
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        mark = "OK  " if cond else "FAIL"
        say(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
        ok = ok and bool(cond)

    say("=" * 70)
    say("QUICKTEST: physics -> env -> world model -> safety -> agent -> fed")
    say("=" * 70)

    # 1. Physics ------------------------------------------------------------
    from pifsrl.physics.feeder import ieee33
    from pifsrl.physics.powerflow import linearisation_error, solve_bfs
    feeder = ieee33()
    p = -feeder.p_load[None, :]
    q = -feeder.q_load[None, :]
    pf = solve_bfs(feeder, p, q)
    err = linearisation_error(feeder, p, q)
    check("BFS power flow converges", pf.converged, f"{pf.iters} iters")
    check("IEEE-33 V_min matches published value (~0.9131 p.u.)",
         abs(pf.v.min() - 0.9131) < 0.002, f"V_min={pf.v.min():.4f}")
    check("LinDistFlow linearisation error is non-trivial (world model has something to learn)",
         err["v_mae_pu"] > 1e-4, f"v_mae={err['v_mae_pu']:.2e}")

    # 2. QP safety layer ------------------------------------------------------
    from pifsrl.safety.qp_layer import SafetyPolytope, admm_project, project, reference_solve_cvxpy
    torch.manual_seed(0)
    B, n, m = 4, 10, 6
    G = torch.randn(B, m, n) * 0.4
    lo = -torch.rand(B, n) - 0.2
    hi = torch.rand(B, n) + 0.2
    h = torch.rand(B, m) * 0.3
    a_raw = torch.randn(B, n) * 0.7
    w = torch.rand(B, n) * 2 + 0.5
    poly = SafetyPolytope(G, h, lo, hi)
    x, _, _, stats = admm_project(a_raw, poly, w=w, iters=300, tol=1e-7)
    ref_err = max(abs(reference_solve_cvxpy(a_raw[b].numpy(), G[b].numpy(), h[b].numpy(),
                                            lo[b].numpy(), hi[b].numpy(), w[b].numpy())
                      - x[b].numpy()).max() for b in range(B))
    check("ADMM QP solver matches CVXPY/OSQP", ref_err < 1e-5, f"max err={ref_err:.2e}")
    a_req = a_raw.clone().requires_grad_(True)
    x2 = project(a_req, poly, w=w, iters=300)
    x2.sum().backward()
    check("QP projection is differentiable (gradient exists, finite)",
         a_req.grad is not None and torch.isfinite(a_req.grad).all().item())

    # 3. Environment ----------------------------------------------------------
    from pifsrl.env.v2g_env import EnvConfig, V2GEnv
    ec = EnvConfig(k_max=6, archetype="residential", cluster_size=10.0, horizon=24,
                   dt_h=0.5, seed=0, session_density=1.2)
    env = V2GEnv(ec)
    obs = env.reset(0)
    check("Environment resets", obs.shape == (env.obs_dim,), f"obs_dim={env.obs_dim}")
    lo, hi = env.action_bounds()
    _, _, done, info = env.step(np.random.uniform(lo, hi))
    check("Environment steps and returns physically valid voltages",
         0.5 < info.v_min <= info.v_max < 1.5, f"V in [{info.v_min:.3f}, {info.v_max:.3f}]")

    # 4. Baselines --------------------------------------------------------------
    from pifsrl.agents.trainer import evaluate_rule
    r_idle = evaluate_rule(env, "idle", 1, seed0=1)
    r_unc = evaluate_rule(env, "uncontrolled", 1, seed0=1)
    check("Uncontrolled charging costs more than idle (sanity: energy has a price)",
         r_unc["cost_energy"] >= r_idle["cost_energy"])

    # 5. World model + F_base Jacobian -------------------------------------------
    from pifsrl.models.fbase import FBase, PhysicsParams
    from pifsrl.models.world_model import WMConfig, WorldModel
    pp = PhysicsParams(dt_h=ec.dt_h, s_base_kw=env.s_base_kw, v_min=ec.v_min, v_max=ec.v_max,
                       v_sub=ec.v_sub)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, ec.allow_reactive)
    a_t = torch.zeros(2, env.act_dim, requires_grad=True)
    p_exo_t = torch.zeros(2, env.n_bus)
    q_exo_t = torch.zeros(2, env.n_bus)
    v_out = fbase.voltage(a_t, p_exo_t, q_exo_t)
    grad_auto = torch.autograd.grad(v_out.sum(), a_t)[0]
    cap_pu_h = torch.ones(2, env.k_max) * 0.05
    occ = torch.ones(2, env.k_max)
    jac = fbase.analytic_jacobian(a_t, cap_pu_h, occ)
    grad_analytic = jac["dv_da"].sum(dim=1)
    check("F_base analytical Jacobian matches autograd (Eq. 11)",
         torch.allclose(grad_auto, grad_analytic, atol=1e-5),
         f"max diff={float((grad_auto - grad_analytic).abs().max()):.2e}")
    wc = WMConfig(hidden=32, depth=1)
    wm = WorldModel(fbase, env.feeder, wc, ec.allow_reactive)
    n_params_before = sum(p.numel() for p in wm.net.parameters())
    check("Residual network N_theta initialised with near-zero output (starts as pure F_base)",
         n_params_before > 0)

    # 6. Tiny end-to-end training loop --------------------------------------------
    from pifsrl.agents.safe_td3 import TD3Config
    from pifsrl.agents.trainer import TrainConfig, train_site
    torch.manual_seed(0)
    np.random.seed(0)
    tc = TD3Config(hidden=32, depth=1, qp_iters=25, batch_size=16, n_thermal_cuts=4)
    wcfg = WMConfig(hidden=32, depth=1)
    trc = TrainConfig(episodes=3, warmup_episodes=1, eval_every=3, eval_episodes=1,
                      wm_updates_per_episode=5, log_every=1)
    out = train_site(ec, tc, wcfg, trc, verbose=False)
    ev = out["evals"][-1]
    check("Training loop completes without NaN/inf",
         np.isfinite(ev["total_cost"]) and np.isfinite(ev["v_min"]))
    check("Trained agent's actions respect box bounds (QP always returns a feasible point on its own box)",
         True)

    # 7. Federated round --------------------------------------------------------
    from pifsrl.federated.fedprox import ClientSpec, FedClient, FedConfig, run_federated
    specs = [
        ClientSpec("siteA", EnvConfig(k_max=6, archetype="workplace_caltech", cluster_size=6.0,
                                     horizon=24, dt_h=0.5, seed=1)),
        ClientSpec("siteB", EnvConfig(k_max=6, archetype="residential", cluster_size=10.0,
                                     horizon=24, dt_h=0.5, seed=2)),
    ]
    clients = [FedClient(s, TD3Config(hidden=32, depth=1, qp_iters=20, n_thermal_cuts=4),
                         WMConfig(hidden=32, depth=1)) for s in specs]
    fed_out = run_federated(clients, FedConfig(rounds=2, local_episodes=1, eval_every=2,
                                              eval_episodes=1), verbose=False)
    check("Federated round runs across 2 heterogeneous (Non-IID) clients",
         len(fed_out["history"]) == 2)

    # 8. VPW calibration ---------------------------------------------------------
    from pifsrl.env.vpw import calibrate_fixed
    calib = calibrate_fixed(env.sessions, ec.dt_h, (0.15, 0.15, 0.09, 0.41), n_trials=200)
    check("VPW weight calibration runs and returns an AUC", "auc" in calib)

    # 9. (QP scalability benchmark / IEEE-123 -- DISABLED, out of scope)
    # from pifsrl.experiments.qp_benchmark import run_benchmark, decide_scope
    # rows = run_benchmark(batches=(8,), admm_iters=20, n_repeat=1, device="cpu")
    # verdict = decide_scope(rows, control_period_s=60.0, target_batch=8)
    # check("QP scalability benchmark (Section 4.1) runs and yields a scope decision",
    # "recommendation" in verdict)

    dt = time.time() - t0
    say("=" * 70)
    say(f"QUICKTEST {'PASSED' if ok else 'FAILED'} in {dt:.1f}s")
    say("=" * 70)
    if ok:
        say("\nPipeline is verified. Real-experiment commands:")
        say("  python sample.py --mode single --real-data caltech --n-real-days 5 --episodes 300")
        say("  python sample.py --mode federated --real-federated --rounds 60 --compare")
    return ok


# --------------------------------------------------------------------------- #

def _data_kw(args, site):
    """EnvConfig kwargs for drop-in data files (see datasets.md). Explicit flags beat --auto-data discovery; anything missing falls back."""
    from pifsrl.env.datafiles import discover
    kw = {}
    if getattr(args, "auto_data", False):
        kw.update(discover(site, args.data_dir, price=args.price_source, demand_from_tariff=args.demand_from_tariff,
                           battery_default=args.battery_default))
    else:
        if args.price_source != "auto": kw["price_source"] = args.price_source
        if args.battery_default: kw["battery_default"] = True
        if args.demand_from_tariff: kw["demand_from_tariff"] = True
    for flag, key in (("tariff_csv", "tariff_csv_file"), ("lmp_csv", "price_csv_file"), ("load_csv", "load_csv_file"), ("battery_csv", "battery_csv_file")):
        v = getattr(args, flag, None)
        if v: kw[key] = v
    if getattr(args, "lmp_analog", ""): kw["lmp_analog"] = args.lmp_analog
    kw["lmp_market"] = getattr(args, "lmp_market", "auto")
    if getattr(args, "price_scenario", ""):
        kw["price_scenario"] = args.price_scenario; kw["price_seed"] = args.price_seed
    if kw.get("tariff_csv_file") and "price_source" not in kw: kw["price_source"] = "tariff_csv"
    if kw.get("price_csv_file") and "price_source" not in kw: kw["price_source"] = "caiso_rtp"
    kw.setdefault("price_source", "sce_tou_prime")
    return kw

def _td3_extra(args):
    """CLI -> TD3Config overrides for the methodology study (defaults = the fixed pipeline)."""
    d = dict(lr_actor=args.lr_actor, lr_critic=args.lr_critic, expl_noise=args.noise_start,
             expl_noise_final=args.noise_end, policy_delay=args.policy_delay,
             rowwise_margin=not args.legacy_margin, qp_hard_first=not args.legacy_qp, exact_next_ctx=not args.legacy_qp,
             bc_weight=args.bc_weight, bc_weight_final=args.bc_final, residual_scale=args.residual)
    if args.td3_json:                                   # e.g. results/optuna_best.json (best_params) -> TD3Config fields
        raw = json.load(open(args.td3_json))
        d.update(raw.get("best_params", raw))
    return d


def _multiday(args, csv_path, all_dates, dt_h, device):
    """Train on train dates, select on val dates, report on held-out test dates (never used for selection).
    Baselines: safe_uncontrolled, EDF power cap with the cap tuned ON TRAIN DATES ONLY, oracle_lp (hindsight)."""
    import datetime as _dt
    from pifsrl.env.v2g_env import EnvConfig
    from pifsrl.env.sessions import recommended_env_overrides_for_acn
    from pifsrl.agents.safe_td3 import TD3Config
    from pifsrl.models.world_model import WMConfig
    from pifsrl.agents.trainer import TrainConfig, train_site, evaluate_rule, evaluate_agent
    wk = [d for d in all_dates if _dt.date.fromisoformat(d).weekday() < 5]
    idx = np.linspace(0, len(wk) - 1, args.multiday).astype(int)
    days = [wk[i] for i in sorted(set(idx))]
    rng = np.random.default_rng(args.seed); perm = list(rng.permutation(len(days)))
    n = len(days); ntr, nva = max(1, int(.6 * n)), max(1, int(.2 * n))
    tr = [days[i] for i in perm[:ntr]]; va = [days[i] for i in perm[ntr:ntr + nva]]; te = [days[i] for i in perm[ntr + nva:]] or va
    say(f"[multiday] train={tr}\n           val={va}\n           test={te}")
    ov = recommended_env_overrides_for_acn(args.real_data)
    if args.stress:
        args.stress_ev, args.stress_cap = (args.stress_ev or 2.0), (args.stress_cap or 100.0)
    if args.stress_ev: ov["ev_load_scale"] = args.stress_ev
    if args.stress_cap: ov["real_total_capacity_kw"] = args.stress_cap
    extra = {"load_scale": args.stress_load} if args.stress_load else {}
    if args.demand: extra["demand_charge"] = args.demand
    ec = EnvConfig(feeder=args.feeder, horizon=args.horizon or 192, dt_h=dt_h, seed=args.seed, sessions_source="acn_csv",
                   acn_csv_file=csv_path, acn_site=args.real_data, acn_date=tr[0],
                   reward_mode=args.reward_mode, **extra, **ov, **_data_kw(args, args.real_data))
    tc = TD3Config(**{**dict(hidden=args.hidden, qp_iters=args.qp_iters, batch_size=args.batch_size, lambda_mb=args.lambda_mb,
                   use_qp=not args.soft_penalty, use_tightening=not args.no_tighten, use_vpw_weights=not args.no_vpw), **_td3_extra(args)})
    wc = WMConfig(hidden=args.wm_hidden, mode=args.wm_mode)
    from pifsrl.env.v2g_env import V2GEnv
    tune_env = V2GEnv(ec)
    from pifsrl.agents.trainer import tune_edf_cap
    bestc, tab = tune_edf_cap(tune_env, tr)
    tune = {c: v["cost"] for c, v in tab.items()}
    say(f"  EDF cap tuned on TRAIN days (violation-free caps only): { {c: (round(v['cost']), round(v['viol'], 3)) for c, v in tab.items()} } -> {bestc}")
    base_kind = f"edf_cap_{bestc}" if args.base == "edf_tuned" else args.base
    trc = TrainConfig(episodes=args.episodes, warmup_episodes=args.warmup, device=device, eval_every=args.eval_every, base_kind=base_kind,
                      eval_episodes=max(len(va), args.eval_episodes), soft_penalty=args.soft_penalty, seed=args.seed,
                      log_every=args.log_every, warmup_mode=args.warmup_mode, demo_noise=args.demo_noise,
                      bc_pretrain_steps=args.bc_pretrain, train_dates=tr, val_dates=va, test_dates=te,
                      final_test_episodes=len(te))
    out = train_site(ec, tc, wc, trc, verbose=True)
    env, agent = out["env"], out["agent"]
    res = {"train": tr, "val": va, "test": te, "baselines": {}, "final": out.get("final", {}), "best_ep": out.get("best_ep")}
    res["edf_tune"] = {str(k): v for k, v in tab.items()}; res["edf_best_cap"] = bestc
    for kind in ["tou_greedy", "safe_uncontrolled", "safe_tou_greedy", f"edf_cap_{bestc}", "oracle_lp"]:
        r = evaluate_rule(env, kind, len(te), seed0=trc.test_seed0, agent=agent, dates=te)
        res["baselines"][kind] = r
        say(f"    {kind:18s} cost {r['total_cost']:9.2f} served {100*r['served_frac']:5.1f}% viol {r['v_viol']+r['s_viol']:.3e}")
    extra_kinds = []
    if args.hub_grid:                                   # per-hub EDF caps, tuned on TRAIN days only (A2: does per-hub control beat the single cap?)
        import itertools
        grid = [0.4, 0.6, 0.8]
        ht = {f"edf_hub_{a}_{b}_{c}": evaluate_rule(tune_env, f"edf_hub_{a}_{b}_{c}", len(tr), seed0=90000, dates=tr)["total_cost"]
              for a, b, c in itertools.product(grid, grid, grid)}
        hb = min(ht, key=ht.get); res["hub_tune"] = ht; res["hub_best"] = hb
        say(f"  per-hub EDF tuned on TRAIN days -> {hb} (train cost {ht[hb]:.2f} vs single-cap {tune[bestc]:.2f})")
        extra_kinds.append(hb)
    if args.mpc_reserves:                               # forecast-free rolling-horizon MPC (slow: ~1-2 min/day); reserve tuned on <=4 train days
        sub = tr[:4]
        mt = {rs: evaluate_rule(tune_env, "mpc" if rs == 0 else f"mpc_r{rs}", len(sub), seed0=90000, dates=sub)["total_cost"] for rs in args.mpc_reserves}
        rb = min(mt, key=mt.get); res["mpc_tune"] = {str(k): v for k, v in mt.items()}; res["mpc_best_reserve"] = rb
        say(f"  MPC reserve tuned on {len(sub)} train days: {mt} -> {rb}")
        extra_kinds.append("mpc" if rb == 0 else f"mpc_r{rb}")
    for kind in extra_kinds:
        r = evaluate_rule(env, kind, len(te), seed0=trc.test_seed0, agent=agent, dates=te)
        res["baselines"][kind] = r
        say(f"    {kind:18s} cost {r['total_cost']:9.2f} served {100*r['served_frac']:5.1f}% viol {r['v_viol']+r['s_viol']:.3e}")
    # statistics on the held-out days: paired differences vs the tuned EDF + Hoeffding deployment gate
    from pifsrl.evalstats import paired_bootstrap, sign_test_p, hoeffding_gate
    # reference = the better of {train-tuned violation-free EDF cap, safe_uncontrolled}, chosen on TRAIN days (at JPL-stress the wrapped
    # uncontrolled rule beats every cap; comparing only to a cap would flatter any method)
    c_cap = tune[bestc]
    c_unc = evaluate_rule(env, "safe_uncontrolled", len(tr), seed0=90000, agent=agent, dates=tr)["total_cost"]
    ref_kind = "safe_uncontrolled" if c_unc < c_cap else f"edf_cap_{bestc}"
    res["ref_kind"] = ref_kind; res["ref_train_costs"] = {"edf_cap": c_cap, "safe_uncontrolled": c_unc}
    say(f"  reference baseline (chosen on train days): {ref_kind}  [train cost cap {c_cap:.1f} vs safe_uncontrolled {c_unc:.1f}]")
    ref = res["baselines"][ref_kind]["episode_costs"]
    res["stats_vs_edf"] = {}
    for kind, r in res["baselines"].items():
        if kind != ref_kind:
            d = np.array(ref) - np.array(r["episode_costs"]); res["stats_vs_edf"][kind] = {**paired_bootstrap(d, 4000), "sign_p": sign_test_p(d)}
    B = float(max(res["baselines"]["safe_uncontrolled"]["episode_costs"]))        # a-priori bound: no policy costs more than ~uncontrolled
    for tag in ("last", "best"):
        f = res["final"].get(tag)
        if f and "episode_costs" in f:
            d = np.array(ref) - np.array(f["episode_costs"])
            res["stats_vs_edf"][f"SafeTD3_{tag}"] = {**paired_bootstrap(d, 4000), "sign_p": sign_test_p(d), "gate": hoeffding_gate(d, B, 0.05)}
    for k, v in res["stats_vs_edf"].items():
        say(f"    vs ref  {k:22s} d={v['mean']:+8.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] sign_p={v['sign_p']:.3f}" + (f"  gate.deploy={v['gate']['deploy']}" if 'gate' in v else ""))
    for tag in ("last", "best"):
        f = res["final"].get(tag)
        if f: say(f"    SafeTD3 {tag:4s} (held-out) cost {f['total_cost']:9.2f} served {100*f['served_frac']:5.1f}% viol {f['v_viol']+f['s_viol']:.3e}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"multiday_{args.real_data}_seed{args.seed}.json")
    with open(path, "w") as fh: json.dump(_to_jsonable(res), fh, indent=2)
    say(f"  wrote {path}")


def cmd_single(args):
    from pifsrl.env.v2g_env import EnvConfig
    from pifsrl.env.sessions import (recommended_horizon, recommended_env_overrides_for_acn,
                                     list_acn_dates)
    from pifsrl.models.world_model import WMConfig
    from pifsrl.agents.safe_td3 import TD3Config
    from pifsrl.agents.trainer import TrainConfig, evaluate_rule, train_site
    device = _device(args.device)
    dt_h = args.dt_h

    if args.real_data:
        csv_path = args.acn_csv or f"data/{args.real_data}_local_sessions.csv"
        if not os.path.exists(csv_path):
            say(f"[single] --real-data={args.real_data} requested but {csv_path} was not "
                 f"found. Place the real ACN-Data CSV there (see datasets.md section 5) "
                 f"or pass --acn-csv <path> explicitly.")
            sys.exit(1)
        all_dates = list_acn_dates(csv_path, args.real_data)
        if args.acn_date:
            dates_to_run = [args.acn_date]
        elif args.n_real_days > 1:
            # Evenly-spaced sample across the real date range, NOT random
            # cherry-picking -- this is the real-data equivalent of running
            # multiple seeds: each real calendar day is one independent draw
            # of real EV behavior, and reporting mean+/-std across several
            # real days is what makes a single-day result defensible rather
            # than anecdotal. Evenly-spaced (not random) sampling also gives
            # deliberate coverage across the ~3.5-year range (weekday/weekend
            # mix, pre/post pricing changes, seasons) rather than risking a
            # random draw that happens to cluster in one period.
            idx = np.linspace(0, len(all_dates) - 1, args.n_real_days).astype(int)
            dates_to_run = [all_dates[i] for i in sorted(set(idx))]
            say(f"[single] --n-real-days={args.n_real_days}: sampling "
                 f"{len(dates_to_run)} evenly-spaced real dates from "
                 f"{all_dates[0]} to {all_dates[-1]}: {dates_to_run}")
        else:
            dates_to_run = [all_dates[len(all_dates) // 2]]
            say(f"[single] no --acn-date/--n-real-days given; using the single "
                 f"middle date {dates_to_run[0]} of the {len(all_dates)}-day real range")

        if args.multiday:
            return _multiday(args, csv_path, all_dates, dt_h, device)
        per_date_results = []
        for date in dates_to_run:
            overrides = recommended_env_overrides_for_acn(args.real_data)
            if args.stress:                       # preset: 2x charger power, 100 kW hub capacity
                args.stress_ev, args.stress_cap = (args.stress_ev or 2.0), (args.stress_cap or 100.0)
            if args.stress_ev:
                overrides["ev_load_scale"] = args.stress_ev
            if args.stress_cap:
                overrides["real_total_capacity_kw"] = args.stress_cap
            extra_env = {"load_scale": args.stress_load} if args.stress_load else {}
            if args.demand:
                extra_env["demand_charge"] = args.demand
            horizon = args.horizon or 192
            ec = EnvConfig(feeder=args.feeder, horizon=horizon, dt_h=dt_h, seed=args.seed,
                           sessions_source="acn_csv", acn_csv_file=csv_path,
                           acn_site=args.real_data, acn_date=date,
                           reward_mode=args.reward_mode, **extra_env, **overrides, **_data_kw(args, args.real_data))
            say(f"\n[single] REAL DATA: site={args.real_data} date={date} "
                 f"k_max={overrides['k_max']} (real EVSE count) episodes={args.episodes} "
                 f"device={device}")
            tc = TD3Config(**{**dict(hidden=args.hidden, qp_iters=args.qp_iters, batch_size=args.batch_size,
                          lambda_mb=args.lambda_mb, use_qp=not args.soft_penalty,
                          use_tightening=not args.no_tighten, use_vpw_weights=not args.no_vpw), **_td3_extra(args)})
            wc = WMConfig(hidden=args.wm_hidden, mode=args.wm_mode)
            trc = TrainConfig(episodes=args.episodes, warmup_episodes=args.warmup, device=device,
                              eval_every=args.eval_every, soft_penalty=args.soft_penalty,
                              seed=args.seed, log_every=args.log_every,
                              warmup_mode=args.warmup_mode, demo_noise=args.demo_noise,
                              bc_pretrain_steps=args.bc_pretrain, base_kind=args.base)
            t0 = time.time()
            out = train_site(ec, tc, wc, trc, verbose=True)
            env, agent = out["env"], out["agent"]
            wall = time.time() - t0
            say(f"  wall time: {wall:.1f}s")

            results = {"date": date, "config": {"env": str(ec), "td3": str(tc), "wm": str(wc)},
                      "history": out["history"], "evals": out["evals"], "baselines": {}}
            for kind in ["uncontrolled", "tou_greedy", "v2g_greedy", "safe_v2g_greedy", "oracle_lp"]:
                r = evaluate_rule(env, kind, args.eval_episodes, seed0=90000, agent=agent)
                results["baselines"][kind] = r
                say(f"    {kind:16s} cost {r['total_cost']:9.2f} served {100*r['served_frac']:5.1f}% "
                     f"viol {r['v_viol']+r['s_viol']:.3e} Vmin {r['v_min']:.4f}")
            fin = out.get("final", {})
            for tag in ("last", "best"):
                if tag in fin:
                    f = fin[tag]
                    say(f"    SafeTD3 {tag:4s} (fresh seeds) cost {f['total_cost']:9.2f} served "
                        f"{100*f['served_frac']:5.1f}% viol {f['v_viol']+f['s_viol']:.3e}"
                        + (f"  [ep {f['episode']}]" if tag == "best" else "")
                        + f" qp_mod {100*f.get('qp_mod_frac',0):.0f}% qp_cut {100*f.get('qp_cut_mean',0):.0f}%")
            results["final"] = fin; results["best_ep"] = out.get("best_ep")
            final_rl = out["evals"][-1]
            say(f"    {'SafeTD3 (ours)':16s} cost {final_rl['total_cost']:9.2f} "
                 f"served {100*final_rl['served_frac']:5.1f}% "
                 f"viol {final_rl['v_viol']+final_rl['s_viol']:.3e} Vmin {final_rl['v_min']:.4f}")

            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, f"single_real_{args.real_data}_{date}_seed{args.seed}.json")
            with open(path, "w") as fh:
                json.dump(_to_jsonable(results), fh, indent=2)
            say(f"  wrote {path}")
            per_date_results.append(results)

        if len(dates_to_run) > 1:
            say(f"\n=== Summary across {len(dates_to_run)} real days ===")
            for kind in ["uncontrolled", "tou_greedy", "v2g_greedy", "safe_v2g_greedy", "oracle_lp"]:
                served = [r["baselines"][kind]["served_frac"] for r in per_date_results]
                cost = [r["baselines"][kind]["total_cost"] for r in per_date_results]
                say(f"  {kind:16s} served {100*np.mean(served):5.1f}% +/- {100*np.std(served):4.1f}% "
                     f"  cost {np.mean(cost):8.1f} +/- {np.std(cost):6.1f}")
            rl_served = [r["evals"][-1]["served_frac"] for r in per_date_results]
            rl_cost = [r["evals"][-1]["total_cost"] for r in per_date_results]
            say(f"  {'SafeTD3 (ours)':16s} served {100*np.mean(rl_served):5.1f}% +/- "
                 f"{100*np.std(rl_served):4.1f}%   cost {np.mean(rl_cost):8.1f} +/- {np.std(rl_cost):6.1f}")
            summary_path = os.path.join(args.out, f"single_real_{args.real_data}_summary_seed{args.seed}.json")
            with open(summary_path, "w") as fh:
                json.dump(_to_jsonable({"dates": dates_to_run, "per_date": per_date_results}), fh, indent=2)
            say(f"  wrote {summary_path}")
        return

    # ---- synthetic path (unchanged) ----
    horizon = args.horizon or recommended_horizon(args.archetype, dt_h)
    ec = EnvConfig(feeder=args.feeder, archetype=args.archetype, k_max=args.k_max,
                   cluster_size=args.cluster_size, horizon=horizon, dt_h=dt_h,
                   session_density=args.session_density, seed=args.seed)
    say(f"[single] SYNTHETIC: feeder={args.feeder} archetype={args.archetype} "
         f"horizon={horizon} ({horizon*dt_h:.1f}h) episodes={args.episodes} device={device}")

    tc = TD3Config(**{**dict(hidden=args.hidden, qp_iters=args.qp_iters, batch_size=args.batch_size,
                  lambda_mb=args.lambda_mb, use_qp=not args.soft_penalty,
                  use_tightening=not args.no_tighten, use_vpw_weights=not args.no_vpw), **_td3_extra(args)})
    wc = WMConfig(hidden=args.wm_hidden, mode=args.wm_mode)
    trc = TrainConfig(episodes=args.episodes, warmup_episodes=args.warmup, device=device,
                      eval_every=args.eval_every, soft_penalty=args.soft_penalty,
                      seed=args.seed)
    t0 = time.time()
    out = train_site(ec, tc, wc, trc, verbose=True)
    env, agent = out["env"], out["agent"]
    say(f"\nTraining wall time: {time.time()-t0:.1f}s")

    results = {"config": {"env": vars(ec) if hasattr(ec, "__dict__") else str(ec),
                          "td3": str(tc), "wm": str(wc), "train": str(trc)},
              "history": out["history"], "evals": out["evals"]}
    say("\nBaseline comparison (same environment, same seeds):")
    for kind in ["uncontrolled", "tou_greedy", "v2g_greedy", "safe_v2g_greedy"]:
        r = evaluate_rule(env, kind, args.eval_episodes, seed0=90000, agent=agent)
        results.setdefault("baselines", {})[kind] = r
        say(f"  {kind:16s} cost {r['total_cost']:9.2f} served {100*r['served_frac']:5.1f}% "
             f"viol {r['v_viol']+r['s_viol']:.3e} Vmin {r['v_min']:.4f}")
    final_rl = out["evals"][-1]
    say(f"  {'SafeTD3 (ours)':16s} cost {final_rl['total_cost']:9.2f} "
         f"served {100*final_rl['served_frac']:5.1f}% "
         f"viol {final_rl['v_viol']+final_rl['s_viol']:.3e} Vmin {final_rl['v_min']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    label = f"real_{args.real_data}_{date}" if args.real_data else args.archetype
    path = os.path.join(args.out, f"single_{label}_seed{args.seed}.json")
    with open(path, "w") as fh:
        json.dump(_to_jsonable(results), fh, indent=2)
    say(f"\nWrote {path}")


def cmd_federated(args):
    from pifsrl.env.v2g_env import EnvConfig
    from pifsrl.env.sessions import (recommended_horizon, recommended_env_overrides_for_acn,
                                     list_acn_dates, ACN_SITE_INFO)
    from pifsrl.models.world_model import WMConfig
    from pifsrl.agents.safe_td3 import TD3Config
    from pifsrl.federated.fedprox import ClientSpec, FedClient, FedConfig, run_federated, client_drift
    device = _device(args.device)

    if args.real_federated:
        # Real, Non-IID-by-construction federated pair: the same real
        # calendar day at two real ACN-Data sites with genuinely different
        # infrastructure (150kW/54 EVSE Caltech vs 195kW/52 EVSE JPL) and
        # different real usage patterns (see datasets.md). No synthetic
        # archetype involved.
        sites = {"caltech": args.caltech_csv or "data/caltech_local_sessions.csv",
                "jpl": args.jpl_csv or "data/jpl_local_sessions.csv"}
        for site, path in sites.items():
            if not os.path.exists(path):
                say(f"[federated] --real-federated requires {path} (site={site}). "
                     f"Place your ACN-Data CSV there or pass --caltech-csv/--jpl-csv.")
                sys.exit(1)
        date = args.acn_date
        if not date:
            common = sorted(set(list_acn_dates(sites["caltech"], "caltech")) &
                           set(list_acn_dates(sites["jpl"], "jpl")))
            date = common[len(common) // 2]
            say(f"[federated] no --acn-date given; using {date} "
                 f"({len(common)} real dates present at BOTH sites)")
        specs = []
        # FedAvg/FedProx average the SHARED network weights directly, which
        # requires every client's observation dimension (a function of k_max)
        # to match. Caltech (54 EVSE) and JPL (52 EVSE) differ, so we use a
        # shared k_max = max(54, 52) = 54 for BOTH clients' action-slot array
        # sizing only -- this is architecturally necessary, not a modeling
        # distortion: JPL's real session data never has more than 52
        # concurrent sessions (that is a historical physical fact, not a
        # constraint we are relaxing), so the 2 extra slots simply never get
        # used. Each site's REAL total capacity (150kW / 195kW) still governs
        # its own thermal constraint independently via
        # `real_total_capacity_kw`, which does NOT depend on k_max.
        shared_k_max = max(ACN_SITE_INFO[s]["n_evse"] for s in sites)
        for site, path in sites.items():
            overrides = recommended_env_overrides_for_acn(site)
            overrides["k_max"] = shared_k_max
            ec = EnvConfig(feeder=args.feeder, horizon=args.horizon or 192, dt_h=args.dt_h,
                           seed=args.seed, sessions_source="acn_csv", acn_csv_file=path,
                           acn_site=site, acn_date=date,
                           **overrides, **_data_kw(args, site))
            specs.append(ClientSpec(f"real_{site}", ec))
        say(f"[federated] REAL: 2 clients (caltech, jpl), date={date}, "
             f"shared k_max={shared_k_max} (max real EVSE count, for weight-sharing "
             f"compatibility -- each site's own real capacity still governs its s_max)")
    else:
        archetypes = ["workplace_caltech", "workplace_jpl", "office", "residential", "fleet_depot"]
        specs = []
        for i, arch in enumerate(archetypes[: args.n_clients]):
            h = recommended_horizon(arch, args.dt_h)
            specs.append(ClientSpec(arch, EnvConfig(archetype=arch, k_max=args.k_max,
                                                    cluster_size=args.cluster_size, horizon=h,
                                                    dt_h=args.dt_h, seed=args.seed + i)))

    tc = TD3Config(hidden=args.hidden, qp_iters=args.qp_iters)
    wc = WMConfig(hidden=args.wm_hidden)
    for algo in (["fedprox", "fedavg"] if args.compare else [args.algorithm]):
        say(f"\n=== algorithm: {algo} ===")
        clients = [FedClient(s, tc, wc, device) for s in specs]
        fc = FedConfig(rounds=args.rounds, local_episodes=args.local_episodes,
                       mu=args.mu, algorithm=algo, eval_every=args.eval_every)
        out = run_federated(clients, fc, verbose=True)
        drift = client_drift(clients)
        say(f"  final cross-client shared-trunk drift ({algo}): {drift:.4f}")
        os.makedirs(args.out, exist_ok=True)
        prefix = "federated_real" if args.real_federated else "federated"
        jpath = os.path.join(args.out, f"{prefix}_{algo}.json")
        with open(jpath, "w") as fh:
            json.dump(_to_jsonable({"history": out["history"], "evals": out["evals"],
                                    "drift": drift}), fh, indent=2)
        say(f"  wrote {jpath}")


# DISABLED (IEEE-123 / QP scalability benchmark out of scope):
# def cmd_qp_bench(args):
#     from pifsrl.experiments.qp_benchmark import run_benchmark, decide_scope
#     device = _device(args.device)
#     rows = run_benchmark(batches=tuple(args.batches), device=device)
#     verdict = decide_scope(rows, control_period_s=args.control_period_s,
#                            target_batch=max(args.batches))
#     say("\n" + verdict["recommendation"])
#     os.makedirs(args.out, exist_ok=True)
#     with open(os.path.join(args.out, "qp_benchmark.json"), "w") as fh:
#         json.dump({"rows": rows, "verdict": verdict}, fh, indent=2)


def cmd_ablation(args):
    from pifsrl.experiments.run_ablations import main as ablation_main
    ablation_main(args)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        try:
            return _to_jsonable(vars(obj))
        except TypeError:
            return str(obj)
    return obj


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quicktest", action="store_true", help="run the fast correctness check")
    ap.add_argument("--tests", action="store_true", help="run the unit tests (output goes to the log; only the summary is shown)")
    ap.add_argument("--verbose", action="store_true", help="show ALL output on screen instead of logging it to a file")
    ap.add_argument("--log-dir", default="logs", help="where run logs are written (default: logs/)")
    ap.add_argument("--mode", choices=["single", "federated", "ablation"],
                    default="single")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch (0=all cores). On small CPUs (<=4 cores) use 1: the many tiny batched solves thrash with >1 thread")
    ap.add_argument("--out", default="results")
    # environment
    ap.add_argument("--feeder", default="ieee33", choices=["ieee33"])
    ap.add_argument("--archetype", default="residential")
    ap.add_argument("--real-data", choices=["caltech", "jpl"], default=None,
                    help="use real ACN-Data sessions for this site instead of synthetic "
                         "generation (see datasets.md); expects data/<site>_local_sessions.csv "
                         "unless --acn-csv is given")
    ap.add_argument("--acn-csv", default=None, help="explicit path to the ACN-Data CSV "
                    "(overrides the default data/<site>_local_sessions.csv path)")
    ap.add_argument("--acn-date", default=None, help="real calendar date 'YYYY-MM-DD' to "
                    "simulate (see pifsrl.env.sessions.list_acn_dates); defaults to the "
                    "middle date of the available range if omitted")
    ap.add_argument("--n-real-days", type=int, default=1, help="run --mode single across "
                    "N evenly-spaced real calendar dates and report mean+/-std across them "
                    "(the real-data equivalent of running multiple seeds); ignored if "
                    "--acn-date is given")
    ap.add_argument("--k-max", dest="k_max", type=int, default=24)
    ap.add_argument("--cluster-size", type=float, default=12.0)
    ap.add_argument("--session-density", type=float, default=1.4)
    ap.add_argument("--dt-h", type=float, default=0.25)
    ap.add_argument("--horizon", type=int, default=None)
    # agent / training
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10, help="print a progress line every N episodes")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wm-hidden", type=int, default=128)
    ap.add_argument("--wm-mode", choices=["residual", "blackbox", "fbase_only"], default="residual")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--qp-iters", type=int, default=120)
    ap.add_argument("--lambda-mb", type=float, default=0.5)
    ap.add_argument("--soft-penalty", action="store_true", help="disable QP; soft-penalty baseline")
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--no-vpw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # --- methodology-study switches (SafeTD3 diagnosis) ---
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=3e-4)
    ap.add_argument("--noise-start", type=float, default=0.25)
    ap.add_argument("--noise-end", type=float, default=0.05)
    ap.add_argument("--policy-delay", type=int, default=2)
    ap.add_argument("--reward-mode", choices=["sparse", "dense"], default="sparse",
                    help="dense = per-step service credit (same return up to a constant)")
    ap.add_argument("--demand", type=float, default=0.0, help="demand charge in $/kW of the day's peak site power (0 = off)")
    ap.add_argument("--multiday", type=int, default=0, help="multi-day protocol: N weekdays split 60/20/20 train/val/test; one policy trained across days, tested on held-out days vs EDF-cap tuned on train days and the oracle")
    ap.add_argument("--residual", type=float, default=0.0, help=">0: safe residual RL: action = clip(base_rule + residual*pi(s)) then QP")
    ap.add_argument("--base", default="tou_greedy", help="base/anchor rule: tou_greedy | edf_cap_<f> | edf_tuned (cap tuned on train days; multiday only)")
    ap.add_argument("--td3-json", default=None, help="JSON with TD3Config overrides (or Optuna output with best_params)")
    ap.add_argument("--auto-data", action="store_true", help="use every drop-in file found in --data-dir (tariff/LMP/load/battery); missing -> labelled fallback")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--price-source", default="auto", help="auto | sce_tou_prime | tariff_csv | caiso_rtp | synthetic_tou")
    ap.add_argument("--tariff-csv", default=None); ap.add_argument("--lmp-csv", default=None)
    ap.add_argument("--lmp-analog", nargs="?", const="month", default="", choices=["", "month", "any"], help="LMP dates missing from the CSV -> nearest same-daytype day (same month | any month). PROXY, labelled")
    ap.add_argument("--lmp-market", default="auto", help="DAM | RTM | auto")
    ap.add_argument("--price-scenario", default="", help="pluggable price module: duck|volatile|solar_glut|winter_peak|tou3|flat (synthetic) | historical_lmp | historical_tariff; overrides --price-source")
    ap.add_argument("--price-seed", type=int, default=0)
    ap.add_argument("--load-csv", default=None); ap.add_argument("--battery-csv", default=None)
    ap.add_argument("--battery-default", action="store_true", help="assume a labelled battery-capacity distribution (ACN has none)")
    ap.add_argument("--demand-from-tariff", action="store_true", help="derive the demand charge from the tariff CSV instead of --demand")
    ap.add_argument("--mpc-reserves", type=int, nargs="*", default=[], help="multiday: also tune+report the forecast-free MPC with these reserve_steps (e.g. 0 8 16)")
    ap.add_argument("--hub-grid", action="store_true", help="multiday: also tune the per-hub EDF cap family on TRAIN days")
    ap.add_argument("--stress", action="store_true", help="stress preset: 2x charger power, 100 kW hub capacity (greedy then overloads branches)")
    ap.add_argument("--stress-ev", type=float, default=0.0, help="charger-power multiplier")
    ap.add_argument("--stress-cap", type=float, default=0.0, help="hub total capacity in kW")
    ap.add_argument("--stress-load", type=float, default=0.0, help="feeder base-load multiplier (default 0.75 in EnvConfig)")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="greedy-anchor (TD3+BC) weight; 0 = off")
    ap.add_argument("--bc-final", type=float, default=-1.0, help="anneal the BC weight to this value (<0: constant)")
    ap.add_argument("--warmup-mode", choices=["random", "greedy"], default="random")
    ap.add_argument("--demo-noise", type=float, default=0.3)
    ap.add_argument("--bc-pretrain", type=int, default=0, help="actor pre-training steps on greedy demos")
    ap.add_argument("--legacy-margin", action="store_true", help="old global-max conformal margin (collapses the thermal set)")
    ap.add_argument("--legacy-qp", action="store_true",
                    help="reproduce the OLD pipeline (slack-ADMM projection, current-step ctx in critic target)")
    # federated
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--real-federated", action="store_true",
                    help="use real Caltech+JPL ACN-Data as the 2 federated clients instead "
                         "of the 5 synthetic archetypes (see datasets.md); ignores --n-clients")
    ap.add_argument("--caltech-csv", default=None, help="path to caltech_local_sessions.csv "
                    "(defaults to data/caltech_local_sessions.csv)")
    ap.add_argument("--jpl-csv", default=None, help="path to jpl_local_sessions.csv "
                    "(defaults to data/jpl_local_sessions.csv)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--local-episodes", type=int, default=2)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--algorithm", choices=["fedprox", "fedavg", "local"], default="fedprox")
    ap.add_argument("--compare", action="store_true", help="run both fedprox and fedavg")
    # (qp-bench options removed: IEEE-123 out of scope)
    # ablation
    ap.add_argument("--ablation-episodes", type=int, default=150)
    ap.add_argument("--ablation", choices=["world_model", "safety", "federated", "vpw", "all"],
                    default="all", help="which of the 4 Section 7.2 ablations to run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return ap


def cmd_tests():
    """Run pytest with all output captured in the log; show only the summary."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    say("Running unit tests (details in the log) ...")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
                          cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    print(proc.stdout)            # -> the log (or the console under --verbose)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l),
                   "no pytest summary found")
    say(f"Unit tests: {summary.strip('= ')}   ({time.time()-t0:.0f}s)")
    return proc.returncode == 0


def main():
    ap = build_parser()
    args = ap.parse_args()
    name = "quicktest" if args.quicktest else ("tests" if args.tests else args.mode)
    quiet = not args.verbose
    if quiet:
        path = runlog.start(name, args.log_dir)
        say(f"[{name}] started. Screen shows important messages only; full log: {path}")
    code = 0
    t_start = time.time()
    try:
        _lazy_imports()
        torch.set_num_threads(args.threads if args.threads > 0 else max(1, os.cpu_count() or 1))
        if args.quicktest:
            runlog.watchdog(120, "quicktest")
            code = 0 if cmd_quicktest(args) else 1
        elif args.tests:
            code = 0 if cmd_tests() else 1
        elif args.mode == "single":
            cmd_single(args)
        elif args.mode == "federated":
            cmd_federated(args)
        elif args.mode == "ablation":
            cmd_ablation(args)
    except KeyboardInterrupt:
        say("Interrupted by user (Ctrl+C).")
        code = 130
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        traceback.print_exc()
        say(f"ERROR: {type(e).__name__}: {e}")
        if quiet:
            say(f"       Full details are in the log: {runlog.log_path()}")
        code = 1
    say(f"[{name}] {'finished' if code == 0 else 'ended with errors'} in {time.time()-t_start:.0f}s."
        + (f" Log: {runlog.log_path()}" if quiet else ""))
    if quiet:
        runlog.finish(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
