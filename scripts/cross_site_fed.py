"""Cross-site (Caltech -> JPL) transfer and federated (Caltech + JPL) experiments, stressed and unstressed.
  --task transfer  : train ONE residual SafeTD3 on Caltech train days, test zero-shot on JPL days (+ Caltech test days)
  --task federated : 2 clients (Caltech, JPL), each with its OWN train days and its OWN EDF cap tuned on local train days;
                     FedProx / FedAvg / local-only; evaluated on each site's held-out days
Both: demand charge, dense reward, k_max padded to 54 for weight sharing (JPL has 52 EVSE; real capacity 195 kW still binds).
Stress = 2x charger power and 2/3 of the nameplate hub capacity (Caltech 100 kW, JPL 130 kW).
Tariff per site is configurable: --caltech-price/--jpl-price in {sce_tou_prime, caiso_rtp}; caiso_rtp uses --lmp-csv if given,
else the labelled SYNTHETIC duck curve."""
import argparse, copy, datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pifsrl.runlog import say
from pifsrl.env.v2g_env import EnvConfig, V2GEnv
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.agents.trainer import TrainConfig, train_site, evaluate_rule, evaluate_agent, make_components

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["transfer", "federated"], required=True)
ap.add_argument("--regime", choices=["normal", "stress"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=60)       # transfer: central episodes; federated: rounds*local_episodes
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--local-episodes", type=int, default=2)
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--algos", default="fedprox,local")
ap.add_argument("--residual", type=float, default=0.1)
ap.add_argument("--demand", type=float, default=1.5)
ap.add_argument("--caltech-price", default="sce_tou_prime"); ap.add_argument("--jpl-price", default="sce_tou_prime")
ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
ap.add_argument("--auto-data", action="store_true", help="wire every drop-in file in --data-dir (tariff_<site>.csv, caiso_sp15_lmp.csv, ...); missing -> fallback")
ap.add_argument("--data-dir", default="data"); ap.add_argument("--lmp-analog", nargs="?", const="month", default="", choices=["", "month", "any"])
ap.add_argument("--demand-from-tariff", action="store_true"); ap.add_argument("--battery-default", action="store_true")
ap.add_argument("--tag", default=None); ap.add_argument("--price-scenario", default=""); ap.add_argument("--price-seed", type=int, default=0)
ap.add_argument("--lmp-csv", default=None)
ap.add_argument("--out", default="results/study4")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
import torch
DEV = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
if DEV.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (install the CUDA build of PyTorch)")
tag = a.tag or f"{a.task}_{a.regime}_{a.caltech_price}_{a.jpl_price}_s{a.seed}"
os.makedirs(a.out, exist_ok=True)
CSV = {"caltech": "data/caltech_local_sessions.csv", "jpl": "data/jpl_local_sessions.csv"}
PRICE = {"caltech": a.caltech_price, "jpl": a.jpl_price}
CAPS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]


def split_days(site, n=20):
    wk = [d for d in list_acn_dates(CSV[site], site) if dt.date.fromisoformat(d).weekday() < 5]
    days = [wk[i] for i in sorted(set(np.linspace(0, len(wk) - 1, n).astype(int)))]
    perm = list(np.random.default_rng(a.seed).permutation(len(days)))
    ntr, nva = int(.6 * len(days)), int(.2 * len(days))
    return [days[i] for i in perm[:ntr]], [days[i] for i in perm[ntr:ntr + nva]], [days[i] for i in perm[ntr + nva:]]


def env_cfg(site, day0, price=None):
    ov = recommended_env_overrides_for_acn(site); ov["k_max"] = 54
    if a.regime == "stress":
        ov["ev_load_scale"] = 2.0; ov["real_total_capacity_kw"] = round(ACN_SITE_INFO[site]["total_capacity_kw"] * 2 / 3, 1)
    kw = {"price_source": price or PRICE[site], "price_csv_file": a.lmp_csv}
    if a.auto_data:
        from pifsrl.env.datafiles import discover
        kw.update(discover(site, a.data_dir, price=(price or PRICE[site]) if (price or PRICE[site]) != "sce_tou_prime" else "auto",
                           demand_from_tariff=a.demand_from_tariff, battery_default=a.battery_default))
        if a.lmp_csv: kw["price_csv_file"] = a.lmp_csv
    if a.lmp_analog: kw["lmp_analog"] = a.lmp_analog
    if a.price_scenario: kw["price_scenario"], kw["price_seed"] = a.price_scenario, a.price_seed
    return EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=a.seed, sessions_source="acn_csv", acn_csv_file=CSV[site],
                     acn_site=site, acn_date=day0, reward_mode="dense", demand_charge=a.demand, **kw, **ov)


def tune_cap(env, days):
    from pifsrl.agents.trainer import tune_edf_cap
    b, tab = tune_edf_cap(env, days, CAPS)
    say(f"  EDF cap tuned on {len(days)} local train days (violation-free only): {b}  { {k: (round(v['cost']), round(v['viol'], 3)) for k, v in tab.items()} }")
    return b, tab


def summarise(env, days, agent, base_kind, seed0, label, fbase_agent):
    out = {}
    for kind in ["safe_uncontrolled", base_kind, "oracle_lp"]:
        r = evaluate_rule(env, kind, len(days), seed0=seed0, agent=fbase_agent, dates=days)
        out[kind] = {k: r[k] for k in ("total_cost", "total_cost_std", "served_frac", "v_viol", "s_viol", "episode_costs")}
    if agent is not None:
        r = evaluate_agent(env, agent, len(days), seed0=seed0, dates=days, base_kind=base_kind)
        out["SafeTD3"] = {k: r[k] for k in ("total_cost", "total_cost_std", "served_frac", "v_viol", "s_viol", "episode_costs")}
    for k, v in out.items():
        say(f"    [{label}] {k:20s} cost {v['total_cost']:8.1f} served {100*v['served_frac']:5.1f}% viol {v['v_viol']+v['s_viol']:.1e}")
    return out


ctr, cva, cte = split_days("caltech"); jtr, jva, jte = split_days("jpl")
say(f"[{tag}] caltech train/val/test = {len(ctr)}/{len(cva)}/{len(cte)}  jpl = {len(jtr)}/{len(jva)}/{len(jte)}")
td3 = TD3Config(hidden=256, qp_iters=120, batch_size=128, lambda_mb=0.0, residual_scale=a.residual, bc_weight=0.0)
wm = WMConfig(hidden=128, mode="residual")
res = {"tag": tag, "days": dict(caltech=[ctr, cva, cte], jpl=[jtr, jva, jte])}

if a.task == "transfer":
    ec_c = env_cfg("caltech", ctr[0]); env_c = V2GEnv(ec_c)
    capc, tune = tune_cap(env_c, ctr); base = f"edf_cap_{capc}"
    trc = TrainConfig(episodes=a.episodes, warmup_episodes=6, eval_every=5, eval_episodes=4, seed=a.seed, log_every=10,
                      warmup_mode="greedy", bc_pretrain_steps=0, base_kind=base, train_dates=ctr, val_dates=cva, test_dates=cte,
                      final_test_episodes=len(cte), device=DEV)
    out = train_site(ec_c, td3, wm, trc, verbose=True)
    agent = out["agent"]; best = out["best_state"]
    res["caltech_in_domain"] = summarise(out["env"], cte, agent, base, 7_000_000, "Caltech test (in-domain)", agent)
    jt = jva + jte
    for pr in sorted({a.jpl_price, "caiso_rtp"}):                      # same tariff as training, and a tariff shift arm
        env_j = V2GEnv(env_cfg("jpl", jt[0], price=pr))
        capj, _ = tune_cap(env_j, jtr)                                   # in-domain reference cap, tuned on JPL train days
        r = summarise(env_j, jt, agent, base, 7_000_000, f"JPL zero-shot [{pr}]", agent)
        r[f"edf_cap_{capj}(JPL-tuned)"] = {k: v for k, v in evaluate_rule(env_j, f"edf_cap_{capj}", len(jt), seed0=7_000_000, dates=jt).items()
                                           if k in ("total_cost", "total_cost_std", "served_frac", "v_viol", "s_viol", "episode_costs")}
        say(f"    [JPL {pr}] EDF cap {capj} (JPL-tuned): cost {r[f'edf_cap_{capj}(JPL-tuned)']['total_cost']:.1f}")
        if best is not None:
            last = copy.deepcopy(agent.state_dict()); agent.load_state_dict(best)
            r["SafeTD3_best_ckpt"] = {k: v for k, v in evaluate_agent(env_j, agent, len(jt), seed0=7_000_000, dates=jt, base_kind=base).items()
                                      if k in ("total_cost", "served_frac", "v_viol", "s_viol")}
            agent.load_state_dict(last)
        res[f"jpl_zero_shot_{pr}"] = r
    res["caltech_tuned_cap"] = capc
else:
    from pifsrl.federated.fedprox import ClientSpec, FedClient, FedConfig, run_federated, client_drift
    specs = []
    for site, tr, va in (("caltech", ctr, cva), ("jpl", jtr, jva)):
        ec = env_cfg(site, tr[0]); e = V2GEnv(ec)
        cap, _ = tune_cap(e, tr)                                          # LOCAL tuning: no data leaves the site
        specs.append(ClientSpec(f"real_{site}", ec, train_dates=tr, val_dates=va, base_kind=f"edf_cap_{cap}"))
    res["local_caps"] = {s.name: s.base_kind for s in specs}
    for algo in a.algos.split(","):
        say(f"\n=== {tag} :: {algo} ===")
        clients = [FedClient(s, copy.deepcopy(td3), wm, DEV) for s in specs]
        fc = FedConfig(rounds=a.rounds, local_episodes=a.local_episodes, mu=a.mu, algorithm=algo, eval_every=5,
                       eval_episodes=4, updates_per_episode=192, batch=128, seed=a.seed)
        o = run_federated(clients, fc, verbose=True)
        r = {"drift": client_drift(clients), "history": o["history"], "evals": o["evals"]}
        for c, (site, te) in zip(clients, (("caltech", cte), ("jpl", jte))):
            r[f"test_{site}"] = summarise(c.env, te, c.agent, c.spec.base_kind, 7_000_000, f"{algo} {site} test", c.agent)
        res[algo] = r
json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1, default=float)
say(f"wrote {a.out}/{tag}.json")
