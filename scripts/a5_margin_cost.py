"""Ablation A5 closed: does the smaller conformal margin of the residual world model buy lower dispatch cost at the same AC safety?
Real ACN days (or synthetic when --site synthetic).  Pipeline per site:
  1. collect one-step transitions under four rules on TRAIN days; further transitions on CAL days (disjoint);
  2. train world models: fbase_only (no network), residual_nosoc (grid-only N_theta), blackbox;  3 seeds for the learned ones;
  3. 1% margins from CAL transitions (tau_v = quantile of max_i |v_true - v_hat|; tau_s = quantile of rating-normalised one-sided branch error);
  4. dispatch: the tuned reference rule is executed on disjoint TEST days through the QP safety layer built with
       (i) zero margin, (ii) fbase margin, (iii) N_theta margin AND N_theta voltage offset, (iv) blackbox margin;
     cost ($/day, simulator) and exact-AC violations are recorded.
    PYTHONPATH=. python scripts/a5_margin_cost.py --site jpl --out docs/study/a5_margin_cost_jpl.json"""
import argparse, datetime as dt, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.replay import ReplayBuffer
from pifsrl.agents.trainer import make_components, run_episode, buffer_dims
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig, WorldModel

ap = argparse.ArgumentParser(); ap.add_argument("--site", default="jpl"); ap.add_argument("--ntr", type=int, default=16); ap.add_argument("--ncal", type=int, default=8); ap.add_argument("--nte", type=int, default=10)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stress", type=float, default=2.0)
ap.add_argument("--ref", default="", help="reference rule (default: safe_uncontrolled for jpl, edf_cap_0.6 for caltech/synthetic)"); ap.add_argument("--out", required=True); a = ap.parse_args()
torch.set_num_threads(2)
if a.site == "synthetic":
    ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=False, allow_reactive=False, ev_load_scale=a.stress)
    dates = [None] * (a.ntr + a.ncal + a.nte); sel = list(range(len(dates)))
else:
    from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
    csv = f"data/{a.site}_local_sessions.csv"
    alld = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
    need = a.ntr + a.ncal + a.nte; dates = [alld[i] for i in sorted(set(np.linspace(0, len(alld) - 1, need).astype(int)))]
    ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
    ov["ev_load_scale"], ov["real_total_capacity_kw"] = a.stress, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * a.stress / 3, 1)
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=dates[0], reward_mode="dense",
                   price_source="sce_tou_prime", demand_charge=1.5, allow_discharge=False, allow_reactive=False, **ov)
rs = np.random.default_rng(7).permutation(len(dates)); D_tr = [dates[i] for i in rs[:a.ntr]]; D_cal = [dates[i] for i in rs[a.ntr:a.ntr + a.ncal]]; D_te = [dates[i] for i in rs[a.ntr + a.ncal:]]
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
ref = a.ref or ("safe_uncontrolled" if a.site == "jpl" else "edf_cap_0.6")
kinds = ["edf_cap_1.0", "edf_cap_0.6", "edf_cap_0.4", "uncontrolled"]


def collect(ds, base_seed):
    buf = ReplayBuffer(400000, buffer_dims(env), "cpu")
    for i, d in enumerate(ds):
        run_episode(env, RuleBasedPolicy(kinds[i % 4], env, seed=base_seed + i, fbase=fbase), buffer=buf, seed=base_seed + i, collect=True, date=d)
    return buf.all()


t0 = time.time(); tr, cal = collect(D_tr, 1000), collect(D_cal, 3000); print("transitions", len(tr["soc"]), len(cal["soc"]), f"{time.time()-t0:.0f}s", flush=True)
smax = fbase.s_max.float()


def errs(wm, b):
    with torch.no_grad():
        o = wm.predict(b["act"], b["p_exo"], b["q_exo"], b["soc"], b["cap"], b["occ"], b["price"])
    return (o["v_hat"] - b["v_true"]).abs().max(1).values, ((b["s_true"] - o["s_hat"]).clamp_min(0) / smax).max(1).values


def train_wm(mode, sd):
    torch.manual_seed(sd); rng = np.random.default_rng(sd)
    wm = WorldModel(fbase, env.feeder, WMConfig(mode=mode), allow_reactive=False)
    if mode != "fbase_only":
        wm.cfg.lambda_soc = 0.0                                              # grid-only residual (SoC head harms, see Sec. on SoC diagnosis)
        for _ in range(a.steps):
            ix = rng.choice(len(tr["soc"]), 256); wm.update({k: v[ix] for k, v in tr.items()})
    return wm


models = {"fbase_only": [train_wm("fbase_only", 0)], "residual_gridonly": [train_wm("residual", s) for s in a.seeds], "blackbox": [train_wm("blackbox", s) for s in a.seeds]}
res = {"config": vars(a), "ref_rule": ref, "train_days": D_tr, "cal_days": D_cal, "test_days": D_te, "variants": {}}


def margins(wm):
    ev, es = errs(wm, cal); h = len(ev) // 2                                  # half for the quantile, half for a coverage check
    q = lambda x, al: float(torch.quantile(x[:h], 1 - al))
    out = {"tau_v_1": q(ev, 0.01), "tau_s_1": q(es, 0.01), "tau_v_5": q(ev, 0.05)}
    out["cover_v_1_check_half"] = float((ev[h:] <= out["tau_v_1"]).float().mean()); out["cover_s_1_check_half"] = float((es[h:] <= out["tau_s_1"]).float().mean())
    return out


def dispatch(wm, tv, ts, use_offset):
    agent.cfg.use_tightening = tv is not None
    agent.wm = wm if use_offset else None
    if tv is not None:
        agent.tighten.margin_v, agent.tighten.margin_v_vec = tv, None
        agent.tighten.margin_s_vec = (ts * smax).numpy(); agent.tighten.margin_s = float(ts * smax.max())
    costs, viol, served = [], [], []
    for i, d in enumerate(D_te):
        st = run_episode(env, RuleBasedPolicy("safe_" + ref if not ref.startswith("safe_") else ref, env, seed=1, fbase=fbase), buffer=None, seed=9000 + i, collect=False, date=d, agent=agent)
        costs.append(st["total_cost"]); viol.append(st["v_viol"] + st["s_viol"]); served.append(st["served_frac"])
    return {"cost": float(np.mean(costs)), "viol_per_day": float(np.mean(viol)), "days_with_violation": float(np.mean(np.array(viol) > 0)), "served": float(np.mean(served)), "per_day": costs}


res["variants"]["zero_margin"] = {"margins": {"tau_v_1": 0.0, "tau_s_1": 0.0}, "runs": [dispatch(None, 0.0, 0.0, False)]}
print("zero", res["variants"]["zero_margin"]["runs"][0]["cost"], res["variants"]["zero_margin"]["runs"][0]["viol_per_day"], flush=True)
for name, wms in models.items():
    v = {"margins": [], "runs": []}
    for wm in wms:
        m = margins(wm); v["margins"].append(m); v["runs"].append(dispatch(wm, m["tau_v_1"], m["tau_s_1"], use_offset=(name == "residual_gridonly")))
        print(name, {k: round(x, 5) for k, x in m.items()}, "cost", round(v["runs"][-1]["cost"], 2), "viol/day", v["runs"][-1]["viol_per_day"], flush=True)
    res["variants"][name] = v
    json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
