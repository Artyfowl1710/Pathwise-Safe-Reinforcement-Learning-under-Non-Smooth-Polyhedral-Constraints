"""Cost breakdown of the oracle plan: LP-model cost vs the same plan priced with the simulator's SoC-dependent charging efficiency.
Plan = no-discharge oracle with the strictly convex tie-break (1e-6).  Components: energy, unmet-energy penalty, demand charge.
    PYTHONPATH=. python scripts/lp_breakdown.py --site jpl --days 8"""
import argparse, datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--site", default="jpl"); ap.add_argument("--days", type=int, default=8); ap.add_argument("--out", default="docs/study/lp_breakdown_{site}.json"); a = ap.parse_args()
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
from pifsrl.env.datafiles import discover
from pifsrl.agents.trainer import make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.oracle import solve_oracle
from pifsrl.diffenv import extract_day, pad_batch
csv = f"data/{a.site}_local_sessions.csv"
dates = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
days = [dates[i] for i in sorted(set(np.linspace(0, len(dates) - 1, 20).astype(int)))][:a.days]
ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
ov["ev_load_scale"], ov["real_total_capacity_kw"] = 2.0, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * 2.0 * 2 / 3, 1)
kw = {"price_source": "sce_tou_prime"}; kw.update(discover(a.site, "data", price="auto", demand_from_tariff=True, battery_default=False))
ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=days[0], reward_mode="dense", demand_charge=1.5, **kw, **ov)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
cfg, S = env.cfg, env.s_base_kw

def price_plan(day, P, curv):
    b = pad_batch([day]); T, K = b["T"], b["K"]; dtt = cfg.dt_h
    soc = np.zeros(K); e = u = 0.0; peak = 0.0
    occ, st, en, cap, soc0, tgt, pr = [b[k][0].numpy() for k in ("occ", "start", "end", "cap_kwh", "soc0", "tgt", "price")]
    for t in range(T):
        soc = np.where(st[t], soc0[t], soc); p = P[t] * (occ[t] > .5)
        eta = cfg.eta_ch - curv * (soc - 0.5) ** 2; capph = np.maximum(cap[t], 1e-6) / S
        soc = np.where(occ[t] > .5, np.clip(soc + eta * p * dtt / np.maximum(capph, 1e-9), 0, 1), soc)
        e += float((p * S * dtt).sum() * pr[t]); u += float((en[t] * np.maximum(tgt[t] - soc, 0) * cap[t]).sum()); peak = max(peak, float(p.sum()))
        soc = soc * (occ[t] > .5)
    return dict(energy=e, unmet=cfg.c_unmet * u, demand=cfg.demand_charge * peak * S, total=e + cfg.c_unmet * u + cfg.demand_charge * peak * S)

rows = []
for d in days:
    day = extract_day(env, fbase, 90000, date=d); env.reset(90000, date=d)
    r = solve_oracle(env, fbase, tie_break=1e-6, no_discharge=True)
    P = np.maximum(r["p"], 0.0)
    rows.append(dict(date=d, lp_model_cost=r["lp_cost"], status=r["status"], plan_lp_eff=price_plan(day, P, 0.0), plan_sim_eff=price_plan(day, P, cfg.eta_curv)))
    print(d, round(r["lp_cost"], 1), {k: round(v, 1) for k, v in rows[-1]["plan_lp_eff"].items()}, "| sim eff", {k: round(v, 1) for k, v in rows[-1]["plan_sim_eff"].items()}, r["status"], flush=True)
M = lambda k, c: float(np.mean([x[k][c] for x in rows]))
print("MEAN lp_model %.1f | plan@LP-eff %.1f | plan@sim-eff %.1f" % (np.mean([x["lp_model_cost"] for x in rows]), M("plan_lp_eff", "total"), M("plan_sim_eff", "total")))
json.dump(rows, open(a.out.format(site=a.site), "w"), indent=1)
