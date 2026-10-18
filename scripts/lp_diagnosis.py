"""Why does the TOU-EV-4 oracle LP report 'solution may be inaccurate' and negative headroom?
For each day: solve the oracle under (a) default Clarabel, (b) tightened tolerances, (c) tie-break regulariser, (d) no discharge;
report solver status, LP-model cost, and the SIMULATOR cost of executing each plan.  Also the same LP model's cost
with the SoC-efficiency taper switched off in the simulator (cfg flag if available) to isolate the model mismatch.
   PYTHONPATH=. python scripts/lp_diagnosis.py --site jpl --days 8"""
import argparse, datetime as dt, json, os, sys, warnings, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--site", default="jpl"); ap.add_argument("--days", type=int, default=8)
ap.add_argument("--out", default="docs/study/lp_diagnosis_{site}.json"); a = ap.parse_args()
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
from pifsrl.env.datafiles import discover
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.agents import oracle as O
csv = f"data/{a.site}_local_sessions.csv"
dates = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
days = [dates[i] for i in sorted(set(np.linspace(0, len(dates) - 1, 20).astype(int)))][:a.days]
ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
ov["ev_load_scale"], ov["real_total_capacity_kw"] = 2.0, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * 2.0 * 2 / 3, 1)
kw = {"price_source": "sce_tou_prime"}; kw.update(discover(a.site, "data", price="auto", demand_from_tariff=True, battery_default=False))
ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site,
               acn_date=days[0], reward_mode="dense", demand_charge=1.5, **kw, **ov)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
variants = {"default": {}, "tight_tol": dict(solver_opts=dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500)),
            "tiebreak_1e-6": dict(tie_break=1e-6), "tiebreak_1e-6+tight": dict(tie_break=1e-6, solver_opts=dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500)),
            "no_discharge": dict(no_discharge=True), "nodis_tie": dict(no_discharge=True, tie_break=1e-6),
            "nodis_tie_derate": dict(no_discharge=True, tie_break=1e-6, eta_derate=0.0132)}
import os
if os.environ.get('ONLY'): variants = {k: v for k, v in variants.items() if k in os.environ['ONLY'].split(',')}
class Plan:
    def __init__(s, kw): s.kw = kw
rows = []
for d in days:
    env.reset(seed=90000, date=d) if hasattr(env, "reset") else None
    r = {"date": d}
    # reference heuristic cost (same as the headroom run)
    ref = RuleBasedPolicy("safe_uncontrolled", env, seed=1, fbase=fbase)
    st = run_episode(env, ref, buffer=None, seed=90000, collect=False, date=d, agent=agent); r["safe_uncontrolled_sim"] = st["total_cost"]
    for name, k in variants.items():
        pol = RuleBasedPolicy("oracle_lp", env, seed=1, fbase=fbase)
        orig = O.solve_oracle; box = {}
        def wrapped(env_, fb_, _k=k, _o=orig, _b=box, **x):
            buf = io.StringIO()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always"); res = _o(env_, fb_, **_k, **x)
            _b["warn"] = [str(i.message)[:80] for i in w]; _b["res"] = res; return res
        O.solve_oracle = wrapped
        try:
            st = run_episode(env, pol, buffer=None, seed=90000, collect=False, date=d, agent=None)
        finally:
            O.solve_oracle = orig
        res = box.get("res") or {}
        r[name] = dict(sim_cost=st["total_cost"], lp_cost=res.get("lp_cost"), status=res.get("status"), iters=res.get("iters"), warn=box.get("warn"), viol=st["v_viol"] + st["s_viol"], served=st["served_frac"])
    rows.append(r); print(d, {k: (round(v["sim_cost"], 1), round(v["lp_cost"], 1) if v["lp_cost"] else None, v["status"]) for k, v in r.items() if isinstance(v, dict)}, "ref", round(r["safe_uncontrolled_sim"], 1), flush=True)
out = (os.environ.get("OUT") or a.out).format(site=a.site); json.dump(rows, open(out, "w"), indent=1)
M = lambda k, f: float(np.mean([x[k][f] for x in rows]))
print("MEAN sim cost: ref %.1f | " % np.mean([x["safe_uncontrolled_sim"] for x in rows]) + " | ".join(f"{k} sim {M(k,'sim_cost'):.1f} lp {M(k,'lp_cost'):.1f}" for k in variants))
