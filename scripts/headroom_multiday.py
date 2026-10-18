"""Learning headroom without any learning: per-day cost of safe_uncontrolled, EDF caps, best-per-day cap (clairvoyant
choice of ONE scalar), fixed cap tuned on a train split, and the hindsight oracle.  Tells us how much a learned
policy could possibly gain over a tuned heuristic.   python scripts/headroom_multiday.py --days 20 --demand 1.5 --stress"""
import argparse, datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pifsrl.env.v2g_env import EnvConfig, V2GEnv
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode

ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=20); ap.add_argument("--demand", type=float, default=1.5)
ap.add_argument("--site", default="caltech", choices=["caltech", "jpl"]); ap.add_argument("--auto-data", action="store_true"); ap.add_argument("--data-dir", default="data")
ap.add_argument("--hub-grid", action="store_true", help="also evaluate per-hub EDF caps (grid {0.4,0.6,0.8}^3) -> A2 kill test: per-hub oracle-choice vs best single cap")
ap.add_argument("--mpc-reserves", type=int, nargs="*", default=[]); ap.add_argument("--lmp-analog", nargs="?", const="month", default="", choices=["", "month", "any"]); ap.add_argument("--demand-from-tariff", action="store_true")
ap.add_argument("--battery-default", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--price-scenario", default=""); ap.add_argument("--price-seed", type=int, default=0)
ap.add_argument("--stress", action="store_true"); ap.add_argument("--ev-scale", type=float, default=2.0, help="with --stress: ev_load_scale"); ap.add_argument("--cap-frac", type=float, default=2/3, help="with --stress: hub capacity as a fraction of the site nameplate x ev-scale"); ap.add_argument("--wep-lmp", action="store_true", help="use data/caiso_sp15_lmp.csv (WEP CAISO hourly) as the energy price instead of the tariff"); ap.add_argument("--skip-oracle", action="store_true"); ap.add_argument("--out", default="docs/study/headroom_multiday.json")
a = ap.parse_args()
csv = f"data/{a.site}_local_sessions.csv"
dates = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
days = [dates[i] for i in sorted(set(np.linspace(0, len(dates) - 1, a.days).astype(int)))]
ov = recommended_env_overrides_for_acn(a.site)
from pifsrl.env.sessions import ACN_SITE_INFO
ov["k_max"] = 54
if a.stress: ov["ev_load_scale"], ov["real_total_capacity_kw"] = a.ev_scale, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * a.ev_scale * a.cap_frac, 1)
kw = {"price_source": "sce_tou_prime"}
if a.auto_data:
    from pifsrl.env.datafiles import discover
    kw.update(discover(a.site, a.data_dir, price="auto", demand_from_tariff=a.demand_from_tariff, battery_default=a.battery_default))
if a.wep_lmp: kw.update(price_source="caiso_rtp", price_csv_file=os.path.join(a.data_dir, "caiso_sp15_lmp.csv"))
if a.lmp_analog: kw["lmp_analog"] = a.lmp_analog
if a.price_scenario: kw["price_scenario"], kw["price_seed"] = a.price_scenario, a.price_seed
ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site,
               acn_date=days[0], reward_mode="dense", demand_charge=a.demand, **kw, **ov)
from pifsrl.agents.trainer import make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
# FIX (2026-10-15): `safe_*` rules must really go through the QP safety layer.  Earlier versions of this script ran them with agent=None,
# i.e. UNPROJECTED, so `safe_uncontrolled` violated the network on 60-75 % of stress days yet was admitted as the 'best heuristic'.
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), 'cpu')
caps = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
import itertools
hubs = [f"edf_hub_{x}_{y}_{z}" for x, y, z in itertools.product([0.4, 0.6, 0.8], repeat=3)] if a.hub_grid else []
mpcs = [("mpc" if rs == 0 else f"mpc_r{rs}") for rs in a.mpc_reserves]
rows = []
for d in days:
    r = {"date": d}
    for kind in ["uncontrolled", "safe_uncontrolled"] + [f"edf_cap_{c}" for c in caps] + ["oa_peak", "oracle_lp"] + hubs + mpcs:
        pol = RuleBasedPolicy(kind, env, seed=1, fbase=fbase)
        st = run_episode(env, pol, buffer=None, seed=90000, collect=False, date=d, agent=(agent if kind.startswith("safe_") else None))
        r[kind] = [st["total_cost"], st["served_frac"], st["v_viol"] + st["s_viol"]]
    rows.append(r); print(d, {k: round(v[0], 1) for k, v in r.items() if k != "date"}, flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(rows, open(a.out, "w"), indent=1)
C = lambda k: np.array([r[k][0] for r in rows])
V = lambda k: np.array([r[k][2] for r in rows])
feas = [c for c in caps if V(f"edf_cap_{c}").mean() <= 1e-6] or caps          # only violation-free caps count as a deployable baseline
print("violation-free caps:", feas)
# every candidate must be violation-free ON THE DAYS EVALUATED (mean viol <= 1e-6); an unprojected rule never qualifies
ok_unc = [k for k in ["safe_uncontrolled"] if V(k).mean() <= 1e-6]
best_day = np.min([C(f"edf_cap_{c}") for c in feas] + [C(k) for k in ok_unc], axis=0)     # safe_uncontrolled is a candidate heuristic too
if hubs:
    best_hub_day = np.min([C(h) for h in hubs], axis=0); best_fixed_hub = min(hubs, key=lambda h: C(h).mean())
    o = C("oracle_lp"); b1 = min([C(f"edf_cap_{c}").mean() for c in feas] + [C(k).mean() for k in ok_unc])
    print(f"A2 per-hub: best fixed single cap {b1:.1f} | best fixed per-hub triple {best_fixed_hub} {C(best_fixed_hub).mean():.1f} | per-day best triple (clairvoyant) {best_hub_day.mean():.1f} | oracle {o.mean():.1f}")
    print(f"   per-hub gain over single cap (fixed): {(b1 - C(best_fixed_hub).mean()) / b1:+.1%}   kill criterion A2: < 5% -> drop per-hub control")
print("MEAN cost  safe_uncontrolled %.1f | fixed best cap %s | per-day best cap %.1f | oracle %.1f" % (
    C("safe_uncontrolled").mean(), {c: round(C(f"edf_cap_{c}").mean(), 1) for c in caps}, best_day.mean(), C("oracle_lp").mean()))
