"""Does bidirectional V2G pay?  Same days, same cost (energy + unmet + degradation + demand charge), discharge off vs on.
Hindsight LP oracle (oracle_lp) with and without discharge, and the causal v2g_greedy rule vs tou_greedy.   PYTHONPATH=. python scripts/v2g_discharge_check.py --days 8
Synthetic sessions and the SCE TOU stand-in price (default env); not real data."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=8); ap.add_argument("--stress", type=float, default=1.0); ap.add_argument("--out", default="docs/study/v2g_discharge_check.json"); a = ap.parse_args()
R = {}
for dis in (False, True):
    ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=dis, allow_reactive=False, ev_load_scale=a.stress)
    env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
    for kind in ("oracle_lp", "tou_greedy", "v2g_greedy"):
        if kind == "v2g_greedy" and not dis: continue
        c, v = [], []
        for s in range(2000, 2000 + a.days):
            st = run_episode(env, RuleBasedPolicy(kind, env, seed=s, fbase=fbase), buffer=None, seed=s, collect=False)
            c.append(st["total_cost"]); v.append(st["v_viol"] + st["s_viol"])
        R[f"{kind}|discharge={dis}"] = {"mean": float(np.mean(c)), "viol": float(np.mean(v)), "per_day": c}
        print(kind, "discharge", dis, round(float(np.mean(c)), 2), "viol", round(float(np.mean(v)), 5), flush=True)
o0, o1 = R["oracle_lp|discharge=False"]["mean"], R["oracle_lp|discharge=True"]["mean"]
t0, g1 = R["tou_greedy|discharge=False"]["mean"], R["v2g_greedy|discharge=True"]["mean"]
R["summary"] = {"oracle_gain_usd_per_day": o0 - o1, "oracle_gain_rel": (o0 - o1) / o0, "v2g_greedy_vs_tou_greedy_rel": (g1 - t0) / t0}
print(R["summary"]); json.dump(R, open(a.out, "w"), indent=1)
