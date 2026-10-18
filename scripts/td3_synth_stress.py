"""Residual SafeTD3 (critic-based, QP-projected) on the SAME synthetic stress days as scripts/diff_env_poc.py -> fills the 'SafeTD3 @ high headroom' cell.
PYTHONPATH=. python scripts/td3_synth_stress.py --episodes 120 --seed 0
NOTE: TD3 sees a fresh random synthetic day every episode (effectively unlimited training days); the pathwise policy sees 30 fixed days."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.trainer import TrainConfig, train_site, evaluate_agent, evaluate_rule, tune_edf_cap
ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=120); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--residual", type=float, default=0.3); ap.add_argument("--lambda-mb", type=float, default=0.0, help="1-step Dyna weight (physics-informed world-model loss on the actor; 0 = model-free)"); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default="docs/study/td3_synth_stress.json"); a = ap.parse_args()
torch.set_num_threads(1)
if a.device == 'cuda' and not torch.cuda.is_available(): raise SystemExit('--device cuda requested but CUDA is unavailable')
ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=a.seed, reward_mode="dense", demand_charge=1.5, allow_discharge=False, allow_reactive=False, ev_load_scale=2.0)
from pifsrl.env.v2g_env import V2GEnv
env0 = V2GEnv(ec)
bestc, tab = tune_edf_cap(env0, [None] * 30, seed0=1000)                       # train seeds 1000..1029, as in the pathwise run
print("tuned cap", bestc, flush=True)
td3 = TD3Config(residual_scale=a.residual, use_tightening=True, lambda_mb=a.lambda_mb)
tc = TrainConfig(episodes=a.episodes, warmup_episodes=8, eval_every=20, eval_episodes=5, base_kind=f"edf_cap_{bestc}", seed=a.seed, warmup_mode="greedy",
                 final_test_episodes=5, log_every=10, device=a.device)
t0 = time.time(); out = train_site(ec, td3, WMConfig(), tc, verbose=True); env, agent = out["env"], out["agent"]
te = 20; seed0 = 2000
res = {"lambda_mb": a.lambda_mb, "seed": a.seed, "cap": bestc, "episodes": a.episodes, "wall_s": time.time() - t0, "best_ep": out["best_ep"]}
res["ref"] = evaluate_rule(env, f"edf_cap_{bestc}", te, seed0=seed0)
res["last"] = evaluate_agent(env, agent, te, seed0=seed0, base_kind=f"edf_cap_{bestc}")
if out["best_state"] is not None:
    agent.load_state_dict(out["best_state"]); res["best"] = evaluate_agent(env, agent, te, seed0=seed0, base_kind=f"edf_cap_{bestc}")
for k in ("ref", "last", "best"):
    if k in res: print(k, round(res[k]["total_cost"], 1), "viol", res[k]["v_viol"] + res[k]["s_viol"], "served", round(res[k]["served_frac"], 3))
json.dump(res, open(a.out, "w"), indent=1, default=float)
