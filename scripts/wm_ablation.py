"""Residual world model ablation (proposal Sec. 7.2, ablation 1):  F_base only  vs  F_base + N_theta (residual)  vs  black-box.
Question: how well does each predict the NEXT-STEP voltage / branch flow / SoC of the exact-AC simulator, as a function of the number of real transitions?
Transitions come from V2GEnv under several tuned-EDF caps + random actions (not from a trained agent), split by DAY into train / held-out.
    PYTHONPATH=. python scripts/wm_ablation.py --device cuda --out docs/study/wm_ablation.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.replay import ReplayBuffer
from pifsrl.agents.trainer import make_components, run_episode, buffer_dims
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig, WorldModel

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--train-days", type=int, default=30); ap.add_argument("--test-days", type=int, default=10)
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 2880]); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--stress", type=float, default=2.0)
ap.add_argument("--out", default="docs/study/wm_ablation.json"); a = ap.parse_args()
if a.device == "cuda" and not torch.cuda.is_available(): sys.exit("CUDA requested but unavailable")
ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=False, allow_reactive=False, ev_load_scale=a.stress)
env, fbase, _, _ = make_components(ec, TD3Config(use_tightening=False), WMConfig(), "cpu")
feeder = env.feeder
kinds = ["edf_cap_1.0", "edf_cap_0.6", "edf_cap_0.4", "uncontrolled"]


def collect(seeds):
    buf = ReplayBuffer(200000, buffer_dims(env), "cpu")
    for i, s in enumerate(seeds):
        pol = RuleBasedPolicy(kinds[i % len(kinds)], env, seed=s, fbase=fbase)
        run_episode(env, pol, buffer=buf, seed=s, collect=True)
    return buf


t0 = time.time(); btr = collect(range(1000, 1000 + a.train_days)); bte = collect(range(2000, 2000 + a.test_days))
print(f"collected {len(btr)} train / {len(bte)} test transitions in {time.time()-t0:.0f}s", flush=True)
test = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in bte.all().items()}
res = []
for mode in ("fbase_only", "residual", "blackbox"):
    for n in a.sizes:
        for sd in a.seeds:
            if mode == "fbase_only" and sd != a.seeds[0]: continue          # deterministic, no training
            torch.manual_seed(sd); rng = np.random.default_rng(sd)
            wm = WorldModel(fbase.to(a.device), feeder, WMConfig(mode=mode), allow_reactive=False).to(a.device)
            idx = rng.choice(len(btr), min(n, len(btr)), replace=False)
            sub = btr.sample(0, idx=idx); sub = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in sub.items()}
            m = len(idx)
            for it in range(a.steps if mode != "fbase_only" else 0):
                b = rng.integers(0, m, size=min(256, m)); wm.update({k: v[b] for k, v in sub.items()})
            ev = wm.evaluate(test); ev.update(mode=mode, n=int(m), seed=sd); res.append(ev)
            print(mode, m, sd, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in ev.items() if k in ("v_mae", "v_max_ae", "s_mae", "soc_mae")}, flush=True)
fbase.to("cpu")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"config": vars(a), "n_train": len(btr), "n_test": len(bte), "rows": res, "note": "targets are the exact-AC simulator's next-step v (p.u.), branch apparent power (p.u.) and SoC; held-out days"}, open(a.out, "w"), indent=1)
print("saved", a.out)
