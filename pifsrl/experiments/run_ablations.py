"""
Section 7.2 ablation runner: the four ablations named in the proposal, each
isolating one architectural component, run across multiple seeds so results
come with a variance estimate rather than a single anecdotal run.

    1. F_base-only vs. F_base + N_theta           (--ablation world_model)
    2. Soft-penalty Safe RL vs. QP hard-safety     (--ablation safety)
    3. FedAvg vs. personalized FedProx             (--ablation federated)
    4. With vs. without VPW prioritisation         (--ablation vpw)

Each ablation trains the SAME environment/seeds under two (or more) arms that
differ in exactly one switch, so the comparison isolates that component rather
than confounding it with unrelated hyperparameter differences.

Usage
-----
    python -m pifsrl.experiments.run_ablations --ablation world_model --seeds 0 1 2 --episodes 150
    python -m pifsrl.experiments.run_ablations --ablation all --seeds 0 1 2 --episodes 150
"""
from __future__ import annotations

from ..runlog import say

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch

from ..env.sessions import recommended_horizon
from ..env.v2g_env import EnvConfig
from ..models.world_model import WMConfig
from ..agents.safe_td3 import TD3Config
from ..agents.trainer import TrainConfig, evaluate_rule, train_site
from ..federated.fedprox import ClientSpec, FedClient, FedConfig, client_drift, run_federated


def _base_env_cfg(seed: int, archetype: str = "residential", k_max: int = 16,
                  cluster_size: float = 10.0, dt_h: float = 0.5) -> EnvConfig:
    h = recommended_horizon(archetype, dt_h)
    return EnvConfig(archetype=archetype, k_max=k_max, cluster_size=cluster_size,
                     horizon=h, dt_h=dt_h, seed=seed, session_density=1.4)


def _mean_std(rows: List[dict], key: str):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


# --------------------------------------------------------------------------- #
def ablation_world_model(seeds, episodes, device, out_dir, **kw) -> Dict:
    """F_base-only vs. residual (F_base + N_theta) vs. black-box surrogate."""
    results = {"residual": [], "fbase_only": [], "blackbox": []}
    for seed in seeds:
        ec = _base_env_cfg(seed)
        for mode in results:
            tc = TD3Config(lambda_mb=(0.5 if mode != "fbase_only" else 0.0))
            wc = WMConfig(mode=mode)
            trc = TrainConfig(episodes=episodes, device=device, seed=seed, eval_episodes=5)
            print(f"[world_model] seed={seed} mode={mode}")
            t0 = time.time()
            out = train_site(ec, tc, wc, trc, verbose=False)
            wall = time.time() - t0
            ev = out["evals"][-1]
            ev["wall_s"] = wall
            ev["seed"] = seed
            # world-model-quality metrics from training history (final episode)
            last_hist = out["history"][-1]
            ev["wm_v_mae"] = last_hist.get("wm_v_mae", float("nan"))
            results[mode].append(ev)
    summary = {mode: {k: _mean_std(rows, k) for k in
                      ["total_cost", "v_viol", "s_viol", "served_frac", "wm_v_mae", "wall_s"]}
              for mode, rows in results.items()}
    _save(out_dir, "ablation_world_model", {"raw": results, "summary": summary})
    return summary


def ablation_safety(seeds, episodes, device, out_dir, **kw) -> Dict:
    """Soft-penalty Safe RL (QP disabled, penalty in reward) vs. QP hard safety."""
    results = {"qp_hard": [], "soft_penalty": []}
    for seed in seeds:
        ec = _base_env_cfg(seed)
        for mode in results:
            soft = (mode == "soft_penalty")
            tc = TD3Config(use_qp=not soft)
            wc = WMConfig()
            trc = TrainConfig(episodes=episodes, device=device, seed=seed, eval_episodes=5,
                              soft_penalty=soft)
            print(f"[safety] seed={seed} mode={mode}")
            out = train_site(ec, tc, wc, trc, verbose=False)
            ev = out["evals"][-1]
            ev["seed"] = seed
            results[mode].append(ev)
    summary = {mode: {k: _mean_std(rows, k) for k in
                      ["total_cost", "v_viol", "s_viol", "served_frac", "n_v_steps"]}
              for mode, rows in results.items()}
    _save(out_dir, "ablation_safety", {"raw": results, "summary": summary})
    return summary


def ablation_federated(seeds, episodes, device, out_dir, n_clients: int = 5, **kw) -> Dict:
    """FedAvg vs. personalized FedProx across Non-IID archetype clients."""
    archetypes = ["workplace_caltech", "workplace_jpl", "office", "residential", "fleet_depot"]
    results = {"fedavg": [], "fedprox": [], "local": []}
    rounds = max(4, episodes // 5)
    for seed in seeds:
        specs = [ClientSpec(a, _base_env_cfg(seed + i, archetype=a, k_max=12))
                for i, a in enumerate(archetypes[:n_clients])]
        for algo in results:
            print(f"[federated] seed={seed} algo={algo}")
            clients = [FedClient(s, TD3Config(), WMConfig(), device) for s in specs]
            fc = FedConfig(rounds=rounds, local_episodes=2, mu=0.05, algorithm=algo, eval_every=rounds)
            out = run_federated(clients, fc, verbose=False)
            final_eval = out["evals"][-1]
            costs = [final_eval[c.spec.name]["total_cost"] for c in clients]
            served = [final_eval[c.spec.name]["served_frac"] for c in clients]
            drift = client_drift(clients)
            results[algo].append({"seed": seed, "mean_cost": float(np.mean(costs)),
                                  "cost_spread": float(np.std(costs)),
                                  "mean_served": float(np.mean(served)), "drift": drift})
    summary = {algo: {k: _mean_std(rows, k) for k in ["mean_cost", "cost_spread", "mean_served", "drift"]}
              for algo, rows in results.items()}
    _save(out_dir, "ablation_federated", {"raw": results, "summary": summary})
    return summary


def ablation_vpw(seeds, episodes, device, out_dir, **kw) -> Dict:
    """With vs. without VPW-weighted QP prioritisation, evaluated against the
    FCFS/SoC-only baseline noted in Section 5.2 (here: the uncontrolled and
    tou_greedy rule baselines serve as that reference point)."""
    results = {"vpw_on": [], "vpw_off": []}
    for seed in seeds:
        ec = _base_env_cfg(seed)
        for mode in results:
            tc = TD3Config(use_vpw_weights=(mode == "vpw_on"))
            wc = WMConfig()
            trc = TrainConfig(episodes=episodes, device=device, seed=seed, eval_episodes=5)
            print(f"[vpw] seed={seed} mode={mode}")
            out = train_site(ec, tc, wc, trc, verbose=False)
            ev = out["evals"][-1]
            ev["seed"] = seed
            # also record the discomfort-weighted cost, VPW's actual target quantity
            ev["mean_vpw"] = out["history"][-1].get("mean_vpw", float("nan"))
            results[mode].append(ev)
        # rule baselines for reference (FCFS proxy = uncontrolled, SoC-only proxy = tou_greedy)
        from ..agents.trainer import make_components
        env, _, _, _ = make_components(ec, TD3Config(), WMConfig(), device)
        for kind in ["uncontrolled", "tou_greedy"]:
            r = evaluate_rule(env, kind, 5, seed0=90000)
            results.setdefault(f"baseline_{kind}", []).append({**r, "seed": seed})
    summary = {mode: {k: _mean_std(rows, k) for k in
                      ["total_cost", "v_viol", "s_viol", "served_frac", "cost_discomfort"]
                      if k in rows[0]}
              for mode, rows in results.items()}
    _save(out_dir, "ablation_vpw", {"raw": results, "summary": summary})
    return summary


def _save(out_dir: str, name: str, payload: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, "w") as fh:
        json.dump(clean(payload), fh, indent=2)
    say(f"Wrote {path}")


ABLATIONS = {
    "world_model": ablation_world_model,
    "safety": ablation_safety,
    "federated": ablation_federated,
    "vpw": ablation_vpw,
}


def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--ablation", choices=list(ABLATIONS) + ["all"], default="all")
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        ap.add_argument("--episodes", type=int, default=150)
        ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        ap.add_argument("--out", default="results/ablations")
        args = ap.parse_args()
    # Called from sample.py: that CLI exposes `--ablation-episodes`/`--seeds`/
    # `--device`/`--out` but not `--ablation` itself (there is no per-ablation
    # selector flag there), so default to running the full grid.  This
    # function must NEVER fall back to parsing sys.argv when an args object is
    # already provided -- sample.py's own flags (e.g. --mode) would not be
    # recognised by this module's parser and the process would exit(2).
    name = getattr(args, "ablation", "all")
    episodes = getattr(args, "ablation_episodes", None) or getattr(args, "episodes", 150)
    seeds = getattr(args, "seeds", [0, 1, 2])
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = getattr(args, "out", "results/ablations")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    todo = list(ABLATIONS) if name == "all" else [name]
    all_summaries = {}
    for n in todo:
        say(f"\nAblation: {n}  (seeds={seeds}, episodes={episodes}, device={device})")
        summary = ABLATIONS[n](seeds, episodes, device, out_dir)
        all_summaries[n] = summary
        print(json.dumps(summary, indent=2, default=float))
    return all_summaries


if __name__ == "__main__":
    main()
