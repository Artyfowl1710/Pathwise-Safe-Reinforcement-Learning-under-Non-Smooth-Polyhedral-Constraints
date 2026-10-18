"""Bit-exact reproducibility checks (CPU): same seed -> identical numbers. These are what a reader should run first."""
import copy, subprocess, sys, os, json
import numpy as np
import torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.agents.trainer import TrainConfig, train_site, evaluate_rule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train(seed):
    ec = EnvConfig(k_max=6, archetype="residential", cluster_size=10.0, horizon=24, dt_h=0.5, seed=0, session_density=1.2, price_scenario="volatile")
    tc = TD3Config(hidden=32, depth=1, qp_iters=25, batch_size=16, n_thermal_cuts=4)
    trc = TrainConfig(episodes=3, warmup_episodes=1, eval_every=3, eval_episodes=1, wm_updates_per_episode=5, log_every=1, seed=seed, device="cpu")
    out = train_site(ec, tc, WMConfig(hidden=32, depth=1), trc, verbose=False)
    return out["evals"][-1]["total_cost"], {k: v.clone() for k, v in out["agent"].state_dict().items()}


def test_training_is_bit_reproducible_on_cpu():
    c1, s1 = _train(7); c2, s2 = _train(7); c3, s3 = _train(8)
    assert c1 == c2 and all(torch.equal(s1[k], s2[k]) for k in s1)
    assert any(not torch.equal(s1[k], s3[k]) for k in s1)             # a different seed really changes the run


def test_rule_baselines_are_deterministic_given_price_scenario():
    ec = EnvConfig(k_max=6, archetype="residential", cluster_size=10.0, horizon=24, dt_h=0.5, seed=0, price_scenario="solar_glut")
    from pifsrl.env.v2g_env import V2GEnv
    e = V2GEnv(ec)
    a = evaluate_rule(e, "uncontrolled", 2, seed0=5)["total_cost"]; b = evaluate_rule(e, "uncontrolled", 2, seed0=5)["total_cost"]
    assert a == b


def test_conformal_claim_reproduces_small():
    """Miniature of scripts/adaptive_conformal_shift.py: the headline qualitative claim must hold exactly and deterministically."""
    from tests.scripts_import import run_small
    r1, r2 = run_small(), run_small()
    assert r1 == r2                                                    # deterministic
    assert r1["fixed"] < 0.5 and abs(r1["aci"] - 0.95) < 0.03
