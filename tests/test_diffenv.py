"""Differentiable twin == V2GEnv for identical actions; gradients are finite and non-trivial."""
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost


class _Replay:
    def __init__(self, P): self.P, self.i = P, 0
    def __call__(self):
        p = self.P[min(self.i, len(self.P) - 1)]; self.i += 1; return p


def _setup():
    ec = EnvConfig(feeder="ieee33", horizon=48, dt_h=0.25, seed=0, reward_mode="dense", demand_charge=1.5,
                   allow_discharge=False, allow_reactive=False, ev_load_scale=2.0)
    env, fbase, _, _ = make_components(ec, TD3Config(use_tightening=False), WMConfig(), "cpu")
    return env, fbase


class _Rand(torch.nn.Module):
    def __init__(self): super().__init__(); self.g = torch.Generator().manual_seed(3)
    def forward(self, feat, t): return torch.rand(feat.shape[:2], generator=self.g, dtype=torch.float64)


def test_twin_matches_env():
    env, fbase = _setup()
    days = [extract_day(env, fbase, s) for s in (11, 12)]
    out = rollout_cost(pad_batch(days), _Rand(), env.cfg, env.s_base_kw, env.cfg.demand_charge, env.cfg.c_unmet, return_traj=True)
    for i, s in enumerate((11, 12)):
        st = run_episode(env, _Replay(out["p"][i].numpy()), buffer=None, seed=s, collect=False)
        assert abs(st["total_cost"] - float(out["total"][i])) < 1e-6 * max(1.0, abs(st["total_cost"]))


def test_gradient_flows():
    env, fbase = _setup()
    b = pad_batch([extract_day(env, fbase, 5)])
    z = torch.zeros(1, b["T"], b["K"], dtype=torch.float64, requires_grad=True)

    class P(torch.nn.Module):
        def forward(self, feat, t): return torch.sigmoid(z[:, t])
    rollout_cost(b, P(), env.cfg, env.s_base_kw, env.cfg.demand_charge, env.cfg.c_unmet)["total"].sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


import pytest
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_twin_gpu_cpu_parity():
    from pifsrl.diffenv import extract_day, pad_batch, rollout_cost
    ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=False, allow_reactive=False, ev_load_scale=2.0)
    env, fbase, _, _ = make_components(ec, TD3Config(use_tightening=False, qp_iters=60), __import__("pifsrl.models.world_model", fromlist=["WMConfig"]).WMConfig(), "cpu")
    D = [extract_day(env, fbase, s) for s in (1, 2)]
    class Half(torch.nn.Module):
        def forward(self, feat, t): return torch.full(feat.shape[:2], 0.6, dtype=torch.float64, device=feat.device)
    c = rollout_cost(pad_batch(D), Half(), env.cfg, env.s_base_kw, 1.5, env.cfg.c_unmet)["total"]
    g = rollout_cost(pad_batch(D, device="cuda"), Half(), env.cfg, env.s_base_kw, 1.5, env.cfg.c_unmet)["total"].cpu()
    assert torch.allclose(c, g, rtol=1e-9)
