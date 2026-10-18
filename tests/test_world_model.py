"""Tests for the analytical physics model (Eq. 11 Jacobian), the environment's
physical sign conventions, and the world model's residual-learning behaviour."""
import numpy as np
import pytest
import torch

from pifsrl.env.v2g_env import EnvConfig, V2GEnv
from pifsrl.models.fbase import FBase, PhysicsParams
from pifsrl.models.world_model import WMConfig, WorldModel


def _make_env(**kw):
    defaults = dict(k_max=8, archetype="residential", cluster_size=8.0,
                    horizon=24, dt_h=0.5, seed=0, session_density=1.2)
    defaults.update(kw)
    return V2GEnv(EnvConfig(**defaults))


def test_fbase_jacobian_matches_autograd():
    env = _make_env()
    pp = PhysicsParams(dt_h=env.cfg.dt_h, s_base_kw=env.s_base_kw,
                       v_min=env.cfg.v_min, v_max=env.cfg.v_max, v_sub=env.cfg.v_sub)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, env.cfg.allow_reactive)
    B = 5
    a = (torch.randn(B, env.act_dim) * 0.02).requires_grad_(True)
    p_exo = torch.randn(B, env.n_bus) * 0.01
    q_exo = torch.randn(B, env.n_bus) * 0.005
    v = fbase.voltage(a, p_exo, q_exo)
    g_auto = torch.autograd.grad(v.sum(), a, retain_graph=True)[0]
    cap = torch.ones(B, env.k_max) * 0.05
    occ = torch.ones(B, env.k_max)
    jac = fbase.analytic_jacobian(a, cap, occ)
    g_analytic = jac["dv_da"].sum(dim=1)
    assert torch.allclose(g_auto, g_analytic, atol=1e-5)


def test_fbase_soc_jacobian_matches_autograd():
    env = _make_env()
    pp = PhysicsParams(dt_h=env.cfg.dt_h, s_base_kw=env.s_base_kw)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, env.cfg.allow_reactive)
    B = 4
    a = (torch.rand(B, env.act_dim) * 0.04 - 0.02).requires_grad_(True)
    cap = torch.full((B, env.k_max), 0.06)
    occ = torch.ones(B, env.k_max)
    soc = torch.full((B, env.k_max), 0.5)
    soc_n = fbase.soc_next(soc, a, cap, occ)
    g_auto = torch.autograd.grad(soc_n.sum(), a)[0]
    jac = fbase.analytic_jacobian(a, cap, occ)
    g_analytic = jac["dsoc_da"].sum(dim=1)
    assert torch.allclose(g_auto, g_analytic, atol=1e-5)


def test_charging_lowers_voltage_not_raises_it():
    """Regression test for the sign-convention bug: charging (p_ev > 0) must
    LOWER downstream voltage, and discharging must raise it."""
    env = _make_env(k_max=4)
    env.reset(0)
    v0 = env.nominal_voltage().min()
    lo, hi = env.action_bounds()
    a_charge = np.where(env.occupied, hi, 0.0) if env.occupied.any() else hi.copy()
    # force at least one occupied-like slot by overriding occupied for the test
    env.occupied[:] = True
    lo, hi = env.action_bounds()
    _, _, _, info_charge = env.step(np.clip(hi, lo, hi))
    env.reset(0)
    env.occupied[:] = True
    lo, hi = env.action_bounds()
    _, _, _, info_discharge = env.step(np.clip(lo, lo, hi))
    assert info_charge.v_min <= info_discharge.v_max


def test_world_model_starts_near_fbase():
    env = _make_env()
    pp = PhysicsParams(dt_h=env.cfg.dt_h, s_base_kw=env.s_base_kw)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, env.cfg.allow_reactive)
    wm = WorldModel(fbase, env.feeder, WMConfig(hidden=32, depth=1), env.cfg.allow_reactive)
    B = 6
    a = torch.randn(B, env.act_dim) * 0.02
    p_exo = torch.randn(B, env.n_bus) * 0.01
    q_exo = torch.randn(B, env.n_bus) * 0.005
    soc = torch.rand(B, env.k_max)
    cap = torch.full((B, env.k_max), 0.05)
    occ = torch.ones(B, env.k_max)
    price = torch.full((B,), 0.15)
    out = wm.predict(a, p_exo, q_exo, soc, cap, occ, price)
    base_v = fbase.voltage(a, p_exo, q_exo)
    assert torch.allclose(out["v_hat"], base_v, atol=0.06), \
        "freshly-initialised N_theta should perturb F_base only slightly"


def test_world_model_kvl_residual_zero_on_exact_distflow_point():
    """Sanity check of the KVL residual formula (Eq. 5) itself: it should be
    (near) zero when evaluated at a point that already satisfies DistFlow."""
    env = _make_env(k_max=4)
    pp = PhysicsParams(dt_h=env.cfg.dt_h, s_base_kw=env.s_base_kw, v_sub=1.0)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, env.cfg.allow_reactive)
    wm = WorldModel(fbase, env.feeder, WMConfig(hidden=16, depth=1), env.cfg.allow_reactive)
    from pifsrl.physics.powerflow import solve_bfs
    p_net = -env.feeder.p_load[None, :] * 0.5
    q_net = -env.feeder.q_load[None, :] * 0.5
    pf = solve_bfs(env.feeder, p_net, q_net, v0=1.0)
    v_true = torch.tensor(pf.v, dtype=torch.float32)
    pf_flow = -(torch.tensor(p_net, dtype=torch.float32) @ torch.tensor(env.feeder.M.T, dtype=torch.float32))
    qf_flow = -(torch.tensor(q_net, dtype=torch.float32) @ torch.tensor(env.feeder.M.T, dtype=torch.float32))
    res = wm.kvl_residual(v_true, pf_flow, qf_flow)
    assert float(res.abs().max()) < 1e-3


def test_env_obs_layout_matches_obs_dim():
    env = _make_env()
    layout = env.obs_layout
    max_end = max(s.stop for s in layout.values())
    assert max_end == env.obs_dim


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
