"""Physics correctness tests.  These pin down the numbers a reviewer would
check first: does the power flow solver reproduce published results, and is
the LinDistFlow linearisation error a genuine, non-trivial quantity."""
import numpy as np
import pytest

from pifsrl.physics.feeder import ieee33   # ieee123_like disabled (123-bus out of scope)
from pifsrl.physics.powerflow import linearisation_error, solve_bfs, solve_lindistflow


def test_ieee33_matches_baran_wu():
    """Baran & Wu (1989) / the standard reference solution for IEEE-33 at
    nominal load: V_min = 0.9131 p.u. at bus 18, active losses = 202.68 kW."""
    f = ieee33()
    p = -f.p_load[None, :]
    q = -f.q_load[None, :]
    res = solve_bfs(f, p, q)
    assert res.converged
    assert abs(res.v.min() - 0.9131) < 2e-3
    assert abs(res.v.argmin() + 2 - 18) <= 1          # bus 18 is the usual min (off-by-one tolerant)
    assert abs(res.p_loss[0] * 1000 - 202.68) < 2.0    # kW


def test_bfs_converges_tightly():
    f = ieee33()
    p, q = -f.p_load[None, :] * 1.3, -f.q_load[None, :] * 1.3   # stressed case
    res = solve_bfs(f, p, q, tol=1e-12, max_iter=100)
    assert res.max_mismatch < 1e-10
    assert res.iters < 30


def test_lindistflow_error_is_nontrivial_and_directionally_correct():
    """The whole Residual-Dyna premise requires this gap to be real: at nominal
    load LinDistFlow must UNDERSTATE the true voltage drop (v_lin > v_true),
    because it drops the (r^2+x^2) I^2 quadratic term, which is strictly a
    further voltage-lowering effect on a radial feeder with lagging loads."""
    f = ieee33()
    p, q = -f.p_load[None, :], -f.q_load[None, :]
    err = linearisation_error(f, p, q)
    assert err["v_mae_pu"] > 5e-4
    assert err["v_bias_pu"] < 0, "LinDistFlow should overestimate voltage (miss the I^2 Z drop)"


def test_feeder_thermal_ratings_positive():
    for name, f in [("33", ieee33())]:   # ("123", ieee123_like(seed=1)) disabled
        assert (f.s_max > 0).all(), name
        assert f.R.shape == (f.n_bus, f.n_bus)
        assert np.allclose(f.R, f.R.T), f"{name}: R must be symmetric (M^T diag M form)"
        assert np.all(np.linalg.eigvalsh(f.R) >= -1e-9), f"{name}: R must be PSD"


# DISABLED (professor's request): IEEE-123 bus work removed from scope. Code kept commented for reference.
# def test_123_bus_radial_and_connected():
#     f = ieee123_like(seed=0)
#     assert f.n_bus == 122          # 123 buses total minus the slack
#     assert f.M.shape == (f.n_branch, f.n_bus)
#     # every non-slack bus reachable: each column of M has >=1 nonzero (path to slack exists)
#     assert (f.M.sum(axis=0) > 0).all()


def test_lindistflow_matches_fbase_numpy_reference():
    """solve_lindistflow (numpy) must agree with the analytical formula used
    inside FBase (torch) -- cross-checked here without importing torch."""
    f = ieee33()
    p, q = -f.p_load[None, :] * 0.7, -f.q_load[None, :] * 0.7
    v_lin, s_lin = solve_lindistflow(f, p, q, v0=1.02)
    v_manual = 1.02 + p @ f.R.T + q @ f.X.T
    assert np.allclose(v_lin, v_manual, atol=1e-10)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
