"""QP safety layer tests: the ADMM solver against CVXPY/OSQP, implicit
gradients against finite differences, and the specific regressions caught
during development (see RUNBOOK.md 'ADMM convergence')."""
import numpy as np
import pytest
import torch

from pifsrl.safety.qp_layer import (SafetyPolytope, admm_project, build_polytope,
                                    project, reference_solve_cvxpy, vpw_weights_to_qp)

torch.manual_seed(0)


def _random_problem(B=4, n=10, m=6, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(B, m, n, generator=g, dtype=dtype) * 0.4
    lo = -torch.rand(B, n, generator=g, dtype=dtype) - 0.2
    hi = torch.rand(B, n, generator=g, dtype=dtype) + 0.2
    h = torch.rand(B, m, generator=g, dtype=dtype) * 0.3
    a_raw = torch.randn(B, n, generator=g, dtype=dtype) * 0.7
    w = torch.rand(B, n, generator=g, dtype=dtype) * 3 + 0.3
    return G, h, lo, hi, a_raw, w


def test_admm_matches_cvxpy_osqp():
    torch.set_default_dtype(torch.float64)
    G, h, lo, hi, a_raw, w = _random_problem()
    poly = SafetyPolytope(G, h, lo, hi)
    x, _, _, stats = admm_project(a_raw, poly, w=w, iters=800, tol=1e-10)
    errs = []
    for b in range(G.shape[0]):
        xr = reference_solve_cvxpy(a_raw[b].numpy(), G[b].numpy(), h[b].numpy(),
                                   lo[b].numpy(), hi[b].numpy(), w[b].numpy())
        errs.append(np.abs(xr - x[b].numpy()).max())
    assert max(errs) < 1e-5, f"ADMM should match CVXPY/OSQP to high precision, got {max(errs):.2e}"


def test_admm_solution_is_feasible():
    torch.set_default_dtype(torch.float64)
    G, h, lo, hi, a_raw, w = _random_problem(seed=1)
    poly = SafetyPolytope(G, h, lo, hi)
    x, _, _, _ = admm_project(a_raw, poly, w=w, iters=500, tol=1e-10)
    viol = (torch.bmm(G, x.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0).max()
    assert float(viol) < 1e-6
    assert (x >= lo - 1e-6).all() and (x <= hi + 1e-6).all()


def test_implicit_gradients_match_finite_differences():
    """The Eq. 14 gradient chain hinges on this.  Uses a NON-uniform weight
    matrix deliberately: with W=I the Jacobian is symmetric and a transpose
    bug in the backward pass is invisible; W != I catches it."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)
    B, n, m = 3, 10, 7
    G = torch.randn(B, m, n) * 0.5
    lo = -torch.rand(B, n) - 0.3
    hi = torch.rand(B, n) + 0.3
    h = torch.rand(B, m) * 0.25
    w = torch.rand(B, n) * 3 + 0.4
    a = (torch.randn(B, n) * 0.9).requires_grad_(True)
    hh = h.clone().requires_grad_(True)

    def f(a_, h_):
        p = SafetyPolytope(G, h_, lo, hi)
        return project(a_, p, w=w, iters=1500, act_tol=1e-6, skip_feasible=False)

    x = f(a, hh)
    gout = torch.randn_like(x)
    (x * gout).sum().backward()
    ga, gh = a.grad.clone(), hh.grad.clone()

    def fd(tensor, idx, eps=1e-6):
        t = tensor.detach().clone()
        t.view(-1)[idx] += eps
        xp = f(t, h) if tensor is a else f(a.detach(), t)
        t2 = tensor.detach().clone()
        t2.view(-1)[idx] -= eps
        xm = f(t2, h) if tensor is a else f(a.detach(), t2)
        return float(((xp - xm) * gout).sum() / (2 * eps))

    rng = np.random.default_rng(0)
    ia = rng.choice(B * n, 10, replace=False)
    ih = rng.choice(B * m, 10, replace=False)
    for i in ia:
        assert abs(fd(a, int(i)) - float(ga.view(-1)[i])) < 5e-4
    for i in ih:
        assert abs(fd(hh, int(i)) - float(gh.view(-1)[i])) < 5e-4


def test_skip_feasible_returns_identity_for_feasible_action():
    """Regression test for the exact bug found during development: an
    already-feasible action must come back unchanged (to within ADMM
    tolerance), not perturbed by a slow-converging solver.  See
    RUNBOOK.md 'ADMM convergence' for the full incident writeup."""
    torch.manual_seed(5)
    B, n_bus, k = 1, 8, 4
    R = torch.eye(n_bus) * 0.05 + 0.01
    X = torch.eye(n_bus) * 0.03 + 0.005
    M = torch.eye(n_bus)[: n_bus - 1].unsqueeze(0)[0]   # dummy branch map (not used heavily here)
    M = torch.zeros(n_bus - 1, n_bus)
    for i in range(n_bus - 1):
        M[i, i + 1] = 1.0
    A = torch.zeros(n_bus, k)
    A[[1, 2, 3, 4], range(k)] = 1.0
    p_exo = torch.full((B, n_bus), -0.02)
    q_exo = torch.full((B, n_bus), -0.01)
    s_max = torch.full((n_bus - 1,), 2.0)
    lo = torch.full((B, 2 * k), -0.02)
    hi = torch.full((B, 2 * k), 0.02)
    poly = build_polytope(R, X, M, A, p_exo, q_exo, s_max, lo, hi,
                          v_min=0.90, v_max=1.10, v0=1.0, n_cuts=4, allow_reactive=True)
    a_raw = torch.zeros(B, 2 * k)          # doing nothing is always feasible here
    out = project(a_raw, poly, iters=50)   # deliberately few iterations
    assert torch.allclose(out, a_raw, atol=1e-8), \
        "an already-feasible action must be returned unchanged, not perturbed by ADMM"


def test_vpw_weights_ordering():
    """Higher willingness -> lower QP deviation cost (Section 3.2, property 1)."""
    vpw = torch.tensor([[0.1, 0.9]])
    occ = torch.tensor([[1.0, 1.0]])
    w = vpw_weights_to_qp(vpw, occ, allow_reactive=False, enabled=True)
    assert w[0, 0] > w[0, 1], "low-willingness driver should have HIGHER deviation weight"


def test_infeasible_polytope_does_not_crash():
    """Section 4's slack relaxation: an empty hard feasible set must still
    return a finite, box-respecting action rather than raising or returning
    NaN/inf."""
    torch.manual_seed(0)
    B, n, m = 2, 6, 4
    G = torch.ones(B, m, n) * 5.0        # deliberately impossible to satisfy
    h = -torch.ones(B, m)                # G @ a <= h with a in a small box is infeasible
    lo = -torch.ones(B, n) * 0.1
    hi = torch.ones(B, n) * 0.1
    a_raw = torch.zeros(B, n)
    poly = SafetyPolytope(G, h, lo, hi, n_slack=0)
    # augment manually with slacks the way build_polytope would
    from pifsrl.safety.qp_layer import _augment_with_slacks
    G2, h2, lo2, hi2 = _augment_with_slacks(G, h, lo, hi, n_v_rows=m, slack_penalty=300.0)
    poly2 = SafetyPolytope(G2, h2, lo2, hi2, n_slack=2)
    out = project(a_raw, poly2, iters=300)
    assert torch.isfinite(out).all()
    assert (out >= lo - 1e-5).all() and (out <= hi + 1e-5).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_hard_first_matches_exact():
    """When idle (a=0) is feasible the hard projection (no slack columns) must
    reproduce the exact QP solution; the old slack-ADMM path did not converge in
    120 iterations and over-shrank the action (incident: SafeTD3 over-conservatism)."""
    torch.set_default_dtype(torch.float64)
    G, h, lo, hi, a_raw, w = _random_problem(B=6, n=12, m=8, seed=3)
    h = h.abs() + 0.02                                   # idle feasible
    poly = SafetyPolytope(G, h, lo, hi)
    x = project(a_raw, poly, w=w, iters=200, hard_first=True, rho_hard=0.1)
    for b in range(G.shape[0]):
        xr = reference_solve_cvxpy(a_raw[b].numpy(), G[b].numpy(), h[b].numpy(),
                                   lo[b].numpy(), hi[b].numpy(), w[b].numpy())
        assert np.abs(xr - x[b].numpy()).max() < 2e-3
    viol = (torch.bmm(G, x.unsqueeze(-1)).squeeze(-1) - h).clamp_min(0).max()
    assert float(viol) < 2e-3


def test_restore_feasibility_is_exact_and_idempotent():
    """The feasibility certificate shrinks toward idle until every row holds, and leaves a
    feasible point untouched."""
    from pifsrl.safety.qp_layer import _restore_feasibility
    torch.set_default_dtype(torch.float64)
    G, h, lo, hi, a_raw, w = _random_problem(B=5, n=9, m=7, seed=11)
    h = h.abs() + 0.01
    x = a_raw.clamp(lo, hi)
    y = _restore_feasibility(x, G, h, floor=1e-9, tol=1e-12)
    assert float((torch.bmm(G, y.unsqueeze(-1)).squeeze(-1) - h).amax()) < 1e-9
    assert (y >= lo - 1e-12).all() and (y <= hi + 1e-12).all()
    z = _restore_feasibility(y, G, h, floor=1e-9, tol=1e-12)
    assert torch.allclose(y, z)


def test_rowwise_margin_is_per_row_and_smaller_than_global_max():
    """Per-branch one-sided margins must not be dominated by the worst branch, and must
    cover that branch's own error with the requested frequency."""
    import numpy as np
    from pifsrl.safety.qp_layer import ResidualTightening
    rng = np.random.default_rng(0)
    n, nb = 4000, 6
    scale = np.array([0.5, 0.5, 1, 1, 2, 40.0])                  # one noisy 'trunk' branch
    s_true = rng.normal(size=(n, nb)) * scale + 100
    s_pred = np.full((n, nb), 100.0)
    v = np.zeros((n, 3))
    g = ResidualTightening(rowwise=False, alpha=0.05).fit(v, v, s_true, s_pred)
    r = ResidualTightening(rowwise=True, alpha=0.05).fit(v, v, s_true, s_pred)
    assert r.margin_s_vec[0] < 0.1 * g.margin_s                  # small branch not inflated by the trunk
    cover = (np.maximum(s_true - s_pred, 0.0)[:, 0] <= r.margin_s_vec[0]).mean()
    assert cover >= 0.94
