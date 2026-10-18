"""
Hard-constraint safety layer (Section 4): a differentiable Euclidean projection
of the raw actor action onto the physically safe polytope.

    a_safe = argmin_a  1/2 || a - a_raw ||^2_W
             s.t.      v_min <= v0 + R p_net(a) + X q_net(a) <= v_max      (Eq. 17)
                       |S_ij(a)| <= S_ij,max                               (Eq. 18)
                       lo <= a <= hi                (rating + SoC feasibility)

IMPLEMENTATION CHOICES AND WHY
------------------------------
1.  **Own solver, not a CVXPYLayers dependency.**  CVXPYLayers/diffcp is a
    per-problem Python-level solve and it is the single biggest wall-clock risk
    in the proposal (Section 4.1 exists because of it).  We implement an
    OSQP-style ADMM that is *batched in PyTorch*: one Cholesky of an
    (n_a x n_a) matrix per batch element, then pure BLAS iterations.  A whole
    minibatch of 256 projections is one set of batched tensor ops, which is what
    makes model-based Dyna training with a safety layer in the loop affordable
    on a 6 GB GPU.  `reference_solve_cvxpy()` cross-checks against CVXPY/OSQP to
    machine precision in the unit tests, so the fast path is *validated*, not
    merely asserted.

2.  **Implicit differentiation, not unrolling.**  Backward uses the KKT system
    of the active set (OptNet, Amos & Kolter 2017).  For a projection with
    quadratic form W this has the closed form

        da*/da_raw = I - W^-1 C_A^T (C_A W^-1 C_A^T)^-1 C_A
        da*/db_A   =      W^-1 C_A^T (C_A W^-1 C_A^T)^-1

    where C_A are the active rows.  The first term is what lets the actor "feel"
    the boundary of the safe set (Eq. 14 keeps an unbroken chain); the second is
    what lets gradients flow back into the *state-dependent* constraint offsets,
    i.e. into F_base and N_theta.  Both are implemented; the second is what makes
    this a genuinely end-to-end differentiable physics stack rather than a
    projection bolted on after the fact.

3.  **Residual-corrected tightening.**  The QP is written against the LINEAR
    model, but the plant is non-linear.  Enforcing the linear constraints
    exactly therefore does NOT imply zero true violations.  We close that gap by
    tightening the offsets with the residual network's own prediction plus a
    conformal margin calibrated on held-out data (`ResidualTightening`).  This
    is the difference between "zero violations w.r.t. our own approximation"
    (not a real claim) and "zero violations w.r.t. the exact AC solve"
    (the claim in the abstract).  It is ablatable: `--no-tighten`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Constraint assembly
# --------------------------------------------------------------------------- #
@dataclass
class SafetyPolytope:
    """G a <= h  together with box bounds lo <= a <= hi (all torch tensors).

    Shapes: G (B, m, n), h (B, m), lo/hi (B, n).
    """
    G: torch.Tensor
    h: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    n_slack: int = 0

    @property
    def n_ineq(self) -> int:
        return self.G.shape[1]


def thermal_cuts(n_cuts: int = 8, device=None, dtype=torch.float32) -> torch.Tensor:
    """Angles of the polygonal outer approximation of |S| <= S_max.

    The apparent-power limit is a disc in (P, Q); we use the regular `n_cuts`-gon
    *inscribed-normal* form  cos(phi) P + sin(phi) Q <= S_max, which is an OUTER
    approximation of the disc (feasible set slightly larger than the true disc)
    -- so we shrink S_max by the polygon's circumradius ratio 1/cos(pi/n_cuts)
    to make it an INNER (conservative) approximation.  Conservative is the only
    acceptable direction for a safety layer.
    """
    phi = torch.arange(n_cuts, device=device, dtype=dtype) * (2 * np.pi / n_cuts)
    return phi


def build_polytope(R: torch.Tensor, X: torch.Tensor, M: torch.Tensor, A: torch.Tensor,
                   p_exo: torch.Tensor, q_exo: torch.Tensor,
                   s_max: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
                   v_min: float, v_max: float, v0: float = 1.0,
                   n_cuts: int = 8, allow_reactive: bool = True,
                   v_offset: Optional[torch.Tensor] = None,
                   margin_v: float = 0.0, margin_s: float = 0.0,
                   include_thermal: bool = True,
                   slack_penalty: Optional[float] = 300.0) -> SafetyPolytope:
    """Assemble the safe polytope in the action variable a = [p_ev ; q_ev].

    Parameters
    ----------
    v_offset : (B, n_bus) optional additive voltage correction (the residual
        network's prediction of  v_true - v_lin ).  Constraints become
        v_min <= v_lin(a) + v_offset <= v_max, i.e. the projection is performed
        against the *corrected* physics rather than the raw linearisation.
    margin_v, margin_s : extra conservative tightening (conformal margin).
    """
    B = p_exo.shape[0]
    dev, dt = p_exo.device, p_exo.dtype
    RA = R @ A                                    # (n_bus, k)
    XA = X @ A
    k = A.shape[1]
    # dv/da = -[R A , X A]  because a>0 is charging (a load).
    if allow_reactive:
        Gv = torch.cat([-RA, -XA], dim=1)         # (n_bus, 2k)
    else:
        Gv = -RA
    Gv = Gv.unsqueeze(0).expand(B, -1, -1)

    v_lin_exo = v0 + p_exo @ R.T + q_exo @ X.T    # (B, n_bus)
    if v_offset is not None:
        v_lin_exo = v_lin_exo + v_offset
    hi_v = (v_max - margin_v) - v_lin_exo         # Gv a <=  hi_v
    lo_v = v_lin_exo - (v_min + margin_v)         # -Gv a <= lo_v

    Gs, hs = [], []
    if include_thermal:
        phi = thermal_cuts(n_cuts, dev, dt)
        shrink = float(np.cos(np.pi / max(n_cuts, 3)))
        s_eff = (s_max * shrink - margin_s).clamp_min(1e-4)          # (n_branch,)
        MA = M @ A                                                   # (n_branch, k)
        pf_exo = -(p_exo @ M.T)                                      # (B, n_branch)
        qf_exo = -(q_exo @ M.T)
        for j in range(n_cuts):
            c, s = torch.cos(phi[j]), torch.sin(phi[j])
            # flow = -M p_net = -M p_exo + M A p_ev  ->  d(flow)/da = +M A
            if allow_reactive:
                row = torch.cat([c * MA, s * MA], dim=1)             # (n_branch, 2k)
            else:
                row = c * MA
            Gs.append(row.unsqueeze(0).expand(B, -1, -1))
            hs.append((s_eff.unsqueeze(0) - c * pf_exo - s * qf_exo))
    G = torch.cat([Gv, -Gv] + Gs, dim=1)
    h = torch.cat([hi_v, lo_v] + hs, dim=1)
    if slack_penalty is not None and slack_penalty > 0:
        G, h, lo, hi = _augment_with_slacks(G, h, lo, hi, n_v_rows=2 * Gv.shape[1],
                                            slack_penalty=slack_penalty)
    return SafetyPolytope(G=G, h=h, lo=lo, hi=hi, n_slack=(2 if slack_penalty else 0))


def _augment_with_slacks(G, h, lo, hi, n_v_rows: int, slack_penalty: float,
                         slack_cap: float = 1.5):
    """Add two scalar slack variables (voltage group, thermal group).

    WHY: the exogenous load alone can already push a weak feeder outside
    [v_min, v_max] -- on IEEE-33 at 100% nominal load V_min is 0.913 p.u., i.e.
    the safe set is EMPTY before any EV acts.  A projection onto an empty set is
    undefined, and an ADMM run on an infeasible QP does not converge, it drifts.
    Any safety layer that is going to be believed has to say what it does in
    that case.  Ours solves

        min 1/2||a - a_raw||_W^2 + 1/2 (xi + rho_s)^2   s.t.  G a <= h + xi,  xi >= 0

    which is an exact-penalty relaxation: while the hard set is non-empty the
    solution is identical to the hard projection (xi* = 0 to machine precision),
    and when it is empty the layer returns the action that is *least infeasible*
    in a uniform-margin sense instead of failing.  Reported violations always
    come from the exact AC solve afterwards, so a relaxation can never hide a
    violation -- it can only avoid a crash.

    `slack_cap` bounds xi at `slack_cap` p.u. rather than an arbitrary "big M".
    This matters more than it looks: mixing a variable whose natural scale is
    O(1e3) with action variables whose natural scale is O(0.05) in the SAME
    linear system gives a condition number on the order of (1e3/0.05)^2 ~ 4e8,
    which no fixed-iteration ADMM converges on without full equilibration (the
    non-convergence this caused -- an already-feasible action being perturbed
    for no constraint-driven reason -- is documented in RUNBOOK.md under 'ADMM
    convergence').  1.5 p.u. is already far more voltage/thermal headroom than
    any single control interval could need, so nothing is lost by capping it.
    """
    B, m, n = G.shape
    dev, dt = G.device, G.dtype
    Sv = torch.zeros(B, m, 2, device=dev, dtype=dt)
    Sv[:, :n_v_rows, 0] = -1.0
    Sv[:, n_v_rows:, 1] = -1.0
    G2 = torch.cat([G, Sv], dim=2)
    cap = torch.full((B, 2), slack_cap, device=dev, dtype=dt)
    lo2 = torch.cat([lo, torch.zeros(B, 2, device=dev, dtype=dt)], dim=1)
    hi2 = torch.cat([hi, cap], dim=1)
    return G2, h, lo2, hi2


# --------------------------------------------------------------------------- #
#  Batched ADMM QP solve (forward)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def admm_project(a_raw: torch.Tensor, poly: SafetyPolytope, w: Optional[torch.Tensor] = None,
                 rho: float = 8.0, sigma: float = 1e-6, alpha: float = 1.6,
                 iters: int = 220, tol: Optional[float] = None,
                 z0: Optional[torch.Tensor] = None, y0: Optional[torch.Tensor] = None,
                 adaptive_rho: bool = False, rho_update_every: int = 25,
                 rho_min: float = 1e-3, rho_max: float = 1e5):
    """OSQP-style ADMM for  min 1/2 (a-ar)^T W (a-ar)  s.t. l <= C a <= u.

    C stacks [G ; I]; the G-rows have lower bound -inf, the identity rows are the
    box.  Returns (a_safe, z, y, stats).  Everything is batched over dim 0.

    ADAPTIVE RHO -- why this is not optional in practice
    ------------------------------------------------------
    A fixed rho converges in <10 iterations on a well-scaled synthetic problem
    (see tests/test_qp_layer.py), but the *real* safety polytope mixes wildly
    different scales: per-EV weights spanning ~[0.25, 16] (willingness-weighted,
    Section 3.2), a big-M slack relaxation column (Section 4's feasibility
    fallback) with a box upper bound orders of magnitude larger than the real
    action box, and thermal/voltage rows with their own natural units. A single
    global rho tuned for one scale converges in tens of iterations on another
    and stalls (visibly: primal residual plateaus around 1e-2 instead of
    shrinking) on a third -- this was caught empirically by cross-checking
    projected actions against a raw, already-feasible action that the solver
    was moving away from for no constraint-driven reason (see RUNBOOK.md,
    'ADMM convergence' section, for the diagnostic that found this).  The
    standard OSQP remedy is to re-estimate rho from the ratio of primal to
    dual residual norms every `rho_update_every` iterations and refactor; we do
    the same, batched.  Because different batch elements can have different
    conditioning, rho is per-batch-element, not a single scalar for the batch.
    """
    B, n = a_raw.shape
    dev, dt = a_raw.device, a_raw.dtype
    if tol is None:                      # float32 cannot reach 1e-7 on p.u. quantities of O(1e-2): stop at ~1 mW
        tol = 2e-6 if dt == torch.float32 else 1e-7
    G, h, lo, hi = poly.G, poly.h, poly.lo, poly.hi
    m = G.shape[1]
    eye = torch.eye(n, device=dev, dtype=dt).expand(B, n, n)
    Gt = G.transpose(1, 2).contiguous()                  # (B, n, m)
    # C = [G ; I] is never materialised: the identity (box) block is applied implicitly, which
    # removes ~70% of the memory traffic of every ADMM iteration.
    def Cmul(v):                                         # (B,n) -> (B,m+n)
        return torch.cat([torch.bmm(G, v.unsqueeze(-1)).squeeze(-1), v], dim=1)

    def Ctmul(u):                                        # (B,m+n) -> (B,n)
        return torch.bmm(Gt, u[:, :m].unsqueeze(-1)).squeeze(-1) + u[:, m:]
    ub = torch.cat([h, hi], dim=1)
    lb = torch.cat([torch.full_like(h, -1e12), lo], dim=1)

    w = torch.ones(B, n, device=dev, dtype=dt) if w is None else w
    P = torch.diag_embed(w)
    q = -(w * a_raw)

    rho_vec = torch.full((B, 1, 1), float(rho), device=dev, dtype=dt)

    def factor(rho_b):
        KKT = P + sigma * eye + rho_b * (Gt @ G + eye) + 1e-9 * eye
        return torch.cholesky_inverse(torch.linalg.cholesky(KKT))   # explicit inverse: 1 bmm / iteration (much faster than 2 triangular solves on CPU)

    L = factor(rho_vec)

    x = a_raw.clone()
    z = torch.clamp(Cmul(x), lb, ub) if z0 is None else z0.clone()
    y = torch.zeros(B, m + n, device=dev, dtype=dt) if y0 is None else y0.clone()

    prim, dual, it = torch.zeros(B), torch.zeros(B), 0
    for it in range(1, iters + 1):
        rho_row = rho_vec.squeeze(-1)                      # (B,1) broadcastable over rows
        rhs = sigma * x - q + Ctmul(rho_row * z - y)
        x_new = torch.bmm(L, rhs.unsqueeze(-1)).squeeze(-1)
        Cx = Cmul(x_new)
        Cx_rel = alpha * Cx + (1 - alpha) * z
        z_new = torch.clamp(Cx_rel + y / rho_row, lb, ub)
        y = y + rho_row * (Cx_rel - z_new)
        prim = (Cx - z_new).abs().amax(dim=1)
        dual = (x_new - x).abs().amax(dim=1)
        x, z = x_new, z_new
        if adaptive_rho and it % rho_update_every == 0 and it < iters:
            # Proper OSQP residuals for rho selection (both evaluated at the
            # same iterate x_new, z_new):
            #   r_prim = C x_new - z_new
            #   r_dual = P x_new + q + C^T y   = w*(x_new - a_raw) + C^T y
            # (P = diag(w), q = -w*a_raw for this projection problem).  An
            # earlier version omitted the P x + q term and only used C^T y,
            # which made the dual residual estimate spuriously large and drove
            # rho to its floor -- verified by the regression this update fixes
            # in tests/test_qp_layer.py::test_adaptive_rho_matches_fixed_rho.
            pr_vec = Cx - z_new
            du_vec = w * (x_new - a_raw) + Ctmul(y)
            pn = pr_vec.norm(dim=1).clamp_min(1e-10)
            dn = du_vec.norm(dim=1).clamp_min(1e-10)
            ratio = (pn / dn).clamp(1e-6, 1e6)
            new_rho = (rho_vec.squeeze(-1).squeeze(-1) * torch.sqrt(ratio)).clamp(rho_min, rho_max)
            changed = (new_rho - rho_vec.squeeze(-1).squeeze(-1)).abs() > 0.2 * rho_vec.squeeze(-1).squeeze(-1)
            if changed.any():
                rho_vec = new_rho.view(B, 1, 1)
                y = y * (rho_vec.squeeze(-1) / rho_row)     # rescale dual variable, OSQP convention
                L = factor(rho_vec)
        if float(prim.max()) < tol and float(dual.max()) < tol:
            break
    # final exact box clamp (cheap, removes any residual ADMM infeasibility on
    # the box rows -- the box is the one set of constraints we can satisfy for free)
    x = torch.clamp(x, lo, hi)
    return x, z, y, {"iters": it, "prim_res": float(prim.max()), "dual_res": float(dual.max()),
                     "rho_final": float(rho_vec.mean())}


# --------------------------------------------------------------------------- #
#  Implicit differentiation (backward)
# --------------------------------------------------------------------------- #
class _ProjectFn(torch.autograd.Function):
    """Differentiable projection with active-set KKT gradients."""

    @staticmethod
    def forward(ctx, a_raw, G, h, lo, hi, w, rho, iters, act_tol):
        poly = SafetyPolytope(G=G, h=h, lo=lo, hi=hi)
        with torch.no_grad():
            x, z, y, stats = admm_project(a_raw, poly, w=w, rho=rho, iters=iters)
        ctx.save_for_backward(x, G, h, lo, hi, w, a_raw)
        ctx.act_tol = act_tol
        ctx.stats = stats
        return x

    @staticmethod
    def backward(ctx, grad_out):
        x, G, h, lo, hi, w, a_raw = ctx.saved_tensors
        tol = ctx.act_tol
        B, n = x.shape
        dev, dt = x.device, x.dtype
        m = G.shape[1]
        # Coordinates sitting ON a box bound are pinned (x_i = lo_i / hi_i): they drop out of the
        # KKT system, so only the ACTIVE polytope rows (typically < 30) enter the solve.  This is
        # exactly equivalent to the full (m+2n)-row active-set system but ~100x cheaper (the
        # old dense solve was the CPU training bottleneck).
        s_hi = hi - x
        s_lo = x - lo
        at_hi = (s_hi.abs() < tol)
        at_lo = (s_lo.abs() < tol)
        fixed = (at_hi | at_lo)
        free = (~fixed).to(dt)                                    # (B,n)
        act_G = ((h - torch.bmm(G, x.unsqueeze(-1)).squeeze(-1)).abs() < tol).to(dt)   # (B,m)
        Winv = 1.0 / w.clamp_min(1e-8)                            # (B, n)
        Winv_ = Winv.unsqueeze(1)
        K = max(1, int(act_G.sum(dim=1).max().item()))
        order = torch.argsort(act_G, dim=1, descending=True, stable=True)[:, :K]     # (B,K)
        Gk = torch.gather(G, 1, order.unsqueeze(-1).expand(B, K, n))
        act_a = torch.gather(act_G, 1, order)                                        # (B,K)
        Ca = Gk * act_a.unsqueeze(-1) * free.unsqueeze(1)                            # active, free cols
        S = torch.bmm(Ca * Winv_, Ca.transpose(1, 2))
        S = S + torch.diag_embed(1.0 - act_a) + 1e-7 * torch.eye(K, device=dev, dtype=dt).expand(B, K, K)
        try:
            Sinv_Ca = torch.linalg.solve(S, Ca * Winv_)                              # (B,K,n)
        except Exception:                                          # pragma: no cover
            Sinv_Ca = torch.linalg.lstsq(S, Ca * Winv_).solution

        # J = dx/da_raw = diag(free) - Winv C^T S^-1 C   (J is NOT symmetric unless W = I;
        # the unit test uses a non-uniform W precisely to catch a transpose error.)
        g = grad_out
        tmp_a = torch.bmm(Sinv_Ca, (g * free).unsqueeze(-1)).squeeze(-1) * act_a    # (B,K)
        corr = torch.bmm(Gk.transpose(1, 2) * act_a.unsqueeze(1), tmp_a.unsqueeze(-1)).squeeze(-1)  # (B,n)
        grad_a = free * (g - corr)
        tmp = torch.zeros(B, m, device=dev, dtype=dt).scatter(1, order, tmp_a)      # nu on the G rows
        grad_h = tmp
        grad_hi = torch.where(at_hi, g - corr, torch.zeros_like(g))
        grad_lo = torch.where(at_lo & ~at_hi, g - corr, torch.zeros_like(g))
        grad_G = None
        if ctx.needs_input_grad[1]:
            grad_G = -tmp.unsqueeze(-1) * x.unsqueeze(1)
        grad_w = None
        return grad_a, grad_G, grad_h, grad_lo, grad_hi, grad_w, None, None, None


def _project_core(a_raw: torch.Tensor, poly: SafetyPolytope, w: torch.Tensor,
                  rho: float, iters: int, act_tol: float) -> torch.Tensor:
    return _ProjectFn.apply(a_raw, poly.G, poly.h, poly.lo, poly.hi, w, rho, iters, act_tol)


def project(a_raw: torch.Tensor, poly: SafetyPolytope, w: Optional[torch.Tensor] = None,
            rho: float = 8.0, iters: int = 220, act_tol: float = 1e-5,
            slack_penalty: float = 300.0, skip_feasible: bool = True,
            feas_tol: float = 1e-6, hard_first: bool = True,
            rho_hard: float = 0.1, fast: bool = False) -> torch.Tensor:
    """Differentiable safe projection.  This is the layer used inside the actor.

    SKIP-IF-FEASIBLE (default on): rows whose raw action is already feasible are
    returned unchanged (ADMM crawls on interior optima; see RUNBOOK.md).

    HARD-FIRST (default on) -- the fix for a measured over-conservatism bug
    -----------------------------------------------------------------------
    The slack-augmented QP (slack columns with a 300x linear penalty next to
    actions of order 1e-2 p.u.) is so badly conditioned that fixed-iteration ADMM
    (120 iterations, as used in training) returns a point that is NOT the
    projection: on real Caltech steps it removed 33% of the requested charging
    where the exact projection of the identical QP removes 12.8% (2.6x too
    conservative; 1000 iterations were needed to converge).  The slack is only
    needed when the hard set is EMPTY.  The idle action a=0 is always inside the
    box, so  h >= 0  (every row satisfied at a=0)  proves the hard set is
    non-empty; those rows are solved WITHOUT slack columns and with a small
    rho (`rho_hard`), which matches the exact solution to 0.05 kW in 120
    iterations (see tests/test_qp_layer.py::test_hard_first_matches_exact).
    Only rows whose hard set is empty (idle infeasible) use the slack path.
    """
    if w is None:
        w = torch.ones_like(a_raw)
    B, n_real = a_raw.shape
    ns = poly.n_slack

    if skip_feasible:
        with torch.no_grad():
            Greal = poly.G[:, :, :n_real]
            resid = torch.bmm(Greal, a_raw.unsqueeze(-1)).squeeze(-1) - poly.h
            box_ok = (a_raw >= poly.lo[:, :n_real] - feas_tol).all(dim=1) & \
                     (a_raw <= poly.hi[:, :n_real] + feas_tol).all(dim=1)
            feasible = (resid <= feas_tol).all(dim=1) & box_ok
        if bool(feasible.all()):
            return a_raw
        idx = torch.nonzero(~feasible, as_tuple=False).squeeze(-1)
    else:
        idx = torch.arange(B, device=a_raw.device)

    solved = _solve_rows(a_raw[idx], _sub_poly(poly, idx), w[idx], rho, iters, act_tol,
                         slack_penalty, hard_first, rho_hard, feas_tol, n_real, fast)
    if idx.numel() == B:
        return solved
    out = a_raw.clone()
    out[idx] = solved
    return out


def _sub_poly(poly: SafetyPolytope, idx: torch.Tensor) -> SafetyPolytope:
    return SafetyPolytope(G=poly.G[idx], h=poly.h[idx], lo=poly.lo[idx], hi=poly.hi[idx],
                          n_slack=poly.n_slack)


def _solve_rows(a_raw, poly, w, rho, iters, act_tol, slack_penalty, hard_first,
                rho_hard, feas_tol, n_real, fast=False) -> torch.Tensor:
    """Solve the projection QP for rows that are infeasible as requested.

    hard_first=True : NO slack columns.  Rows whose RHS h_j is negative (conformal tightening
        ate the headroom at idle, or the exogenous load already violates the limit) are relaxed
        to h_j := 0, i.e. "do not make this constraint worse than idling".  The idle action a=0
        then always satisfies every row, so the feasible set is never empty and a well-conditioned
        small-rho ADMM matches the exact projection.  (Measured on the real Caltech day: after
        conformal calibration ~30% of training samples have an infeasible-at-idle row; the old
        slack-penalty ADMM spent ~97% of the time there and over-shrank the action.)
    hard_first=False: legacy slack-augmented QP (kept for the reproduction arm).
    """
    ns = poly.n_slack
    if not (hard_first and ns):
        return _solve_with_slack(a_raw, poly, w, rho, iters, act_tol, slack_penalty)
    G = poly.G[:, :, :n_real]
    h = poly.h.clamp_min(0.0)
    lo, hi = poly.lo[:, :n_real], poly.hi[:, :n_real]
    if fast:      # no-grad target / imagination projections: 64 tightest rows, no verification pass
        x = _solve_row_generation(a_raw, G, h, lo, hi, w, rho_hard, iters, act_tol,
                                  ks=(min(64, G.shape[1] - 1),), viol_tol=float("inf"))
    else:
        x = _solve_row_generation(a_raw, G, h, lo, hi, w, rho_hard, iters, act_tol)
    return _restore_feasibility(x, G, h)


def _restore_feasibility(x, G, h, floor: float = 1e-4, tol: float = 1e-6):
    """Exact feasibility certificate on the linear rows.

    ADMM (fixed iterations) and the row-generation tolerance can leave a few-kW residual on a
    binding row.  Because idle (a = 0) satisfies every row (h >= 0 after the relaxation), any
    feasible-direction shrink theta*x with theta in [0,1] is a *convex combination of idle and x*
    and therefore stays in the box; theta is the largest value that makes every row feasible.
    theta == 1 (no change) whenever the solver already returned a feasible point.
    The shrink factor is treated as a constant in the backward pass."""
    with torch.no_grad():
        gx = torch.bmm(G, x.unsqueeze(-1)).squeeze(-1)                  # (B,m)
        cap_h = torch.clamp(h, min=floor)
        bad = gx > cap_h + tol
        ratio = torch.where(bad, cap_h / gx.clamp_min(1e-12), torch.ones_like(gx))
        theta = ratio.amin(dim=1, keepdim=True).clamp(0.0, 1.0)
    return x * theta


def _solve_row_generation(a_raw, G, h, lo, hi, w, rho, iters, act_tol,
                          ks=(48, 128), viol_tol=4e-3) -> torch.Tensor:
    """Constraint generation: only the ~20 polytope rows nearest to binding can be active at the
    projection, so solve with the K rows of smallest slack, verify against ALL rows, and enlarge K
    only if something is violated.  Cuts the per-iteration cost ~5-8x (it is memory-bound in
    bmm(G,.)); the result is identical to the full solve whenever the verification passes."""
    B, m, n = G.shape
    x_ref = a_raw
    x = a_raw
    for K in ks:
        if K >= m:
            break
        slack = h - torch.bmm(G, x_ref.detach().unsqueeze(-1)).squeeze(-1)           # (B,m)
        idx = torch.topk(-slack, K, dim=1).indices                                    # K tightest rows
        Gs = torch.gather(G, 1, idx.unsqueeze(-1).expand(B, K, n))
        hs = torch.gather(h, 1, idx)
        sub = SafetyPolytope(G=Gs, h=hs, lo=lo, hi=hi, n_slack=0)
        x = _project_core(a_raw, sub, w, rho, iters, act_tol)
        with torch.no_grad():
            viol = (torch.bmm(G, x.unsqueeze(-1)).squeeze(-1) - h).amax().item()
        if viol <= viol_tol:
            return x
        x_ref = x
    full = SafetyPolytope(G=G, h=h, lo=lo, hi=hi, n_slack=0)
    return _project_core(a_raw, full, w, rho, iters, act_tol)


def _solve_with_slack(a_raw, poly, w, rho, iters, act_tol, slack_penalty) -> torch.Tensor:
    ns = poly.n_slack
    if ns:
        B = a_raw.shape[0]
        pad = torch.full((B, ns), -float(slack_penalty), device=a_raw.device, dtype=a_raw.dtype)
        a_in = torch.cat([a_raw, pad], dim=1)
        w_in = torch.cat([w, torch.ones(B, ns, device=w.device, dtype=w.dtype)], dim=1)
    else:
        a_in, w_in = a_raw, w
    out = _project_core(a_in, poly, w_in, rho, iters, act_tol)
    return out[:, :a_raw.shape[1]] if ns else out


def slack_values(a_raw: torch.Tensor, poly: SafetyPolytope, w=None, rho: float = 8.0,
                 iters: int = 220, slack_penalty: float = 300.0) -> torch.Tensor:
    """Diagnostic: how infeasible was the hard safe set?  (0 = strictly feasible)"""
    if poly.n_slack == 0:
        return torch.zeros(a_raw.shape[0], 0)
    B = a_raw.shape[0]
    w = torch.ones_like(a_raw) if w is None else w
    a_in = torch.cat([a_raw, torch.full((B, poly.n_slack), -float(slack_penalty),
                                        device=a_raw.device, dtype=a_raw.dtype)], dim=1)
    w_in = torch.cat([w, torch.ones(B, poly.n_slack, device=w.device, dtype=w.dtype)], dim=1)
    with torch.no_grad():
        x, _, _, _ = admm_project(a_in, poly, w=w_in, rho=rho, iters=iters)
    return x[:, a_raw.shape[1]:]


def vpw_weights_to_qp(vpw: torch.Tensor, occupied: torch.Tensor, allow_reactive: bool,
                      w_floor: float = 0.25, w_ceiling: float = 4.0,
                      enabled: bool = True) -> torch.Tensor:
    """Map willingness W in [0,1] to the QP metric W_qp (Section 3.2, property 1).

    Deviation cost is *decreasing* in willingness, so the projection preferentially
    moves the setpoints of high-willingness drivers and leaves reluctant drivers
    close to their requested schedule.  Unoccupied slots get a large weight so the
    solver has no incentive to move them (their box is {0} anyway).
    """
    if not enabled:
        base = torch.ones_like(vpw)
    else:
        base = w_ceiling - (w_ceiling - w_floor) * vpw.clamp(0, 1)
    base = torch.where(occupied > 0.5, base, torch.full_like(base, w_ceiling * 4))
    return torch.cat([base, base], dim=-1) if allow_reactive else base


# --------------------------------------------------------------------------- #
#  Conformal / residual-informed tightening
# --------------------------------------------------------------------------- #
@dataclass
class ResidualTightening:
    """Calibrated conservative margin that converts 'feasible under the linear
    model' into 'feasible under the exact AC model' with probability >= 1-alpha.

    Two modes:

    * ``rowwise=False`` (LEGACY): one scalar per constraint family =
      (1-alpha)-quantile over samples of  max_{buses/branches} |e| .  The worst
      branch (the heavily loaded trunk, whose loss error is tens of kVA) sets the margin
      that is then applied to *every* row, including the small hub branches whose
      thermal limit is only 73-100 kVA.  Measured on the real Caltech feeder this margin
      (~32 kVA) is comparable to the hub headroom, so after the first calibration the
      safe set collapses to "do not charge more" and the policy collapses to idle.
    * ``rowwise=True`` (default in the fixed pipeline): one margin PER bus / PER branch =
      finite-sample-corrected (1-alpha)-quantile of that row's own error (one-sided for
      thermal: only under-estimated flow matters), optionally capped at
      ``cap_frac`` x the row's limit so calibration noise can never empty a row.
      Marginal (per-row) coverage 1-alpha instead of joint coverage; for a joint
      guarantee use alpha/m (Bonferroni) -- exposed as ``bonferroni``.
    """
    margin_v: float = 0.0              # scalar summaries (max over rows) for logging
    margin_s: float = 0.0
    alpha: float = 0.01
    n_calib: int = 0
    rowwise: bool = False
    bonferroni: bool = False
    cap_frac: float = 0.35
    margin_v_vec: Optional[np.ndarray] = None
    margin_s_vec: Optional[np.ndarray] = None

    @staticmethod
    def _q(x: np.ndarray, level: float, axis=0):
        n = x.shape[axis]
        lv = min(1.0, np.ceil((n + 1) * level) / max(n, 1))      # finite-sample split-conformal level
        return np.quantile(x, lv, axis=axis, method="higher")

    def fit(self, v_true: np.ndarray, v_pred: np.ndarray,
            s_true: Optional[np.ndarray] = None, s_pred: Optional[np.ndarray] = None):
        e = np.abs(v_true - v_pred)
        if not self.rowwise:
            self.margin_v = float(np.quantile(e.max(axis=-1), 1.0 - self.alpha)) if e.size else 0.0
            self.margin_v_vec = None
            if s_true is not None and s_pred is not None and s_true.size:
                es = np.abs(s_true - s_pred)
                self.margin_s = float(np.quantile(es.max(axis=-1), 1.0 - self.alpha))
            self.margin_s_vec = None
        else:
            a = self.alpha / max(e.shape[-1], 1) if self.bonferroni else self.alpha
            self.margin_v_vec = self._q(e, 1.0 - a, axis=0).astype(np.float32) if e.size else None
            self.margin_v = float(self.margin_v_vec.max()) if self.margin_v_vec is not None else 0.0
            if s_true is not None and s_pred is not None and s_true.size:
                es = np.maximum(s_true - s_pred, 0.0)                # one-sided: under-estimated flow
                a_s = self.alpha / max(es.shape[-1], 1) if self.bonferroni else self.alpha
                self.margin_s_vec = self._q(es, 1.0 - a_s, axis=0).astype(np.float32)
                self.margin_s = float(self.margin_s_vec.max())
        self.n_calib = int(e.shape[0]) if e.ndim > 1 else 1
        return self

    def as_dict(self) -> dict:
        d = {"margin_v": self.margin_v, "margin_s": self.margin_s,
             "alpha": self.alpha, "n_calib": self.n_calib}
        if self.margin_s_vec is not None:
            d["margin_s_mean"] = float(np.mean(self.margin_s_vec))
        return d


# --------------------------------------------------------------------------- #
#  Reference solver for validation
# --------------------------------------------------------------------------- #
def reference_solve_cvxpy(a_raw: np.ndarray, G: np.ndarray, h: np.ndarray,
                          lo: np.ndarray, hi: np.ndarray,
                          w: Optional[np.ndarray] = None) -> np.ndarray:  # pragma: no cover
    """Ground-truth QP solve via CVXPY/OSQP -- used only in tests/benchmarks."""
    import cvxpy as cp
    n = a_raw.shape[0]
    w = np.ones(n) if w is None else w
    a = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum(cp.multiply(w, cp.square(a - a_raw))))
    cons = [G @ a <= h, a >= lo, a <= hi]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=80000)
    if a.value is None:
        prob.solve(solver=cp.CLARABEL)
    return np.asarray(a.value, dtype=np.float64)
