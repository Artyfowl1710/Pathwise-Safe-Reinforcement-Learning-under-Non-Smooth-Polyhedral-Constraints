"""
Ground-truth non-linear power flow: backward/forward sweep (BFS) for radial
networks, plus the LinDistFlow approximation for comparison.

WHY THIS MATTERS FOR THE PROPOSAL
---------------------------------
The whole premise of the Residual-Dyna model (Eq. 10) is that

        s_true = F_base(s, a) + Delta,      Delta = "unmodelled residual"

is a *real* quantity, not a placeholder.  In this repository the residual is
generated honestly: the environment steps the **exact** non-linear DistFlow
recursion (Eqs. 1-6 of the spec, solved by BFS to 1e-10 p.u.), while F_base is
the **linearised** model of Eq. 7.  Delta is therefore exactly the linearisation
error -- second-order voltage-drop terms, I^2*Z loss terms, and the
voltage-dependence of constant-power loads.  It is O(1e-3) p.u. at nominal load
and grows to O(1e-2) p.u. under heavy/reverse flow, i.e. it is the same order as
the 0.05 p.u. safety margin the QP has to defend.  That is precisely why the
residual is load-bearing rather than cosmetic.

The BFS solver is fully vectorised over a batch of scenarios so that world-model
datasets and Monte-Carlo evaluations are cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .feeder import Feeder


@dataclass
class PFResult:
    v: np.ndarray          # (B, n) voltage magnitude [p.u.] at non-slack buses
    i_branch: np.ndarray   # (B, n_branch) branch current magnitude [p.u.]
    s_branch: np.ndarray   # (B, n_branch) branch apparent power magnitude [p.u.]
    p_loss: np.ndarray     # (B,) total active loss [p.u.]
    p_slack: np.ndarray    # (B,) substation active import [p.u.]
    iters: int
    converged: bool
    max_mismatch: float


def solve_bfs(feeder: Feeder, p_net: np.ndarray, q_net: np.ndarray,
              v0: float = 1.0, tol: float = 1e-10, max_iter: int = 60) -> PFResult:
    """Exact radial AC power flow by backward/forward sweep.

    Parameters
    ----------
    p_net, q_net : (B, n) or (n,) *net injections* in p.u. (generation positive,
        load negative).  i.e. ``p_net = p_v2g - p_load``.

    Notes
    -----
    Fixed point:  V = v0*1 - Zpath @ conj(S_draw / V),  S_draw = -(p_net + j q_net)
    with ``Zpath = M^T diag(z) M``.  For radial networks this iteration is the
    textbook BFS and converges linearly with a contraction factor ~ |S|*|Z|,
    i.e. in <10 iterations for any realistic distribution feeder.
    """
    p_net = np.atleast_2d(p_net)
    q_net = np.atleast_2d(q_net)
    B, n = p_net.shape
    assert n == feeder.n_bus, f"expected {feeder.n_bus} buses, got {n}"

    z = feeder.z_pu                                  # (n_branch,)
    M = feeder.M                                     # (n_branch, n)
    zpath = M.T @ np.diag(z) @ M                     # (n, n) complex

    s_draw = -(p_net + 1j * q_net)                   # power drawn from the grid
    v = np.full((B, n), v0, dtype=np.complex128)
    mismatch, it = np.inf, 0
    for it in range(1, max_iter + 1):
        i_draw = np.conj(s_draw / v)                 # (B, n)
        v_new = v0 - i_draw @ zpath.T                # (B, n)
        mismatch = float(np.max(np.abs(v_new - v))) if B else 0.0
        v = v_new
        if mismatch < tol:
            break

    i_draw = np.conj(s_draw / v)
    i_branch = i_draw @ M.T                          # (B, n_branch) complex
    v_from = np.concatenate([np.full((B, 1), v0, dtype=np.complex128), v], axis=1)
    from_idx = feeder.branches[:, 0].astype(int)
    s_branch = v_from[:, from_idx] * np.conj(i_branch)
    p_loss = np.real((np.abs(i_branch) ** 2) @ feeder.r_pu)
    p_slack = np.real(s_branch[:, feeder.M[:, :].sum(axis=1) == feeder.n_bus].sum(axis=1)) \
        if (feeder.M.sum(axis=1) == feeder.n_bus).any() else np.real(s_branch[:, 0])

    return PFResult(v=np.abs(v), i_branch=np.abs(i_branch), s_branch=np.abs(s_branch),
                    p_loss=p_loss, p_slack=p_slack, iters=it,
                    converged=bool(mismatch < 1e-6), max_mismatch=float(mismatch))


def solve_lindistflow(feeder: Feeder, p_net: np.ndarray, q_net: np.ndarray,
                      v0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """First-order LinDistFlow: V ~= v0 + R p_net + X q_net, S_flow = -M p_net etc.

    Returns (v, s_branch_magnitude).  This is exactly what ``F_base`` computes
    (see models/fbase.py) -- kept here in numpy for validation and analysis.
    """
    p_net = np.atleast_2d(p_net)
    q_net = np.atleast_2d(q_net)
    v = v0 + p_net @ feeder.R.T + q_net @ feeder.X.T
    pf = -(p_net @ feeder.M.T)
    qf = -(q_net @ feeder.M.T)
    return v, np.sqrt(pf ** 2 + qf ** 2)


def linearisation_error(feeder: Feeder, p_net: np.ndarray, q_net: np.ndarray) -> dict:
    """Quantify the residual that N_theta is tasked with learning."""
    exact = solve_bfs(feeder, p_net, q_net)
    v_lin, s_lin = solve_lindistflow(feeder, p_net, q_net)
    dv = exact.v - v_lin
    ds = exact.s_branch - s_lin
    return {
        "v_mae_pu": float(np.abs(dv).mean()),
        "v_max_abs_pu": float(np.abs(dv).max()),
        "v_bias_pu": float(dv.mean()),
        "s_mae_pu": float(np.abs(ds).mean()),
        "v_min_exact": float(exact.v.min()),
        "v_max_exact": float(exact.v.max()),
        "loss_mean_pu": float(exact.p_loss.mean()),
        "pf_iters": exact.iters,
        "converged": exact.converged,
    }
