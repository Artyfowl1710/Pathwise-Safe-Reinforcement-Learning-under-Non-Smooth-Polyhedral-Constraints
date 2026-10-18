"""
Dynamic V2G Participation Willingness (VPW) -- Section 5.1/5.2 of the spec.

    W_k(t) = w1 * T_park/T_req
           + w2 * (SoC - SoC_min)/(SoC_tgt - SoC_min)
           + w3 * (lambda - lambda_avg)/(lambda_max - lambda_min)
           - w4 * C_deg(P_dis)/Rev_max                                  (Eq. 19)

The spec is explicit that this is a *secondary, non-load-bearing* contribution
and that the proxy signal is not validated ground truth.  This module is written
to honour that: the calibrator optimises w against revealed behaviour
(opt-in / early-departure), reports its own rank correlation with the held-out
latent willingness, and the evaluation always includes FCFS and SoC-only
prioritisation baselines so that "VPW helps" is a measured claim, never an
assumption.

W enters the system in exactly two places, both auditable:
  1. the *weighting matrix* of the QP safety projection (safety/qp_layer.py),
     so low-willingness drivers are expensive to deviate from their raw action
     and therefore absorb less of the feasibility correction;
  2. a discomfort term in the reward.
It never enters the physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .sessions import Session


@dataclass
class VPWWeights:
    w1: float = 0.35     # parking-time slack
    w2: float = 0.30     # SoC headroom
    w3: float = 0.25     # price signal
    w4: float = 0.20     # degradation aversion
    bias: float = 0.20

    def as_array(self) -> np.ndarray:
        return np.array([self.w1, self.w2, self.w3, self.w4, self.bias])

    @staticmethod
    def from_array(a: Sequence[float]) -> "VPWWeights":
        return VPWWeights(*[float(x) for x in a])


def vpw_features(t_park_remaining_h: np.ndarray, t_req_h: np.ndarray,
                 soc: np.ndarray, soc_min: np.ndarray, soc_tgt: np.ndarray,
                 price: float, price_avg: float, price_min: float, price_max: float,
                 c_deg_norm: np.ndarray) -> np.ndarray:
    """Return (K, 4) feature matrix of Eq. 19 (each column already normalised)."""
    f1 = np.clip(t_park_remaining_h / np.maximum(t_req_h, 1e-3), 0.0, 3.0) / 3.0
    f2 = np.clip((soc - soc_min) / np.maximum(soc_tgt - soc_min, 1e-3), 0.0, 1.5) / 1.5
    denom = max(price_max - price_min, 1e-6)
    f3 = np.full_like(f1, np.clip((price - price_avg) / denom, -1.0, 1.0))
    f4 = np.clip(c_deg_norm, 0.0, 1.0)
    return np.stack([f1, f2, f3, f4], axis=-1)


def willingness(features: np.ndarray, w: VPWWeights) -> np.ndarray:
    """Eq. 19, squashed to [0, 1]."""
    raw = (w.w1 * features[..., 0] + w.w2 * features[..., 1]
           + w.w3 * features[..., 2] - w.w4 * features[..., 3] + w.bias)
    return 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.5)))


# --------------------------------------------------------------------------- #
#  Stage 1: fixed-heuristic calibration by grid / random search against the
#  revealed-behaviour proxy (Section 5.2.1)
# --------------------------------------------------------------------------- #
def _session_features(sessions: List[Session], dt_h: float,
                      price_stats: Tuple[float, float, float, float]) -> np.ndarray:
    price, p_avg, p_min, p_max = price_stats
    n = len(sessions)
    t_park = np.array([s.dwell * dt_h for s in sessions])
    t_req = np.array([s.energy_req_kwh / max(s.p_rate_kw, 1e-6) for s in sessions])
    soc = np.array([s.soc_arr for s in sessions])
    soc_tgt = np.array([s.soc_tgt for s in sessions])
    soc_min = np.full(n, 0.20)
    cdeg = np.clip(np.array([s.p_rate_kw for s in sessions]) / 50.0, 0, 1)
    return vpw_features(t_park, t_req, soc, soc_min, soc_tgt,
                        price, p_avg, p_min, p_max, cdeg)


def calibrate_fixed(sessions: List[Session], dt_h: float,
                    price_stats: Tuple[float, float, float, float],
                    n_trials: int = 4000, seed: int = 0) -> dict:
    """Random search over the 5-dim weight simplex maximising agreement with the
    revealed-behaviour proxy target  y = opt_in - early_leave  in {-1,0,1}.

    Objective: Spearman-style rank agreement (AUC of W against y>0 vs y<0),
    which is invariant to the arbitrary sigmoid scaling.
    """
    if not sessions:
        return {"weights": VPWWeights(), "auc": float("nan"), "n": 0}
    F = _session_features(sessions, dt_h, price_stats)
    y = np.array([s.opt_in - s.early_leave for s in sessions])
    pos, neg = y > 0, y < 0
    if pos.sum() < 3 or neg.sum() < 3:
        return {"weights": VPWWeights(), "auc": float("nan"), "n": len(sessions)}

    rng = np.random.default_rng(seed)
    best, best_auc = VPWWeights(), -1.0
    for _ in range(n_trials):
        w = VPWWeights.from_array(np.concatenate([rng.dirichlet(np.ones(4)) * rng.uniform(0.6, 1.6),
                                                  [rng.uniform(-0.2, 0.6)]]))
        s = willingness(F, w)
        auc = _auc(s[pos], s[neg])
        if auc > best_auc:
            best, best_auc = w, auc
    return {"weights": best, "auc": float(best_auc), "n": len(sessions)}


def _auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney U / ROC-AUC without sklearn."""
    allv = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        mean_rank = np.zeros(len(uniq))
        for i in range(len(uniq)):
            mean_rank[i] = ranks[inv == i].mean()
        ranks = mean_rank[inv]
    n1, n2 = len(pos_scores), len(neg_scores)
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n2))


# --------------------------------------------------------------------------- #
#  Stage 2 (stretch): per-user learned weights by maximum-likelihood inverse RL
# --------------------------------------------------------------------------- #
def calibrate_irl(sessions: List[Session], dt_h: float,
                  price_stats: Tuple[float, float, float, float],
                  lr: float = 0.15, epochs: int = 400, l2: float = 1e-3,
                  seed: int = 0) -> dict:
    """Logistic maximum-likelihood fit of w to the revealed-behaviour proxy.

    This is the 'inverse RL against the same proxy signal' of Section 5.2.2 in
    its simplest identifiable form: the driver's binary participate/decline
    decision is modelled as a Bernoulli policy whose logit is Eq. 19, and w is
    the maximum-likelihood reward parameter.  Reported *alongside* the fixed
    heuristic with a rank-correlation sanity check, never assumed superior.
    """
    if len(sessions) < 20:
        return {"weights": VPWWeights(), "auc": float("nan"), "n": len(sessions)}
    F = _session_features(sessions, dt_h, price_stats)
    y01 = np.array([1.0 if (s.opt_in - s.early_leave) > 0 else 0.0 for s in sessions])
    mask = np.array([(s.opt_in - s.early_leave) != 0 for s in sessions])
    F, y01 = F[mask], y01[mask]
    if len(y01) < 10 or y01.std() < 1e-6:
        return {"weights": VPWWeights(), "auc": float("nan"), "n": int(mask.sum())}
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.1, 5)                # [w1,w2,w3,-w4,bias]
    Fa = np.concatenate([F * np.array([1, 1, 1, -1]), np.ones((len(F), 1))], axis=1)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-Fa @ theta))
        g = Fa.T @ (p - y01) / len(y01) + l2 * theta
        theta -= lr * g
    w = VPWWeights(w1=float(theta[0]), w2=float(theta[1]), w3=float(theta[2]),
                   w4=float(-theta[3]), bias=float(theta[4]))
    s = willingness(F, w)
    auc = _auc(s[y01 > 0.5], s[y01 < 0.5])
    return {"weights": w, "auc": float(auc), "n": int(mask.sum())}


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho (no scipy dependency in the hot path)."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")
