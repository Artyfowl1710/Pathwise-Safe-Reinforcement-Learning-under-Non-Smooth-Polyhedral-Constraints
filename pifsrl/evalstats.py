"""Statistics for honest policy comparison: paired day-blocked bootstrap, sign test, Hoeffding deployment gate, IQM.

All inputs are per-unit costs (unit = one held-out day, optionally pooled over seeds) for the SAME days under two policies;
d_i = J_base(i) - J_method(i) is the improvement (positive = method better)."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def paired_bootstrap(d: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(m, alpha / 2)), "hi": float(np.quantile(m, 1 - alpha / 2)), "n": int(len(d))}


def sign_test_p(d: Sequence[float]) -> float:
    """Two-sided exact sign test on d != 0."""
    d = np.asarray(d, float); d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = int((d > 0).sum())
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def hoeffding_gate(d: Sequence[float], B: float, delta: float = 0.05) -> Dict[str, float]:
    """Deploy the learner iff mean(d) - B*sqrt(2 ln(1/delta)/n) > 0, where |d_i| <= B is an A-PRIORI bound.
    Then P(deploy | E d <= 0) <= delta (one-sided Hoeffding), days exchangeable with deployment days."""
    d = np.asarray(d, float)
    n = len(d)
    lb = float(d.mean() - B * math.sqrt(2.0 * math.log(1.0 / delta) / max(n, 1)))
    return {"deploy": bool(lb > 0), "lower_bound": lb, "mean": float(d.mean()), "n": int(n), "B": float(B), "delta": float(delta),
            "n_needed_if_effect_equals_mean": (int(math.ceil(2 * math.log(1 / delta) * (B / d.mean()) ** 2)) if d.mean() > 0 else None)}


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean (Agarwal et al. 2021)."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    lo, hi = int(math.floor(0.25 * n)), int(math.ceil(0.75 * n))
    return float(x[lo:max(hi, lo + 1)].mean())
