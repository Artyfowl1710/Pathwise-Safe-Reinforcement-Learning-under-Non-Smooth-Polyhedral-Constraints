"""Adaptive conformal margin (Gibbs & Candes 2021) for the LinDistFlow -> AC residual.

Split-conformal margins (ResidualTightening) are valid only if calibration and deployment are exchangeable.
Under a load shift the coverage drops. ACI keeps the LONG-RUN miscoverage frequency at alpha with no distributional
assumption: alpha_{t+1} = alpha_t + gamma (alpha - err_t), margin_t = Q_{1-alpha_t}(recent scores).
Guarantee (their Prop. 4.1): |mean_t err_t - alpha| <= (max(alpha_1,1-alpha_1)+gamma)/(gamma T).
Scores here are one scalar per time step (max over rows of the one-sided residual), i.e. JOINT coverage."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class AdaptiveConformal:
    def __init__(self, alpha: float = 0.05, gamma: float = 0.02, window: int = 500, init_scores: Optional[np.ndarray] = None):
        self.alpha, self.gamma, self.window = alpha, gamma, window
        self.alpha_t = alpha
        self.scores = deque([] if init_scores is None else list(map(float, init_scores)), maxlen=window)
        self.err_hist = []

    def margin(self) -> float:
        if not self.scores:
            return 0.0
        a = self.alpha_t
        if a <= 0:
            return float(max(self.scores) * 1.5 + 1e-9)          # alpha_t<=0 -> "infinite" margin (cap)
        if a >= 1:
            return 0.0
        return float(np.quantile(np.fromiter(self.scores, float), 1.0 - a, method="higher"))

    def update(self, score: float) -> bool:
        """Call after the true residual score is revealed. Returns True if the margin covered it."""
        covered = score <= self.margin()
        err = 0.0 if covered else 1.0
        self.alpha_t += self.gamma * (self.alpha - err)
        self.err_hist.append(err)
        self.scores.append(float(score))
        return covered

    def coverage(self, last: Optional[int] = None) -> float:
        e = self.err_hist[-last:] if last else self.err_hist
        return 1.0 - float(np.mean(e)) if e else float("nan")
