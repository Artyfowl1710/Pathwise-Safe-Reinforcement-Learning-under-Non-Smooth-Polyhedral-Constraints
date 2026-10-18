import numpy as np
from pifsrl.safety.adaptive import AdaptiveConformal


def test_aci_longrun_coverage_under_shift():
    rng = np.random.default_rng(0)
    cal = rng.normal(1.0, 0.1, 1000)
    a = AdaptiveConformal(alpha=0.1, gamma=0.02, window=500, init_scores=cal)
    for t in range(4000):
        a.update(rng.normal(1.0 + 1.0 * (t > 500), 0.1))        # +10 sigma mean shift
    assert abs(a.coverage() - 0.9) < 0.03
    fixed = np.quantile(cal, 0.9)
    assert np.mean(rng.normal(2.0, 0.1, 2000) <= fixed) < 0.05  # the fixed margin would have failed
