import numpy as np
from pifsrl.evalstats import paired_bootstrap, sign_test_p, hoeffding_gate, iqm


def test_bootstrap_ci_contains_true_mean_and_is_ordered():
    d = np.random.default_rng(0).normal(5.0, 2.0, 40)
    r = paired_bootstrap(d, n_boot=2000)
    assert r["lo"] < r["mean"] < r["hi"] and r["lo"] > 3.0


def test_sign_test():
    assert sign_test_p([1] * 10) < 0.01 and sign_test_p([1, -1] * 5) == 1.0


def test_gate_refuses_small_n_and_accepts_large_effect():
    d = np.full(4, 20.0)
    assert not hoeffding_gate(d, B=300.0, delta=0.05)["deploy"]          # 4 days can never certify
    assert hoeffding_gate(np.full(200, 60.0), B=300.0, delta=0.05)["deploy"]
    assert not hoeffding_gate(-np.abs(np.random.default_rng(1).normal(5, 1, 100)), B=300.0)["deploy"]


def test_gate_type_one_error_under_null():
    rng = np.random.default_rng(0); B = 10.0; fires = 0
    for _ in range(2000):
        d = rng.uniform(-B, B, 30)                  # E d = 0 -> must fire <= delta of the time
        fires += hoeffding_gate(d, B, 0.05)["deploy"]
    assert fires / 2000 <= 0.05


def test_iqm_robust_to_outlier():
    assert abs(iqm([1, 2, 3, 4, 5, 6, 7, 1000]) - 4.5) < 1.5
