import numpy as np, pandas as pd, pytest
from pifsrl.env.pricing import REGISTRY, SYNTHETIC_SCENARIOS, build_price_model
from pifsrl.env.v2g_env import EnvConfig, V2GEnv


@pytest.mark.parametrize("name", SYNTHETIC_SCENARIOS)
def test_scenarios_deterministic_finite_and_seeded(name):
    m = REGISTRY[name]
    a, b = m("2019-07-10", 192, 0.25, 0), m("2019-07-10", 192, 0.25, 0)
    assert np.array_equal(a, b) and a.shape == (192,) and np.isfinite(a).all() and m.provenance == "synthetic"
    if name not in ("flat", "tou3"):
        assert not np.array_equal(a, m("2019-07-10", 192, 0.25, 1))            # price_seed changes the path
        assert not np.array_equal(a, m("2019-07-11", 192, 0.25, 0))            # date changes the path


def _env(**kw):
    return V2GEnv(EnvConfig(k_max=6, archetype="residential", cluster_size=10.0, horizon=24, dt_h=1.0, seed=0, **kw))


def test_price_module_is_the_only_seam_synthetic_to_historical(tmp_path):
    """Swapping synthetic -> historical changes ONLY env.price; observation/action dimensions are identical, so agents are untouched."""
    ts = pd.date_range("2019-07-09 20:00", periods=96, freq="h", tz="UTC")
    pd.DataFrame({"INTERVALSTARTTIME_GMT": ts.strftime("%Y-%m-%dT%H:%M:%S-00:00"), "MW": 40.0}).to_csv(tmp_path / "l.csv", index=False)
    e1 = _env(price_scenario="duck", acn_date="2019-07-10")
    e2 = _env(price_scenario="historical_lmp", price_csv_file=str(tmp_path / "l.csv"), acn_date="2019-07-10")
    o1, o2 = e1.reset(seed=0), e2.reset(seed=0)
    assert o1.shape == o2.shape and e1.price.shape == e2.price.shape
    assert e2.price_label == "historical:historical_lmp" and e1.price_label == "synthetic:duck"
    assert np.allclose(e2.price, 0.04 + 0.12, atol=1e-6) and not np.allclose(e1.price, e2.price)


def test_historical_falls_back_to_synthetic_when_file_missing():
    e = _env(price_scenario="historical_lmp", price_csv_file="/nonexistent.csv", acn_date="2019-07-10")
    e.reset(seed=0)
    assert np.allclose(e.price, REGISTRY["duck"]("2019-07-10", 24, 1.0, 0))


def test_price_independent_of_env_seed():
    e = _env(price_scenario="volatile", acn_date="2019-07-10")
    e.reset(seed=1); p1 = e.price.copy(); e.reset(seed=2); assert np.array_equal(p1, e.price)


def test_unknown_scenario_is_a_clear_error():
    with pytest.raises(ValueError):
        build_price_model("nope")
