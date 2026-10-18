"""
Tests for the REAL ACN-Data integration. Unlike every other test file in this
suite, these tests require the actual data files (not included in the
repository -- see datasets.md for why, and where to get them). They are
skipped automatically if the files aren't present, so the rest of the suite
and `sample.py --quicktest` never depend on this data being available.

Run explicitly once you have the files:
    python -m pytest tests/test_real_data.py -v
"""
import os

import numpy as np
import pytest

CALTECH_CSV = os.environ.get("ACN_CALTECH_CSV", "data/caltech_local_sessions.csv")
JPL_CSV = os.environ.get("ACN_JPL_CSV", "data/jpl_local_sessions.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CALTECH_CSV) and os.path.exists(JPL_CSV)),
    reason=f"Real ACN-Data CSVs not found at {CALTECH_CSV} / {JPL_CSV}. "
           f"See datasets.md for how to obtain them from your ACN-Data account.")


def _import():
    from pifsrl.env.sessions import (list_acn_dates, load_acn_sessions_real,
                                     recommended_env_overrides_for_acn, ACN_SITE_INFO)
    return list_acn_dates, load_acn_sessions_real, recommended_env_overrides_for_acn, ACN_SITE_INFO


def test_schema_matches_expected_columns():
    """Pin the exact schema this loader is written against, so a future ACN
    export with different columns fails loudly here instead of silently
    inside the parser."""
    import pandas as pd
    df = pd.read_csv(CALTECH_CSV, nrows=5)
    expected = {"_id", "userInputs", "sessionID", "stationID", "spaceID", "siteID",
               "clusterID", "connectionTime", "disconnectTime", "kWhDelivered",
               "doneChargingTime", "timezone", "userID", "site_location"}
    assert expected.issubset(set(df.columns))


def test_list_acn_dates_covers_known_range():
    list_acn_dates, _, _, _ = _import()
    dates = list_acn_dates(CALTECH_CSV, "caltech")
    assert len(dates) > 500          # ACN-Data spans ~3.5 years at both sites
    assert dates[0] <= "2018-04-26"  # data starts 2018-04-25
    assert dates[-1] >= "2021-09-01"  # data ends 2021-09-14


@pytest.mark.parametrize("site,csv_path", [("caltech", CALTECH_CSV), ("jpl", JPL_CSV)])
def test_real_sessions_load_and_are_physically_sane(site, csv_path):
    list_acn_dates, load_acn_sessions_real, _, _ = _import()
    dates = list_acn_dates(csv_path, site)
    rng = np.random.default_rng(0)
    buses = np.arange(1, 33)
    # pick a date well inside the range with likely activity
    test_date = dates[len(dates) // 2]
    sessions = load_acn_sessions_real(csv_path, site, buses, horizon=192,
                                      dt_h=0.25, date=test_date, rng=rng)
    assert len(sessions) > 0
    for s in sessions:
        assert s.t_dep > s.t_arr, "a real session must have positive dwell"
        assert s.energy_req_kwh > 0, "a real session must have positive requested energy"
        assert s.p_rate_kw > 0
        assert s.opt_in in (0, 1)
        assert s.early_leave in (0, 1)
        # soc_arr/soc_tgt are a bookkeeping device (see loader docstring), not
        # literal battery SoC -- but must still be internally consistent
        assert s.soc_arr == 0.0 and s.soc_tgt == 1.0


def test_no_nan_or_inf_in_real_session_fields():
    list_acn_dates, load_acn_sessions_real, _, _ = _import()
    dates = list_acn_dates(CALTECH_CSV, "caltech")
    rng = np.random.default_rng(0)
    buses = np.arange(1, 33)
    for test_date in dates[100:105]:
        sessions = load_acn_sessions_real(CALTECH_CSV, "caltech", buses, horizon=192,
                                          dt_h=0.25, date=test_date, rng=rng)
        for s in sessions:
            assert np.isfinite(s.energy_req_kwh)
            assert np.isfinite(s.p_rate_kw)


def test_unknown_date_raises_informative_error():
    list_acn_dates, load_acn_sessions_real, _, _ = _import()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="No sessions found"):
        load_acn_sessions_real(CALTECH_CSV, "caltech", np.arange(1, 33), 192, 0.25,
                               date="1999-01-01", rng=rng)


def test_recommended_overrides_prevent_the_four_known_hazards():
    """Regression test for the soc_max / cluster_size / k_max / real_total_capacity_kw
    integration hazards caught while wiring real data into V2GEnv (see sessions.py
    docstring for `recommended_env_overrides_for_acn` and datasets.md)."""
    _, _, recommended_env_overrides_for_acn, ACN_SITE_INFO = _import()
    ov = recommended_env_overrides_for_acn("caltech")
    assert ov["soc_max"] == 1.0, "must not clip real session targets by the battery-SoC margin"
    assert ov["cluster_size"] == 1.0, "must not multiply real per-EV energy by a synthetic cluster factor"
    assert ov["k_max"] == ACN_SITE_INFO["caltech"]["n_evse"], \
        "must not artificially cap concurrency below the real physical station count"
    assert ov["real_total_capacity_kw"] == ACN_SITE_INFO["caltech"]["total_capacity_kw"], \
        "must not leave the hub thermal limit on the synthetic 22kW-per-slot default"


def test_real_data_flows_through_full_environment():
    """End-to-end: real sessions + real tariff through the actual V2GEnv,
    not just the loader in isolation."""
    from pifsrl.env.v2g_env import EnvConfig, V2GEnv
    from pifsrl.env.sessions import list_acn_dates, recommended_env_overrides_for_acn

    dates = list_acn_dates(CALTECH_CSV, "caltech")
    overrides = recommended_env_overrides_for_acn("caltech")
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0,
                   sessions_source="acn_csv", acn_csv_file=CALTECH_CSV,
                   acn_site="caltech", acn_date=dates[500],
                   price_source="sce_tou_prime", **overrides)
    env = V2GEnv(ec)
    obs = env.reset(0)
    assert obs.shape == (env.obs_dim,)
    assert env.k_max == overrides["k_max"]

    lo, hi = env.action_bounds()
    total_viol = 0.0
    done = False
    while not done:
        _, _, done, info = env.step(np.zeros_like(lo))  # idle: must never violate
        total_viol += info.v_violation + info.s_violation
    assert total_viol == 0.0, "idle action on real data must never violate physics"
    assert env.energy_requested_kwh > 0, "a real day should have real recorded demand"


def test_price_source_missing_acn_date_raises():
    """acn_csv sessions_source requires acn_date -- silently defaulting to some
    arbitrary date would make results non-reproducible and untraceable to a
    specific real day."""
    from pifsrl.env.v2g_env import EnvConfig, V2GEnv
    from pifsrl.env.sessions import recommended_env_overrides_for_acn

    overrides = recommended_env_overrides_for_acn("caltech")
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0,
                   sessions_source="acn_csv", acn_csv_file=CALTECH_CSV,
                   acn_site="caltech", acn_date=None, **overrides)
    env = V2GEnv(ec)
    with pytest.raises(ValueError, match="acn_date"):
        env.reset(0)


def test_real_total_capacity_overrides_synthetic_nameplate():
    """Regression test for the hub thermal-limit fix: without
    `real_total_capacity_kw`, s_max was built from a synthetic '22kW per
    slot' guess with no connection to either real site; with it, s_max must
    change and must scale with the real capacity figure (150kW Caltech vs
    195kW JPL)."""
    from pifsrl.env.v2g_env import EnvConfig, V2GEnv
    from pifsrl.env.sessions import recommended_env_overrides_for_acn

    ov_caltech = recommended_env_overrides_for_acn("caltech")
    ov_jpl = recommended_env_overrides_for_acn("jpl")
    assert ov_caltech["real_total_capacity_kw"] == 150.0
    assert ov_jpl["real_total_capacity_kw"] == 195.0

    ec_synthetic = EnvConfig(feeder="ieee33", k_max=54, horizon=48, dt_h=0.25, seed=0)
    ec_real = EnvConfig(feeder="ieee33", k_max=54, horizon=48, dt_h=0.25, seed=0,
                        real_total_capacity_kw=150.0)
    s_max_synthetic = V2GEnv(ec_synthetic).feeder.s_max.copy()
    s_max_real = V2GEnv(ec_real).feeder.s_max.copy()
    assert not np.allclose(s_max_synthetic, s_max_real), \
        "real_total_capacity_kw must actually change the thermal limit, not be ignored"


def test_real_federated_two_sites_share_k_max_for_weight_compatibility():
    """Regression test for the FedAvg dimension-mismatch bug: Caltech (54
    EVSE) and JPL (52 EVSE) have different real k_max, but shared-trunk
    weight averaging requires identical observation dimensions. Both real
    clients must end up with the same (max) k_max, while keeping their own
    distinct real_total_capacity_kw."""
    from pifsrl.env.sessions import recommended_env_overrides_for_acn, ACN_SITE_INFO

    ov_caltech = recommended_env_overrides_for_acn("caltech")
    ov_jpl = recommended_env_overrides_for_acn("jpl")
    assert ov_caltech["k_max"] != ov_jpl["k_max"], \
        "sanity: the two real sites really do have different native EVSE counts"

    shared_k_max = max(ACN_SITE_INFO["caltech"]["n_evse"], ACN_SITE_INFO["jpl"]["n_evse"])
    ov_caltech["k_max"] = shared_k_max
    ov_jpl["k_max"] = shared_k_max

    from pifsrl.env.v2g_env import EnvConfig, V2GEnv
    dates = None
    from pifsrl.env.sessions import list_acn_dates
    common = sorted(set(list_acn_dates(CALTECH_CSV, "caltech")) &
                    set(list_acn_dates(JPL_CSV, "jpl")))
    date = common[len(common) // 2]

    ec_c = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0,
                     sessions_source="acn_csv", acn_csv_file=CALTECH_CSV,
                     acn_site="caltech", acn_date=date, price_source="sce_tou_prime", **ov_caltech)
    ec_j = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0,
                     sessions_source="acn_csv", acn_csv_file=JPL_CSV,
                     acn_site="jpl", acn_date=date, price_source="sce_tou_prime", **ov_jpl)
    env_c, env_j = V2GEnv(ec_c), V2GEnv(ec_j)
    env_c.reset(0)
    env_j.reset(0)
    assert env_c.obs_dim == env_j.obs_dim, \
        "shared k_max must produce matching observation dimensions across real clients"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
