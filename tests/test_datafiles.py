import numpy as np, pandas as pd
from pifsrl.env.datafiles import load_tariff, load_profile_shape, load_battery_dist, apply_battery, _parse_months
from pifsrl.env.sessions import Session


def _tariff(tmp_path):
    rows = [
        dict(site="caltech", effective_from="2018-01-01", effective_to="2021-12-31", months="6-9", daytype="weekday", start_hour=0, end_hour=12, energy_usd_per_kwh=0.10, demand_usd_per_kw_month=15.0),
        dict(site="caltech", effective_from="2018-01-01", effective_to="2021-12-31", months="6-9", daytype="weekday", start_hour=12, end_hour=18, energy_usd_per_kwh=0.30, demand_usd_per_kw_month=15.0),
        dict(site="caltech", effective_from="2018-01-01", effective_to="2021-12-31", months="6-9", daytype="weekday", start_hour=18, end_hour=24, energy_usd_per_kwh=0.12, demand_usd_per_kw_month=15.0),
        dict(site="caltech", effective_from="2018-01-01", effective_to="2021-12-31", months="10-5", daytype="all", start_hour=0, end_hour=24, energy_usd_per_kwh=0.09, demand_usd_per_kw_month=12.0),
    ]
    p = tmp_path / "tariff_caltech.csv"; pd.DataFrame(rows).to_csv(p, index=False); return str(p)


def test_months_parsing_wraps():
    assert _parse_months("6-9") == [6, 7, 8, 9] and _parse_months("10-5") == [10, 11, 12, 1, 2, 3, 4, 5] and _parse_months("all") == list(range(1, 13))


def test_tariff_prices_demand_and_gaps(tmp_path):
    t = load_tariff(_tariff(tmp_path), "caltech")
    p = t.price("2019-07-10", 96, 0.25)                       # Wednesday in summer
    h = np.arange(96) * 0.25
    assert np.allclose(p[h < 12], 0.10) and np.allclose(p[(h >= 12) & (h < 18)], 0.30) and np.allclose(p[h >= 18], 0.12)
    assert abs(t.demand_per_day("2019-07-10") - 15.0 / 30.4) < 1e-9
    assert t.price("2019-07-13", 96, 0.25) is None            # summer weekend not covered -> caller falls back
    assert np.allclose(t.price("2019-01-15", 96, 0.25), 0.09)  # winter via wrapped months


def test_missing_or_bad_files_return_none(tmp_path):
    assert load_tariff(str(tmp_path / "nope.csv")) is None and load_battery_dist(None) is None
    bad = tmp_path / "b.csv"; pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    assert load_tariff(str(bad)) is None and load_battery_dist(str(bad)) is None
    assert load_profile_shape(str(bad), 96, 0.25, "2019-07-10") is None


def test_load_shape_normalised(tmp_path):
    ts = pd.date_range("2019-07-09 20:00", periods=48, freq="h", tz="UTC")
    pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"), "load": np.linspace(10, 30, 48)}).to_csv(tmp_path / "l.csv", index=False)
    s = load_profile_shape(str(tmp_path / "l.csv"), 96, 0.25, "2019-07-10")
    assert s is not None and s.max() == 1.0 and s.min() >= 0.05


def test_battery_what_if_keeps_requested_energy(tmp_path):
    pd.DataFrame({"model": ["a", "b"], "capacity_kwh": [40, 80], "weight": [1, 1]}).to_csv(tmp_path / "b.csv", index=False)
    d = load_battery_dist(str(tmp_path / "b.csv"))
    ss = [Session(0, 1, 0, 10, 0.0, 1.0, 12.0, 6.6, float("nan")), Session(1, 1, 0, 10, 0.0, 1.0, 70.0, 6.6, float("nan"))]
    apply_battery(ss, d, np.random.default_rng(1))
    for s, e in zip(ss, (12.0, 70.0)):
        assert abs(s.energy_req_kwh - e) < 1e-6 and s.cap_kwh >= e / 0.95 - 1e-9 and 0 <= s.soc_arr <= 0.95
