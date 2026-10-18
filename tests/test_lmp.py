import numpy as np, pandas as pd
from pifsrl.env.sessions import lmp_csv_price


def _csv(tmp_path):
    rows = []
    for d in pd.date_range("2021-09-14", "2021-09-20", freq="D"):
        for h in range(24):
            ts = (d + pd.Timedelta(hours=h + 7)).strftime("%Y-%m-%dT%H:%M:00-00:00")   # 07:00Z = 00:00 PDT
            rows += [dict(INTERVALSTARTTIME_GMT=ts, MARKET_RUN_ID="DAM", LMP_TYPE="LMP", NODE="TH_SP15_GEN-APND", MW=30 + h),
                     dict(INTERVALSTARTTIME_GMT=ts, MARKET_RUN_ID="DAM", LMP_TYPE="MCC", NODE="TH_SP15_GEN-APND", MW=999),
                     dict(INTERVALSTARTTIME_GMT=ts, MARKET_RUN_ID="RTM", LMP_TYPE="LMP", NODE="TH_SP15_GEN-APND", MW=500)]
    p = tmp_path / "lmp.csv"; pd.DataFrame(rows).to_csv(p, index=False); return str(p)


def test_raw_oasis_filters_type_and_market(tmp_path):
    p = _csv(tmp_path)
    v = lmp_csv_price(p, 96, 0.25, "2021-09-15", adder=0.0)
    assert v is not None and abs(v[0] - 0.030) < 1e-6 and abs(v[4 * 10] - 0.040) < 1e-6      # DAM LMP only, not MCC/RTM
    assert abs(lmp_csv_price(p, 96, 0.25, "2021-09-15", adder=0.0, market="RTM")[0] - 0.5) < 1e-6


def test_missing_date_none_unless_analog(tmp_path):
    p = _csv(tmp_path)
    assert lmp_csv_price(p, 96, 0.25, "2019-09-17", adder=0.0) is None
    a = lmp_csv_price(p, 96, 0.25, "2019-09-17", adder=0.0, analog=True)
    assert a is not None and abs(a[0] - 0.030) < 1e-6
    assert lmp_csv_price(p, 96, 0.25, "2019-03-05", adder=0.0, analog=True) is None        # no same-month day exists


def test_analog_any_month(tmp_path):
    p = _csv(tmp_path)
    a = lmp_csv_price(p, 96, 0.25, "2019-03-05", adder=0.0, analog="any")
    assert a is not None and abs(a[0] - 0.030) < 1e-6
