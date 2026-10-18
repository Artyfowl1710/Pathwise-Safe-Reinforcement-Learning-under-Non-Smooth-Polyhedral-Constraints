import numpy as np, pandas as pd
from pifsrl.env.sessions import caiso_rtp_synthetic_price, lmp_csv_price


def test_synthetic_duck_curve_shape_and_determinism():
    p = caiso_rtp_synthetic_price(192, 0.25, "2019-07-10")
    q = caiso_rtp_synthetic_price(192, 0.25, "2019-07-10")
    assert np.allclose(p, q) and p.shape == (192,)
    h = np.arange(192) * 0.25
    assert p[(h >= 17) & (h < 21)].mean() > p[(h >= 11) & (h < 15)].mean() + 0.02   # evening ramp above solar dip


def test_lmp_csv_roundtrip_and_graceful_fallback(tmp_path):
    ts = pd.date_range("2019-07-09 20:00", periods=96, freq="h", tz="UTC")          # covers 2019-07-10 local day
    pd.DataFrame({"INTERVALSTARTTIME_GMT": ts.strftime("%Y-%m-%dT%H:%M:%S-00:00"), "MW": 40.0}).to_csv(tmp_path / "l.csv", index=False)
    p = lmp_csv_price(str(tmp_path / "l.csv"), 96, 0.25, "2019-07-10", adder=0.1)
    assert p is not None and np.allclose(p, 0.04 + 0.1, atol=1e-6)
    assert lmp_csv_price(str(tmp_path / "missing.csv"), 96, 0.25, "2019-07-10") is None          # caller falls back to synthetic
    assert lmp_csv_price(str(tmp_path / "l.csv"), 96, 0.25, "2021-01-01") is None                 # date not covered
