"""Drop-in data files with graceful fallbacks.

Every loader returns ``None`` (never raises) when its file is missing, unreadable, or does not cover the requested
date; the caller then uses a *labelled* fallback and ``warn_once`` prints exactly one line per (kind, reason).
Formats are specified in datasets.md / data/README.md and validated by ``scripts/check_data.py``.

  tariff   data/tariff_<site>.csv   historical retail TOU schedule (+ optional demand charge)
  lmp      data/caiso_sp15_lmp.csv  wholesale prices (see sessions.lmp_csv_price)
  load     data/feeder_load.csv     feeder base-load time series (shape only; level comes from the feeder case)
  battery  data/battery_models.csv  EV battery-capacity distribution for the V2G what-if
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_WARNED: set = set()


def warn_once(kind: str, msg: str) -> None:
    key = (kind, msg)
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        from ..runlog import say
        say(f"[data:{kind}] {msg}")
    except Exception:                                    # pragma: no cover
        print(f"[data:{kind}] {msg}")


# --------------------------------------------------------------------------- tariff
TARIFF_COLUMNS = ["site", "effective_from", "effective_to", "months", "daytype", "start_hour", "end_hour",
                  "energy_usd_per_kwh", "demand_usd_per_kw_month"]


def _parse_months(spec) -> List[int]:
    s = str(spec).strip().lower()
    if s in ("", "all", "nan"):
        return list(range(1, 13))
    if "," in s:
        return [int(x) for x in s.split(",")]
    if "-" in s:
        a, b = (int(x) for x in s.split("-"))
        return list(range(a, b + 1)) if a <= b else list(range(a, 13)) + list(range(1, b + 1))
    return [int(s)]


class Tariff:
    """Historical retail TOU tariff. ``price(date, horizon, dt_h)`` -> $/kWh array or None if the schedule does not
    fully cover the day; ``demand_per_day(date)`` -> $/kW-day (monthly demand charge / 30.4) or None."""

    def __init__(self, df, site: Optional[str] = None):
        import pandas as pd
        df = df.copy()
        if site is not None and "site" in df.columns:
            sel = df[df["site"].astype(str).str.lower().isin([site.lower(), "all", "nan", ""])]
            df = sel if len(sel) else df
        df["_from"] = pd.to_datetime(df["effective_from"], errors="coerce")
        df["_to"] = pd.to_datetime(df["effective_to"], errors="coerce")
        self.df = df.reset_index(drop=True)

    def _rows(self, date: str):
        d = _dt.date.fromisoformat(date)
        ts = np.datetime64(d)
        wk = "weekday" if d.weekday() < 5 else "weekend"
        out = []
        for _, r in self.df.iterrows():
            if r["_from"] == r["_from"] and np.datetime64(r["_from"].date()) > ts:
                continue
            if r["_to"] == r["_to"] and np.datetime64(r["_to"].date()) < ts:
                continue
            if d.month not in _parse_months(r.get("months", "all")):
                continue
            dt_ = str(r.get("daytype", "all")).strip().lower()
            if dt_ not in ("all", "nan", "", wk):
                continue
            out.append(r)
        return out

    def price(self, date: str, horizon: int, dt_h: float) -> Optional[np.ndarray]:
        rows = self._rows(date)
        if not rows:
            return None
        t = np.arange(horizon) * dt_h % 24.0
        p = np.full(horizon, np.nan)
        for r in rows:
            a, b = float(r["start_hour"]), float(r["end_hour"])
            m = (t >= a) & (t < b) if a <= b else ((t >= a) | (t < b))
            p[m & np.isnan(p)] = float(r["energy_usd_per_kwh"])
        return None if np.isnan(p).any() else p

    def demand_per_day(self, date: str) -> Optional[float]:
        if "demand_usd_per_kw_month" not in self.df.columns:
            return None
        vals = [float(r["demand_usd_per_kw_month"]) for r in self._rows(date)
                if r["demand_usd_per_kw_month"] == r["demand_usd_per_kw_month"]]
        return (max(vals) / 30.4) if vals else None


def load_tariff(path: Optional[str], site: Optional[str] = None) -> Optional[Tariff]:
    if not path or not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        need = {"effective_from", "start_hour", "end_hour", "energy_usd_per_kwh"}
        if not need.issubset(df.columns):
            warn_once("tariff", f"{path}: missing columns {sorted(need - set(df.columns))} -> fallback")
            return None
        return Tariff(df, site)
    except Exception as e:                                 # pragma: no cover
        warn_once("tariff", f"{path}: unreadable ({e}) -> fallback")
        return None


# --------------------------------------------------------------------------- feeder load shape
def load_profile_shape(path: Optional[str], horizon: int, dt_h: float, date: str,
                       tz: str = "America/Los_Angeles") -> Optional[np.ndarray]:
    """Peak-normalised (<=1) shape for ``date`` from ``timestamp,load`` rows; None if the file/date is missing."""
    if not path or not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        tcol = next((c for c in ("timestamp", "time", "datetime", "INTERVALSTARTTIME_GMT") if c in df.columns), None)
        lcol = next((c for c in ("load", "load_mw", "load_kw", "MW", "value") if c in df.columns), None)
        if tcol is None or lcol is None:
            warn_once("load", f"{path}: need a timestamp column and a load column -> generator fallback")
            return None
        ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert(tz)
        s = pd.Series(pd.to_numeric(df[lcol], errors="coerce").values, index=ts).dropna().sort_index()
        day = pd.Timestamp(date, tz=tz)
        grid = day + pd.to_timedelta(np.arange(horizon) * dt_h, unit="h")
        sub = s[(s.index >= day - pd.Timedelta(hours=2)) & (s.index <= grid[-1] + pd.Timedelta(hours=2))]
        if len(sub) < 12:
            return None
        v = np.interp(grid.as_unit("ns").asi8, sub.index.as_unit("ns").asi8, sub.values)
        return np.clip(v / max(v.max(), 1e-9), 0.05, 1.0)
    except Exception as e:                                 # pragma: no cover
        warn_once("load", f"{path}: unreadable ({e}) -> generator fallback")
        return None


# --------------------------------------------------------------------------- battery distribution
DEFAULT_BATTERY_ASSUMPTION: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((40.0, 60.0, 75.0), (0.40, 0.40, 0.20))


def load_battery_dist(path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path or not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if "capacity_kwh" not in df.columns:
            warn_once("battery", f"{path}: needs a capacity_kwh column -> default behaviour")
            return None
        w = df["weight"].to_numpy(float) if "weight" in df.columns else np.ones(len(df))
        c = df["capacity_kwh"].to_numpy(float)
        ok = (c > 5) & (w > 0)
        if not ok.any():
            return None
        return c[ok], w[ok] / w[ok].sum()
    except Exception as e:                                 # pragma: no cover
        warn_once("battery", f"{path}: unreadable ({e}) -> default behaviour")
        return None


def apply_battery(sessions, dist, rng: np.random.Generator) -> None:
    """V2G what-if: give each real session a battery. Requested energy E is kept; capacity C is sampled (and raised to
    >= E/0.95); arrival SoC = 1 - E/C, target 1.0.  Never called unless a distribution was explicitly enabled."""
    caps, w = dist
    for s in sessions:
        e = s.cap_kwh * max(s.soc_tgt - s.soc_arr, 0.0)
        c = float(rng.choice(caps, p=w))
        c = max(c, e / 0.95)
        s.cap_kwh = c
        s.soc_arr = float(np.clip(1.0 - e / c, 0.0, 0.95))
        s.soc_tgt = 1.0


# --------------------------------------------------------------------------- discovery
FILES = {
    "tariff": "tariff_{site}.csv",
    "lmp": "caiso_sp15_lmp.csv",
    "load": "feeder_load.csv",
    "battery": "battery_models.csv",
}


def discover(site: str, data_dir: str = "data", price: str = "auto", demand_from_tariff: bool = False,
             battery_default: bool = False) -> Dict:
    """EnvConfig kwargs for whatever real files exist in ``data_dir`` (missing ones are simply not wired, so the
    environment falls back by itself).  price in {auto, sce_tou_prime, tariff_csv, caiso_rtp}: ``auto`` = historical tariff
    CSV if present, else the SCE stand-in."""
    P = lambda k: os.path.join(data_dir, FILES[k].format(site=site))
    kw: Dict = {}
    tar, lmp = P("tariff"), P("lmp")
    if price == "auto":
        price = "tariff_csv" if os.path.exists(tar) else "sce_tou_prime"
    kw["price_source"] = price
    if price == "tariff_csv":
        kw["tariff_csv_file"] = tar
        kw["demand_from_tariff"] = bool(demand_from_tariff)
    if price == "caiso_rtp" and os.path.exists(lmp):
        kw["price_csv_file"] = lmp
    if os.path.exists(P("load")):
        kw["load_csv_file"] = P("load")
    if os.path.exists(P("battery")):
        kw["battery_csv_file"] = P("battery")
    if battery_default:
        kw["battery_default"] = True
    return kw
