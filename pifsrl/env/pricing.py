"""Pluggable price module.  The ONLY thing the environment, the agents, the baselines and the oracle consume is the array
``env.price`` ($/kWh on the env grid).  Everything that produces that array lives behind the ``PriceModel`` interface, so a
synthetic scenario can be replaced by historical CAISO data (or a tariff CSV) by changing ONE config string -- no change to
any RL agent, baseline, safety layer or evaluation code.

    cfg.price_scenario = "duck"                # synthetic, seeded
    cfg.price_scenario = "historical_lmp"      # real CAISO CSV (cfg.price_csv_file), same interface
    cfg.price_scenario = "historical_tariff"   # real tariff CSV (cfg.tariff_csv_file)

Determinism: a synthetic scenario's output depends ONLY on (scenario name, calendar date, price_seed).  It is independent of the
agent, of the training seed and of the env RNG, so every method is compared on the identical price path for a given day.
Every model carries a ``provenance`` tag in {"synthetic", "historical", "standin"} that is written into result JSONs."""
from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .sessions import caiso_rtp_synthetic_price, lmp_csv_price, sce_tou_prime_price, tou_price


def _rng(name: str, date: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{name}|{date}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _t(horizon, dt_h):
    return np.arange(horizon) * dt_h % 24.0


def _ar1(r, n, phi, sd):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + r.normal(0, sd)
    return e


@dataclass
class PriceModel:
    name: str
    provenance: str                                   # synthetic | historical | standin
    fn: Callable[[str, int, float, int], np.ndarray]  # (date, horizon, dt_h, seed) -> $/kWh
    description: str = ""

    def __call__(self, date: str, horizon: int, dt_h: float, seed: int = 0) -> np.ndarray:
        p = np.asarray(self.fn(date, horizon, dt_h, seed), dtype=np.float64)
        assert p.shape == (horizon,) and np.isfinite(p).all(), f"price model {self.name} returned a bad array"
        return p

    @property
    def label(self) -> str:
        return f"{self.provenance}:{self.name}"


# ------------------------------------------------------------------ synthetic scenarios (all $/kWh, seeded by date)
def _season(date):
    return {6: 1.25, 7: 1.4, 8: 1.4, 9: 1.3}.get(_dt.date.fromisoformat(date).month, 1.0)


def _duck(date, H, dt, seed):
    return caiso_rtp_synthetic_price(H, dt, date, _rng("duck", date, seed), 0.12)


def _volatile(date, H, dt, seed):                      # duck + heavy-tailed scarcity spikes (1-3 events/day, 0.5-2 h, up to ~$1/kWh)
    r = _rng("volatile", date, seed)
    p = caiso_rtp_synthetic_price(H, dt, date, r, 0.12)
    for _ in range(int(r.integers(1, 4))):
        lo_, hi_ = int(14 / dt), min(H, int(22 / dt))
        if lo_ >= hi_:                                   # short test horizons: spikes anywhere in the last third
            lo_, hi_ = H * 2 // 3, H
        c = int(r.integers(lo_, max(hi_, lo_ + 1)))
        w = max(1, int(r.uniform(0.5, 2.0) / dt))
        p[c:c + w] += min(r.pareto(2.5) * 0.15 + 0.10, 0.9)
    return p


def _solar_glut(date, H, dt, seed):                    # deep midday trough (negative wholesale), steep evening ramp
    r = _rng("solar_glut", date, seed); t = _t(H, dt); s = _season(date)
    base = np.interp(t, [0, 5, 8, 10, 12, 15, 17, 19, 21, 24], [35, 30, 20, -15, -30, -10, 55, 95, 60, 35]) * s
    return np.clip(base + _ar1(r, H, 0.9, 5.0), -60, 400) / 1000.0 + 0.12


def _flat(date, H, dt, seed):                          # null scenario: no price signal -> any price-aware gain must vanish
    return np.full(H, 0.20)


def _tou3(date, H, dt, seed):
    return tou_price(H, dt, (0.09, 0.17, 0.41))


def _winter_peak(date, H, dt, seed):                   # morning + evening double peak
    r = _rng("winter_peak", date, seed); t = _t(H, dt)
    base = np.interp(t, [0, 5, 7, 9, 12, 16, 18, 21, 24], [28, 26, 55, 60, 30, 35, 75, 70, 32])
    return np.clip(base + _ar1(r, H, 0.9, 4.0), 5, 300) / 1000.0 + 0.12


# ------------------------------------------------------------------ adapters for real data (same interface)
def historical_lmp(path: Optional[str], adder: float = 0.12, market: str = "auto", analog=False, fallback: str = "duck") -> PriceModel:
    def fn(date, H, dt, seed):
        v = lmp_csv_price(path, H, dt, date, adder, market=market, analog=analog) if path else None
        return v if v is not None else REGISTRY[fallback](date, H, dt, seed)
    return PriceModel("historical_lmp", "historical", fn, f"CAISO LMP CSV {path}; missing dates fall back to '{fallback}' (synthetic)")


def historical_tariff(path: Optional[str], site: Optional[str], fallback: str = "sce_tou_prime") -> PriceModel:
    from .datafiles import load_tariff
    tar = load_tariff(path, site)

    def fn(date, H, dt, seed):
        v = tar.price(date, H, dt) if tar is not None else None
        return v if v is not None else sce_tou_prime_price(H, dt, date)
    return PriceModel("historical_tariff", "historical", fn, f"tariff CSV {path}; uncovered dates fall back to the SCE 2026 stand-in")


REGISTRY: Dict[str, PriceModel] = {}


def register_price_model(m: PriceModel) -> PriceModel:
    REGISTRY[m.name] = m
    return m


for _n, _f, _d in (("duck", _duck, "CAISO-like duck curve + AR(1)"), ("volatile", _volatile, "duck + heavy-tailed scarcity spikes"),
                   ("solar_glut", _solar_glut, "negative midday wholesale, steep evening ramp"), ("flat", _flat, "constant 0.20 $/kWh (null scenario)"),
                   ("tou3", _tou3, "3-tier TOU 0.09/0.17/0.41"), ("winter_peak", _winter_peak, "morning+evening double peak")):
    register_price_model(PriceModel(_n, "synthetic", _f, _d))
register_price_model(PriceModel("sce_tou_prime", "standin", lambda d, H, dt, s: sce_tou_prime_price(H, dt, d), "SCE TOU-D-PRIME 2026 residential stand-in"))

SYNTHETIC_SCENARIOS = ["duck", "volatile", "solar_glut", "winter_peak", "tou3", "flat"]


def build_price_model(name: str, *, csv: Optional[str] = None, tariff_csv: Optional[str] = None, site: Optional[str] = None,
                      adder: float = 0.12, market: str = "auto", analog=False) -> PriceModel:
    if name == "historical_lmp":
        return historical_lmp(csv, adder, market, analog)
    if name == "historical_tariff":
        return historical_tariff(tariff_csv, site)
    if name not in REGISTRY:
        raise ValueError(f"unknown price scenario '{name}'. Known: {sorted(REGISTRY)} + historical_lmp, historical_tariff")
    return REGISTRY[name]
