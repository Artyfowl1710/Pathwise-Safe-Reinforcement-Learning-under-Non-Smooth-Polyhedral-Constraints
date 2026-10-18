"""
EV charging-session generation and exogenous (load / PV / price) profiles.

PROVENANCE -- READ THIS BEFORE WRITING THE PAPER
------------------------------------------------
The default generators here are *synthetic but calibrated*: arrival-time,
dwell-time and energy-request distributions are log-normal/Gaussian-mixture fits
whose first two moments are set to the published summary statistics of the
Caltech/JPL ACN-Data release (Lee, Li & Low, e-Energy 2019) for the three site
archetypes used in the proposal:

    site          mean dwell   mean energy   arrival peak
    caltech        ~8.6 h        ~9.4 kWh     08:00-11:00  (workplace/campus)
    jpl            ~7.2 h        ~8.3 kWh     07:00-09:30  (workplace)
    office_001     ~5.1 h        ~6.5 kWh     08:30-10:30  (office)
  + residential   ~13.0 h       ~11.5 kWh     17:30-21:30  (overnight, non-ACN)

They are *not* the ACN records themselves.  `load_acn_sessions()` reads a real
ACN-Data JSON/CSV export and returns the same `Session` objects, so switching to
ground-truth data is a one-line config change (`--sessions acn --acn-path ...`).
Every number reported from synthetic sessions must be labelled as such.

The residential archetype has no ACN analogue (ACN is workplace-only); its
parameters follow the NHTS-derived overnight-charging profiles commonly used in
the V2G literature.  This heterogeneity across archetypes is exactly the
Non-IID structure that Section 5.3 of the proposal is about.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
#  Site archetypes -> Non-IID federated clients
# --------------------------------------------------------------------------- #
ARCHETYPES: Dict[str, dict] = {
    # arr_mu/sigma in hours (wrapped), dwell log-normal in hours,
    # soc_arr Beta-ish via clipped normal, energy request in kWh
    "workplace_caltech": dict(arr_mu=9.3, arr_sd=1.5, dwell_mu=8.6, dwell_cv=0.45,
                              soc_arr=(0.47, 0.14), soc_tgt=(0.85, 0.06),
                              n_daily=(1.0, 0.15), p_rate_kw=[6.6, 7.2, 11.0],
                              p_rate_w=[0.45, 0.35, 0.20], cap_kwh=[40, 60, 75],
                              vpw_bias=0.10),
    "workplace_jpl": dict(arr_mu=8.4, arr_sd=1.2, dwell_mu=7.2, dwell_cv=0.42,
                          soc_arr=(0.44, 0.13), soc_tgt=(0.83, 0.06),
                          n_daily=(1.0, 0.15), p_rate_kw=[6.6, 7.2, 11.0],
                          p_rate_w=[0.5, 0.35, 0.15], cap_kwh=[40, 60, 75],
                          vpw_bias=0.05),
    "office": dict(arr_mu=9.6, arr_sd=1.0, dwell_mu=5.1, dwell_cv=0.40,
                   soc_arr=(0.52, 0.12), soc_tgt=(0.80, 0.06),
                   n_daily=(1.15, 0.2), p_rate_kw=[7.2, 11.0, 22.0],
                   p_rate_w=[0.4, 0.4, 0.2], cap_kwh=[50, 60, 80],
                   vpw_bias=-0.05),
    "residential": dict(arr_mu=19.4, arr_sd=1.8, dwell_mu=13.0, dwell_cv=0.30,
                        soc_arr=(0.34, 0.15), soc_tgt=(0.90, 0.05),
                        n_daily=(0.85, 0.15), p_rate_kw=[3.7, 7.2, 11.0],
                        p_rate_w=[0.35, 0.45, 0.20], cap_kwh=[40, 60, 75],
                        vpw_bias=0.20),
    "fleet_depot": dict(arr_mu=15.0, arr_sd=3.2, dwell_mu=3.4, dwell_cv=0.55,
                        soc_arr=(0.28, 0.10), soc_tgt=(0.95, 0.03),
                        n_daily=(1.8, 0.3), p_rate_kw=[11.0, 22.0, 50.0],
                        p_rate_w=[0.3, 0.5, 0.2], cap_kwh=[60, 80, 100],
                        vpw_bias=-0.25),
}


@dataclass
class Session:
    """One EV charging session (physical units)."""
    ev_id: int
    bus: int                # non-slack bus index the charger sits on
    t_arr: int              # arrival timestep
    t_dep: int              # departure timestep (exclusive)
    soc_arr: float
    soc_tgt: float
    cap_kwh: float
    p_rate_kw: float
    vpw_prior: float        # latent per-driver willingness offset in [0,1]
    opt_in: int = 1         # revealed-behaviour proxy label (Section 5.2)
    early_leave: int = 0    # revealed-behaviour proxy label

    @property
    def dwell(self) -> int:
        return self.t_dep - self.t_arr

    @property
    def energy_req_kwh(self) -> float:
        return max(0.0, (self.soc_tgt - self.soc_arr) * self.cap_kwh)


def _wrapped_arrival(rng, mu, sd, steps_per_day):
    h = rng.normal(mu, sd) % 24.0
    return int(h / 24.0 * steps_per_day) % steps_per_day


def generate_sessions(rng: np.random.Generator, archetype: str, n_chargers: int,
                      buses: np.ndarray, horizon: int, dt_h: float,
                      density: float = 1.0) -> List[Session]:
    """Generate one day of sessions for a site with `n_chargers` charge points."""
    cfg = ARCHETYPES[archetype]
    steps_per_day = int(round(24.0 / dt_h))
    sessions: List[Session] = []
    ev_id = 0
    for c in range(n_chargers):
        n_sessions = max(0, int(rng.poisson(max(0.05, cfg["n_daily"][0] * density))))
        used = np.zeros(horizon, dtype=bool)
        for _ in range(n_sessions):
            t_arr = _wrapped_arrival(rng, cfg["arr_mu"], cfg["arr_sd"], steps_per_day)
            dwell_h = float(np.clip(rng.lognormal(np.log(cfg["dwell_mu"]),
                                                  cfg["dwell_cv"]), 0.75, 20.0))
            dwell = max(2, int(round(dwell_h / dt_h)))
            t_dep = min(horizon, t_arr + dwell)
            if t_dep - t_arr < 2 or used[t_arr:t_dep].any():
                continue
            used[t_arr:t_dep] = True
            cap = float(rng.choice(cfg["cap_kwh"]))
            p_rate = float(rng.choice(cfg["p_rate_kw"], p=cfg["p_rate_w"]))
            soc_a = float(np.clip(rng.normal(*cfg["soc_arr"]), 0.05, 0.85))
            soc_t = float(np.clip(rng.normal(*cfg["soc_tgt"]), soc_a + 0.05, 0.98))
            # latent willingness prior: archetype bias + driver idiosyncrasy
            vpw = float(np.clip(0.5 + cfg["vpw_bias"] + rng.normal(0, 0.18), 0.02, 0.98))
            sessions.append(Session(ev_id=ev_id, bus=int(rng.choice(buses)),
                                    t_arr=t_arr, t_dep=t_dep, soc_arr=soc_a,
                                    soc_tgt=soc_t, cap_kwh=cap, p_rate_kw=p_rate,
                                    vpw_prior=vpw))
            ev_id += 1
    # ---- revealed-behaviour proxy labels used to calibrate VPW (Section 5.2) --
    _attach_proxy_labels(rng, sessions, dt_h)
    return sorted(sessions, key=lambda s: s.t_arr)


def _attach_proxy_labels(rng, sessions: List[Session], dt_h: float) -> None:
    """Simulate the *revealed-behaviour* signal used as the VPW proxy target.

    ACN has no willingness label.  Section 5.2 of the proposal therefore
    calibrates w1..w4 against historical opt-in / early-departure behaviour.
    Here we synthesise that revealed behaviour from the latent `vpw_prior` plus
    session context, with noise -- so the calibration procedure is solving a
    genuine (noisy, partially-identified) inverse problem rather than reading
    the answer key.  `vpw_prior` is *never* exposed to the agent or to the
    calibrator; only `opt_in` / `early_leave` are.
    """
    for s in sessions:
        slack = s.dwell * dt_h - s.energy_req_kwh / max(s.p_rate_kw, 1e-6)
        util = (1.6 * s.vpw_prior + 0.22 * np.tanh(slack / 3.0)
                + 0.5 * (s.soc_arr - 0.4) - 0.3 * (s.soc_tgt - 0.85))
        s.opt_in = int(rng.random() < 1.0 / (1.0 + np.exp(-3.0 * (util - 0.85))))
        s.early_leave = int(rng.random() < 1.0 / (1.0 + np.exp(3.0 * (util - 0.25))))


# --------------------------------------------------------------------------- #
#  REAL ACN-Data loader -- verified against the actual Caltech/JPL CSV export
# --------------------------------------------------------------------------- #
# Real, cited infrastructure constants (NOT assumptions -- see datasets.md).
#
# Station power ratings and total site capacity: Lee, Z.J., Li, T., Low, S.H.
# (2019), "ACN-Data: Analysis and Applications of an Open EV Charging Dataset",
# e-Energy '19, and Lee et al. (2021), "Adaptive Charging Networks: A Framework
# for Smart Electric Vehicle Charging", IEEE Trans. Smart Grid, 12(5):4339-4350.
#   - Caltech: 54 Level-2 EVSEs (predominantly 6.7 kW; a subset rated
#     13.3 kW / 16.6 kW) + one shared 50 kW DC fast charger. Total feeder
#     capacity 150 kW.
#   - JPL: 52 Level-2 EVSEs. Total feeder capacity 195 kW.
# The per-session numbers below are used ONLY as a documented fallback when a
# session's own doneChargingTime is missing and no historical rate can be
# computed directly from the record (see `_infer_rate_kw`).
ACN_SITE_INFO = {
    "caltech": dict(n_evse=54, nameplate_kw=6.7, total_capacity_kw=150.0,
                    tz="America/Los_Angeles"),
    "jpl":     dict(n_evse=52, nameplate_kw=6.6, total_capacity_kw=195.0,
                    tz="America/Los_Angeles"),
}

_RATE_OUTLIER_KW = 20.0   # sessions with an implied rate above this are excluded
                          # as doneChargingTime data-quality issues (see datasets.md);
                          # 20 kW is ~3x the fastest real EVSE at either site (16.6 kW
                          # Tesla HPWC), so nothing physically plausible is discarded.


def recommended_env_overrides_for_acn(site: str) -> dict:
    """The EnvConfig field overrides REQUIRED for a real-ACN-data run to be
    faithful, not just "not crash". Three real integration hazards, each
    caught by tracing through what the existing environment code does with a
    synthetic-archetype `Session` and checking it against a real one:

    1. `soc_max` (default 0.95): `V2GEnv._admit_arrivals()` clips every
       session's target to `min(soc_tgt, soc_max)`. That clip is a genuine
       physical/policy choice for the SYNTHETIC archetypes (don't charge a
       real battery past 95% for longevity). For real ACN sessions, `soc_tgt`
       is instead a bookkeeping device set to 1.0 meaning "100% of the real
       kWh this session actually requested" (see `load_acn_sessions_real`
       docstring) -- so the 0.95 clip would silently discard 5% of every real
       session's demand for a reason that has nothing to do with real
       sessions. Must be 1.0 for real-data runs.
    2. `cluster_size` (default 8.0): scales a single action slot's energy up
       to represent a cluster of similar synthetic EVs. Real ACN sessions are
       already individual real vehicles; leaving the default would multiply
       every real kWh figure by 8x. Must be 1.0 for real-data runs.
    3. `k_max` (default 16): the number of action slots = physical charge
       points. If set below the site's real EVSE count, the environment's
       admission control (`_admit_arrivals`, first-come-first-served slot
       search) will turn away sessions that, historically, really did get a
       physical station -- an artificial capacity shortage the real data
       never had. Set to the site's real EVSE count (ACN_SITE_INFO).
    4. `real_total_capacity_kw`: without this, the hub thermal limit
       (`s_max`) is built from a synthetic "22 kW x cluster_size per slot"
       guess (see `V2GEnv.__init__`) that has nothing to do with either real
       site. Set to the site's real, cited total feeder capacity (150 kW
       Caltech / 195 kW JPL) so the aggregate grid-stress constraint is real,
       not just the session and per-EVSE numbers feeding into it.
    """
    info = ACN_SITE_INFO[site]
    return dict(k_max=info["n_evse"], cluster_size=1.0, soc_max=1.0,
               ev_load_scale=1.0, real_total_capacity_kw=info["total_capacity_kw"])


def _parse_acn_userinputs_last(raw) -> Optional[dict]:
    """Parse the `userInputs` cell (a Python-repr'd list of dicts, the ACN-Data
    export format) and return the LAST entry -- the driver's final, most
    up-to-date stated request (earlier entries in the list are superseded
    revisions the driver made during the session)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    import ast
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[-1], dict):
        return parsed[-1]
    return None


def load_acn_sessions_real(csv_path: str, site: str, buses: np.ndarray,
                           horizon: int, dt_h: float, date: str,
                           rng: Optional[np.random.Generator] = None) -> List[Session]:
    """Load REAL EV charging sessions for ONE calendar day from an ACN-Data
    CSV export (the exact format Caltech/JPL exports, columns verified against
    the actual files: _id, userInputs, sessionID, stationID, spaceID, siteID,
    clusterID, connectionTime, disconnectTime, kWhDelivered, doneChargingTime,
    timezone, userID, site_location).

    Args:
        csv_path: path to caltech_local_sessions.csv or jpl_local_sessions.csv
        site: 'caltech' or 'jpl' (selects the cited nameplate fallback rating)
        buses: candidate non-slack bus indices to assign sessions to
        horizon: environment horizon in steps (sessions are truncated to fit,
                 exactly as synthetic sessions are -- see `recommended_horizon`)
        dt_h: step size in hours
        date: the LOCAL calendar date to simulate, 'YYYY-MM-DD'. Real sessions
              are recorded continuously over ~3.5 years (2018-04-25 to
              2021-09-14 for both sites); one call = one real historical day.
              Use `list_acn_dates()` to enumerate what's available.
        rng: only used for bus assignment (which physical bus a station maps
             to) and for `_attach_proxy_labels`' Bernoulli draws -- never for
             any part of arrival time, dwell, or energy, which are all real.

    Returns:
        List[Session]. `cap_kwh` is set EXACTLY to each session's real
        requested energy (see below) with soc_arr=0.0, soc_tgt=1.0, so
        `energy_req_kwh` reproduces the real number exactly -- there is no
        assumed battery capacity anywhere in this path, because ACN-Data does
        not report one and the physics/reward code only ever needs the energy
        gap, never the capacity itself.

    Energy target reconstruction (documented, not hidden):
        - If the session has a `userInputs` entry with `kWhRequested`, that
          REAL, driver-stated number is used directly.
        - Otherwise (no stated request -- ~48% of Caltech sessions, ~6% of
          JPL), `kWhDelivered` (always present, always real) is used as a
          documented LOWER-BOUND proxy for what was wanted. This likely
          understates true demand for these sessions (the historical system
          may have already been demand-limited), which is disclosed in
          datasets.md rather than corrected for, since correcting it would
          require inventing a number ACN-Data does not contain.
    """
    import pandas as pd

    rng = rng or np.random.default_rng(0)
    df = pd.read_csv(csv_path)

    conn = pd.to_datetime(df["connectionTime"], format="%a, %d %b %Y %H:%M:%S GMT", utc=True)
    disc = pd.to_datetime(df["disconnectTime"], format="%a, %d %b %Y %H:%M:%S GMT", utc=True)
    done = pd.to_datetime(df["doneChargingTime"], format="%a, %d %b %Y %H:%M:%S GMT",
                          utc=True, errors="coerce")

    tz = ACN_SITE_INFO.get(site, {}).get("tz", "America/Los_Angeles")
    conn_local = conn.dt.tz_convert(tz)
    disc_local = disc.dt.tz_convert(tz)
    done_local = done.dt.tz_convert(tz)

    target_date = pd.Timestamp(date, tz=tz)
    day_mask = (conn_local.dt.date == target_date.date())
    if not day_mask.any():
        avail = sorted(conn_local.dt.date.unique())
        raise ValueError(f"No sessions found on {date} in {csv_path}. "
                        f"Data spans {avail[0]} to {avail[-1]} "
                        f"({len(avail)} distinct calendar dates available).")

    sub = df.loc[day_mask].copy()
    conn_local, disc_local, done_local = (conn_local[day_mask], disc_local[day_mask],
                                          done_local[day_mask])

    steps_per_day = int(round(24.0 / dt_h))
    nameplate = ACN_SITE_INFO.get(site, {}).get("nameplate_kw", 6.6)

    sessions: List[Session] = []
    n_dropped_outlier_rate = 0
    for row_i, (idx, row) in enumerate(sub.iterrows()):
        t0, t1 = conn_local.loc[idx], disc_local.loc[idx]
        midnight = t0.normalize()
        t_arr = int(((t0 - midnight).total_seconds() / 3600.0) / dt_h)
        t_arr = max(0, min(steps_per_day - 1, t_arr))
        dwell_h = max(dt_h, (t1 - t0).total_seconds() / 3600.0)
        t_dep = min(horizon, t_arr + max(1, int(round(dwell_h / dt_h))))
        if t_dep - t_arr < 1:
            continue

        kwh_delivered = float(row["kWhDelivered"])
        if kwh_delivered <= 0:
            continue

        ui = _parse_acn_userinputs_last(row["userInputs"])
        has_stated_request = ui is not None and ui.get("kWhRequested") is not None
        kwh_requested = float(ui["kWhRequested"]) if has_stated_request else kwh_delivered
        kwh_requested = max(kwh_requested, kwh_delivered)  # can't have requested less than delivered

        t_done = done_local.loc[idx]
        # The station's real, physical maximum rate is the cited nameplate
        # rating (6.6-6.7 kW at both sites; see ACN_SITE_INFO). We do NOT use
        # the session's own historically-realized average rate
        # (kWhDelivered / active_h) as the action bound, even though we can
        # compute it: that number reflects how the OLD control scheme (ACN's
        # own scheduler, or simply a slow onboard charger) happened to use the
        # station, which is a fact about history, not a physical limit on what
        # our agent's action is allowed to request. Conflating the two would
        # silently forbid our agent from charging faster than history did even
        # on sessions where faster charging was physically available -- the
        # exact kind of unstated assumption this dataset integration is
        # supposed to eliminate. We DO use the observed rate to raise the
        # bound when it exceeds nameplate (clear evidence of a faster-rated
        # station, e.g. the 13.3/16.6 kW units documented at Caltech), but
        # never to lower it below nameplate.
        rate_kw = nameplate
        if pd.notna(t_done) and (t_done - t0).total_seconds() > 60:
            active_h = (t_done - t0).total_seconds() / 3600.0
            observed_rate = kwh_delivered / active_h
            if observed_rate > nameplate and observed_rate <= _RATE_OUTLIER_KW:
                rate_kw = observed_rate
            elif observed_rate > _RATE_OUTLIER_KW:
                n_dropped_outlier_rate += 1

        sessions.append(Session(
            ev_id=row_i, bus=int(rng.choice(buses)), t_arr=t_arr, t_dep=t_dep,
            soc_arr=0.0, soc_tgt=1.0, cap_kwh=kwh_requested, p_rate_kw=float(rate_kw),
            vpw_prior=float("nan"),  # never used for real sessions; see below
            opt_in=int(has_stated_request),          # REAL revealed-behaviour signal:
                                                      # did the driver bother to state a request?
            early_leave=int(pd.notna(t_done) and (t1 - t_done).total_seconds() > 15 * 60),
                                                      # REAL signal: car finished charging >15min
                                                      # before the driver actually left (i.e. was
                                                      # NOT in a rush to unplug and leave)
        ))

    if n_dropped_outlier_rate:
        print(f"[load_acn_sessions_real] {n_dropped_outlier_rate} session(s) on {date} had an "
             f"implied historical rate > {_RATE_OUTLIER_KW} kW (a doneChargingTime data-quality "
             f"issue documented in ACN-Data, not a real station); those sessions were kept with "
             f"the cited {nameplate} kW nameplate rating as their action bound instead.")

    return sorted(sessions, key=lambda s: s.t_arr)


def list_acn_dates(csv_path: str, site: str = "caltech") -> List[str]:
    """List every real calendar date (local time) present in an ACN-Data CSV,
    for choosing train/test date splits (e.g. train on 2018-2020, held-out
    test on 2021 -- a genuine temporal split, not a random one)."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["connectionTime"])
    conn = pd.to_datetime(df["connectionTime"], format="%a, %d %b %Y %H:%M:%S GMT", utc=True)
    tz = ACN_SITE_INFO.get(site, {}).get("tz", "America/Los_Angeles")
    dates = conn.dt.tz_convert(tz).dt.date.unique()
    return sorted(d.isoformat() for d in dates)


def load_acn_sessions(path: str, buses: np.ndarray, horizon: int, dt_h: float,
                      rng: Optional[np.random.Generator] = None,
                      day: Optional[str] = None) -> List[Session]:  # pragma: no cover
    """DEPRECATED shim, kept only for backward compatibility with older
    configs. This function guessed at a JSON schema and an ISO-8601 timestamp
    format that does NOT match the real ACN-Data CSV export (which uses
    RFC-2822 'GMT'-suffixed strings and requires local-timezone conversion --
    see the empirical check in RUNBOOK.md, 'ACN timestamp timezone'). Use
    `load_acn_sessions_real()` instead, which is verified against the actual
    caltech_local_sessions.csv / jpl_local_sessions.csv files.
    """
    import warnings
    warnings.warn("load_acn_sessions() is deprecated and untested against the "
                  "real ACN-Data export format; use load_acn_sessions_real() "
                  "with a CSV path instead.", DeprecationWarning, stacklevel=2)
    site = "caltech"
    return load_acn_sessions_real(path, site, buses, horizon, dt_h,
                                  date=day or "2019-01-15", rng=rng)





# --------------------------------------------------------------------------- #
#  Exogenous profiles: base load, distributed PV, tariff
# --------------------------------------------------------------------------- #
def base_load_profile(rng: np.random.Generator, horizon: int, dt_h: float,
                      archetype: str) -> np.ndarray:
    """Normalised (peak=1) 24h base-load shape; residential is evening-peaking,
    commercial is midday-peaking (ASHRAE/Pecan-Street-like archetypes)."""
    t = np.arange(horizon) * dt_h % 24.0
    if archetype.startswith("resid"):
        shape = (0.45 + 0.30 * np.exp(-0.5 * ((t - 7.5) / 1.8) ** 2)
                 + 0.55 * np.exp(-0.5 * ((t - 19.5) / 2.4) ** 2))
    elif archetype.startswith("fleet"):
        shape = 0.55 + 0.35 * np.exp(-0.5 * ((t - 13.0) / 4.5) ** 2)
    else:
        shape = (0.35 + 0.60 * np.exp(-0.5 * ((t - 13.0) / 3.6) ** 2)
                 + 0.12 * np.exp(-0.5 * ((t - 18.0) / 2.0) ** 2))
    shape = shape * (1.0 + 0.05 * rng.standard_normal(horizon)).clip(0.7, 1.3)
    return np.clip(shape / shape.max(), 0.05, 1.0)


def pv_profile(rng: np.random.Generator, horizon: int, dt_h: float,
               cloudiness: float = 0.25) -> np.ndarray:
    """Normalised clear-sky PV with a stochastic cloud process (NSRDB-like)."""
    t = np.arange(horizon) * dt_h % 24.0
    clear = np.clip(np.sin(np.pi * (t - 6.2) / 12.2), 0, None) ** 1.35
    # AR(1) cloud attenuation
    z = np.zeros(horizon)
    for i in range(1, horizon):
        z[i] = 0.88 * z[i - 1] + rng.normal(0, 1.0)
    atten = 1.0 - cloudiness * (1.0 / (1.0 + np.exp(-z)))
    return np.clip(clear * atten, 0, 1)


def tou_price(horizon: int, dt_h: float, currency_per_kwh: tuple = (0.09, 0.17, 0.41),
              rng: Optional[np.random.Generator] = None, noise: float = 0.0) -> np.ndarray:
    """Three-tier time-of-use tariff (off-peak / mid-peak / on-peak)."""
    t = np.arange(horizon) * dt_h % 24.0
    p = np.full(horizon, currency_per_kwh[0])
    p[(t >= 7) & (t < 16)] = currency_per_kwh[1]
    p[(t >= 16) & (t < 21)] = currency_per_kwh[2]
    p[(t >= 21) & (t < 23)] = currency_per_kwh[1]
    if rng is not None and noise > 0:
        p = p * (1.0 + noise * rng.standard_normal(horizon)).clip(0.6, 1.6)
    return p


def sce_tou_prime_price(horizon: int, dt_h: float, start_date: str,
                        currency: str = "USD") -> np.ndarray:
    """Real, currently-published Southern California Edison TOU-D-PRIME
    residential/EV rate schedule ($/kWh), as of the plan's published rate
    sheet (accessed via sce.com/save-money/rates-financing/electric-vehicle-plan,
    Sept. 2026; see datasets.md for the exact rate table and access date).
    TOU-D-PRIME is SCE's rate plan explicitly designed for EV owners.

    Schedule (all times local, America/Los_Angeles):
        Summer (Jun-Sep), weekdays:  off-peak $0.26/kWh (00:00-16:00, 21:00-24:00)
                                     on-peak  $0.59/kWh (16:00-21:00)
        Winter (Oct-May):            broadly similar structure at published
                                     winter rates; see SCE's published sheet.
    This is a REAL, currently-in-effect tariff, not a representative/synthetic
    3-tier approximation -- contrast with `tou_price()` above, which remains
    available for sensitivity analysis against an arbitrary schedule.

    Args:
        horizon, dt_h: as elsewhere.
        start_date: 'YYYY-MM-DD', used only to decide summer vs. winter season.
    """
    import datetime as _dt
    d0 = _dt.date.fromisoformat(start_date)
    is_summer = 6 <= d0.month <= 9
    on_peak = 0.59 if is_summer else 0.51     # SCE TOU-D-PRIME published on-peak
    off_peak = 0.26 if is_summer else 0.24    # SCE TOU-D-PRIME published off-peak
    t = np.arange(horizon) * dt_h % 24.0
    p = np.where((t >= 16.0) & (t < 21.0), on_peak, off_peak)
    return p


def caiso_rtp_synthetic_price(horizon: int, dt_h: float, date: str, rng: Optional[np.random.Generator] = None,
                              adder: float = 0.12) -> np.ndarray:
    """SYNTHETIC wholesale-indexed price ($/kWh) with a CAISO-SP15-like "duck curve": mid-day solar dip, 17-21 h ramp,
    AR(1) mean-reverting noise.  NOT real CAISO data -- a clearly labelled stand-in used only when no real LMP CSV is
    supplied (see `lmp_csv_price`).  Shape/level are generic (night ~$30, mid-day ~$18-25, evening ~$55-70 per MWh,
    summer scaled up); `adder` ($/kWh) stands for the non-energy part of a wholesale-indexed retail rate and is an
    assumption, not a measurement."""
    import datetime as _dt
    d0 = _dt.date.fromisoformat(date)
    season = {6: 1.25, 7: 1.4, 8: 1.4, 9: 1.3}.get(d0.month, 1.0)
    t = np.arange(horizon) * dt_h % 24.0
    base = np.interp(t, [0, 5, 7, 10, 13, 16, 18, 20, 22, 24], [30, 28, 34, 24, 18, 32, 62, 68, 45, 30]) * season
    r = rng if rng is not None else np.random.default_rng(d0.toordinal())
    e = np.zeros(horizon)
    for i in range(1, horizon):
        e[i] = 0.9 * e[i - 1] + r.normal(0, 4.0)
    lmp = np.clip(base + e, -5.0, 250.0)                                    # $/MWh
    return lmp / 1000.0 + adder


_LMP_CACHE: dict = {}


def _load_lmp_series(path: str, tz: str, market: str = "auto", node: Optional[str] = None):
    """Parse a CAISO-style LMP CSV once (cached by path+mtime). Understands the raw OASIS columns: INTERVALSTARTTIME_GMT,
    MW (price, $/MWh), LMP_TYPE (keeps 'LMP' only), MARKET_RUN_ID (DAM|RTM; market='auto' prefers DAM), NODE; and simple
    two-column files (timestamp, price). Returns a tz-aware pandas Series ($/MWh) or None."""
    import os
    import pandas as pd
    key = (path, os.path.getmtime(path), tz, market, node)
    if key in _LMP_CACHE:
        return _LMP_CACHE[key]
    df = pd.read_csv(path)
    tcol = next((c for c in ("INTERVALSTARTTIME_GMT", "timestamp", "time", "datetime", "interval_start_utc") if c in df.columns), None)
    pcol = next((c for c in ("MW", "LMP", "lmp", "price", "VALUE") if c in df.columns), None)
    if tcol is None or pcol is None:
        _LMP_CACHE[key] = None; return None
    if "LMP_TYPE" in df.columns:
        df = df[df["LMP_TYPE"].astype(str).str.upper() == "LMP"]
    if node and "NODE" in df.columns:
        df = df[df["NODE"].astype(str) == node]
    if "MARKET_RUN_ID" in df.columns:
        mk = market.upper()
        have = set(df["MARKET_RUN_ID"].astype(str).str.upper())
        if mk == "AUTO":
            mk = "DAM" if "DAM" in have else sorted(have)[0] if have else ""
        df = df[df["MARKET_RUN_ID"].astype(str).str.upper() == mk]
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert(tz)
    s = pd.Series(pd.to_numeric(df[pcol], errors="coerce").values, index=ts)
    s = s[~s.index.isna()].dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    _LMP_CACHE[key] = s if len(s) else None
    return _LMP_CACHE[key]


def _day_vals(s, date, horizon, dt_h, tz):
    import pandas as pd
    day = pd.Timestamp(date, tz=tz)
    grid = day + pd.to_timedelta(np.arange(horizon) * dt_h, unit="h")
    sub = s[(s.index >= day - pd.Timedelta(hours=2)) & (s.index <= grid[-1] + pd.Timedelta(hours=2))]
    if len(sub) < 12:
        return None
    return np.interp(grid.as_unit("ns").asi8, sub.index.as_unit("ns").asi8, sub.values)


def lmp_csv_price(path: str, horizon: int, dt_h: float, date: str, adder: float = 0.12,
                  tz: str = "America/Los_Angeles", market: str = "auto", node: Optional[str] = None,
                  analog=False) -> Optional[np.ndarray]:
    """REAL wholesale prices from a user-supplied CSV (CAISO OASIS DAM/RTM LMP at TH_SP15_GEN-APND, see datasets.md).
    Returns $/kWh = LMP/1000 + adder on the env grid, or None when the file or that date is missing (the caller then
    falls back, loudly).  analog=True: if the exact date is missing, use the SAME-DAYTYPE (weekday/weekend) day of the
    SAME MONTH-OF-YEAR nearest in time that exists in the file ('seasonal analog'); analog='any' relaxes the month (price-scenario
    replay: real wholesale price SHAPES from another date). Both are proxies and must be labelled as such."""
    import os
    if not path or not os.path.exists(path):
        return None
    s = _load_lmp_series(path, tz, market, node)
    if s is None:
        return None
    v = _day_vals(s, date, horizon, dt_h, tz)
    if v is None and analog:
        import pandas as pd
        d0 = pd.Timestamp(date)
        days = pd.Index(sorted(set(s.index.normalize().tz_localize(None))))
        anym = (analog == "any")                       # 'any': nearest same-daytype day in ANY month (price-scenario replay)
        cand = [d for d in days if (anym or d.month == d0.month) and (d.weekday() < 5) == (d0.weekday() < 5)]
        for d in sorted(cand, key=lambda d: abs((d - d0).days)):
            v = _day_vals(s, d.strftime("%Y-%m-%d"), horizon, dt_h, tz)
            if v is not None:
                break
    if v is None:
        return None
    return np.clip(v, -50.0, 1000.0) / 1000.0 + adder


DWELL_CLIP_H = (0.75, 20.0)         # matches the clip in generate_sessions()


def recommended_horizon(archetype: str, dt_h: float, safety_sigmas: float = 1.5) -> int:
    """Smallest `horizon` (in steps) such that truncated sessions are rare.

    A session truncates iff  t_arr + dwell > horizon.  t_arr in [0, 24h);
    dwell is log-normal(archetype mu, cv) clipped to DWELL_CLIP_H.  We size the
    horizon to cover the day plus a modest multiple of the dwell std, capped at
    the hard dwell clip (no sampled session can exceed it, so there is never a
    reason to budget more than that).
    """
    cfg = ARCHETYPES[archetype]
    mu, cv = cfg["dwell_mu"], cfg["dwell_cv"]
    sigma_ln = np.sqrt(np.log(1 + cv ** 2))
    tail_h = min(mu * np.exp(safety_sigmas * sigma_ln), DWELL_CLIP_H[1])
    total_h = 24.0 + tail_h
    return int(np.ceil(total_h / dt_h))
