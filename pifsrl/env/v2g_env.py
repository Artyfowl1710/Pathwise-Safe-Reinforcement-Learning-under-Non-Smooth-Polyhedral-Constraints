"""
Bidirectional V2G coordination environment on a radial ADN.

GROUND TRUTH
------------
`step()` advances the *exact* non-linear system:
  * network state via backward/forward-sweep AC power flow (physics/powerflow.py),
    i.e. Eqs. (1)-(6) of the spec solved to 1e-10 p.u.;
  * battery state via an SoC-dependent-efficiency model
        eta(SoC) = eta0 - a_eff (SoC - 0.5)^2
    which is a standard first-order representation of the real charge-acceptance
    taper.  F_base assumes constant eta (Eq. 8), so this term is a second,
    *non-network* source of residual -- important, because it proves N_theta is
    learning system-level model error and not just LinDistFlow curvature.

INTERFACE
---------
Actions are the per-slot (P, Q) setpoints of every connected EV, normalised to
[-1, 1] and scaled by the charger rating.  The environment exposes
`constraint_context()` returning everything the QP safety layer needs
(R, X, aggregation matrix A, branch matrix M, exogenous bus injections, per-EV
box bounds), so the safety layer never has to reach into environment internals.

Slot semantics: `k_max` physical charge points.  A session occupies one slot for
its whole dwell.  Multiple slots may map to the same bus -- that is the
contention case of Section 3.2(1) and it is the common case, not an edge case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..physics.feeder import Feeder, get_feeder
from ..physics.powerflow import solve_bfs
from .sessions import (ACN_SITE_INFO, ARCHETYPES, Session, base_load_profile,
                       generate_sessions, load_acn_sessions, load_acn_sessions_real,
                       pv_profile, recommended_env_overrides_for_acn,
                       sce_tou_prime_price, tou_price, caiso_rtp_synthetic_price, lmp_csv_price)
from .datafiles import (load_tariff, load_profile_shape, load_battery_dist, apply_battery, warn_once,
                        DEFAULT_BATTERY_ASSUMPTION)
from .vpw import VPWWeights, vpw_features, willingness


@dataclass
class EnvConfig:
    feeder: str = "ieee33"
    archetype: str = "workplace_caltech"
    k_max: int = 16                    # charge points (= action slots)
    horizon: int = 112                 # control-loop length. NOTE: arrivals are
                                       # always confined to the first 24h (see
                                       # sessions._wrapped_arrival), so `horizon`
                                       # must be >= 24h/dt_h + the archetype's
                                       # long-tail dwell, or sessions get
                                       # truncated by the episode boundary and
                                       # every method's achievable service level
                                       # is capped by a calendar artefact rather
                                       # than by policy quality. Use
                                       # `recommended_horizon()` below when
                                       # changing archetype or dt_h.
    dt_h: float = 0.25
    v_min: float = 0.95
    v_max: float = 1.05
    v_sub: float = 1.03           # substation / OLTC setpoint (p.u.)
    thermal_headroom: float = 1.35
    allow_reactive: bool = True        # Q setpoints controllable (4-quadrant inverters)
    soc_min: float = 0.20
    soc_max: float = 0.95
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    eta_curv: float = 0.22             # SoC-dependent efficiency curvature (truth only)
    price_tiers: Tuple[float, float, float] = (0.09, 0.17, 0.41)
    sell_ratio: float = 0.92           # export tariff as a fraction of import
    deg_beta: float = 0.055            # Eq. 9 beta  [currency per (p.u. power)^gamma . h]
    deg_gamma_exp: float = 1.35        # Eq. 9 exponent
    deg_dod: float = 0.020             # Eq. 9 gamma (1-SoC)^2 term
    c_unmet: float = 2.00              # currency per kWh of unserved energy at departure
                                       # (~2x the on-peak tariff: unserved range is
                                       #  worth more to a driver than the kWh cost)
    c_discomfort: float = 0.05         # (1 - W) weighted discharge discomfort
    c_violation: float = 60.0          # SOFT-penalty baselines only
    load_scale: float = 0.75           # multiplier on feeder nominal load
    ev_load_scale: float = 1.0         # multiplier on charger ratings (stress knob)
    n_hubs: int = 3                    # number of charging hubs (see siting note)
    cluster_size: float = 8.0          # EVs represented by ONE action slot (see note)
    conn_factor: float = 0.55          # non-firm connection rating as a fraction of
                                       # downstream charger nameplate capacity
    pv_penetration: float = 0.35       # PV capacity as fraction of feeder peak load
    session_density: float = 1.0
    sessions_source: str = "synthetic"  # "synthetic" | "acn_csv" (real) | "acn" (deprecated JSON path)
    acn_path: Optional[str] = None      # deprecated: only used by sessions_source="acn"
    acn_csv_file: Optional[str] = None  # path to caltech_local_sessions.csv / jpl_local_sessions.csv
    acn_site: str = "caltech"           # "caltech" or "jpl" -- selects real nameplate/capacity constants
    acn_date: Optional[str] = None      # 'YYYY-MM-DD', a real calendar date present in the CSV
    price_source: str = "synthetic_tou"  # "synthetic_tou" | "sce_tou_prime" (cited retail tariff) | "caiso_rtp" (real LMP CSV if price_csv_file is given, else labelled synthetic)
    price_csv_file: Optional[str] = None  # real LMP CSV for price_source="caiso_rtp" (docs/DATASETS.md)
    tariff_csv_file: Optional[str] = None  # data/tariff_<site>.csv: historical retail TOU (+demand) for price_source="tariff_csv"
    demand_from_tariff: bool = False     # take demand_charge ($/kW-day) from the tariff CSV when it provides one
    load_csv_file: Optional[str] = None  # data/feeder_load.csv: real base-load SHAPE (fallback: archetype generator)
    battery_csv_file: Optional[str] = None  # data/battery_models.csv: capacity distribution for the V2G what-if
    battery_default: bool = False        # use the ASSUMED {40,60,75} kWh distribution when no CSV is given (V2G what-if)
    allow_discharge: bool = True         # False = physically unidirectional chargers (p >= 0), in env bounds AND in the oracle/MPC
    price_scenario: str = ""            # pluggable price module (pifsrl/env/pricing.py): duck|volatile|solar_glut|winter_peak|tou3|flat|historical_lmp|historical_tariff; "" = legacy price_source
    price_seed: int = 0                  # synthetic scenarios depend ONLY on (scenario, date, price_seed)
    lmp_analog: object = False             # if the exact date is absent from the LMP CSV, use the nearest same-month same-daytype day (PROXY; label it)
    lmp_market: str = "auto"            # DAM | RTM | auto (prefer DAM) when the CSV has MARKET_RUN_ID
    price_adder: float = 0.12            # $/kWh added to the wholesale price (non-energy part of an indexed retail rate; assumption)
    real_total_capacity_kw: Optional[float] = None  # if set, overrides the synthetic
                                                     # 22kW-per-slot hub nameplate guess with
                                                     # a real, cited feeder capacity (see
                                                     # sessions.ACN_SITE_INFO); used for
                                                     # real-data runs only.
    demand_charge: float = 0.0         # currency per kW of the day's PEAK site charging power (commercial-tariff demand charge); 0 = off
    reward_scale: float = 10.0
    reward_mode: str = "sparse"        # "sparse": unmet charged at departure | "dense": per-step service credit (same return up to a policy-independent constant)
    stochastic_exogenous: bool = True
    seed: int = 0


@dataclass
class StepInfo:
    cost_energy: float = 0.0
    cost_deg: float = 0.0
    cost_unmet: float = 0.0
    cost_discomfort: float = 0.0
    cost_demand: float = 0.0
    v_violation: float = 0.0      # sum over buses of p.u. excursion beyond [v_min,v_max]
    v_violation_max: float = 0.0
    n_v_violations: int = 0
    s_violation: float = 0.0      # p.u. apparent-power overload summed over branches
    n_s_violations: int = 0
    v_min: float = 1.0
    v_max: float = 1.0
    p_ev_total: float = 0.0
    p_grid: float = 0.0
    losses: float = 0.0
    energy_served_kwh: float = 0.0
    n_connected: int = 0
    mean_vpw: float = 0.0
    unmet_kwh: float = 0.0
    requested_kwh: float = 0.0
    qp_slack: float = 0.0


class V2GEnv:
    """Single-site (single-feeder) V2G environment.  One episode = one day."""

    def __init__(self, cfg: EnvConfig, feeder: Optional[Feeder] = None,
                 vpw_weights: Optional[VPWWeights] = None):
        self.cfg = cfg
        self.feeder = feeder if feeder is not None else get_feeder(cfg.feeder)
        self.n_bus = self.feeder.n_bus
        self.k_max = cfg.k_max
        self.vpw_w = vpw_weights or VPWWeights()
        self.rng = np.random.default_rng(cfg.seed)
        self.s_base_kw = self.feeder.s_base_mva * 1e3

        # ---- charger siting -------------------------------------------------
        # Chargers are NOT scattered uniformly.  Real V2G capacity arrives as
        # *hubs* (a depot, a workplace car park) connected at one or two points,
        # and the electrical impact of 1 MW at one deep bus is completely
        # different from 30 kW at each of 32 buses: with R_ii ~ 0.069 p.u. at
        # bus 18, a concentrated 500 kW moves the voltage 0.035 p.u., while the
        # same total power spread uniformly moves it by a third of that.  The
        # scattered layout makes the safety constraint vacuous and would let us
        # report "zero violations" for free.  We therefore site the fleet as
        # `n_hubs` hubs on the electrically weakest buses, which is both
        # realistic and the honest stress case.
        depth = self.feeder.depth
        order = np.argsort(-depth)
        hubs = [int(b) for b in order[:max(1, cfg.n_hubs)]]
        self.hub_buses = np.array(hubs)
        w = np.array([0.6 ** i for i in range(len(hubs))]); w /= w.sum()
        self.charger_buses = np.array(
            [hubs[int(np.argmax(np.cumsum(w) > ((i + 0.5) / cfg.k_max)))] for i in range(cfg.k_max)])
        # ---- branch ratings -------------------------------------------------
        # Two components, because both are real:
        #   (a) headroom over the served base load, and
        #   (b) a *non-firm connection rating* for the charging hub: the branch
        #       feeding a hub is reinforced when the hub is connected, but only
        #       to `conn_factor` of its nameplate, because utilities grant
        #       flexible / non-firm connection agreements precisely so they do
        #       not have to build for 100% simultaneity.  That agreement is the
        #       constraint V2G coordination exists to respect, so hard-coding a
        #       rating that already covers full simultaneity would delete the
        #       problem.
        pf_nom = self.feeder.M @ (self.feeder.p_load * cfg.load_scale)
        qf_nom = self.feeder.M @ (self.feeder.q_load * cfg.load_scale)
        nameplate = np.zeros(self.n_bus)
        if cfg.real_total_capacity_kw is not None:
            # Real-data path: distribute the site's REAL, cited total feeder
            # capacity (e.g. 150 kW Caltech, 195 kW JPL -- see
            # sessions.ACN_SITE_INFO) across hub buses using the same
            # electrical-depth weighting as everywhere else, instead of the
            # synthetic "22 kW x cluster_size per slot" guess below. Each
            # individual session's own action bound is still its own real
            # p_rate_kw (see Session.p_rate_kw, set in load_acn_sessions_real);
            # this only fixes the AGGREGATE hub thermal limit.
            total_pu = cfg.real_total_capacity_kw / self.s_base_kw
            for hub, wi in zip(hubs, w):
                nameplate[hub] += wi * total_pu
        else:
            max_rate_pu = 22.0 * cfg.cluster_size * cfg.ev_load_scale / self.s_base_kw
            for b in self.charger_buses:
                nameplate[b] += max_rate_pu
        self.feeder.s_max = (cfg.thermal_headroom * np.maximum(np.sqrt(pf_nom ** 2 + qf_nom ** 2), 1e-3)
                             + cfg.conn_factor * (self.feeder.M @ nameplate))
        # aggregation matrix A (Section 3.2): (n_bus, k_max)
        self.A = np.zeros((self.n_bus, cfg.k_max))
        self.A[self.charger_buses, np.arange(cfg.k_max)] = 1.0

        self.obs_dim = 2 * self.n_bus + 9 + 8 * self.k_max
        self.act_dim = 2 * self.k_max if cfg.allow_reactive else cfg.k_max
        self._reset_state()

    # ------------------------------------------------------------------ #
    #  Episode setup
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.t = 0
        self.slot_sess: List[Optional[Session]] = [None] * self.k_max
        self.soc = np.zeros(self.k_max)
        self.occupied = np.zeros(self.k_max, dtype=bool)
        self.energy_requested_kwh = 0.0
        self.peak_kw = 0.0

    def reset(self, seed: Optional[int] = None, date: Optional[str] = None) -> np.ndarray:
        """`date` (YYYY-MM-DD) overrides cfg.acn_date for this episode: multi-day training/evaluation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        day = date or cfg.acn_date
        self._reset_state()

        # exogenous day
        shape = (load_profile_shape(cfg.load_csv_file, cfg.horizon, cfg.dt_h, day) if (cfg.load_csv_file and day) else None)
        if cfg.load_csv_file and shape is None:
            warn_once('load', f'no usable real load shape for {day} in {cfg.load_csv_file} -> archetype generator')
        gen_shape = base_load_profile(self.rng, cfg.horizon, cfg.dt_h, cfg.archetype)
        shape = shape if shape is not None else gen_shape
        diversity = 1.0 if not cfg.stochastic_exogenous else float(np.clip(self.rng.normal(1.0, 0.08), 0.7, 1.3))
        self.p_bus_load = cfg.load_scale * diversity * self.feeder.p_load[None, :] * shape[:, None]
        self.q_bus_load = cfg.load_scale * diversity * self.feeder.q_load[None, :] * shape[:, None]
        pv = pv_profile(self.rng, cfg.horizon, cfg.dt_h,
                        cloudiness=0.25 if cfg.stochastic_exogenous else 0.0)
        pv_cap = cfg.pv_penetration * self.feeder.p_load.sum()
        pv_alloc = self.feeder.p_load / max(self.feeder.p_load.sum(), 1e-9)
        self.p_bus_pv = pv_cap * pv[:, None] * pv_alloc[None, :]
        # price: pluggable module (agents only ever see env.price), else the legacy price_source branches
        if cfg.price_scenario:
            from .pricing import build_price_model
            pm = build_price_model(cfg.price_scenario, csv=cfg.price_csv_file, tariff_csv=cfg.tariff_csv_file, site=cfg.acn_site,
                                   adder=cfg.price_adder, market=cfg.lmp_market, analog=cfg.lmp_analog)
            self.price = pm(day or "2019-07-01", cfg.horizon, cfg.dt_h, cfg.price_seed)
            self.price_label = pm.label
        elif cfg.price_source == "sce_tou_prime":
            date_for_season = day or "2019-07-01"
            self.price = sce_tou_prime_price(cfg.horizon, cfg.dt_h, date_for_season)
        elif cfg.price_source == "tariff_csv":
            d = day or "2019-07-01"
            tar = load_tariff(cfg.tariff_csv_file, cfg.acn_site)
            real = tar.price(d, cfg.horizon, cfg.dt_h) if tar is not None else None
            if real is None:
                warn_once("tariff", f"no usable historical tariff for site={cfg.acn_site} date={d} in {cfg.tariff_csv_file} "
                                    "-> SCE TOU-D-PRIME 2026 residential STAND-IN (label results accordingly)")
                real = sce_tou_prime_price(cfg.horizon, cfg.dt_h, d)
            elif cfg.demand_from_tariff:
                dd = tar.demand_per_day(d)
                if dd is not None:
                    cfg.demand_charge = float(dd)
            self.price = real
        elif cfg.price_source == "caiso_rtp":
            d = day or "2019-07-01"
            real = lmp_csv_price(cfg.price_csv_file, cfg.horizon, cfg.dt_h, d, cfg.price_adder, market=cfg.lmp_market, analog=cfg.lmp_analog) if cfg.price_csv_file else None
            if real is None:
                if not getattr(V2GEnv, "_warned_synth_lmp", False):
                    from ..runlog import say
                    say("[price] no usable real LMP CSV for this date -> using the SYNTHETIC CAISO-like duck curve "
                        "(labelled synthetic; supply price_csv_file for real CAISO data)")
                    V2GEnv._warned_synth_lmp = True
                real = caiso_rtp_synthetic_price(cfg.horizon, cfg.dt_h, d, None, cfg.price_adder)
            self.price = real
        else:
            self.price = tou_price(cfg.horizon, cfg.dt_h, cfg.price_tiers,
                                   self.rng if cfg.stochastic_exogenous else None, noise=0.05)

        # net exogenous bus injection (generation positive)
        self.p_exo = self.p_bus_pv - self.p_bus_load          # (T, n_bus) p.u.
        self.q_exo = -self.q_bus_load

        # sessions: real ACN-Data CSV (verified loader), the deprecated JSON
        # path, or the synthetic (calibrated but not real) generator.
        if cfg.sessions_source == "acn_csv" and cfg.acn_csv_file:
            if not day:
                raise ValueError("EnvConfig.acn_date must be set to a real calendar date "
                                "(see sessions.list_acn_dates()) when sessions_source='acn_csv'")
            self.sessions = load_acn_sessions_real(
                cfg.acn_csv_file, cfg.acn_site, np.arange(self.n_bus),
                cfg.horizon, cfg.dt_h, day, self.rng)
            bd = load_battery_dist(cfg.battery_csv_file) if cfg.battery_csv_file else None
            if cfg.battery_csv_file and bd is None:
                warn_once('battery', f'{cfg.battery_csv_file} unusable -> ' + ('ASSUMED default distribution' if cfg.battery_default else 'no battery assumption'))
            if bd is None and cfg.battery_default:
                bd = (np.array(DEFAULT_BATTERY_ASSUMPTION[0]), np.array(DEFAULT_BATTERY_ASSUMPTION[1]))
            if bd is not None:
                apply_battery(self.sessions, bd, np.random.default_rng(0 if seed is None else int(seed)))
        elif cfg.sessions_source == "acn" and cfg.acn_path:
            self.sessions = load_acn_sessions(cfg.acn_path, np.arange(self.n_bus),
                                              cfg.horizon, cfg.dt_h, self.rng)
        else:
            self.sessions = generate_sessions(self.rng, cfg.archetype, cfg.k_max,
                                              np.arange(self.n_bus), cfg.horizon,
                                              cfg.dt_h, cfg.session_density)
        self.pending = sorted(self.sessions, key=lambda s: s.t_arr)
        self._ptr = 0
        self.departed: List[Tuple[Session, float]] = []
        self._admit_arrivals()
        return self._obs()

    def _admit_arrivals(self):
        while self._ptr < len(self.pending) and self.pending[self._ptr].t_arr <= self.t:
            s = self.pending[self._ptr]
            self._ptr += 1
            if s.t_dep <= self.t:
                continue
            free = np.where(~self.occupied)[0]
            if len(free) == 0:
                continue                      # charger contention: session is turned away
            k = int(free[0])
            # A target above soc_max is unreachable by construction and would
            # show up as permanent "unmet demand" that no controller could ever
            # fix -- a silent floor on every method's service quality.  Clip it.
            s.soc_tgt = min(s.soc_tgt, self.cfg.soc_max)
            self.slot_sess[k] = s
            self.soc[k] = s.soc_arr
            self.occupied[k] = True
            self.energy_requested_kwh += s.energy_req_kwh * self.cfg.cluster_size

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #
    def _slot_arrays(self):
        cfg = self.cfg
        p_rate = np.zeros(self.k_max); cap = np.zeros(self.k_max)
        t_rem = np.zeros(self.k_max); soc_tgt = np.zeros(self.k_max)
        for k, s in enumerate(self.slot_sess):
            if s is None or not self.occupied[k]:
                continue
            p_rate[k] = s.p_rate_kw * cfg.ev_load_scale * cfg.cluster_size / self.s_base_kw
            cap[k] = s.cap_kwh * cfg.cluster_size
            t_rem[k] = max(0, s.t_dep - self.t) * cfg.dt_h
            soc_tgt[k] = s.soc_tgt
        return p_rate, cap, t_rem, soc_tgt

    def current_vpw(self) -> np.ndarray:
        cfg = self.cfg
        p_rate, cap, t_rem, soc_tgt = self._slot_arrays()
        e_need = np.maximum(0.0, soc_tgt - self.soc) * cap
        t_req = e_need / np.maximum(p_rate * self.s_base_kw, 1e-6)
        cdeg = np.clip(p_rate * self.s_base_kw / 50.0, 0, 1)
        F = vpw_features(t_rem, np.maximum(t_req, 1e-3), self.soc,
                         np.full(self.k_max, cfg.soc_min), np.maximum(soc_tgt, cfg.soc_min + 1e-3),
                         float(self.price[min(self.t, cfg.horizon - 1)]),
                         float(self.price.mean()), float(self.price.min()), float(self.price.max()),
                         cdeg)
        w = willingness(F, self.vpw_w)
        return np.where(self.occupied, w, 0.0)

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        t = min(self.t, cfg.horizon - 1)
        p_rate, cap, t_rem, soc_tgt = self._slot_arrays()
        e_need = np.maximum(0.0, soc_tgt - self.soc) * cap
        hour = (t * cfg.dt_h) % 24.0
        pr = self.price / max(self.price.max(), 1e-9)
        glob = np.array([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            pr[t], pr[min(t + 1, cfg.horizon - 1)], pr[min(t + 4, cfg.horizon - 1)],
            self.p_exo[t].sum(), self.q_exo[t].sum(), self.occupied.mean(),
            self.peak_kw / 100.0,                       # running peak of the day (demand-charge state)
        ])
        slots = np.stack([
            self.occupied.astype(float),
            self.soc,
            soc_tgt,
            t_rem / (cfg.horizon * cfg.dt_h),
            e_need / 100.0,
            p_rate * self.s_base_kw / 50.0,
            self.current_vpw(),
            self.charger_buses / max(self.n_bus, 1),
        ], axis=0).reshape(-1)
        return np.concatenate([glob, self.p_exo[t], self.q_exo[t], slots]).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Constraint context consumed by the QP safety layer
    # ------------------------------------------------------------------ #
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-slot physical (p.u.) box bounds honouring rating + SoC feasibility.

        Charging is capped so SoC cannot exceed soc_max within dt, discharging so
        SoC cannot fall below soc_min -- so the box alone already guarantees the
        battery constraint exactly; the QP only has to handle *network* coupling.
        """
        cfg = self.cfg
        p_rate, cap, _, _ = self._slot_arrays()
        cap_pu_h = np.maximum(cap, 1e-6) / self.s_base_kw     # p.u.-hours of storage
        head = (cfg.soc_max - self.soc) * cap_pu_h / (cfg.eta_ch * cfg.dt_h)
        foot = (self.soc - cfg.soc_min) * cap_pu_h * cfg.eta_dis / cfg.dt_h
        lo = np.where(self.occupied, -np.minimum(p_rate, np.maximum(foot, 0.0)), 0.0)
        if not cfg.allow_discharge:
            lo = np.zeros_like(lo)
        hi = np.where(self.occupied, np.minimum(p_rate, np.maximum(head, 0.0)), 0.0)
        if cfg.allow_reactive:
            qlim = np.where(self.occupied, p_rate * 0.44, 0.0)   # ~0.9 pf inverter envelope
            lo = np.concatenate([lo, -qlim])
            hi = np.concatenate([hi, qlim])
        return lo, hi

    def constraint_context(self) -> Dict[str, np.ndarray]:
        t = min(self.t, self.cfg.horizon - 1)
        lo, hi = self.action_bounds()
        return {
            "R": self.feeder.R, "X": self.feeder.X, "M": self.feeder.M, "A": self.A,
            "s_max": self.feeder.s_max,
            "p_exo": self.p_exo[t].copy(), "q_exo": self.q_exo[t].copy(),
            "lo": lo, "hi": hi,
            "vpw": self.current_vpw(), "occupied": self.occupied.copy(),
            "v_min": self.cfg.v_min, "v_max": self.cfg.v_max, "v_sub": self.cfg.v_sub,
        }

    # ------------------------------------------------------------------ #
    #  Dynamics
    # ------------------------------------------------------------------ #
    def split_action(self, a_phys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.allow_reactive:
            return a_phys[:self.k_max], a_phys[self.k_max:]
        return a_phys, np.zeros(self.k_max)

    def scale_action(self, a_norm: np.ndarray) -> np.ndarray:
        """Map policy output in [-1,1] to physical p.u. setpoints inside the box."""
        lo, hi = self.action_bounds()
        return lo + (np.clip(a_norm, -1, 1) + 1.0) * 0.5 * (hi - lo)

    def true_soc_next(self, soc: np.ndarray, p: np.ndarray, cap_pu_h: np.ndarray) -> np.ndarray:
        """SoC-dependent-efficiency battery model (ground truth)."""
        cfg = self.cfg
        eta_c = cfg.eta_ch - cfg.eta_curv * (soc - 0.5) ** 2
        eta_d = cfg.eta_dis - cfg.eta_curv * (soc - 0.5) ** 2
        pch = np.maximum(p, 0.0); pdis = np.maximum(-p, 0.0)
        d = (eta_c * pch - pdis / np.maximum(eta_d, 0.3)) * cfg.dt_h / np.maximum(cap_pu_h, 1e-9)
        return np.clip(soc + d, 0.0, 1.0)

    def step(self, a_phys: np.ndarray, soft_penalty: bool = False):
        """Advance one control interval with the given *physical* p.u. action."""
        cfg = self.cfg
        t = self.t
        info = StepInfo()
        p_ev, q_ev = self.split_action(np.asarray(a_phys, dtype=np.float64))
        lo, hi = self.action_bounds()
        full = np.concatenate([p_ev, q_ev]) if cfg.allow_reactive else p_ev
        full = np.clip(full, lo, hi)          # box is enforced unconditionally
        p_ev, q_ev = self.split_action(full)
        p_ev = np.where(self.occupied, p_ev, 0.0)
        q_ev = np.where(self.occupied, q_ev, 0.0)

        # --- network: exact AC solve -------------------------------------
        # p_ev > 0 is CHARGING, i.e. a load: it enters the injection balance
        # with a MINUS sign.  (Eq. 7 of the spec writes Pnet = P_V2G - P_L with
        # P_V2G as injection; our action is the mirror of that, so the sign lives
        # here and in FBase.net_injection -- both, consistently.)
        p_net = self.p_exo[t] - self.A @ p_ev
        q_net = self.q_exo[t] - self.A @ q_ev
        pf = solve_bfs(self.feeder, p_net, q_net, v0=cfg.v_sub)
        v = pf.v[0]
        s_br = pf.s_branch[0]
        under = np.maximum(cfg.v_min - v, 0.0)
        over = np.maximum(v - cfg.v_max, 0.0)
        info.v_violation = float(under.sum() + over.sum())
        info.v_violation_max = float(max(under.max(), over.max()))
        info.n_v_violations = int(((under > 1e-9) | (over > 1e-9)).sum())
        ovl = np.maximum(s_br - self.feeder.s_max, 0.0)
        info.s_violation = float(ovl.sum())
        info.n_s_violations = int((ovl > 1e-9).sum())
        info.v_min, info.v_max = float(v.min()), float(v.max())
        info.p_grid = float(pf.p_slack[0]); info.losses = float(pf.p_loss[0])

        # --- batteries ----------------------------------------------------
        _, cap, _, soc_tgt = self._slot_arrays()
        cap_pu_h = np.maximum(cap, 1e-6) / self.s_base_kw
        soc_next = self.true_soc_next(self.soc, p_ev, cap_pu_h)
        soc_next = np.where(self.occupied, soc_next, 0.0)
        service_gain = 0.0
        if cfg.reward_mode == "dense":
            g = np.minimum(soc_next, soc_tgt) - np.minimum(self.soc, soc_tgt)
            service_gain = float((np.where(self.occupied, g, 0.0) * cap).sum() * cfg.cluster_size)

        # --- economics ----------------------------------------------------
        price = float(self.price[t])
        kwh = p_ev * self.s_base_kw * cfg.dt_h                 # +charge / -discharge
        buy = np.maximum(kwh, 0.0).sum() * price
        sell = np.maximum(-kwh, 0.0).sum() * price * cfg.sell_ratio
        info.cost_energy = float(buy - sell)
        p_dis_pu = np.maximum(-p_ev, 0.0)
        cdeg = (cfg.deg_beta * (p_dis_pu * self.s_base_kw / 10.0) ** cfg.deg_gamma_exp * cfg.dt_h
                + cfg.deg_dod * (1.0 - self.soc) ** 2 * (p_dis_pu > 1e-9))
        info.cost_deg = float(np.where(self.occupied, cdeg, 0.0).sum())
        w = self.current_vpw()
        info.cost_discomfort = float((cfg.c_discomfort * (1.0 - w)
                                      * p_dis_pu * self.s_base_kw * cfg.dt_h).sum())
        info.p_ev_total = float(p_ev.sum() * self.s_base_kw)
        info.energy_served_kwh = float(kwh.sum())
        info.n_connected = int(self.occupied.sum())
        info.mean_vpw = float(w[self.occupied].mean()) if self.occupied.any() else 0.0

        # --- departures ---------------------------------------------------
        self.soc = soc_next
        self.t += 1
        unmet = 0.0
        for k, s in enumerate(self.slot_sess):
            if s is not None and self.occupied[k] and self.t >= s.t_dep:
                deficit = max(0.0, s.soc_tgt - self.soc[k]) * s.cap_kwh * cfg.cluster_size
                unmet += deficit
                self.departed.append((s, float(self.soc[k])))
                self.occupied[k] = False
                self.slot_sess[k] = None
                self.soc[k] = 0.0
        info.cost_unmet = float(cfg.c_unmet * unmet)
        info.unmet_kwh = float(unmet)
        info.requested_kwh = float(self.energy_requested_kwh)
        if self.t < cfg.horizon:
            self._admit_arrivals()

        p_site_kw = max(float(info.p_ev_total), 0.0)
        info.cost_demand = float(cfg.demand_charge * max(0.0, p_site_kw - self.peak_kw))   # telescopes to c * day-peak
        self.peak_kw = max(self.peak_kw, p_site_kw)
        total = (info.cost_energy + info.cost_deg + info.cost_unmet + info.cost_discomfort + info.cost_demand)
        if soft_penalty:
            total += cfg.c_violation * (info.v_violation + info.s_violation)
        if cfg.reward_mode == "dense":
            total = total - info.cost_unmet - cfg.c_unmet * service_gain   # credit now instead of penalty at departure
        reward = -total / cfg.reward_scale
        done = self.t >= cfg.horizon
        if done:                              # settle anyone still plugged in
            resid = 0.0
            for k, s in enumerate(self.slot_sess):
                if s is not None and self.occupied[k]:
                    resid += max(0.0, s.soc_tgt - self.soc[k]) * s.cap_kwh * cfg.cluster_size
                    self.departed.append((s, float(self.soc[k])))
            info.cost_unmet += float(cfg.c_unmet * resid)
            info.unmet_kwh += float(resid)
            if cfg.reward_mode != "dense":
                reward -= cfg.c_unmet * resid / cfg.reward_scale
        return self._obs(), float(reward), bool(done), info

    # ------------------------------------------------------------------ #
    def physical_state(self) -> Dict[str, np.ndarray]:
        """The part of the state the world model is responsible for predicting."""
        return {"soc": self.soc.copy(), "occupied": self.occupied.copy()}

    def nominal_voltage(self) -> np.ndarray:
        t = min(self.t, self.cfg.horizon - 1)
        pf = solve_bfs(self.feeder, self.p_exo[t], self.q_exo[t], v0=self.cfg.v_sub)
        return pf.v[0]

    # ------------------------------------------------------------------ #
    @property
    def obs_layout(self) -> Dict[str, slice]:
        """Index map into the flat observation.

        obs = [ global(9) | p_exo(n) | q_exo(n) | slots(8k) ]
        slots rows (in order): occupied, soc, soc_tgt, t_rem, e_need, p_rate, vpw, bus
        The world model / Dyna rollout overwrites exactly the `soc` and `e_need`
        blocks, which are the only action-dependent channels.
        """
        n, k = self.n_bus, self.k_max
        off = 9 + 2 * n
        names = ["occupied", "soc", "soc_tgt", "t_rem", "e_need", "p_rate", "vpw", "bus"]
        return {nm: slice(off + i * k, off + (i + 1) * k) for i, nm in enumerate(names)}

    def slot_meta(self) -> Dict[str, np.ndarray]:
        """soc_tgt / capacity / departing flags used by the analytic reward model."""
        p_rate, cap, t_rem, soc_tgt = self._slot_arrays()
        departing = np.array([1.0 if (s is not None and self.occupied[k] and s.t_dep == self.t + 1)
                              else 0.0 for k, s in enumerate(self.slot_sess)])
        return {"soc_tgt": soc_tgt, "cap_pu_h": np.maximum(cap, 1e-6) / self.s_base_kw,
                "departing": departing, "p_rate": p_rate}
