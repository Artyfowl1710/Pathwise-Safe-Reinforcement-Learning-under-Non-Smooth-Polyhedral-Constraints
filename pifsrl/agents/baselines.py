"""
Non-RL baselines.  Every headline claim in the proposal ("~50% cost reduction",
"zero violations") is meaningless without these, because the comparator decides
the number.  We therefore report against a *ladder* of increasingly strong
baselines, not just the weakest one:

  1. `uncontrolled`  - plug-and-charge at full rate until target.  This is the
     status quo and the honest denominator for "cost reduction".
  2. `tou_greedy`    - deadline-aware ToU-shifted unidirectional smart charging.
     A strong, realistic, deployed-today comparator.  Beating *this* is the
     claim that matters.
  3. `v2g_greedy`    - myopic ToU arbitrage with V2G discharge, no network model.
     Cheapest of all but violates constraints -- it exists to show that the cost
     ranking and the safety ranking are genuinely in tension.
  4. `safe_v2g_greedy` - the same rule passed through the QP projection: isolates
     "how much of the benefit is the safety layer vs the learned policy".
  5. `oracle_lp`     - perfect-foresight convex dispatch (LinDistFlow + full-day
     lookahead).  A lower bound on achievable cost; the RL agent should approach
     but not beat it.  Reported as the normalising upper bound of performance.

Note 3-5 are optional (need cvxpy for 5); 1, 2, 4 always run.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class RuleBasedPolicy:
    def __init__(self, kind: str, env, seed: int = 0, fbase=None):
        self.kind = kind
        self.env = env
        self.fbase = fbase
        self._plan = None
        self.rng = np.random.default_rng(seed)



    def _mpc_action(self, env, p_lo, p_hi):
        """Forecast-aware rolling-horizon dispatch: LP over the rest of the day for the sessions connected NOW (no arrival
        forecast), known prices/exogenous load, demand-charge peak bounded below by today's running peak, same network rows
        as the safety layer.  Re-plans when the connected set changes or every `replan_every` steps."""
        from .oracle import solve_oracle
        k = env.k_max
        sig = frozenset(id(s) for j, s in enumerate(env.slot_sess) if (s is not None and env.occupied[j]))
        new_arrival = not sig <= getattr(self, "_mpc_sig", frozenset())          # departures never need a re-plan
        if self._plan is None or new_arrival or env.t - getattr(self, "_mpc_t0", -10**9) >= getattr(self, "replan_every", 8):
            rs = int(self.kind.split("mpc_r")[-1]) if "mpc_r" in self.kind else 0
            res = solve_oracle(env, self.fbase, known_only=True, reserve_steps=rs)
            self._mpc_plan = res["p"] if res is not None else None
            self._mpc_mask = res["mask"] if res is not None else None
            self._mpc_sig, self._mpc_t0, self._plan = sig, env.t, True
        i = env.t - self._mpc_t0
        if self._mpc_plan is None or i >= len(self._mpc_plan):
            return np.where(env.occupied, p_hi, 0.0)               # LP failed: plug-and-charge (the QP wrapper stays on)
        planned = np.clip(self._mpc_plan[i], p_lo, p_hi)
        covered = self._mpc_mask[i]                                   # slots inside an LP window; others (reserve tail) plug-and-charge
        return np.where(env.occupied, np.where(covered, planned, p_hi), 0.0)

    @staticmethod
    def _oa_budget(env, need, t_rem, p_hi, slack=1.0):
        """Online 'Optimal-Available' peak shaving (Yao-Demers-Shenker / Bansal-Kimbrel-Pruhs speed-scaling idea):
        the smallest constant site budget L (p.u.) such that earliest-deadline-first finishes every CURRENTLY connected
        session by its deadline, ignoring future arrivals.  The budget never drops below the peak already paid for
        (demand charge is a max, so staying under the running peak is free)."""
        cfg = env.cfg
        k = env.k_max
        act = np.where(env.occupied & (need > 1e-9))[0]
        peak_pu = float(getattr(env, "peak_kw", 0.0)) / env.s_base_kw
        if len(act) == 0:
            return peak_pu
        n_left = np.maximum(1, np.ceil(t_rem[act] / cfg.dt_h - 1e-9).astype(int))
        order = act[np.argsort(n_left)]
        nl = dict(zip(act, n_left)); T = int(n_left.max())

        def feasible(L):
            rem = {j: need[j] for j in act}
            for tau in range(T):
                b = L
                for j in order:
                    if tau >= nl[j] or rem[j] <= 1e-9 or b <= 0:
                        continue
                    g = min(p_hi[j] * cfg.dt_h * cfg.eta_ch, b * cfg.dt_h * cfg.eta_ch, rem[j])
                    rem[j] -= g; b -= g / (cfg.dt_h * cfg.eta_ch)
            return all(rem[j] <= 1e-6 for j in act)

        hi_L = float(sum(p_hi[j] for j in act)); lo_L = 0.0
        if not feasible(hi_L):
            return hi_L                                   # cannot finish everyone: charge as fast as allowed
        for _ in range(14):
            mid = 0.5 * (lo_L + hi_L)
            if feasible(mid): hi_L = mid
            else: lo_L = mid
        return max(slack * hi_L, peak_pu)

    def __call__(self, obs=None) -> np.ndarray:
        env, cfg = self.env, self.env.cfg
        lo, hi = env.action_bounds()
        k = env.k_max
        p = np.zeros(k)
        _, cap, t_rem, soc_tgt = env._slot_arrays()
        cap_pu_h = np.maximum(cap, 1e-6) / env.s_base_kw
        t = min(env.t, cfg.horizon - 1)
        price = env.price[t]
        p_lo, p_hi = lo[:k], hi[:k]

        # energy still required, and the power needed to just make the deadline
        need = np.maximum(0.0, soc_tgt - env.soc) * cap_pu_h / max(cfg.eta_ch, 1e-6)
        t_left_h = np.maximum(t_rem, cfg.dt_h)
        p_must = need / t_left_h                       # p.u. required average power

        if self.kind in ("uncontrolled", "safe_uncontrolled"):
            p = np.where(env.occupied, p_hi, 0.0)
        elif self.kind in ("tou_greedy", "safe_tou_greedy"):
            cheap = price <= np.quantile(env.price, 0.45)
            urgent = p_must > 0.75 * np.maximum(p_hi, 1e-9)
            go = (cheap | urgent) & (need > 1e-9)
            p = np.where(env.occupied & go, p_hi, 0.0)
        elif self.kind in ("v2g_greedy", "safe_v2g_greedy"):
            expensive = price >= np.quantile(env.price, 0.8)
            cheap = price <= np.quantile(env.price, 0.4)
            urgent = p_must > 0.6 * np.maximum(p_hi, 1e-9)
            p = np.where(env.occupied & (cheap | urgent) & (need > 1e-9), p_hi, 0.0)
            disch = env.occupied & expensive & (~urgent) & (env.soc > soc_tgt - 0.15)
            p = np.where(disch, p_lo, p)
        elif self.kind == "oracle_lp":
            if self._plan is None:
                from .oracle import solve_oracle
                import os as _os
                res = solve_oracle(env, self.fbase, **(dict(no_discharge=True, tie_break=1e-6, eta_derate=0.0132) if _os.environ.get('ORACLE_FIX') else {}))
                self._plan = res["p"] if res is not None else np.zeros((cfg.horizon, k))
            p = np.where(env.occupied, self._plan[t], 0.0)
        elif self.kind.replace("safe_", "", 1).startswith("mpc"):
            p = self._mpc_action(env, p_lo, p_hi)
        elif self.kind.startswith("edf_hub_") or self.kind.startswith("safe_edf_hub_"):
            # per-hub EDF caps: "edf_hub_0.6_0.5_1.0" = one cap fraction per hub (hub order = env.hub_buses), each applied to
            # that hub's share of site capacity (split by the number of slots wired to it)
            fr = [float(x) for x in self.kind.split("hub_")[-1].split("_")]
            site_kw = cfg.real_total_capacity_kw or float((p_hi[: k] * env.s_base_kw).sum())
            nslots = np.array([(env.charger_buses == hb).sum() for hb in env.hub_buses], dtype=float)
            share = nslots / max(nslots.sum(), 1.0)
            p = np.zeros(k)
            for h_i, hb in enumerate(env.hub_buses):
                budget = fr[min(h_i, len(fr) - 1)] * site_kw * share[h_i] / env.s_base_kw
                slots = np.where((env.charger_buses == hb) & env.occupied & (need[:k] > 1e-9))[0]
                for j in slots[np.argsort(t_rem[slots])]:
                    give = min(p_hi[j], budget)
                    p[j] = give; budget -= give
                    if budget <= 0:
                        break
        elif self.kind.replace("safe_", "", 1).startswith("edf_cap") or self.kind in ("oa_peak", "safe_oa_peak"):
            # deadline-aware load levelling: earliest-deadline-first under a site power cap
            # (the classic ACN-style scheduler family).  "edf_cap_0.6" = cap at 60 % of site capacity.
            frac = float(self.kind.split("_")[-1]) if self.kind.split("_")[-1].replace(".", "").isdigit() else 0.6
            site_kw = cfg.real_total_capacity_kw or float((p_hi[: k] * env.s_base_kw).sum())
            budget = frac * site_kw / env.s_base_kw
            if self.kind in ("oa_peak", "safe_oa_peak"):
                budget = self._oa_budget(env, need, t_rem, p_hi, getattr(self, "oa_slack", 1.0))
            order = np.argsort(np.where(env.occupied & (need > 1e-9), t_rem, np.inf))
            p = np.zeros(k)
            for j in order:
                if not (env.occupied[j] and need[j] > 1e-9) or budget <= 0:
                    continue
                give = min(p_hi[j], budget)
                p[j] = give
                budget -= give
        elif self.kind == "random":
            p = self.rng.uniform(p_lo, p_hi)
        elif self.kind == "idle":
            p = np.zeros(k)
        else:
            raise KeyError(self.kind)

        p = np.clip(p, p_lo, p_hi)
        if cfg.allow_reactive:
            # mild volt-var support: absorb/inject Q against the local deviation
            q = np.zeros(k)
            if self.kind in ("v2g_greedy", "safe_v2g_greedy"):
                v = env.nominal_voltage()
                dev = 1.0 - v[env.charger_buses]
                q = np.clip(-dev * 8.0, -1, 1) * hi[k:]   # q>0 absorbs => lowers V
            return np.concatenate([p, q])
        return p


def perfect_foresight_lp(env, relax_network: bool = False) -> Optional[Dict[str, float]]:  # pragma: no cover
    """Perfect-foresight convex dispatch over the whole day (LinDistFlow network).

    Provides the *lower bound on cost* that any causal policy can aspire to.
    Requires cvxpy; returns None if unavailable.  Uses the linear network model,
    so its 'violations' are with respect to the same approximation the QP uses --
    it is a cost reference, not a safety reference.
    """
    try:
        import cvxpy as cp
    except Exception:
        return None
    cfg = env.cfg
    T, k = cfg.horizon, env.k_max
    occ = np.zeros((T, k)); rate = np.zeros((T, k)); cap = np.zeros(k)
    soc0 = np.zeros(k); tgt = np.zeros(k); dep = np.zeros((T, k))
    for s in env.sessions:
        j = None
        for kk in range(k):
            if occ[s.t_arr:s.t_dep, kk].sum() == 0:
                j = kk
                break
        if j is None:
            continue
        occ[s.t_arr:s.t_dep, j] = 1
        rate[s.t_arr:s.t_dep, j] = s.p_rate_kw * cfg.ev_load_scale / env.s_base_kw
        cap[j] = max(cap[j], s.cap_kwh / env.s_base_kw)
        if s.t_arr == 0 or soc0[j] == 0:
            soc0[j] = s.soc_arr
        tgt[j] = max(tgt[j], s.soc_tgt)
        dep[min(s.t_dep, T - 1), j] = 1

    p = cp.Variable((T, k))
    soc = cp.Variable((T + 1, k))
    cons = [soc[0] == soc0, p <= cp.multiply(occ, rate), p >= -cp.multiply(occ, rate),
            soc >= cfg.soc_min * (cap > 0), soc <= cfg.soc_max]
    for t in range(T):
        cons.append(soc[t + 1] == soc[t] + cp.multiply(p[t], cfg.dt_h / np.maximum(cap, 1e-6)) * cfg.eta_ch)
    if not relax_network:
        for t in range(T):
            v = cfg.v_sub + (env.p_exo[t] - env.A @ p[t]) @ env.feeder.R.T + env.q_exo[t] @ env.feeder.X.T
            cons += [v >= cfg.v_min, v <= cfg.v_max]
    unmet = cp.Variable(k, nonneg=True)
    cons.append(unmet >= cp.multiply(tgt - soc[T], (cap > 0)))
    price = env.price
    cost = cp.sum(cp.multiply(price[:, None], cp.pos(p)) * env.s_base_kw * cfg.dt_h) \
        - cfg.sell_ratio * cp.sum(cp.multiply(price[:, None], cp.pos(-p)) * env.s_base_kw * cfg.dt_h) \
        + cfg.c_unmet * cp.sum(cp.multiply(unmet, cap * env.s_base_kw))
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if p.value is None:
        return None
    return {"cost": float(prob.value), "status": prob.status}
