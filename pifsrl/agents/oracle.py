"""Perfect-foresight convex dispatch ("oracle MPC").

A full-day LP/convex program that knows every arrival, departure, price and load in advance and
respects the SAME linearised network rows (voltage + thermal polygon) the safety layer projects
onto.  It is the reference upper bound on what any causal controller (including SafeTD3) can
achieve; the interesting numbers are the *gap* to it and whether SafeTD3 closes more of the gap
than the safe-greedy rule.  Executed in the true (exact AC) environment, so its reported
violations are real, not LinDistFlow-internal.

Approximations (documented in DOC.md): reactive power fixed at 0; linear degradation is replaced by
the convex power law of Eq. 9 without the DoD term; discomfort term omitted; SoC-dependent charge
efficiency curvature (truth only) ignored.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch


def _idle_rollout(env, steps=None):
    """Admission is independent of the action (depends on time only), so an idle rollout of a
    copy of the environment reveals which session sits in which slot at every step."""
    clone = copy.deepcopy(env)
    T, k = (env.cfg.horizon if steps is None else int(steps)), env.k_max
    occ = np.zeros((T, k), bool); rate = np.zeros((T, k)); cap_kwh = np.zeros((T, k))
    soc0 = np.zeros((T, k)); tgt = np.zeros((T, k)); sid = -np.ones((T, k), int)
    lo = np.zeros((T, clone.act_dim)); hi = np.zeros((T, clone.act_dim))
    for t in range(T):
        _, cap, _, soc_tgt = clone._slot_arrays()
        l, h = clone.action_bounds()
        lo[t], hi[t] = l, h
        for j, s in enumerate(clone.slot_sess):
            if s is not None and clone.occupied[j]:
                occ[t, j] = True
                sid[t, j] = id(s)
                rate[t, j] = s.p_rate_kw * clone.cfg.ev_load_scale * clone.cfg.cluster_size / clone.s_base_kw
                cap_kwh[t, j] = s.cap_kwh * clone.cfg.cluster_size
                soc0[t, j] = clone.soc[j]
                tgt[t, j] = min(s.soc_tgt, clone.cfg.soc_max)
        clone.step(np.zeros(clone.act_dim))
    return dict(occ=occ, rate=rate, cap_kwh=cap_kwh, soc0=soc0, tgt=tgt, sid=sid, lo=lo, hi=hi)


def solve_oracle(env, fbase, margin_v: float = 0.002, margin_s: float = 0.02,
                 n_cuts: int = 8, known_only: bool = False, reserve_steps: int = 0,
                 tie_break: float = 0.0, solver_opts: Optional[Dict] = None, no_discharge: bool = False, eta_derate: float = 0.0) -> Optional[Dict]:
    """Perfect-foresight dispatch from the env's CURRENT time to the end of the day (t=0 -> the classic oracle).
    known_only=True makes it a rolling-horizon MPC step: only sessions already connected NOW are known (no arrival
    forecast), prices and exogenous load are assumed known, and the demand-charge peak is bounded below by the peak
    already paid today."""
    import cvxpy as cp
    from ..safety.qp_layer import build_polytope
    cfg = env.cfg
    t_now = int(env.t)
    T, K = cfg.horizon - t_now, env.k_max
    if T <= 0:
        return None
    r = _idle_rollout(env, T)
    occ, rate = r["occ"], r["rate"]
    if known_only:
        present = set(int(s) for s in r["sid"][0][occ[0]])
        occ = occ & np.isin(r["sid"], list(present)) if present else np.zeros_like(occ)
        if reserve_steps > 0:                       # plan to finish `reserve_steps` before the deadline (hedge vs unseen arrivals)
            occ = occ.copy()
            for sid_ in present:
                for j in range(K):
                    idx = np.where((r["sid"][:, j] == sid_) & occ[:, j])[0]
                    if len(idx) > reserve_steps + 2:
                        occ[idx[-reserve_steps:], j] = False
    S = env.s_base_kw
    dev = fbase.R.device

    # --- network rows (same polytope as the safety layer), p-columns only (q = 0) ---
    with torch.no_grad():
        poly = build_polytope(
            R=fbase.R, X=fbase.X, M=fbase.M, A=fbase.A,
            p_exo=torch.as_tensor(env.p_exo[t_now:t_now + T], dtype=torch.float32, device=dev),
            q_exo=torch.as_tensor(env.q_exo[t_now:t_now + T], dtype=torch.float32, device=dev),
            s_max=fbase.s_max, lo=torch.as_tensor(r["lo"], dtype=torch.float32, device=dev),
            hi=torch.as_tensor(r["hi"], dtype=torch.float32, device=dev),
            v_min=fbase.p.v_min, v_max=fbase.p.v_max, v0=fbase.p.v_sub, n_cuts=n_cuts,
            allow_reactive=fbase.allow_reactive, margin_v=margin_v, margin_s=margin_s)
    act = np.where(occ.any(0))[0]                                   # only slots that ever hold a (known) session
    if len(act) == 0:
        return {"p": np.zeros((T, K)), "mask": np.zeros((T, K), bool), "lp_cost": 0.0, "status": "empty"}
    G = poly.G[:, :, :K].cpu().numpy().astype(np.float64)[:, :, act]       # (T, m, Ka)
    h = np.maximum(poly.h.cpu().numpy().astype(np.float64), 0.0)  # idle-feasible relaxation

    Ka = len(act)
    rate = rate[:, act]
    Pch = cp.Variable((T, Ka), nonneg=True)
    Pdis = cp.Variable((T, Ka), nonneg=True)
    cons = [Pch <= rate, Pdis <= rate]
    if no_discharge or not getattr(cfg, "allow_discharge", True):
        cons.append(Pdis == 0)
    unmet_terms, seg_cost = [], 0
    c_pu = np.maximum(r["cap_kwh"], 1e-6) / S
    for ka, k in enumerate(act):
        t = 0
        while t < T:
            if not occ[t, k]:
                t += 1
                continue
            t0 = t
            while t < T and occ[t, k] and r["sid"][t, k] == r["sid"][t0, k]:
                t += 1
            t1 = t                                           # session occupies [t0, t1)
            L = t1 - t0
            soc = cp.Variable(L + 1)
            lo_soc = min(r["soc0"][t0, k], cfg.soc_min)
            cons += [soc[0] == r["soc0"][t0, k], soc >= lo_soc, soc <= cfg.soc_max]
            cons.append(soc[1:] == soc[:-1] + cfg.dt_h * ((cfg.eta_ch - eta_derate) * Pch[t0:t1, ka]
                                                         - Pdis[t0:t1, ka] / cfg.eta_dis) / c_pu[t0:t1, k])
            u = cp.Variable(nonneg=True)
            cons.append(u >= (r["tgt"][t0, k] - soc[L]) * r["cap_kwh"][t0, k])
            unmet_terms.append(u)
    # network rows (drop rows that can never bind given the charger ratings)
    for t in range(T):
        reach = (np.abs(G[t]) * rate[t][None, :]).sum(1)
        rows = np.where(reach > h[t] - 1e-9)[0]
        if len(rows):
            cons.append(G[t][rows] @ (Pch[t] - Pdis[t]) <= h[t][rows])
    price = np.asarray(env.price[t_now:t_now + T], dtype=np.float64)
    kwh_ch = cp.multiply(price[:, None], Pch) * S * cfg.dt_h
    kwh_dis = cp.multiply(price[:, None], Pdis) * S * cfg.dt_h
    deg = 0.0 if no_discharge else cfg.deg_beta * cfg.dt_h * cp.sum(cp.power(Pdis * S / 10.0 + 1e-9, cfg.deg_gamma_exp))   # no power cone when discharge is off
    cost = cp.sum(kwh_ch) - cfg.sell_ratio * cp.sum(kwh_dis) + deg + cfg.c_unmet * sum(unmet_terms)
    if getattr(cfg, "demand_charge", 0.0) > 0:                   # day-peak demand charge (kW of aggregate EV power)
        peak = cp.Variable(nonneg=True)
        cons.append(S * cp.sum(Pch - Pdis, axis=1) <= peak)
        cons.append(peak >= float(getattr(env, "peak_kw", 0.0)) if known_only else peak >= 0)
        cost = cost + cfg.demand_charge * peak
    obj = cost + (tie_break * cp.sum_squares(Pch) if tie_break > 0 else 0)   # strictly convex tie-break on degenerate price blocks
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL, **(solver_opts or {}))
    except Exception:
        return None
    if Pch.value is None:
        return None
    pfull = np.zeros((T, K)); pfull[:, act] = Pch.value - Pdis.value
    return {"p": pfull, "mask": occ, "lp_cost": float(prob.value - (tie_break * float(np.sum(np.square(Pch.value))) if tie_break > 0 else 0.0)), "status": prob.status, "iters": getattr(prob.solver_stats, "num_iters", None)}
