"""Differentiable twin of the V2G day-rollout (PyTorch, float64).

What is differentiable here (and what is not):
  * The environment's day is a deterministic function of the action sequence once the exogenous data
    (sessions, prices, base load) are fixed: admission/departure depend on TIME only (see oracle._idle_rollout),
    so the whole day is a smooth-almost-everywhere map  actions -> SoC trajectory -> cost.
  * SoC dynamics (SoC-dependent efficiency, clip), energy cost, unmet-energy penalty at departure, day-peak demand charge
    and the LinDistFlow voltage/thermal rows are all written with torch ops -> exact pathwise gradients (kinks: clip, relu, max).
  * The EXACT AC (BFS) solve in V2GEnv.step is not used inside the twin; the network enters through the same
    linearised rows the safety layer projects onto (build_polytope).  `scale_to_rows` is a differentiable
    "shrink toward idle" feasibility restoration (idle is feasible after h:=max(h,0)), a cheap stand-in for the QP.
  * Not differentiable: discrete arrivals/departures/contention (treated as data), degradation/discomfort terms (V2G disabled here).
`rollout_cost` returns the same cost components as V2GEnv (energy + unmet + demand charge); `tests/test_diffenv.py` and
`scripts/diff_env_poc.py` verify the twin against V2GEnv.step with identical actions.
"""
from __future__ import annotations

import numpy as np
import torch

from .agents.oracle import _idle_rollout


def extract_day(env, fbase, seed: int, margin_v: float = 0.002, margin_s: float = 0.02, n_cuts: int = 8, date=None):
    """Reset `env` on (seed,date) and freeze everything exogenous into numpy arrays."""
    from .safety.qp_layer import build_polytope
    env.reset(seed, date=date)
    cfg, T, K = env.cfg, env.cfg.horizon, env.k_max
    r = _idle_rollout(env, T)
    env.reset(seed, date=date)                                    # _idle_rollout works on a copy; reset again to be safe
    dev = fbase.R.device
    with torch.no_grad():
        poly = build_polytope(
            R=fbase.R, X=fbase.X, M=fbase.M, A=fbase.A,
            p_exo=torch.as_tensor(env.p_exo[:T], dtype=torch.float32, device=dev),
            q_exo=torch.as_tensor(env.q_exo[:T], dtype=torch.float32, device=dev),
            s_max=fbase.s_max, lo=torch.as_tensor(r["lo"], dtype=torch.float32, device=dev),
            hi=torch.as_tensor(r["hi"], dtype=torch.float32, device=dev),
            v_min=fbase.p.v_min, v_max=fbase.p.v_max, v0=fbase.p.v_sub, n_cuts=n_cuts,
            allow_reactive=fbase.allow_reactive, margin_v=margin_v, margin_s=margin_s)
    G = poly.G[:, :, :K].cpu().numpy().astype(np.float64)
    h = np.maximum(poly.h.cpu().numpy().astype(np.float64), 0.0)
    return dict(occ=r["occ"], rate=r["rate"], cap_kwh=r["cap_kwh"], soc0=r["soc0"], tgt=r["tgt"], sid=r["sid"],
                price=np.asarray(env.price[:T], dtype=np.float64), G=G, h=h, T=T, K=K)


def pad_batch(days, m_pad=None, device="cpu"):
    """Stack day dicts into float64 tensors (B,...).  Network rows are padded with (0 <= big) so shapes agree."""
    B = len(days)
    m = max(d["G"].shape[1] for d in days)
    out = {}
    for k in ("occ", "rate", "cap_kwh", "soc0", "tgt", "price"):
        out[k] = torch.as_tensor(np.stack([d[k] for d in days]), dtype=torch.float64)
    sid = np.stack([d["sid"] for d in days])
    out["sid"] = torch.as_tensor(sid, dtype=torch.int64)
    G = np.zeros((B, days[0]["T"], m, days[0]["K"])); h = np.full((B, days[0]["T"], m), 1e6)
    for i, d in enumerate(days):
        mi = d["G"].shape[1]; G[i, :, :mi] = d["G"]; h[i, :, :mi] = d["h"]
    out["G"], out["h"] = torch.as_tensor(G, dtype=torch.float32), torch.as_tensor(h, dtype=torch.float32)   # float32 storage (cast per step)
    # segment boundaries: a slot-session starts when occupied and (t==0 or previous step not the same session)
    occ = out["occ"] > 0.5
    prev_same = torch.zeros_like(occ); prev_same[:, 1:] = occ[:, 1:] & occ[:, :-1] & (out["sid"][:, 1:] == out["sid"][:, :-1])
    nxt_same = torch.zeros_like(occ); nxt_same[:, :-1] = occ[:, :-1] & occ[:, 1:] & (out["sid"][:, :-1] == out["sid"][:, 1:])
    out["start"], out["end"] = occ & ~prev_same, occ & ~nxt_same
    out["B"], out["T"], out["K"] = B, days[0]["T"], days[0]["K"]
    if str(device) != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out


def scale_to_rows(p, G, h, ste_beta=0.0):
    """Largest alpha in [0,1] with G (alpha p) <= h for every row (h >= 0, so alpha=0 is feasible).  Differentiable."""
    gp = torch.einsum("bmk,bk->bm", G, p)
    ratio = torch.where(gp > 1e-12, h / gp.clamp_min(1e-12), torch.full_like(gp, float("inf")))
    if ste_beta > 0:                                  # straight-through: exact min forward, softmin gradient backward
        r = ratio.clamp(max=1.0)
        alpha = -smooth_max_ste(-r, dim=1, beta=ste_beta)
    else:
        alpha = ratio.min(dim=1).values.clamp(max=1.0)
    return p * alpha.unsqueeze(1), alpha


def smooth_max_ste(x, dim=-1, beta=20.0):
    """Forward = exact max (simulator fidelity preserved); backward = softmax/logsumexp gradient (spreads over near-peak entries)."""
    hard = x.max(dim=dim).values
    soft = torch.logsumexp(beta * x, dim=dim) / beta
    return hard.detach() + (soft - soft.detach())


def scale_to_rows_sens(p, G, h, ste_beta=0.0):
    """Two-stage restoration.  Stage 1: every VIOLATED row i scales only the chargers that load it (G_ik>0) by h_i/(Gp)_i, so a congested lateral does not
    throttle chargers that do not touch it.  Stage 2: the uniform ray scale of `scale_to_rows` (guarantees G p<=h even with mixed-sign rows)."""
    gp = torch.einsum("bmk,bk->bm", G, p)
    viol = gp > h + 1e-12
    ratio = torch.where(viol, h / gp.clamp_min(1e-12), torch.full_like(gp, float("inf")))
    own = torch.where((G > 1e-12) & viol.unsqueeze(2), ratio.unsqueeze(2), torch.full_like(G, float("inf"))).min(dim=1).values
    p1 = p * own.clamp(max=1.0)
    return scale_to_rows(p1, G, h, ste_beta)[0]


def relu_ste(x, beta=20.0):
    """Forward = exact relu; backward = gradient of softplus(beta x)/beta (a sigmoid), so the violated-row term has a non-zero slope at the boundary."""
    hard = torch.relu(x)
    soft = torch.nn.functional.softplus(x, beta=beta)
    return hard.detach() + (soft - soft.detach())


def project_pgd(p, G, h, p_min, p_max, n_steps=10, step=1.0, beta=0.0, avg=True):
    """Unrolled orthogonal-projection restoration (alternating row projections, Jacobi form).
    Each step moves p along -G_i^T * viol_i / ||G_i||^2 (the exact projection onto the violated half-space i), is clamped to the box [p_min, p_max], and is repeated
    n_steps times.  A final `scale_to_rows` is kept ONLY as a floating-point feasibility certificate (alpha is ~1 after the unrolled steps), so the tangential
    component of the gradient along an active wall survives: d p~/d p ~ I - g g^T/||g||^2 instead of the radial null direction of pure ray scaling.
    p, p_min, p_max: (B,K); G: (B,m,K); h: (B,m)."""
    rn = G.pow(2).sum(dim=-1) + 1e-8                                  # (B,m) squared row norms
    for _ in range(n_steps):
        gp = torch.einsum("bmk,bk->bm", G, p)
        viol = relu_ste(gp - h, beta) if beta > 0 else torch.relu(gp - h)
        w = viol / rn
        if avg:                                                         # Cimmino weighting: mean (not sum) of the violated-row projections; the summed Jacobi step overshoots when many near-parallel rows are violated
            w = w / (viol > 0).double().sum(-1, keepdim=True).clamp_min(1.0)
        p = torch.minimum(torch.maximum(p - step * torch.einsum("bmk,bm->bk", G, w), p_min), p_max)
    return scale_to_rows(p, G, h)


def base_fraction(kind, pmax, need, tl, occ, budget_pu):
    """Fraction-of-pmax allocation of the tuned reference rule.  kind='full': charge every vehicle that still needs energy at its maximum rate
    (the projected-uncontrolled rule).  kind=float c: earliest-deadline-first under a site budget `budget_pu` (the EDF-cap rule; budget already = c * site capacity in p.u.)."""
    want = (occ > 0.5) & (need > 1e-9)
    if kind == "full":
        return want.double()
    order = torch.argsort(torch.where(want, tl, torch.full_like(tl, 1e9)), dim=1)
    ps = torch.gather(pmax * want.double(), 1, order)
    excl = torch.cumsum(ps, 1) - ps
    give = (budget_pu - excl).clamp_min(0).minimum(ps)
    out = torch.zeros_like(pmax).scatter(1, order, give)
    return out / pmax.clamp_min(1e-9) * want.double()


def rollout_cost(batch, policy, cfg, s_base_kw, demand_charge, c_unmet, restore=True, smooth_peak=0.0, return_traj=False, restore_mode="uniform", ste_beta=0.0, base=None, beta_res=0.5, budget_pu=None, pgd_steps=10):
    """Closed-loop rollout of `policy(feat)->u in [0,1]` per slot; returns per-day cost components (B,) (and the actions)."""
    B, T, K = batch["B"], batch["T"], batch["K"]
    dt = cfg.dt_h
    dev = batch["occ"].device
    soc = torch.zeros(B, K, dtype=torch.float64, device=dev)
    cap_pu_h = batch["cap_kwh"].clamp_min(1e-6) / s_base_kw
    capn = batch["rate"].sum(2).max(1).values.clamp_min(1e-9)                           # (B,) site nameplate (p.u.)
    price = batch["price"]; prank = (price.unsqueeze(2) >= price.unsqueeze(1)).double().mean(2)   # fraction of the day that is not cheaper
    pmaxn = price.max(1).values.clamp_min(1e-9)
    energy = torch.zeros(B, dtype=torch.float64, device=dev); unmet = torch.zeros(B, dtype=torch.float64, device=dev)
    site_prev = torch.zeros(B, dtype=torch.float64, device=dev); peak = torch.zeros(B, dtype=torch.float64, device=dev)
    peaks, P, sites = [], [], []
    for t in range(T):
        occ = batch["occ"][:, t]
        soc = torch.where(batch["start"][:, t], batch["soc0"][:, t], soc)
        head = (cfg.soc_max - soc).clamp_min(0) * cap_pu_h[:, t] / (cfg.eta_ch * dt)
        pmax = torch.minimum(batch["rate"][:, t], head) * occ
        tl = torch.zeros_like(soc)
        # steps left in the session = number of future occupied steps with the same sid (inclusive)
        same = (batch["sid"][:, t:, :] == batch["sid"][:, t:t + 1, :]) & (batch["occ"][:, t:, :] > 0.5)
        tl = (torch.cumprod(same.double(), dim=1)).sum(1)                                # consecutive-run length from t
        need = (batch["tgt"][:, t] - soc).clamp_min(0) * cap_pu_h[:, t] / cfg.eta_ch
        t_left_h = (tl * dt).clamp_min(dt)
        feat = torch.stack([
            occ, (price[:, t] / pmaxn).unsqueeze(1).expand(B, K), prank[:, t].unsqueeze(1).expand(B, K), soc,
            (batch["tgt"][:, t] - soc).clamp_min(0), tl / T,
            (need / (batch["rate"][:, t].clamp_min(1e-9) * t_left_h)).clamp(0, 2) * occ,
            batch["rate"][:, t] / batch["rate"].max().clamp_min(1e-9),
            (site_prev / capn).unsqueeze(1).expand(B, K), (peak / capn).unsqueeze(1).expand(B, K)], dim=2)
        u = policy(feat, t)
        if base is not None:                                            # residual policy: u = clip(base + beta * delta), delta in [-1,1] from the network
            x_ = base_fraction(base, pmax, need, tl, occ, budget_pu) + beta_res * u
            u = x_.clamp(0.0, 1.0).detach() + (x_ - x_.detach()) * ((x_ >= 0.0) & (x_ <= 1.0)).double()   # clamp with an INCLUSIVE-boundary gradient: torch.clamp has zero gradient exactly at the bound, which is where the reference rule sits (u in {0,1})
        p = u * pmax
        if restore:
            Gt, ht = batch["G"][:, t].double(), batch["h"][:, t].double()
            if restore_mode == "pgd":
                p, _ = project_pgd(p, Gt, ht, torch.zeros_like(pmax), pmax, n_steps=pgd_steps, beta=ste_beta)
            elif restore_mode == "sens":
                p = scale_to_rows_sens(p, Gt, ht, ste_beta)
            else:
                p, _ = scale_to_rows(p, Gt, ht, ste_beta)
        eta_c = cfg.eta_ch - cfg.eta_curv * (soc - 0.5) ** 2
        soc = (soc + eta_c * p * dt / cap_pu_h[:, t].clamp_min(1e-9)).clamp(0.0, 1.0) * occ + soc * (1 - occ)
        energy = energy + (p * s_base_kw * dt).sum(1) * price[:, t]
        end = batch["end"][:, t]
        unmet = unmet + (end.double() * (batch["tgt"][:, t] - soc).clamp_min(0) * batch["cap_kwh"][:, t]).sum(1)
        site = p.sum(1); site_prev = site
        peak = torch.maximum(peak, site)
        sites.append(site)
        P.append(p)
        soc = soc * occ                                                   # a departed slot resets (re-filled by `start`)
    cost_e, cost_u = energy, c_unmet * unmet
    if ste_beta > 0:                                   # same value as `peak`, but the gradient reaches every near-peak step
        st = torch.stack(sites, 1).clamp_min(0)
        soft = torch.logsumexp(ste_beta * st / capn.unsqueeze(1), dim=1) * capn / ste_beta
        peak = st.max(dim=1).values.detach() + (soft - soft.detach())
    cost_d = demand_charge * peak * s_base_kw
    out = dict(total=cost_e + cost_u + cost_d, energy=cost_e, unmet=cost_u, demand=cost_d, unmet_kwh=unmet, peak_kw=peak * s_base_kw)
    if return_traj:
        out["p"] = torch.stack(P, 1)
    return out


def rollout_cost_v2g(batch, policy, cfg, s_base_kw, demand_charge, c_unmet, base=None, beta_res=0.5, budget_pu=None, pgd_steps=10, return_traj=False, smooth_ind=False):
    """Bidirectional rollout.  policy(feat,t) -> delta in [-1,1]; u = clip(base_fraction + beta_res*delta, -1, 1) with an inclusive-boundary gradient;
    u>=0 is a fraction of the charge limit, u<0 a fraction of the SoC-feasible discharge limit.  Restoration: project_pgd on the signed power with box [-pdis_max, pch_max].
    SoC: eta_c(s) p+ - p-/eta_d(s) (same taper as the simulator).  Cost: buy - sell_ratio*sell, Eq.-9 degradation (power term + DoD term; the discomfort term c_disc*(1-w)
    is not in the twin because the viewpoint weights w are not part of the extracted day), unmet penalty at departure, demand charge on the signed site power."""
    B, T, K = batch["B"], batch["T"], batch["K"]; dt = cfg.dt_h; dev = batch["occ"].device
    soc = torch.zeros(B, K, dtype=torch.float64, device=dev)
    cap_pu_h = batch["cap_kwh"].clamp_min(1e-6) / s_base_kw
    capn = batch["rate"].sum(2).max(1).values.clamp_min(1e-9)
    price = batch["price"]; prank = (price.unsqueeze(2) >= price.unsqueeze(1)).double().mean(2); pmaxn = price.max(1).values.clamp_min(1e-9)
    energy = torch.zeros(B, dtype=torch.float64, device=dev); unmet = torch.zeros_like(energy); deg = torch.zeros_like(energy)
    site_prev = torch.zeros_like(energy); peak = torch.zeros_like(energy); P = []
    for t in range(T):
        occ = batch["occ"][:, t]
        soc = torch.where(batch["start"][:, t], batch["soc0"][:, t], soc)
        head = (cfg.soc_max - soc).clamp_min(0) * cap_pu_h[:, t] / (cfg.eta_ch * dt)
        foot = (soc - cfg.soc_min).clamp_min(0) * cap_pu_h[:, t] * cfg.eta_dis / dt
        pch_max = torch.minimum(batch["rate"][:, t], head) * occ; pdis_max = torch.minimum(batch["rate"][:, t], foot) * occ
        same = (batch["sid"][:, t:, :] == batch["sid"][:, t:t + 1, :]) & (batch["occ"][:, t:, :] > 0.5)
        tl = (torch.cumprod(same.double(), dim=1)).sum(1)
        need = (batch["tgt"][:, t] - soc).clamp_min(0) * cap_pu_h[:, t] / cfg.eta_ch
        t_left_h = (tl * dt).clamp_min(dt)
        feat = torch.stack([
            occ, (price[:, t] / pmaxn).unsqueeze(1).expand(B, K), prank[:, t].unsqueeze(1).expand(B, K), soc,
            (batch["tgt"][:, t] - soc).clamp_min(0), tl / T,
            (need / (batch["rate"][:, t].clamp_min(1e-9) * t_left_h)).clamp(0, 2) * occ,
            batch["rate"][:, t] / batch["rate"].max().clamp_min(1e-9),
            (site_prev / capn).unsqueeze(1).expand(B, K), (peak / capn).unsqueeze(1).expand(B, K),
            pdis_max / batch["rate"][:, t].clamp_min(1e-9)], dim=2)
        delta = policy(feat, t)
        x_ = base_fraction(base, pch_max, need, tl, occ, budget_pu) + beta_res * delta
        u = x_.clamp(-1.0, 1.0).detach() + (x_ - x_.detach()) * ((x_ >= -1.0) & (x_ <= 1.0)).double()
        p = torch.where(u >= 0, u * pch_max, u * pdis_max)
        Gt, ht = batch["G"][:, t].double(), batch["h"][:, t].double()
        p, _ = project_pgd(p, Gt, ht, -pdis_max, pch_max, n_steps=pgd_steps)
        pch, pdis = p.clamp_min(0), (-p).clamp_min(0)
        eta_c = cfg.eta_ch - cfg.eta_curv * (soc - 0.5) ** 2; eta_d = (cfg.eta_dis - cfg.eta_curv * (soc - 0.5) ** 2).clamp_min(0.3)
        dsoc = (eta_c * pch - pdis / eta_d) * dt / cap_pu_h[:, t].clamp_min(1e-9)
        soc_pre = soc
        soc = (soc + dsoc).clamp(0.0, 1.0) * occ + soc * (1 - occ)
        kwh = p * s_base_kw * dt
        energy = energy + (kwh.clamp_min(0).sum(1) - cfg.sell_ratio * (-kwh).clamp_min(0).sum(1)) * price[:, t]
        deg = deg + (occ * (cfg.deg_beta * ((pdis * s_base_kw / 10.0) + 1e-9) ** cfg.deg_gamma_exp * dt
                            + cfg.deg_dod * (1.0 - soc_pre) ** 2 * (torch.tanh(pdis * s_base_kw / 0.5) if smooth_ind else (pdis > 1e-9).double()))).sum(1)
        end = batch["end"][:, t]
        unmet = unmet + (end.double() * (batch["tgt"][:, t] - soc).clamp_min(0) * batch["cap_kwh"][:, t]).sum(1)
        site = p.sum(1); site_prev = site; peak = torch.maximum(peak, site); P.append(p)
        soc = soc * occ
    cost_d = demand_charge * peak.clamp_min(0) * s_base_kw
    out = dict(total=energy + deg + c_unmet * unmet + cost_d, energy=energy, deg=deg, unmet=c_unmet * unmet, demand=cost_d, unmet_kwh=unmet)
    if return_traj: out["p"] = torch.stack(P, 1)
    return out
