"""
Training / evaluation loop for a single site (one federated client = one feeder).

ORDER OF OPERATIONS PER EPISODE
-------------------------------
  1. collect one day of real transitions with the current (safe) policy
  2. fit N_theta on the replay buffer                        <- world model
  3. re-calibrate the conformal safety margin on held-out data  <- safety layer
  4. TD3 updates, including the differentiable-model term       <- policy

Step 3 is what makes the safety claim falsifiable: the margin is fitted on data
the projection has *not* been tuned on, and the reported violation count is
always measured on the exact non-linear AC solve inside the environment, never
on the model the agent used to plan.
"""
from __future__ import annotations

from ..runlog import say

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ..env.v2g_env import EnvConfig, StepInfo, V2GEnv
from ..models.fbase import FBase, PhysicsParams
from ..models.world_model import WMConfig, WorldModel
from ..safety.qp_layer import ResidualTightening
from .baselines import RuleBasedPolicy
from .replay import ReplayBuffer
from .safe_td3 import SafeTD3, TD3Config


@dataclass
class TrainConfig:
    episodes: int = 60
    warmup_episodes: int = 5
    updates_per_step: int = 1
    wm_updates_per_episode: int = 30
    wm_batch: int = 256
    eval_every: int = 10
    eval_episodes: int = 3
    calib_every: int = 5
    calib_frac: float = 0.3
    buffer_size: int = 60000
    soft_penalty: bool = False        # True for the soft-penalty Safe RL arm
    seed: int = 0
    device: str = "cpu"
    log_every: int = 10
    warmup_mode: str = "random"       # "random" | "greedy" (noisy demonstrations of the tou_greedy rule)
    demo_noise: float = 0.3
    bc_pretrain_steps: int = 0         # actor pre-training steps on the greedy demonstrations (no QP needed)
    base_kind: str = "tou_greedy"            # rule used as the anchor / residual base (e.g. 'edf_cap_0.6')
    train_dates: Optional[List[str]] = None   # multi-day: each training episode samples one of these real dates
    val_dates: Optional[List[str]] = None     # checkpoint selection (fixed seeds)
    test_dates: Optional[List[str]] = None    # final last-vs-best comparison (never used for selection)
    keep_best: bool = True            # keep the best checkpoint (by validation total_cost, zero-violation)
    test_seed0: int = 7_000_000       # fresh seeds for the FINAL last-vs-best comparison (never used to select)
    final_test_episodes: int = 5


def make_components(env_cfg: EnvConfig, td3_cfg: TD3Config, wm_cfg: WMConfig,
                    device: str = "cpu", feeder=None):
    env = V2GEnv(env_cfg, feeder=feeder)
    pp = PhysicsParams(dt_h=env_cfg.dt_h, eta_ch=env_cfg.eta_ch, eta_dis=env_cfg.eta_dis,
                       soc_min=env_cfg.soc_min, soc_max=env_cfg.soc_max,
                       s_base_kw=env.s_base_kw, deg_beta=env_cfg.deg_beta,
                       deg_gamma_exp=env_cfg.deg_gamma_exp, deg_dod=env_cfg.deg_dod,
                       c_unmet=env_cfg.c_unmet, c_discomfort=env_cfg.c_discomfort,
                       c_violation=env_cfg.c_violation, sell_ratio=env_cfg.sell_ratio,
                       reward_scale=env_cfg.reward_scale, reward_mode=env_cfg.reward_mode, v_min=env_cfg.v_min, v_max=env_cfg.v_max,
                       v_sub=env_cfg.v_sub)
    fbase = FBase(env.feeder, pp, env.k_max, env.A, env_cfg.allow_reactive).to(device)
    wm = WorldModel(fbase, env.feeder, wm_cfg, env_cfg.allow_reactive).to(device)
    agent = SafeTD3(env.obs_dim, env.act_dim, fbase, wm, td3_cfg, env.obs_layout, device)
    return env, fbase, wm, agent


def buffer_dims(env: V2GEnv) -> Dict[str, int]:
    n, k, a = env.n_bus, env.k_max, env.act_dim
    return {"obs": env.obs_dim, "next_obs": env.obs_dim, "act_norm": a, "act_phys": a,
            "p_exo": n, "q_exo": n, "soc": k, "soc_next": k, "cap": k, "occ": k,
            "v_true": n, "s_true": env.feeder.n_branch, "lo": a, "hi": a, "vpw": k,
            "soc_tgt": k, "departing": k,
            "p_exo_n": n, "q_exo_n": n, "lo_n": a, "hi_n": a, "vpw_n": k, "occ_n": k, "cap_n": k, "act_greedy": a, "act_base_n": a}


class EpisodeStats:
    """Aggregate a day of StepInfo into the metrics we actually report."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.d = dict(ret=0.0, cost_energy=0.0, cost_deg=0.0, cost_unmet=0.0,
                      cost_discomfort=0.0, cost_demand=0.0, v_viol=0.0, v_viol_max=0.0, n_v_steps=0,
                      s_viol=0.0, n_s_steps=0, v_min=2.0, v_max=0.0, peak_kw=0.0,
                      energy_kwh=0.0, losses=0.0, steps=0, mean_vpw=0.0, n_conn=0.0,
                      discharge_kwh=0.0, unmet_kwh=0.0, requested_kwh=0.0)
        self.qp_dist, self.qp_mod, self.qp_cut = [], [], []

    def add_qp(self, dist2_kw2: float, modified: bool, cut_frac: float):
        """One QP-intervention sample: ||raw - safe||^2 (kW^2, p dims), whether any slot
        moved by >0.1 kW, and the fraction of the raw charging power the projection removed."""
        self.qp_dist.append(dist2_kw2); self.qp_mod.append(float(modified)); self.qp_cut.append(cut_frac)

    def add(self, r: float, i: StepInfo):
        d = self.d
        d["ret"] += r
        d["cost_energy"] += i.cost_energy
        d["cost_deg"] += i.cost_deg
        d["cost_unmet"] += i.cost_unmet
        d["cost_discomfort"] += i.cost_discomfort
        d["cost_demand"] += i.cost_demand
        d["v_viol"] += i.v_violation
        d["v_viol_max"] = max(d["v_viol_max"], i.v_violation_max)
        d["n_v_steps"] += int(i.n_v_violations > 0)
        d["s_viol"] += i.s_violation
        d["n_s_steps"] += int(i.n_s_violations > 0)
        d["v_min"] = min(d["v_min"], i.v_min)
        d["v_max"] = max(d["v_max"], i.v_max)
        d["peak_kw"] = max(d["peak_kw"], abs(i.p_grid) * 1000.0)
        d["energy_kwh"] += i.energy_served_kwh
        d["discharge_kwh"] += max(0.0, -i.energy_served_kwh)
        d["losses"] += i.losses
        d["mean_vpw"] += i.mean_vpw
        d["n_conn"] += i.n_connected
        d["unmet_kwh"] += i.unmet_kwh
        d["requested_kwh"] = max(d["requested_kwh"], i.requested_kwh)
        d["steps"] += 1

    def result(self) -> Dict[str, float]:
        d = dict(self.d)
        s = max(d["steps"], 1)
        d["mean_vpw"] /= s
        d["n_conn"] /= s
        d["total_cost"] = d["cost_energy"] + d["cost_deg"] + d["cost_unmet"] + d["cost_discomfort"] + d["cost_demand"]
        d["violation_rate"] = d["n_v_steps"] / s
        d["served_frac"] = float(np.clip(1.0 - d["unmet_kwh"] / max(d["requested_kwh"], 1e-9), 0.0, 1.0))
        # NOTE: unmet_kwh is measured against each session's TARGET SoC at
        # departure, while requested_kwh is measured against its (arrival ->
        # target) gap. An aggressive/undertrained discharge policy can leave a
        # session below where it ARRIVED, making raw unmet exceed raw
        # requested (a genuine, not-a-bug edge case -- see RUNBOOK.md). Clip
        # here so the reported percentage stays interpretable; the raw
        # d["unmet_kwh"] / d["requested_kwh"] remain in the JSON for anyone
        # who wants to inspect that dynamic directly.
        d["any_violation"] = float((d["v_viol"] + d["s_viol"]) > 1e-9)
        # QP intervention (raw policy action vs executed safe action), steps with EVs present
        if self.qp_dist:
            q = np.array(self.qp_dist); c = np.array(self.qp_cut)
            d.update(qp_dist_mean=float(q.mean()), qp_dist_median=float(np.median(q)),
                     qp_dist_p90=float(np.percentile(q, 90)), qp_dist_p95=float(np.percentile(q, 95)),
                     qp_dist_p99=float(np.percentile(q, 99)), qp_mod_frac=float(np.mean(self.qp_mod)),
                     qp_heavy_frac=float(np.mean(c > 0.5)), qp_cut_mean=float(c.mean()))
        else:
            d.update(qp_dist_mean=0.0, qp_dist_median=0.0, qp_dist_p90=0.0, qp_dist_p95=0.0,
                     qp_dist_p99=0.0, qp_mod_frac=0.0, qp_heavy_frac=0.0, qp_cut_mean=0.0)
        return d


def _make_ctx(env: V2GEnv):
    ctx = env.constraint_context()
    meta = env.slot_meta()
    ctx_t = {"p_exo": ctx["p_exo"], "q_exo": ctx["q_exo"], "lo": ctx["lo"], "hi": ctx["hi"],
             "vpw": ctx["vpw"], "occ": ctx["occupied"].astype(np.float32),
             "soc": env.soc.copy(), "cap": meta["cap_pu_h"],
             "price": np.array([env.price[min(env.t, env.cfg.horizon - 1)]])}
    return ctx, meta, ctx_t


def _base_norm(env, kind, ctx_t):
    """Normalised ([-1,1] in the current box) action of the rule-based base/anchor policy."""
    g = RuleBasedPolicy(kind, env)()
    return np.clip(2 * (g - ctx_t["lo"]) / (ctx_t["hi"] - ctx_t["lo"] + 1e-9) - 1, -1, 1)


def run_episode(env: V2GEnv, policy, buffer: Optional[ReplayBuffer] = None,
                agent: Optional[SafeTD3] = None, noise: float = 0.0,
                seed: Optional[int] = None, soft_penalty: bool = False,
                random_actions: bool = False, collect: bool = True,
                qp_timer: Optional[list] = None,
                demo_noise: Optional[float] = None, date: Optional[str] = None,
                base_kind: str = "tou_greedy") -> Dict[str, float]:
    """One day.  `policy` is either a SafeTD3 or a RuleBasedPolicy."""
    obs = env.reset(seed, date=date)
    st = EpisodeStats()
    done = False
    k = env.k_max
    ctx, meta, ctx_t = _make_ctx(env)
    base_norm_next = None
    while not done:
        t0 = time.perf_counter()
        a_raw_box = None                                  # the policy's action BEFORE the QP (physical p.u.)
        g_norm = None
        if isinstance(policy, SafeTD3) and (buffer is not None or policy.cfg.residual_scale > 0):
            g_norm = base_norm_next if base_norm_next is not None else _base_norm(env, base_kind, ctx_t)
        if isinstance(policy, SafeTD3):
            if demo_noise is not None and g_norm is not None:
                a_norm, a_phys = policy.act_from_norm(
                    np.clip(g_norm + demo_noise * np.random.randn(*g_norm.shape), -1, 1), ctx_t)
            elif random_actions:
                a_norm, a_phys = policy.act_random(ctx_t, base_norm=g_norm)
            else:
                a_norm, a_phys = policy.act(obs, ctx_t, noise=noise, base_norm=g_norm)
            a_raw_box = ctx_t["lo"] + (np.clip(a_norm, -1, 1) + 1.0) * 0.5 * (ctx_t["hi"] - ctx_t["lo"])
            if not policy.cfg.use_qp:
                a_raw_box = None                          # soft-penalty arm has no projection
        else:
            a_phys = policy()
            a_norm = np.zeros_like(a_phys)
            if agent is not None:                 # safe_* rule variants
                a_raw_box = np.clip(a_phys, ctx_t["lo"], ctx_t["hi"])
                import torch as _t
                c = {kk: _t.as_tensor(np.atleast_2d(v), dtype=_t.float32, device=agent.device)
                     for kk, v in ctx_t.items()}
                a_box = _t.as_tensor(np.atleast_2d(a_phys), dtype=_t.float32, device=agent.device)
                a_norm_t = 2 * (a_box - c["lo"]) / (c["hi"] - c["lo"] + 1e-9) - 1
                a_phys = (agent.safe_action(a_norm_t.clamp(-1, 1), c, differentiable=False)
                          .squeeze(0).detach().cpu().numpy())
        if qp_timer is not None:
            qp_timer.append(time.perf_counter() - t0)
        if a_raw_box is not None and ctx["occupied"].any():
            dp = (a_raw_box - a_phys)[:k] * env.s_base_kw
            tot_raw = float(a_raw_box[:k].sum())
            cut = 0.0 if tot_raw < 1e-9 else float(1.0 - a_phys[:k].sum() / tot_raw)
            st.add_qp(float((dp ** 2).sum()), bool(np.abs(dp).max() > 0.1), cut)
        soc_before = env.soc.copy()
        t_idx = env.t
        p_exo, q_exo = env.p_exo[t_idx].copy(), env.q_exo[t_idx].copy()
        price = float(env.price[t_idx])
        next_obs, r, done, info = env.step(a_phys, soft_penalty=soft_penalty)
        st.add(r, info)
        ctx_n, meta_n, ctx_t_n = _make_ctx(env)            # constraint context of the NEXT state
        base_norm_next = (_base_norm(env, base_kind, ctx_t_n) if (isinstance(policy, SafeTD3) and not done and
                          (buffer is not None or policy.cfg.residual_scale > 0)) else None)
        if buffer is not None and collect:
            from ..physics.powerflow import solve_bfs
            pf = solve_bfs(env.feeder, p_exo - env.A @ env.split_action(a_phys)[0],
                           q_exo - env.A @ env.split_action(a_phys)[1], v0=env.cfg.v_sub)
            buffer.add(obs=obs, next_obs=next_obs, act_norm=a_norm, act_phys=a_phys,
                       rew=r, done=float(done), p_exo=p_exo, q_exo=q_exo,
                       soc=soc_before, soc_next=env.soc.copy(), cap=meta["cap_pu_h"],
                       occ=ctx["occupied"].astype(np.float32), price=price,
                       v_true=pf.v[0], s_true=pf.s_branch[0], lo=ctx["lo"], hi=ctx["hi"],
                       vpw=ctx["vpw"], soc_tgt=meta["soc_tgt"], departing=meta["departing"],
                       t=float(t_idx),
                       p_exo_n=ctx_t_n["p_exo"], q_exo_n=ctx_t_n["q_exo"], lo_n=ctx_t_n["lo"],
                       hi_n=ctx_t_n["hi"], vpw_n=ctx_t_n["vpw"], occ_n=ctx_t_n["occ"],
                       cap_n=ctx_t_n["cap"], price_n=float(ctx_t_n["price"][0]),
                       act_greedy=g_norm if g_norm is not None else np.zeros_like(a_norm),
                       act_base_n=(base_norm_next if base_norm_next is not None else np.zeros_like(a_norm)))
        obs = next_obs
        ctx, meta, ctx_t = ctx_n, meta_n, ctx_t_n
    return st.result()


def calibrate_safety(agent: SafeTD3, wm: WorldModel, buffer: ReplayBuffer,
                     frac: float = 0.3, alpha: float = 0.01) -> Dict[str, float]:
    """Split-conformal margin on a held-out slice of the buffer."""
    n = len(buffer)
    if n < 64:
        return agent.tighten.as_dict()
    idx = np.arange(n)
    hold = idx[int(n * (1 - frac)):]
    b = buffer.sample(len(hold), idx=hold)
    with torch.no_grad():
        out = wm.predict(b["act_phys"], b["p_exo"], b["q_exo"], b["soc"], b["cap"],
                         b["occ"], b["price"])
    agent.tighten.alpha = alpha
    agent.tighten.fit(b["v_true"].cpu().numpy(), out["v_hat"].cpu().numpy(),
                      b["s_true"].cpu().numpy(), out["s_hat"].cpu().numpy())
    return agent.tighten.as_dict()


def train_site(env_cfg: EnvConfig, td3_cfg: TD3Config, wm_cfg: WMConfig,
               tr_cfg: TrainConfig, verbose: bool = True,
               log_fn=None, feeder=None, eval_fn=None) -> Dict:
    torch.manual_seed(tr_cfg.seed)
    np.random.seed(tr_cfg.seed)
    env, fbase, wm, agent = make_components(env_cfg, td3_cfg, wm_cfg, tr_cfg.device, feeder)
    buf = ReplayBuffer(tr_cfg.buffer_size, buffer_dims(env), tr_cfg.device)
    rng = np.random.default_rng(tr_cfg.seed)
    history: List[Dict] = []
    evals: List[Dict] = []
    t_start = time.time()
    best = {"score": float("inf"), "ep": -1, "state": None}

    for ep in range(tr_cfg.episodes):
        frac = ep / max(tr_cfg.episodes - 1, 1)
        noise = td3_cfg.expl_noise + (td3_cfg.expl_noise_final - td3_cfg.expl_noise) * frac
        warm = ep < tr_cfg.warmup_episodes
        if td3_cfg.bc_weight > 0 and td3_cfg.bc_weight_final >= 0:
            agent.bc_now = td3_cfg.bc_weight + (td3_cfg.bc_weight_final - td3_cfg.bc_weight) * frac
        demo = tr_cfg.demo_noise if (warm and tr_cfg.warmup_mode == "greedy") else None
        day = (tr_cfg.train_dates[int(rng.integers(len(tr_cfg.train_dates)))]
               if tr_cfg.train_dates else None)
        stats = run_episode(env, agent, buffer=buf, noise=noise, seed=tr_cfg.seed * 1000 + ep,
                            soft_penalty=tr_cfg.soft_penalty, random_actions=(warm and demo is None),
                            demo_noise=demo, date=day, base_kind=tr_cfg.base_kind)
        if warm and ep == tr_cfg.warmup_episodes - 1 and tr_cfg.bc_pretrain_steps > 0:
            bc_loss = float("nan")
            for _ in range(tr_cfg.bc_pretrain_steps):
                bc_loss = agent.bc_pretrain(buf.sample(td3_cfg.batch_size, rng))
            agent.actor_t.load_state_dict(agent.actor.state_dict())
            say(f"  BC pre-training done: actor->greedy MSE {bc_loss:.4f}", flush=True)
        wm_info = {}
        if len(buf) >= 64:
            for _ in range(tr_cfg.wm_updates_per_episode):
                wm_info = wm.update(buf.sample(tr_cfg.wm_batch, rng))
        if (not warm) and len(buf) >= 64:
            if ep % tr_cfg.calib_every == 0:
                calibrate_safety(agent, wm, buf, tr_cfg.calib_frac)
            n_upd = env_cfg.horizon * tr_cfg.updates_per_step
            for _ in range(n_upd):
                agent.update(buf.sample(td3_cfg.batch_size, rng), soft_penalty=tr_cfg.soft_penalty)
        rec = {"episode": ep, "noise": noise, **stats, **wm_info,
               **{f"tight_{k}": v for k, v in agent.tighten.as_dict().items()},
               "wall_s": time.time() - t_start}
        history.append(rec)
        if log_fn:
            log_fn(rec)
        if verbose and (ep % tr_cfg.log_every == 0 or ep == tr_cfg.episodes - 1):
            say(f"  ep {ep:3d} | ret {stats['ret']:8.2f} | cost {stats['total_cost']:8.2f} "
                  f"| Vmin {stats['v_min']:.4f} | viol {stats['v_viol'] + stats['s_viol']:.2e} "
                  f"| unmet {stats['cost_unmet']:6.2f} | wm_v_mae "
                  f"{wm_info.get('wm_v_mae', float('nan')):.2e}", flush=True)
        if (ep + 1) % tr_cfg.eval_every == 0 or ep == tr_cfg.episodes - 1:
            # FIXED validation seeds -> evals are comparable across checkpoints
            ev = evaluate_agent(env, agent, tr_cfg.eval_episodes, seed0=90000,
                                soft_penalty=tr_cfg.soft_penalty, dates=tr_cfg.val_dates, base_kind=tr_cfg.base_kind)
            ev["episode"] = ep
            evals.append(ev)
            if eval_fn is not None:
                eval_fn(ep, ev)
            score = ev["total_cost"] + 1e3 * ev.get("any_violation", 0.0)
            if tr_cfg.keep_best and score < best["score"] and not warm:
                import copy
                best.update(score=score, ep=ep, state=copy.deepcopy(agent.state_dict()))
    # ---- last-vs-best on FRESH seeds (never used for selection) ----
    final = {}
    if tr_cfg.keep_best and tr_cfg.final_test_episodes > 0:
        final["last"] = evaluate_agent(env, agent, tr_cfg.final_test_episodes, seed0=tr_cfg.test_seed0,
                                       soft_penalty=tr_cfg.soft_penalty, dates=tr_cfg.test_dates, base_kind=tr_cfg.base_kind)
        if best["state"] is not None:
            last_state = copy.deepcopy(agent.state_dict()) if False else {k: v.clone() for k, v in agent.state_dict().items()}
            agent.load_state_dict(best["state"])
            final["best"] = evaluate_agent(env, agent, tr_cfg.final_test_episodes, seed0=tr_cfg.test_seed0,
                                           soft_penalty=tr_cfg.soft_penalty, dates=tr_cfg.test_dates, base_kind=tr_cfg.base_kind)
            final["best"]["episode"] = best["ep"]
            agent.load_state_dict(last_state)           # leave the agent as the LAST policy
    return {"history": history, "evals": evals, "final": final, "best_ep": best["ep"],
            "best_state": best["state"], "agent": agent, "wm": wm, "env": env,
            "buffer": buf, "wall_s": time.time() - t_start}


def evaluate_agent(env: V2GEnv, agent: SafeTD3, n_episodes: int = 5, seed0: int = 90000,
                   soft_penalty: bool = False, qp_timer: Optional[list] = None,
                   dates: Optional[List[str]] = None, base_kind: str = "tou_greedy") -> Dict[str, float]:
    rows = []
    for i in range(n_episodes):
        rows.append(run_episode(env, agent, buffer=None, noise=0.0, seed=seed0 + i,
                                soft_penalty=soft_penalty, collect=False, qp_timer=qp_timer,
                                date=(dates[i % len(dates)] if dates else None), base_kind=base_kind))
    return aggregate(rows)


def evaluate_rule(env: V2GEnv, kind: str, n_episodes: int = 5, seed0: int = 90000,
                  agent: Optional[SafeTD3] = None, dates: Optional[List[str]] = None) -> Dict[str, float]:
    rows = []
    for i in range(n_episodes):
        pol = RuleBasedPolicy(kind, env, seed=seed0 + i, fbase=(agent.fbase if agent is not None else None))
        use_agent = agent if kind.startswith("safe_") else None
        rows.append(run_episode(env, pol, buffer=None, seed=seed0 + i, collect=False,
                                agent=use_agent, date=(dates[i % len(dates)] if dates else None)))
    return aggregate(rows)


def tune_edf_cap(env, days, caps=(1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3), seed0: int = 90000, viol_tol: float = 1e-6):
    """Pick the EDF power-cap fraction on TRAIN days: argmin cost among caps whose mean (voltage+thermal) violation <= viol_tol
    (an unwrapped cap that violates the network is not a deployable baseline). Falls back to the least-violating cap.
    Returns (best_cap, {cap: {"cost","viol"}})."""
    tab = {}
    for c in caps:
        r = evaluate_rule(env, f"edf_cap_{c}", len(days), seed0=seed0, dates=days)
        tab[c] = {"cost": r["total_cost"], "viol": r["v_viol"] + r["s_viol"]}
    ok = [c for c in caps if tab[c]["viol"] <= viol_tol]
    best = min(ok, key=lambda c: tab[c]["cost"]) if ok else min(caps, key=lambda c: tab[c]["viol"])
    return best, tab


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std())
    out["n_episodes"] = len(rows)
    out["episode_costs"] = [float(r["total_cost"]) for r in rows]          # per-episode (paired by seed/date index)
    out["episode_served"] = [float(r["served_frac"]) for r in rows]
    return out
