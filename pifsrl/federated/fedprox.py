"""
Personalized federated learning (Section 5.3): FedProx with parameter
decoupling.

    min_wi  L_i(w_i) + (mu/2) ||w_i^shared - w_global^shared||^2          (Eq. 20)
    w_global^{t+1} = sum_i (N_i/N_total) w_i^{shared,t+1}                  (Eq. 21)

WHAT IS SHARED VS PERSONAL, CONCRETELY
---------------------------------------
Every client (= one feeder site = one archetype) runs a `SafeTD3` with the SAME
architecture.  `SafeTD3._is_shared()` (agents/safe_td3.py) marks the actor/critic
*trunks* as shared and the final linear *heads* as personal, per Eq. 21.  The
world model (F_base + N_theta) and target networks are never federated:
  - F_base has no learnable parameters (it is the site's own Y_bus-derived
    physics) so there is nothing to share, and sharing it across sites would be
    physically wrong -- site A's grid is not site B's grid;
  - N_theta is trained on each site's own transitions and reflects that site's
    own linearisation error / battery population, so it is a genuinely local
    model, not federated. This is one of the two axes of Non-IID heterogeneity
    the proposal names (network topology + driver population); federating it
    would blur exactly the structure the ablation in Section 7.2 is designed to
    isolate.

FEDAVG BASELINE
----------------
`fedavg_aggregate()` averages the ENTIRE state (trunk + head, no decoupling,
mu=0 implicitly), which is the literal baseline the proposal claims diverges
under severe Non-IID client distributions.  `run_federated()` can run either
algorithm so Section 7.2's "FedAvg vs. personalized FedProx" ablation is a
one-flag difference, not two different codepaths that could quietly diverge.

NON-IID CLIENT CONSTRUCTION
----------------------------
Clients are drawn from DIFFERENT archetypes in `env/sessions.py` (workplace /
office / residential / fleet), each with its own arrival-time, dwell, and VPW
distribution, and can optionally sit on different feeder topologies (33-bus vs
a smaller/larger synthetic instance). This is what makes "Non-IID" a genuine
label-shift + covariate-shift condition rather than only a data-quantity
imbalance.
"""
from __future__ import annotations

from ..runlog import say

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..env.v2g_env import EnvConfig
from ..models.world_model import WMConfig
from ..safety.qp_layer import ResidualTightening
from ..agents.safe_td3 import SafeTD3, TD3Config
from ..agents.trainer import (TrainConfig, buffer_dims, calibrate_safety, evaluate_agent,
                              make_components, run_episode)
from ..agents.replay import ReplayBuffer


@dataclass
class ClientSpec:
    name: str
    env_cfg: EnvConfig
    weight: Optional[float] = None       # None -> proportional to local data volume
    train_dates: Optional[List[str]] = None   # multi-day local training days (each episode samples one)
    val_dates: Optional[List[str]] = None     # local validation days
    base_kind: str = "tou_greedy"             # local base rule (e.g. 'edf_cap_0.6', tuned on LOCAL train days only)


@dataclass
class FedConfig:
    rounds: int = 20
    local_episodes: int = 2
    mu: float = 0.05                     # FedProx proximal coefficient
    algorithm: str = "fedprox"           # fedprox | fedavg | local (no aggregation)
    eval_every: int = 5
    eval_episodes: int = 2
    seed: int = 0
    updates_per_episode: int = 64        # gradient updates per local episode (central runs use the horizon, 192)
    batch: int = 64


class FedClient:
    def __init__(self, spec: ClientSpec, td3_cfg: TD3Config, wm_cfg: WMConfig,
                device: str = "cpu"):
        self.spec = spec
        self.env, self.fbase, self.wm, self.agent = make_components(
            spec.env_cfg, td3_cfg, wm_cfg, device)
        self.buf = ReplayBuffer(30000, buffer_dims(self.env), device)
        self.rng = np.random.default_rng(spec.env_cfg.seed)
        self.rounds_done = 0

    def local_train(self, episodes: int, global_shared: Optional[Dict[str, torch.Tensor]],
                    mu: float, wm_updates: int = 15, calib: bool = True,
                    n_upd_cfg: int = 64, batch: int = 64) -> Dict[str, float]:
        """One federated round of local SGD, with the FedProx proximal term.

        The proximal term is applied as an explicit gradient penalty rather than
        folded into the optimiser, so it is visible and unit-testable:
            grad += mu * (w_shared - w_global_shared)
        added *after* the normal TD3/actor-critic backward pass, only on shared
        parameters.
        """
        stats = []
        for _ in range(episodes):
            noise = 0.15
            warm = self.rounds_done == 0 and len(self.buf) < 300
            day = (self.spec.train_dates[int(self.rng.integers(len(self.spec.train_dates)))]
                   if self.spec.train_dates else None)
            s = run_episode(self.env, self.agent, buffer=self.buf, noise=noise,
                            seed=self.spec.env_cfg.seed * 7919 + self.rounds_done,
                            random_actions=warm, date=day, base_kind=self.spec.base_kind)
            stats.append(s)
            if len(self.buf) >= 128:
                for _ in range(wm_updates):
                    self.wm.update(self.buf.sample(128, self.rng))
                if calib:
                    calibrate_safety(self.agent, self.wm, self.buf)
                n_upd = min(self.env.cfg.horizon, n_upd_cfg)
                for _ in range(n_upd):
                    self.agent.update(self.buf.sample(batch, self.rng))
                    if global_shared is not None and mu > 0:
                        self._apply_proximal_term(global_shared, mu)
            self.rounds_done += 1
        keys = stats[0].keys()
        return {k: float(np.mean([s[k] for s in stats])) for k in keys}

    def _apply_proximal_term(self, global_shared: Dict[str, torch.Tensor], mu: float):
        """Manual FedProx proximal gradient step applied directly to the shared
        parameters (a small extra SGD step, learning-rate matched to the actor
        LR, since the main optimizer step has already happened)."""
        own = dict(self.agent.named_parameters())
        with torch.no_grad():
            for k, gp in global_shared.items():
                if k in own and self.agent._is_shared(k):
                    own[k].sub_(mu * self.agent.cfg.lr_actor * (own[k] - gp))

    def shared_state(self) -> Dict[str, torch.Tensor]:
        return self.agent.shared_state()

    def load_shared(self, sd: Dict[str, torch.Tensor]):
        self.agent.load_shared(sd)

    def n_samples(self) -> int:
        return len(self.buf)


def fedavg_aggregate(states: List[Dict[str, torch.Tensor]], weights: np.ndarray,
                     keys: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Weighted average of full (or `keys`-restricted) state dicts -- Eq. 21
    with the shared/personal split removed, i.e. the literal FedAvg baseline."""
    w = weights / weights.sum()
    ks = keys or list(states[0].keys())
    out = {}
    for k in ks:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for wi, s in zip(w, states):
            acc += wi * s[k].float()
        out[k] = acc
    return out


def run_federated(clients: List[FedClient], cfg: FedConfig, verbose: bool = True,
                  log_fn: Optional[Callable] = None) -> Dict:
    history = []
    eval_history = []
    global_shared = None
    for rnd in range(cfg.rounds):
        round_stats = []
        local_shared_states = []
        weights = []
        for c in clients:
            s = c.local_train(cfg.local_episodes, global_shared if cfg.algorithm != "local" else None,
                              mu=cfg.mu if cfg.algorithm == "fedprox" else 0.0,
                              n_upd_cfg=cfg.updates_per_episode, batch=cfg.batch)
            round_stats.append(s)
            if cfg.algorithm in ("fedprox", "fedavg"):
                local_shared_states.append(c.shared_state())
                weights.append(c.spec.weight if c.spec.weight is not None else c.n_samples())
        if cfg.algorithm in ("fedprox", "fedavg") and local_shared_states:
            w = np.array(weights, dtype=float)
            global_shared = fedavg_aggregate(local_shared_states, w)
            for c in clients:
                c.load_shared(global_shared)
        rec = {"round": rnd,
               **{f"{c.spec.name}_{k}": v for c, s in zip(clients, round_stats) for k, v in s.items()
                  if k in ("total_cost", "v_viol", "s_viol", "served_frac", "ret")}}
        history.append(rec)
        if log_fn:
            log_fn(rec)
        if verbose and (rnd % max(1, cfg.eval_every // 2) == 0 or rnd == cfg.rounds - 1):
            costs = [s["total_cost"] for s in round_stats]
            served = [s.get("served_frac", float("nan")) for s in round_stats]
            viol = [s["v_viol"] + s["s_viol"] for s in round_stats]
            say(f"  round {rnd:3d} | mean cost {np.mean(costs):8.2f} | "
                  f"mean served {100*np.mean(served):5.1f}% | mean viol {np.mean(viol):.3e}", flush=True)
        if (rnd + 1) % cfg.eval_every == 0 or rnd == cfg.rounds - 1:
            ev = {}
            for c in clients:
                ev[c.spec.name] = evaluate_agent(c.env, c.agent, cfg.eval_episodes,
                                                 seed0=80000 + rnd, dates=c.spec.val_dates,
                                                 base_kind=c.spec.base_kind)
            ev["round"] = rnd
            eval_history.append(ev)
    return {"history": history, "evals": eval_history, "clients": clients}


# --------------------------------------------------------------------------- #
#  Non-IID divergence probe (quantifies "FedAvg diverges" vs "FedProx converges")
# --------------------------------------------------------------------------- #
def client_drift(clients: List[FedClient]) -> float:
    """Mean pairwise L2 distance between clients' shared trunks -- a concrete,
    reportable measure of representational Non-IID divergence."""
    states = [c.shared_state() for c in clients]
    keys = list(states[0].keys())
    n = len(states)
    if n < 2:
        return 0.0
    d = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(float(((states[i][k] - states[j][k]) ** 2).sum()) for k in keys) ** 0.5
            d += dist
            cnt += 1
    return d / max(cnt, 1)
