"""
Safe TD3 with a hard QP projection inside the policy, and an optional
differentiable-model policy gradient (the thing the Residual-Dyna architecture
is *for*).

THE GRADIENT CHAIN (Eq. 14), concretely
---------------------------------------
    obs --actor--> a_norm --affine--> a_box --QP projection--> a_safe
                                                   |
                          +------------------------+------------------------+
                          |                                                 |
                    critic term                                   model-based term
                 grad_a Q(s, a_safe)                 grad_a [ r_hat(s,a) + gamma Q(s', s'_hat(a)) ]
                                                        via  dF_base/da + dN_theta/da   (Eq. 11)

Both terms flow through the *same* implicit QP Jacobian, so the actor is
optimising subject to a differentiable model of the safe set rather than being
clipped after the fact.  `--lambda-mb 0` disables the second term, which is the
ablation isolating the value of differentiable physics from the value of the
safety layer.

SAFE EXPLORATION
----------------
Exploration noise is injected *before* the projection.  Because the projection
is a hard feasibility operator, the behaviour policy is safe by construction at
every step of training -- including step 0, before any learning has happened.
This is the operational difference from soft-penalty Safe RL, and it is what the
'zero violations during exploration' claim means.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.fbase import FBase
from ..models.world_model import WorldModel, mlp
from ..safety.qp_layer import (ResidualTightening, SafetyPolytope, build_polytope,
                               project, vpw_weights_to_qp)

ACT_SCALE = 100.0        # p.u. -> O(1) for network inputs (10 kW on 1 MVA base = 0.01)


@dataclass
class TD3Config:
    hidden: int = 256
    depth: int = 2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    expl_noise: float = 0.25
    expl_noise_final: float = 0.05
    batch_size: int = 128
    lambda_mb: float = 0.5           # weight on the differentiable-model term
    grad_clip: float = 5.0
    # --- architecture switches used by the ablation grid -------------------
    use_qp: bool = True              # False -> soft-penalty baseline (clip only)
    use_tightening: bool = True      # residual-informed conformal margin
    use_vpw_weights: bool = True     # willingness-weighted projection metric
    qp_iters: int = 120
    qp_rho: float = 8.0
    # --- greedy-anchored actor (TD3+BC style; Fujimoto & Gu 2021) ---
    bc_weight: float = 0.0           # >0 enables:  L_a = -(Q + lambda_mb*MB)/|Q|_mean + bc_weight*||pi(s)-a_greedy||^2
    bc_weight_final: float = -1.0    # <0: constant; else linearly annealed to this value over training
    rowwise_margin: bool = True      # per-bus / per-branch conformal margins (False = legacy global max)
    margin_cap_frac: float = 0.35    # cap a thermal margin at this fraction of the row's limit
    residual_scale: float = 0.0      # >0: pre-QP action = clip(base + residual_scale * pi(s)) (safe residual RL around a rule)
    qp_iters_target: int = 40        # iterations for the NO-GRAD target/imagination projections (fast, approximate)
    qp_hard_first: bool = True       # exact-ish hard projection when idle is feasible (fixes ADMM over-conservatism)
    qp_rho_hard: float = 0.1
    exact_next_ctx: bool = True      # critic target projects a' with the NEXT step's constraints
    n_thermal_cuts: int = 8
    # --- federated: which parameters are "shared" vs "personal" -----------
    head_keys: Tuple[str, ...] = ("actor.head", "critic1.head", "critic2.head")


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2):
        super().__init__()
        self.trunk = mlp([obs_dim] + [hidden] * depth, out_act=nn.SiLU)
        self.head = nn.Linear(hidden, act_dim)
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs):
        return torch.tanh(self.head(self.trunk(obs)))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=2):
        super().__init__()
        self.trunk = mlp([obs_dim + act_dim] + [hidden] * depth, out_act=nn.SiLU)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        return self.head(self.trunk(torch.cat([obs, act], dim=-1))).squeeze(-1)


class SafeTD3(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, fbase: FBase, world_model: WorldModel,
                 cfg: TD3Config, obs_layout: Dict[str, slice], device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.fbase = fbase
        self.wm = world_model
        self.layout = obs_layout
        self.actor = Actor(obs_dim, act_dim, cfg.hidden, cfg.depth).to(device)
        self.critic1 = Critic(obs_dim, act_dim, cfg.hidden, cfg.depth).to(device)
        self.critic2 = Critic(obs_dim, act_dim, cfg.hidden, cfg.depth).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic1_t = copy.deepcopy(self.critic1)
        self.critic2_t = copy.deepcopy(self.critic2)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(list(self.critic1.parameters()) +
                                      list(self.critic2.parameters()), lr=cfg.lr_critic)
        self.tighten = ResidualTightening(rowwise=cfg.rowwise_margin, cap_frac=cfg.margin_cap_frac)
        self.bc_now = cfg.bc_weight
        self.total_it = 0
        self.qp_stats = {"n_solves": 0, "n_active": 0.0, "n_clipped": 0}

    # ------------------------------------------------------------------ #
    #  Action pipeline
    # ------------------------------------------------------------------ #
    def compose(self, res: torch.Tensor, base: Optional[torch.Tensor]) -> torch.Tensor:
        """Residual policy: base rule action + scaled network correction (identity when residual_scale == 0)."""
        if self.cfg.residual_scale > 0 and base is not None:
            return (base + self.cfg.residual_scale * res).clamp(-1, 1)
        return res

    def scale_to_box(self, a_norm, lo, hi):
        return lo + (a_norm + 1.0) * 0.5 * (hi - lo)

    def _margin(self, kind: str):
        t = self.tighten
        vec = t.margin_v_vec if kind == "v" else t.margin_s_vec
        if vec is None:
            return t.margin_v if kind == "v" else t.margin_s
        m = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        if kind == "s":                      # never let calibration noise empty a row
            m = torch.minimum(m, t.cap_frac * self.fbase.s_max * float(np.cos(np.pi / max(self.cfg.n_thermal_cuts, 3))))
        return m

    def make_polytope(self, ctx: Dict[str, torch.Tensor], v_offset=None) -> SafetyPolytope:
        c = self.cfg
        return build_polytope(
            R=self.fbase.R, X=self.fbase.X, M=self.fbase.M, A=self.fbase.A,
            p_exo=ctx["p_exo"], q_exo=ctx["q_exo"], s_max=self.fbase.s_max,
            lo=ctx["lo"], hi=ctx["hi"], v_min=self.fbase.p.v_min, v_max=self.fbase.p.v_max,
            v0=self.fbase.p.v_sub,
            n_cuts=c.n_thermal_cuts, allow_reactive=self.fbase.allow_reactive,
            v_offset=v_offset,
            margin_v=self._margin("v") if c.use_tightening else 0.0,
            margin_s=self._margin("s") if c.use_tightening else 0.0,
        )

    def safe_action(self, a_norm: torch.Tensor, ctx: Dict[str, torch.Tensor],
                    differentiable: bool = True, fast: bool = False) -> torch.Tensor:
        a_box = self.scale_to_box(a_norm, ctx["lo"], ctx["hi"])
        if not self.cfg.use_qp:
            return a_box                                  # soft-penalty arm: box only
        v_off = None
        if self.cfg.use_tightening and self.wm is not None:
            v_off = self.wm.voltage_offset(a_box.detach(), ctx["p_exo"], ctx["q_exo"],
                                           ctx["soc"], ctx["cap"], ctx["occ"], ctx["price"])
        poly = self.make_polytope(ctx, v_offset=v_off)
        w = vpw_weights_to_qp(ctx["vpw"], ctx["occ"], self.fbase.allow_reactive,
                              enabled=self.cfg.use_vpw_weights)
        a_safe = project(a_box, poly, w=w, rho=self.cfg.qp_rho, iters=(self.cfg.qp_iters_target if (fast and self.cfg.qp_hard_first) else self.cfg.qp_iters),
                           hard_first=self.cfg.qp_hard_first, rho_hard=self.cfg.qp_rho_hard, fast=fast)
        if not differentiable:
            a_safe = a_safe.detach()
        return a_safe

    @torch.no_grad()
    def act(self, obs: np.ndarray, ctx: Dict[str, np.ndarray], noise: float = 0.0,
            deterministic: bool = False, base_norm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        c = {k: torch.as_tensor(np.atleast_2d(v), dtype=torch.float32, device=self.device)
             for k, v in ctx.items()}
        a_norm = self.actor(o)
        if not deterministic and noise > 0:
            a_norm = (a_norm + noise * torch.randn_like(a_norm)).clamp(-1, 1)
        if self.cfg.residual_scale > 0 and base_norm is not None:
            a_norm = self.compose(a_norm, torch.as_tensor(base_norm, dtype=torch.float32, device=self.device).unsqueeze(0))
        a_safe = self.safe_action(a_norm, c, differentiable=False)
        return a_norm.squeeze(0).cpu().numpy(), a_safe.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_from_norm(self, a_norm_np: np.ndarray, ctx: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Project a given normalised action (e.g. a greedy demonstration) through the safety layer."""
        c = {k: torch.as_tensor(np.atleast_2d(v), dtype=torch.float32, device=self.device)
             for k, v in ctx.items()}
        a = torch.as_tensor(a_norm_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_safe = self.safe_action(a, c, differentiable=False)
        return a_norm_np.astype(np.float32), a_safe.squeeze(0).cpu().numpy()

    def bc_pretrain(self, batch: Dict[str, torch.Tensor]) -> float:
        """One supervised step: actor output -> greedy normalised action (no QP, cheap)."""
        tgt = torch.zeros_like(batch["act_greedy"]) if self.cfg.residual_scale > 0 else batch["act_greedy"]
        loss = F.mse_loss(self.actor(batch["obs"]), tgt)
        self.opt_a.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_a.step()
        return float(loss.detach())

    @torch.no_grad()
    def act_random(self, ctx: Dict[str, np.ndarray], base_norm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        c = {k: torch.as_tensor(np.atleast_2d(v), dtype=torch.float32, device=self.device)
             for k, v in ctx.items()}
        a_norm = torch.rand(1, self.actor.head.out_features, device=self.device) * 2 - 1
        if self.cfg.residual_scale > 0 and base_norm is not None:
            a_norm = self.compose(a_norm, torch.as_tensor(base_norm, dtype=torch.float32, device=self.device).unsqueeze(0))
        a_safe = self.safe_action(a_norm, c, differentiable=False)
        return a_norm.squeeze(0).cpu().numpy(), a_safe.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Imagined next observation (exogenous replay + model-predicted SoC)
    # ------------------------------------------------------------------ #
    def imagine_next_obs(self, next_obs_real: torch.Tensor, soc_hat: torch.Tensor,
                         cap: torch.Tensor) -> torch.Tensor:
        """Overwrite the action-dependent channels of the recorded next obs.

        Arrivals, departures, prices and loads are exogenous: they do not depend
        on the action, so replaying them from the logged trajectory is exact, not
        an approximation.  Only SoC and the derived energy-need channel are
        action-dependent, and those come from the world model.  This keeps the
        Dyna rollout honest: every synthetic quantity is genuinely predicted.
        """
        o = next_obs_real.clone()
        sl_soc, sl_tgt, sl_need = self.layout["soc"], self.layout["soc_tgt"], self.layout["e_need"]
        occ_next = o[:, self.layout["occupied"]]
        soc_new = torch.where(occ_next > 0.5, soc_hat, o[:, sl_soc])
        o[:, sl_soc] = soc_new
        o[:, sl_need] = torch.clamp(o[:, sl_tgt] - soc_new, min=0) * cap * self.fbase.p.s_base_kw / 100.0
        return o

    # ------------------------------------------------------------------ #
    def update(self, batch: Dict[str, torch.Tensor], soft_penalty: bool = False) -> Dict[str, float]:
        cfg = self.cfg
        self.total_it += 1
        obs, next_obs = batch["obs"], batch["next_obs"]
        act = batch["act_phys"] * ACT_SCALE
        rew, done = batch["rew"], batch["done"]
        keys = ["p_exo", "q_exo", "lo", "hi", "vpw", "occ", "soc", "cap", "price"]
        ctx_cur = {k: batch[k] for k in keys}
        if cfg.exact_next_ctx and "lo_n" in batch:
            ctx_next = {"p_exo": batch["p_exo_n"], "q_exo": batch["q_exo_n"], "lo": batch["lo_n"],
                        "hi": batch["hi_n"], "vpw": batch["vpw_n"], "occ": batch["occ_n"],
                        "soc": batch["soc_next"], "cap": batch["cap_n"], "price": batch["price_n"]}
        else:
            ctx_next = ctx_cur

        # ---------------- critic ----------------
        with torch.no_grad():
            a_next = self.actor_t(next_obs)
            noise = (torch.randn_like(a_next) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            a_next = (a_next + noise).clamp(-1, 1)
            if cfg.residual_scale > 0:
                a_next = self.compose(a_next, batch["act_base_n"])
            # NOTE: the target action is projected with the *next* step's box but
            # the current step's network context; this is a deliberate,
            # documented approximation (the alternative is storing a second full
            # constraint context per transition, doubling buffer memory for a
            # second-order effect).
            a_next_safe = self.safe_action(a_next, ctx_next, differentiable=False, fast=True) * ACT_SCALE
            q1t = self.critic1_t(next_obs, a_next_safe)
            q2t = self.critic2_t(next_obs, a_next_safe)
            target = rew + cfg.gamma * (1 - done) * torch.min(q1t, q2t)
        q1, q2 = self.critic1(obs, act), self.critic2(obs, act)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                 cfg.grad_clip)
        self.opt_c.step()

        info = {"critic_loss": float(loss_c.detach()), "q_mean": float(q1.mean().detach()),
                "target_mean": float(target.mean())}

        # ---------------- actor (delayed) ----------------
        if self.total_it % cfg.policy_delay == 0:
            res = self.actor(obs)
            a_norm = self.compose(res, batch["act_greedy"]) if cfg.residual_scale > 0 else res
            a_safe = self.safe_action(a_norm, ctx_cur, differentiable=True)
            q_term = self.critic1(obs, a_safe * ACT_SCALE).mean()
            loss_a = -q_term
            mb_term = torch.tensor(0.0, device=obs.device)
            if cfg.lambda_mb > 0 and self.wm is not None:
                pred = self.wm.predict(a_safe, batch["p_exo"], batch["q_exo"], batch["soc"],
                                       batch["cap"], batch["occ"], batch["price"])
                r_hat = self.fbase.analytic_reward(
                    a_safe, batch["soc"], batch["cap"], batch["occ"], batch["price"],
                    batch["vpw"], v=pred["v_hat"], s=pred["s_hat"], soft_penalty=soft_penalty,
                    soc_tgt=batch["soc_tgt"], departing=batch["departing"])
                nobs_hat = self.imagine_next_obs(next_obs, pred["soc_hat"], batch["cap"])
                with torch.no_grad():
                    a_n2 = self.compose(self.actor_t(nobs_hat), batch["act_base_n"]) if cfg.residual_scale > 0 else self.actor_t(nobs_hat)
                q_next = self.critic1_t(nobs_hat, self.safe_action(a_n2, ctx_next,
                                                                   differentiable=False, fast=True) * ACT_SCALE)
                mb_term = (r_hat + cfg.gamma * (1 - done) * q_next).mean()
                loss_a = loss_a - cfg.lambda_mb * mb_term
            bc_term = torch.tensor(0.0, device=obs.device)
            if self.bc_now > 0 and "act_greedy" in batch:
                # TD3+BC: scale the value terms by 1/|Q| so the BC weight is dimensionless
                qabs = q_term.detach().abs().clamp_min(1e-3)
                loss_a = loss_a / qabs
                bc_term = F.mse_loss(a_norm, batch["act_greedy"])      # residual mode: penalises deviation from base
                loss_a = loss_a + self.bc_now * bc_term
            self.opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            gn = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.opt_a.step()
            self._soft_update()
            info.update({"actor_loss": float(loss_a.detach()), "mb_term": float(mb_term.detach()), "bc_term": float(bc_term.detach()),
                         "actor_gnorm": float(gn)})
        return info

    def _soft_update(self):
        with torch.no_grad():
            for net, tgt in ((self.actor, self.actor_t), (self.critic1, self.critic1_t),
                             (self.critic2, self.critic2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    # ------------------------------------------------------------------ #
    #  Federated parameter split (Section 5.3): shared trunk / personal head
    # ------------------------------------------------------------------ #
    def shared_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if self._is_shared(k)}

    def load_shared(self, sd: Dict[str, torch.Tensor]):
        own = self.state_dict()
        for k, v in sd.items():
            if k in own:
                own[k].copy_(v)

    def _is_shared(self, key: str) -> bool:
        if key.startswith(("actor_t", "critic1_t", "critic2_t", "wm", "fbase")):
            return False
        return not any(h in key for h in self.cfg.head_keys)
