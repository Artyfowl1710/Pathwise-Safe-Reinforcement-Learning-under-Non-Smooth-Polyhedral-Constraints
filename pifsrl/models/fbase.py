"""
F_base: the explicit, analytical, differentiable physics engine (Eq. 10's first
term) -- LinDistFlow network model + SoC dynamics + the closed-form reward.

The entire point of the architecture is that this is NOT a learned surrogate:
every output is an algebraic function of (state, action) with an exact Jacobian

        dF_base/da  =  [  R    X            ]        (voltages)
                       [ eta_ch dt / E_cap  ]        (SoC, charging)
                       [ -dt /(eta_dis E_cap) ]      (SoC, discharging)      (Eq. 11)

`analytic_jacobian()` returns that matrix in closed form and
`tests/test_gradients.py` checks it against torch.autograd to 1e-10.  That
equality is the concrete, checkable version of the proposal's central claim
("unbroken end-to-end analytical gradient backpropagation").

The reward is also analytic, which matters more than it looks: it means Dyna
rollouts need a learned model only for *dynamics*, never for reward, so imagined
returns cannot be corrupted by reward-model error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PhysicsParams:
    dt_h: float = 0.25
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.20
    soc_max: float = 0.95
    s_base_kw: float = 1000.0
    deg_beta: float = 0.055
    deg_gamma_exp: float = 1.35
    deg_dod: float = 0.020
    c_unmet: float = 0.90
    c_discomfort: float = 0.05
    c_violation: float = 60.0
    sell_ratio: float = 0.92
    reward_scale: float = 10.0
    reward_mode: str = "sparse"
    v_min: float = 0.95
    v_max: float = 1.05
    v_sub: float = 1.03


class FBase(nn.Module):
    """Analytical base model.  Stateless w.r.t. learning (no parameters)."""

    def __init__(self, feeder, params: PhysicsParams, k_max: int, A: np.ndarray,
                 allow_reactive: bool = True):
        super().__init__()
        self.p = params
        self.k_max = k_max
        self.allow_reactive = allow_reactive
        self.n_bus = feeder.n_bus
        self.n_branch = feeder.n_branch
        self.register_buffer("R", torch.tensor(feeder.R, dtype=torch.float32))
        self.register_buffer("X", torch.tensor(feeder.X, dtype=torch.float32))
        self.register_buffer("M", torch.tensor(feeder.M, dtype=torch.float32))
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("s_max", torch.tensor(feeder.s_max, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    def split(self, a: torch.Tensor):
        if self.allow_reactive:
            return a[..., :self.k_max], a[..., self.k_max:]
        return a, torch.zeros_like(a)

    def net_injection(self, a: torch.Tensor, p_exo: torch.Tensor, q_exo: torch.Tensor):
        """Section 3.2: bus-level aggregation is a *linear* sum, so the Jacobian
        d p_net,i / d p_k = 1 and Eq. 11 extends to the multi-EV case exactly."""
        p_ev, q_ev = self.split(a)
        p_net = p_exo - p_ev @ self.A.T          # charging is a load: minus sign
        q_net = q_exo - q_ev @ self.A.T
        return p_net, q_net

    def voltage(self, a, p_exo, q_exo, v0: Optional[float] = None):
        v0 = self.p.v_sub if v0 is None else v0
        p_net, q_net = self.net_injection(a, p_exo, q_exo)
        return v0 + p_net @ self.R.T + q_net @ self.X.T

    def branch_flow(self, a, p_exo, q_exo):
        p_net, q_net = self.net_injection(a, p_exo, q_exo)
        pf = -(p_net @ self.M.T)                 # power flowing DOWN the branch
        qf = -(q_net @ self.M.T)
        return pf, qf

    def soc_next(self, soc, a, cap_pu_h, occupied):
        p_ev, _ = self.split(a)
        pch = torch.clamp(p_ev, min=0.0)
        pdis = torch.clamp(-p_ev, min=0.0)
        d = (self.p.eta_ch * pch - pdis / self.p.eta_dis) * self.p.dt_h / cap_pu_h.clamp_min(1e-9)
        return torch.where(occupied > 0.5, (soc + d).clamp(0.0, 1.0), torch.zeros_like(soc))

    def forward(self, a, p_exo, q_exo, soc, cap_pu_h, occupied, v0: Optional[float] = None) -> Dict[str, torch.Tensor]:
        v = self.voltage(a, p_exo, q_exo, v0)
        pf, qf = self.branch_flow(a, p_exo, q_exo)
        s = torch.sqrt(pf ** 2 + qf ** 2 + 1e-12)
        return {"v": v, "s": s, "pf": pf, "qf": qf,
                "soc_next": self.soc_next(soc, a, cap_pu_h, occupied)}

    # ------------------------------------------------------------------ #
    def analytic_jacobian(self, a: torch.Tensor, cap_pu_h: torch.Tensor,
                          occupied: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Eq. 11 in closed form.

        Returns
        -------
        dv_da   : (B, n_bus, n_act)   = [R A , X A]
        dsoc_da : (B, k, n_act)       diagonal in the P block, zero in Q
        """
        B = a.shape[0]
        RA = (self.R @ self.A)
        XA = (self.X @ self.A)
        dv = torch.cat([-RA, -XA], dim=1) if self.allow_reactive else -RA
        dv = dv.unsqueeze(0).expand(B, -1, -1)

        p_ev, _ = self.split(a)
        charging = (p_ev >= 0).float()
        g = (charging * self.p.eta_ch + (1 - charging) / self.p.eta_dis) \
            * self.p.dt_h / cap_pu_h.clamp_min(1e-9)
        g = g * (occupied > 0.5).float()
        n_act = a.shape[-1]
        dsoc = torch.zeros(B, self.k_max, n_act, device=a.device, dtype=a.dtype)
        idx = torch.arange(self.k_max, device=a.device)
        dsoc[:, idx, idx] = g
        return {"dv_da": dv, "dsoc_da": dsoc}

    # ------------------------------------------------------------------ #
    def analytic_reward(self, a, soc, cap_pu_h, occupied, price, vpw,
                        v: Optional[torch.Tensor] = None,
                        s: Optional[torch.Tensor] = None,
                        soft_penalty: bool = False,
                        soc_tgt: Optional[torch.Tensor] = None,
                        departing: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Closed-form, differentiable replica of `V2GEnv.step`'s reward.

        Verified against the environment to <1e-6 in tests/test_reward_model.py.
        """
        p = self.p
        price = price.reshape(a.shape[0], -1)[:, 0]
        p_ev, _ = self.split(a)
        p_ev = p_ev * (occupied > 0.5).float()
        kwh = p_ev * p.s_base_kw * p.dt_h
        buy = torch.clamp(kwh, min=0).sum(-1) * price
        sell = torch.clamp(-kwh, min=0).sum(-1) * price * p.sell_ratio
        cost = buy - sell
        pdis = torch.clamp(-p_ev, min=0)
        cdeg = (p.deg_beta * (pdis * p.s_base_kw / 10.0).clamp_min(1e-9) ** p.deg_gamma_exp * p.dt_h
                + p.deg_dod * (1 - soc) ** 2 * (pdis > 1e-9).float())
        cost = cost + (cdeg * (occupied > 0.5).float()).sum(-1)
        cost = cost + (p.c_discomfort * (1 - vpw) * pdis * p.s_base_kw * p.dt_h).sum(-1)
        if soft_penalty and v is not None:
            under = torch.clamp(p.v_min - v, min=0).sum(-1)
            over = torch.clamp(v - p.v_max, min=0).sum(-1)
            pen = under + over
            if s is not None:
                pen = pen + torch.clamp(s - self.s_max.unsqueeze(0), min=0).sum(-1)
            cost = cost + p.c_violation * pen
        if soc_tgt is not None and departing is not None:
            soc_n = self.soc_next(soc, a, cap_pu_h, occupied)
            deficit = torch.clamp(soc_tgt - soc_n, min=0) * cap_pu_h * p.s_base_kw
            if p.reward_mode == "dense":
                gain = (torch.minimum(soc_n, soc_tgt) - torch.minimum(soc, soc_tgt)) * cap_pu_h * p.s_base_kw
                cost = cost - p.c_unmet * (gain * (occupied > 0.5).float()).sum(-1)
            else:
                cost = cost + p.c_unmet * (deficit * departing).sum(-1)
        return -cost / p.reward_scale
