"""
The Differentiable Residual-Dyna World Model:   s_hat = F_base(s,a) + N_theta(s,a)

N_theta is deliberately small.  It is not trying to learn the grid; F_base
already knows the grid.  It is learning a structured, low-magnitude correction:

  * the LinDistFlow linearisation error (second-order voltage drop + I^2 Z
    losses + the voltage-dependence of constant-power loads), and
  * the SoC-dependent charge-acceptance taper that Eq. 8's constant-eta model
    misses.

Because the correction is small and smooth, a 2x128 MLP suffices, the model
trains in seconds, and -- this is the part that matters for the proposal's
sample-efficiency claim -- it generalises from very few real transitions.  A
black-box surrogate asked to predict v directly has to learn the whole 32-bus
power flow from scratch and needs orders of magnitude more data to reach the
same voltage accuracy.  `--wm-mode blackbox` runs exactly that ablation.

PHYSICS-INFORMED REGULARISER  (lambda_phys L_KCL/KVL of Eq. 15)
---------------------------------------------------------------
We penalise violation of the *exact* DistFlow KVL recursion (Eq. 5) evaluated on
the corrected prediction:

  res_b = v_i - v_j - 2(r_b P_b + x_b Q_b) + (r_b^2 + x_b^2) (P_b^2+Q_b^2)/v_i

with v = squared voltage.  Note this is the term LinDistFlow *drops*, so the
regulariser is not redundant with F_base: it pushes N_theta towards corrections
that are consistent with the full non-linear branch equations rather than
arbitrary function fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fbase import FBase, PhysicsParams


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class ResidualNet(nn.Module):
    """N_theta.  Outputs (dv, ds_flow, dsoc), each scaled to a small range."""

    def __init__(self, feat_dim: int, n_bus: int, n_branch: int, k_max: int,
                 hidden: int = 128, depth: int = 2,
                 v_scale: float = 0.05, s_scale: float = 0.25, soc_scale: float = 0.05):
        super().__init__()
        self.n_bus, self.n_branch, self.k_max = n_bus, n_branch, k_max
        out = n_bus + n_branch + k_max
        self.net = mlp([feat_dim] + [hidden] * depth + [out])
        # last layer starts near zero: the model begins life as pure F_base,
        # which is a meaningful prior and stabilises early training
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        self.register_buffer("scales", torch.tensor(
            [v_scale] * n_bus + [s_scale] * n_branch + [soc_scale] * k_max))

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        o = torch.tanh(self.net(feat)) * self.scales
        nb, nbr = self.n_bus, self.n_branch
        return {"dv": o[..., :nb], "ds": o[..., nb:nb + nbr], "dsoc": o[..., nb + nbr:]}


class BlackBoxNet(nn.Module):
    """Ablation: predict the next physical state directly (no F_base prior).

    This is the 'static PINN surrogate / black-box' arm of Table 2.  Identical
    capacity budget x4 so the comparison is not a straw man.
    """

    def __init__(self, feat_dim, n_bus, n_branch, k_max, hidden=256, depth=3):
        super().__init__()
        self.n_bus, self.n_branch, self.k_max = n_bus, n_branch, k_max
        self.net = mlp([feat_dim] + [hidden] * depth + [n_bus + n_branch + k_max])

    def forward(self, feat):
        o = self.net(feat)
        nb, nbr = self.n_bus, self.n_branch
        return {"dv": o[..., :nb], "ds": o[..., nb:nb + nbr], "dsoc": o[..., nb + nbr:]}


@dataclass
class WMConfig:
    hidden: int = 128
    depth: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_phys: float = 1e-2
    lambda_soc: float = 5.0
    lambda_flow: float = 0.2
    mode: str = "residual"        # residual | blackbox | fbase_only
    grad_clip: float = 5.0


class WorldModel(nn.Module):
    """F_base (+ N_theta) with training, calibration and rollout utilities."""

    def __init__(self, fbase: FBase, feeder, cfg: WMConfig, allow_reactive: bool = True):
        super().__init__()
        self.fbase = fbase
        self.cfg = cfg
        self.allow_reactive = allow_reactive
        n_bus, n_branch, k = fbase.n_bus, fbase.n_branch, fbase.k_max
        self.feat_dim = 2 * n_bus + (2 * k if allow_reactive else k) + 3 * k + 2
        if cfg.mode == "blackbox":
            self.net: Optional[nn.Module] = BlackBoxNet(self.feat_dim, n_bus, n_branch, k,
                                                        hidden=max(cfg.hidden * 2, 256),
                                                        depth=cfg.depth + 1)
        elif cfg.mode == "fbase_only":
            self.net = None
        else:
            self.net = ResidualNet(self.feat_dim, n_bus, n_branch, k,
                                   hidden=cfg.hidden, depth=cfg.depth)
        self.register_buffer("r_br", torch.tensor(feeder.r_pu, dtype=torch.float32))
        self.register_buffer("x_br", torch.tensor(feeder.x_pu, dtype=torch.float32))
        self.register_buffer("Mb", torch.tensor(feeder.M, dtype=torch.float32))
        self.register_buffer("from_idx", torch.tensor(feeder.branches[:, 0].astype(np.int64)))
        self.register_buffer("to_idx", torch.tensor(feeder.branches[:, 1].astype(np.int64)))
        self.opt = None
        if self.net is not None:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                         weight_decay=cfg.weight_decay)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _flat_price(price: torch.Tensor, B: int) -> torch.Tensor:
        """Accept price as (B,), (B,1) or (1,1); always return (B,1)."""
        return price.reshape(B, -1)[:, :1]

    def features(self, a, p_exo, q_exo, soc, cap_pu_h, occupied, price):
        pr = self._flat_price(price, p_exo.shape[0])
        return torch.cat([p_exo, q_exo, a, soc, cap_pu_h, occupied,
                          pr, (soc * occupied).sum(-1, keepdim=True)], dim=-1)

    def predict(self, a, p_exo, q_exo, soc, cap_pu_h, occupied, price,
                detach_residual: bool = False) -> Dict[str, torch.Tensor]:
        base = self.fbase(a, p_exo, q_exo, soc, cap_pu_h, occupied)
        if self.net is None:
            return {**base, "v_hat": base["v"], "s_hat": base["s"],
                    "soc_hat": base["soc_next"], "dv": torch.zeros_like(base["v"])}
        feat = self.features(a, p_exo, q_exo, soc, cap_pu_h, occupied, price)
        res = self.net(feat)
        if detach_residual:
            res = {k: v.detach() for k, v in res.items()}
        if self.cfg.mode == "blackbox":
            v_hat, s_hat = res["dv"], res["ds"]
            soc_hat = res["dsoc"]
        else:
            v_hat = base["v"] + res["dv"]
            s_hat = base["s"] + res["ds"]
            soc_hat = torch.where(occupied > 0.5,
                                  (base["soc_next"] + res["dsoc"]).clamp(0, 1),
                                  torch.zeros_like(soc))
        return {**base, "v_hat": v_hat, "s_hat": s_hat, "soc_hat": soc_hat, **res}

    # ------------------------------------------------------------------ #
    def kvl_residual(self, v_hat, pf, qf) -> torch.Tensor:
        """Exact DistFlow KVL residual (Eq. 5) on the corrected prediction."""
        vv = v_hat ** 2                                       # squared voltage
        ones = torch.ones(vv.shape[0], 1, device=vv.device, dtype=vv.dtype)
        v_full = torch.cat([ones, vv], dim=1)
        vi = v_full[:, self.from_idx]
        vj = v_full[:, self.to_idx]
        i2 = (pf ** 2 + qf ** 2) / vi.clamp_min(1e-6)
        res = vi - vj - 2 * (self.r_br * pf + self.x_br * qf) + (self.r_br ** 2 + self.x_br ** 2) * i2
        return res

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.predict(batch["act"], batch["p_exo"], batch["q_exo"], batch["soc"],
                           batch["cap"], batch["occ"], batch["price"])
        l_v = F.mse_loss(out["v_hat"], batch["v_true"])
        l_s = F.mse_loss(out["s_hat"], batch["s_true"])
        m = (batch["occ"] > 0.5).float()
        l_soc = (((out["soc_hat"] - batch["soc_next"]) ** 2) * m).sum() / m.sum().clamp_min(1.0)
        l_phys = (self.kvl_residual(out["v_hat"], out["pf"], out["qf"]) ** 2).mean()
        loss = l_v + self.cfg.lambda_flow * l_s + self.cfg.lambda_soc * l_soc \
            + self.cfg.lambda_phys * l_phys
        return loss, {"wm_loss": float(loss.detach()), "wm_v_mse": float(l_v.detach()),
                      "wm_s_mse": float(l_s.detach()),
                      "wm_soc_mse": float(l_soc.detach()), "wm_kvl": float(l_phys.detach()),
                      "wm_v_mae": float((out["v_hat"] - batch["v_true"]).abs().mean().detach())}

    def update(self, batch, steps: int = 1) -> Dict[str, float]:
        if self.net is None:
            with torch.no_grad():
                _, info = self.loss(batch)
            return info
        info = {}
        for _ in range(steps):
            loss, info = self.loss(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
            self.opt.step()
        return info

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def voltage_offset(self, a, p_exo, q_exo, soc, cap_pu_h, occupied, price) -> torch.Tensor:
        """N_theta's voltage correction, used to tighten the QP (Section 4)."""
        if self.net is None or self.cfg.mode == "blackbox":
            return torch.zeros(p_exo.shape[0], self.fbase.n_bus,
                               device=p_exo.device, dtype=p_exo.dtype)
        feat = self.features(a, p_exo, q_exo, soc, cap_pu_h, occupied, price)
        return self.net(feat)["dv"]

    @torch.no_grad()
    def evaluate(self, batch) -> Dict[str, float]:
        out = self.predict(batch["act"], batch["p_exo"], batch["q_exo"], batch["soc"],
                           batch["cap"], batch["occ"], batch["price"])
        m = (batch["occ"] > 0.5).float()
        return {
            "v_mae": float((out["v_hat"] - batch["v_true"]).abs().mean()),
            "v_max_ae": float((out["v_hat"] - batch["v_true"]).abs().max()),
            "v_rmse": float(((out["v_hat"] - batch["v_true"]) ** 2).mean().sqrt()),
            "soc_mae": float((((out["soc_hat"] - batch["soc_next"]).abs()) * m).sum()
                             / m.sum().clamp_min(1.0)),
            "s_mae": float((out["s_hat"] - batch["s_true"]).abs().mean()),
            "base_v_mae": float((out["v"] - batch["v_true"]).abs().mean()),
        }
