"""Structured replay buffer.

Unlike a vanilla (s, a, r, s') buffer, we store the *physical* fields the world
model and the safety layer need (exogenous injections, SoC, capacities, box
bounds, measured voltages/flows).  This is what lets the same buffer serve three
consumers -- the critic, the world-model regression, and the conformal
calibration of the safety margin -- without re-simulating anything.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

FIELDS_2D = ["obs", "next_obs", "act_norm", "act_phys", "p_exo", "q_exo", "soc",
             "soc_next", "cap", "occ", "v_true", "s_true", "lo", "hi", "vpw",
             "soc_tgt", "departing",
             # constraint context at the NEXT state (used for the critic target's
             # projected action; before this was the CURRENT step's context)
             "p_exo_n", "q_exo_n", "lo_n", "hi_n", "vpw_n", "occ_n", "cap_n",
             "act_greedy", "act_base_n"]      # normalised action of the deadline-aware greedy rule (BC anchor target)
FIELDS_1D = ["rew", "done", "price", "t", "price_n"]


class ReplayBuffer:
    def __init__(self, capacity: int, dims: Dict[str, int], device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.data = {k: np.zeros((capacity, dims[k]), dtype=np.float32) for k in FIELDS_2D}
        for k in FIELDS_1D:
            self.data[k] = np.zeros((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, **kw):
        i = self.ptr
        for k, v in kw.items():
            if k in self.data:
                self.data[k][i] = v
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: Optional[np.random.Generator] = None,
               idx: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        if idx is None:
            rng = rng or np.random.default_rng()
            idx = rng.integers(0, self.size, size=min(batch, self.size))
        out = {k: torch.as_tensor(v[idx], device=self.device) for k, v in self.data.items()}
        out["act"] = out["act_phys"]
        return out

    def all(self) -> Dict[str, torch.Tensor]:
        return self.sample(self.size, idx=np.arange(self.size))

    def __len__(self):
        return self.size
