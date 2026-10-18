"""Tests for the federated layer: the shared/personal parameter split, FedAvg
aggregation arithmetic, and that heterogeneous clients actually produce
different local updates (i.e. Non-IID-ness is real, not accidentally IID)."""
import numpy as np
import pytest
import torch

from pifsrl.env.v2g_env import EnvConfig
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.federated.fedprox import ClientSpec, FedClient, FedConfig, fedavg_aggregate, run_federated


def _tiny_client(name, archetype, seed):
    ec = EnvConfig(k_max=4, archetype=archetype, cluster_size=6.0, horizon=16,
                   dt_h=0.5, seed=seed, session_density=1.0)
    tc = TD3Config(hidden=16, depth=1, qp_iters=15, n_thermal_cuts=4)
    wc = WMConfig(hidden=16, depth=1)
    return FedClient(ClientSpec(name, ec), tc, wc)


def test_shared_head_split_is_disjoint_and_covers_actor_critic():
    c = _tiny_client("a", "residential", 0)
    shared = set(c.shared_state().keys())
    full = set(dict(c.agent.named_parameters()).keys())
    heads = {k for k in full if any(h in k for h in c.agent.cfg.head_keys)}
    assert shared.isdisjoint(heads), "shared set must not include head parameters"
    assert shared | heads >= {k for k in full if k.startswith(("actor.", "critic1.", "critic2."))}


def test_world_model_and_targets_never_shared():
    c = _tiny_client("a", "residential", 0)
    shared = c.shared_state()
    for k in shared:
        assert not k.startswith("wm."), "world model must never be federated (site-specific physics)"
        assert not k.startswith(("actor_t", "critic1_t", "critic2_t")), \
            "target networks are not part of the federated exchange"


def test_fedavg_aggregate_is_weighted_mean():
    s1 = {"w": torch.tensor([1.0, 2.0])}
    s2 = {"w": torch.tensor([3.0, 4.0])}
    out = fedavg_aggregate([s1, s2], np.array([1.0, 3.0]))
    expected = (1 * s1["w"] + 3 * s2["w"]) / 4
    assert torch.allclose(out["w"], expected)


def test_load_shared_round_trips():
    c1 = _tiny_client("a", "workplace_caltech", 1)
    c2 = _tiny_client("b", "residential", 2)
    s1 = c1.shared_state()
    c2.load_shared(s1)
    s2_after = c2.shared_state()
    for k in s1:
        assert torch.allclose(s1[k], s2_after[k]), f"parameter {k} did not transfer"


def test_heterogeneous_clients_produce_different_local_updates():
    """A minimal, concrete Non-IID check: two structurally different
    archetypes (short-dwell workplace vs long-dwell residential), trained
    for one local round from an IDENTICAL shared initialisation, must diverge
    -- if they didn't, the clients would be accidentally IID and the whole
    federated ablation (Section 7.2) would be measuring nothing."""
    torch.manual_seed(0)
    c1 = _tiny_client("wp", "workplace_caltech", 10)
    c2 = _tiny_client("res", "residential", 11)
    shared0 = c1.shared_state()
    c2.load_shared(shared0)
    c1.local_train(episodes=10, global_shared=None, mu=0.0, wm_updates=3)
    c2.local_train(episodes=10, global_shared=None, mu=0.0, wm_updates=3)
    s1, s2 = c1.shared_state(), c2.shared_state()
    dist = sum(float(((s1[k] - s2[k]) ** 2).sum()) for k in s1) ** 0.5
    assert dist > 1e-8, "heterogeneous clients should not learn identical shared trunks"


def test_run_federated_smoke():
    clients = [_tiny_client(n, a, i) for i, (n, a) in
              enumerate([("wp", "workplace_caltech"), ("res", "residential")])]
    out = run_federated(clients, FedConfig(rounds=2, local_episodes=1, eval_every=2,
                                          eval_episodes=1), verbose=False)
    assert len(out["history"]) == 2
    assert len(out["evals"]) >= 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
