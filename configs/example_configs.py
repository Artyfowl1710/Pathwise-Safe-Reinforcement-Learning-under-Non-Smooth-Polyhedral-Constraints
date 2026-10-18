"""
Example, documented configurations for each experiment type. Import these
directly, or use them as a reference for CLI flags in sample.py.

    from configs.example_configs import CANONICAL_SINGLE_SITE
    from pifsrl.agents.trainer import train_site
    out = train_site(**CANONICAL_SINGLE_SITE)
"""
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.env.sessions import recommended_horizon
from pifsrl.models.world_model import WMConfig
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.agents.trainer import TrainConfig

# --------------------------------------------------------------------------- #
# Canonical single-site scenario: residential overnight charging, IEEE-33,
# a 3-hub fleet concentrated on the electrically weakest buses so the network
# constraint genuinely binds (see env/v2g_env.py siting note). This is the
# scenario used throughout RUNBOOK.md's incident reports.
# --------------------------------------------------------------------------- #
_dt_h = 0.25
_horizon = recommended_horizon("residential", _dt_h)

CANONICAL_SINGLE_SITE = dict(
    env_cfg=EnvConfig(feeder="ieee33", archetype="residential", k_max=24,
                      cluster_size=12.0, horizon=_horizon, dt_h=_dt_h,
                      session_density=1.4, seed=0),
    td3_cfg=TD3Config(hidden=256, depth=2, qp_iters=150, batch_size=128,
                      lambda_mb=0.5, n_thermal_cuts=8),
    wm_cfg=WMConfig(hidden=128, depth=2, mode="residual"),
    tr_cfg=TrainConfig(episodes=300, warmup_episodes=8, eval_every=20,
                       eval_episodes=5, seed=0),
)

# Same scenario, soft-penalty baseline (Section 7.2 ablation #2 arm B)
SOFT_PENALTY_BASELINE = dict(CANONICAL_SINGLE_SITE)
SOFT_PENALTY_BASELINE["td3_cfg"] = TD3Config(hidden=256, depth=2, use_qp=False)
SOFT_PENALTY_BASELINE["tr_cfg"] = TrainConfig(episodes=300, warmup_episodes=8,
                                              soft_penalty=True, seed=0)

# --------------------------------------------------------------------------- #
# Federated scenario: 5 Non-IID archetypes, one feeder each. mu=0.05 is a
# typical FedProx proximal coefficient (Li et al. 2020 use 0.01-1 depending
# on heterogeneity; 0.05 is a reasonable starting point for the archetype
# heterogeneity here -- sweep it if your ablation wants a mu-sensitivity plot).
# --------------------------------------------------------------------------- #
FEDERATED_ARCHETYPES = ["workplace_caltech", "workplace_jpl", "office",
                        "residential", "fleet_depot"]
FEDERATED_MU = 0.05
FEDERATED_ROUNDS = 60
FEDERATED_LOCAL_EPISODES = 2

# DISABLED (IEEE-123 / QP scalability benchmark out of scope):
# # --------------------------------------------------------------------------- #
# # QP scalability benchmark (Section 4.1). batch=200 matches the proposal's
# # "batches of 50, 200, 500 concurrent sessions" from ACN-JPL session-overlap
# # statistics; control_period_s=60 assumes a 1-minute V2G dispatch loop --
# # change to 300 for a 5-minute loop, which is more typical of utility DERMS
# # dispatch intervals and gives much more headroom.
# # --------------------------------------------------------------------------- #
# QP_BENCHMARK_BATCHES = [50, 200, 500]
# QP_BENCHMARK_CONTROL_PERIOD_S = 60.0
