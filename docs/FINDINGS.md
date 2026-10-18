# Empirical Findings & Scientific Conclusions

This document formalizes the scientific claims (**C0–C7**), evidence files, and conclusions of the **PIFSRL-V2G** study. Every claim corresponds to an executable test/script referenced in [`REPRODUCE.md`](../REPRODUCE.md).

---

## Overview of Core Findings

| ID | Topic | Headline Finding | Status | Evidence File |
| :--- | :--- | :--- | :--- | :--- |
| **C0** | **Conformal Margins** | Global conformal margins are overly conservative on feeders with heterogeneous branch limits; row-wise margins maintain coverage with minimal conservatism. | Confirmed | `docs/study/conformal_rowwise_vs_global.json` |
| **C1** | **Adaptive Calibration** | Static split-conformal voltage margins lose coverage under distribution load shift; Adaptive Conformal Inference (ACI) preserves target coverage. | Confirmed | `docs/study/adaptive_conformal_shift.json` |
| **C2** | **Tree Decomposition Lemma** | Cutting a radial distribution feeder couples sub-blocks through at most 3 boundary scalars, enabling exact block-decomposition. | Formally Verified | `pifsrl/physics/feeder.py`, `scripts/poc_tree_decomposition.py` |
| **C3** | **Tree QP Projection** | Decomposed QP projection achieves exact equivalence (gap $\le 10^{-4}$) to monolithic projection. | Confirmed | `docs/study/bench_tree_qp.json` |
| **C4** | **Optimization Headroom** | On real multi-day benchmarks with stress and demand charges, the headroom between a tuned heuristic (capped EDF) and the perfect-foresight LP oracle is modest (~5.2%). | Confirmed | `docs/study/headroom_multiday.json` |
| **C5** | **Safe Residual RL** | Residual SafeTD3 around a tuned EDF base guarantees 0 physical constraint violations and stays within 2–9% of the tuned heuristic without catastrophic degradation. | Confirmed | `docs/study/R*.json`, `docs/study/M*.json` |
| **C6** | **Cross-Site Transfer / FL** | Site policies transfer poorly without personalization due to non-IID charging patterns; FedProx matches local performance while enabling federated coordination. | Confirmed | `docs/study/transfer_*.json`, `docs/study/federated_*.json` |
| **C7** | **Price-Scenario Robustness** | Safety guarantees hold invariant to exogenous pricing scenarios (duck, volatile, solar glut, flat). | Confirmed | `docs/study/suite/price_scenarios.md` |

---

## Detailed Claims & Evidence

### C0. Row-Wise vs. Global Conformal Tightening
- **Claim:** A single global conformal margin computes a scalar bound over all lines and buses. Because line thermal capacities vary widely across the feeder (from 115 kVA on small branches to >1000 kVA at the substation), a global margin (45.3 kVA) consumes up to 39% of capacity on minor branches. Row-wise conformal margins provide per-constraint coverage with an order of magnitude smaller margin on critical branches.
- **Verification:** `python scripts/conformal_rowwise_vs_global.py`
- **Output:** Test coverage $\ge 0.984$ across both schemes; row-wise margins limit maximum margin to 7.1% of branch rating.

### C1. Adaptive Conformal Inference under Distribution Shift
- **Claim:** In distribution networks, exogenous load can shift significantly (e.g., seasonal or event-driven load multiplication). Under a 1.25× load shift, a fixed pre-calibrated conformal margin experiences empirical coverage drop from 95% down to ~8.1%. Adaptive Conformal Inference (Gibbs & Candès 2021) dynamically tracks the error quantile and maintains $95.0\% \pm 0.5\%$ empirical coverage.
- **Verification:** `python scripts/adaptive_conformal_shift.py`
- **Miniature Deterministic Test:** `pytest tests/test_reproducibility.py`

### C2. Radial Feeder Decomposition
- **Claim:** For any radial electrical distribution network modeled under LinDistFlow, isolating any sub-tree results in boundary interactions parameterized by only 3 boundary variables: upstream real power ($P$), reactive power ($Q$), and squared voltage ($V^2$).
- **Verification:** `python scripts/poc_tree_decomposition.py` (passes 27/27 cuts on IEEE-33 and 80/80 on IEEE-123).

### C3. Tree QP Decomposed Projection Equivalence
- **Claim:** Decomposed ADMM projections over partitioned feeder blocks converge to the identical optimal action as the global monolithic QP (residual gap $\le 10^{-4}$), validating distributed safety filtering without central state aggregation.
- **Verification:** `python scripts/bench_tree_qp.py`

### C4. Learning Headroom Over Tuned Heuristics
- **Claim:** On 20 real days from ACN-Data (Caltech site) under grid stress and SCE commercial demand charges, the performance gap between a tuned Earliest Deadline First (EDF) cap ($400.5) and the theoretical upper bound (perfect foresight LP oracle at $379.5) is only 5.2%. 
- **Significance:** Uncontrolled charging incurs heavy cost ($471.4); a well-tuned heuristic captures >75% of the total recoverable surplus.
- **Verification:** `python scripts/headroom_multiday.py --site caltech --days 20 --demand 1.5 --stress`

### C5. Safe Residual RL Guarantee
- **Claim:** Unconstrained RL frequently violates distribution voltage and thermal bounds. By wrapping the policy with a differentiable QP projection layer and parameterizing the policy as a residual around a stable baseline ($\pi(s) = \pi_{\text{base}}(s) + \Delta_\theta(s)$):
  1. Constraint violations remain strictly zero across all test days.
  2. The policy avoids catastrophic policy collapse, tracking within 2% to 9% of the hand-tuned base.
- **Verification:** `python scripts/run_suite.py --only multiday --sites caltech --regimes stress --seeds 0 1 2`

### C6. Federated Coordination Across Non-IID Sites
- **Claim:** EV arrival processes and user departure willingness (VPW) differ significantly between work sites (e.g., JPL vs. Caltech). Naive policy transfer from Caltech to JPL suffers significant performance degradation (+35% to +45% cost penalty). Federated learning via FedProx coordinates shared network representations while preserving site-specific personalization.
- **Verification:** `python scripts/run_suite.py --only transfer,federated`

### C7. Invariance to Exogenous Price Distributions
- **Claim:** While economic dispatch costs depend strongly on the tariff structure (e.g., CAISO LMP vs. TOU-D-PRIME), the physical safety guarantee (zero voltage drops below 0.95 p.u. and branch currents within rating) holds invariant to tariff scenario choice.
- **Verification:** `python scripts/run_suite.py --only scenarios --days 20`
