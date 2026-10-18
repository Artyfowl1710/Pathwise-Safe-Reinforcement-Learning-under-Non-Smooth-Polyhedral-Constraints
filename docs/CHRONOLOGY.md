# Chronology and idea ledger (what you proposed, what happened to it, what I added and why)

## Timeline
* **Proposal (original_proposal.pdf)**: physics-informed federated safe RL for V2G; residual-Dyna world model F_base + N_theta; differentiable QP safety layer; conformal tightening; FedProx; willingness-aware (VPW) bidding; IEEE-33/123.
* **Build phase (DOC.md, RUNBOOK.md)**: simulator on ACN data, SafeTD3, QP layer, world model, FedProx, ablation runner. Debugging of the world-model sign error, conformal over-conservatism, buffer next-step context.
* **Oct 15** you asked for a literature review and a findings-driven paper, including the differentiable-environment idea. Result: LITERATURE.md; the finding that tuned heuristics leave little headroom.
* **Oct 15-16** headroom study with a violation-free tuned reference; residual SafeTD3 around EDF (never wins); conformal row-wise vs global and ACI under load shift; the twin (`diffenv.py`) and pathwise learning; reviewer round 1 fixes (Fig. 1 label, unprojected reference bug).
* **Oct 16** real ACN CSVs, tariff, WEP price files added; real-day twin runs.
* **Oct 17** reviewer round 2; this update: GPU plumbing, chronological split, projection-price test, congestion sweep, world-model ablation, `RUN_ALL.ps1`.

## Your ideas: status
| your idea | status | why / evidence |
|---|---|---|
| Residual-Dyna world model F_base + N_theta | implemented, tested; **ablation 1 was never run until today; the earlier paper drafts did not report it (my omission)**. Result now in paper Sec. 8 | black-box ~70x worse; residual helps worst-case voltage 2.8x and flows 3x, not mean voltage much, and worsens SoC (unexplained). Whether it helps the policy (A5) is not tested |
| Differentiable QP safety layer | used everywhere | works; hard-safety gives zero violations; soft-penalty ablation implemented, not run for the paper |
| Conformal tightening | kept and extended | found global margin wrong object (row-wise), joint coverage gap, load-shift inflation physical |
| "Residual" RL | done in a different form: residual SafeTD3 around a tuned rule | never beats the tuned rule with real-day budgets; the reason is small headroom |
| Federated personalised learning (FedProx) | pilot only | no benefit over local tuning in 2 sites; needs a 3rd site |
| VPW willingness bidding, V2G discharge | disabled in headline runs | my own 8-day check: hindsight-oracle gain 15 % ($57/day), causal v2g_greedy 16 % worse; the collaborator's E9 (<= $0.9/day) was not reproduced; VPW ablation implemented, not run |
| IEEE-123 scaling | lemma verified, solver not faster | consensus-ADMM exact but slower than monolithic |
| Differentiable environment (your Oct 15 idea) | built (`diffenv.py`) | exact fidelity 1e-16; pathwise wins only where headroom is large (synthetic) |

## What I added on my own, and why
* **Headroom gate** (H = (J_ref - J_oracle)/J_ref): needed because RL claims are meaningless without knowing whether anything is left to gain; added after the first tuned baselines were already near the LP bound.
* **Projection of the heuristic reference** and violation gating: fixed an unfair comparison (the unprojected rule violated the network).
* **Row-wise/Bonferroni/max-score margins, ACI, load-normalised score**: your conformal layer was over-conservative; I tested why and rejected my own hypothesis (normalisation) with data.
* **Sens+STE restoration, projection-price test, chronological split, congestion sweep**: added in response to reviewer critiques to separate "learning failed" from "restoration lost energy" and from "no headroom to gain".
* **Reproducibility package** (pins, SHA-256, tests) and `RUN_ALL.ps1`.

## Update (Oct 17, afternoon)
Your residual network now has a result worth keeping: the grid-only N_theta cuts the one-step conformal margin 2.6x (voltage) and 9.8x (branch S); its SoC head was the harm and the 0.024 SoC "error" was arrival/departure steps (reviewer's diagnosis, verified). Still not tested: effect on policy cost and on real days.

## Update (Oct 17, evening)
Reviewer's "projection-price" hypothesis tested properly (closed-loop, validated): at JPL the earlier 10-18 % learning loss was the ray-scaling restoration (the reference rule itself is +17.9 % under ray scaling); one of three pathwise seeds beats the reference by 7 % under the exact QP; the other two never learned. My first, open-loop version of this test was invalid and was discarded. Pending when this was written: 2019 chronological split (queued, CPU-bound).
