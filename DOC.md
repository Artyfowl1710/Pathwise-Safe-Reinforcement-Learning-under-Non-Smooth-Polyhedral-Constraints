# DOC.md — What happened, why, what we fixed, what is still open

Project: **PIFSRL-V2G** — Physics-Informed Federated Safe RL for vehicle-to-grid coordination
(SafeTD3 + differentiable QP safety layer + residual world model + FedProx), IEEE-33 feeder,
real ACN-Data sessions (Caltech/JPL), real SCE TOU-D-PRIME tariff.

This file is the **audit trail for the write-up**: every shortcoming found, the evidence, the
cause, the fix, and what is *not* fixed. Numbers marked **[run]** came from a run in this
repository (JSON/screen outputs in `docs/study/`, `results/`); numbers marked **[prof]** came from
the professor's 5-day Caltech run; nothing here is estimated from memory.

---------------------------------------------------------------------------------------------------

## 0. TL;DR (read this first)

> **Status 2026-10-12 (latest, supersedes older claims below).**
> (1) Safety is solid: 0 violations for every learned and heuristic-with-QP policy, QP row-wise conformal margins, feasibility certificate.
> (2) **Learning headroom is small** (§6.6): on 20 real days with demand charge + stress, oracle beats a tuned fixed EDF cap by only 5.2 %;
> a per-day scalar cap already reaches the oracle.  (3) **Safe residual SafeTD3 around the tuned EDF** removes the late-degradation
> problem and is within +1.8 % (res 0.1) to +6.4 % (res 0.3) of the EDF over 3 seeds, but **does not beat it** (§6.7-6.10).
> (4) The external Optuna/Antigravity result (291.9) is a train-day validation number, not comparable to held-out test costs (§6.8).
> (5) Therefore the defensible paper is a **safety + benchmark + headroom-methodology paper**, not "RL beats heuristics"; see §9.1.

1. **Starting point.** Pipeline, real-data loader, world model, safety layer all worked. SafeTD3
   was *worse than greedy* after 300 episodes on every real Caltech day **[prof]**.
2. **Why (measured).**
   (a) The unmet-energy penalty is **sparse and delayed** (paid only at departure, one scalar for
   54 slots) → credit-assignment failure.
   (b) The QP's **ADMM solver was unconverged** and *over-conservative* (removed ~33 % of requested
   charging where the exact projection removes ~13 %), and ~30 % of training samples fell into a
   slow, ill-conditioned slack path after conformal tightening.
   (c) Evaluation was **optimistic** (best of 15 noisy 3-episode evals).
   (d) On *unstressed* real days **greedy is already near-optimal and almost never violates**, so
   there is little to gain and nothing for the safety layer to do.
3. **Fixes.** Dense service reward, hard-projection QP with row generation and an exact
   feasibility certificate, next-step constraint context in the critic target, fixed-seed
   validation + best/last on fresh seeds, QP-intervention logging.
4. **Effect.** With the fixes SafeTD3 reaches greedy-level quality in ≈10 episodes (cost 281,
   86.5 % served at ep 9) instead of staying at idle level (624 / 51 %) **[run]**; it then
   **oscillates** (281 → 407 → 295 → 351) — an optimisation-stability problem.
5. **Key late finding (S14).** The post-calibration collapse/oscillation is caused by a *global*
   conformal margin (45.8 kVA) that is larger than most of the hub-branch headroom (73–128 kVA);
   per-row margins (0–7 kVA on those branches) remove it.
6. **New (this iteration).** Greedy-anchored actor (TD3+BC style + greedy demonstrations), a
   perfect-foresight **oracle LP**, and a **stress benchmark** where greedy genuinely violates the
   grid (2× charger power, 100 kW hub: uncontrolled overloads branches by 0.22 p.u.).
7. **Honest status.** Not yet publishable at an A* main track: single day, single seed, 40
   episodes, no multi-seed statistics, no theory section. The roadmap in §9 is what closes that.

---------------------------------------------------------------------------------------------------

## 1. System in one page

* **State** (120-d): hour (sin/cos), price now/+1/+4, exogenous P/Q sums, per-slot occupied, SoC,
  target, time remaining, energy need, rate, willingness (VPW), bus.
* **Action** (108-d): (P, Q) set-point per charge point (54 slots × 2), normalised in [-1, 1] and
  affinely mapped to the per-step box [lo, hi] (rating + SoC feasibility).
* **Safety layer.** a_safe = argmin ½‖a − a_raw‖²_W s.t. G a ≤ h, lo ≤ a ≤ hi, where the rows G, h
  are LinDistFlow voltage rows + an 8-cut thermal polygon, tightened by a conformal margin
  (margin_v ≈ 1.4–2.7e-3 p.u., margin_s ≈ 0.02–0.035 p.u. ≈ 20–35 kVA). Differentiable via implicit
  KKT gradients (OptNet-style, `_ProjectFn`).
* **Actor loss.** −Q(s, a_safe) − λ_mb·(r̂ + γ Q_t(s', a')) where r̂ and s' come from the residual
  world model (analytic F_base reward + learned residual N_θ).
* **Critic.** Twin critics, target-policy smoothing, delayed actor updates; critic sees the
  *post-QP physical action*.
* **Reward** (per step): −(energy cost + degradation + discomfort + unmet energy)/10.
  `c_unmet = 2.0 /kWh`.
* **Real data.** Caltech: 54 EVSE, 150 kW feeder capacity, hubs at buses 16/15/14, 192 steps × 15
  min. Branch limits ≈ 1.35 × nominal flow + 0.55 × nameplate.

---------------------------------------------------------------------------------------------------

## 2. Chronology (including our own wrong turns — keep them, reviewers like honesty)

| # | What we believed | What we found |
|---|---|---|
| 1 | "The quicktest hang is caused by duplicate CSVs / Windows file locking." | Wrong. Quicktest never reads the CSVs. Retracted; added a watchdog and file-level logging (`runlog.py`). |
| 2 | "The polytope declares idle infeasible, so the QP forces discharging." | Wrong *at the time it was measured* (idle was always feasible with zero margin). The real cause was ADMM non-convergence with slack columns. (Later: after conformal calibration idle **is** infeasible on ~30 % of samples — a *different* mechanism, see S3.) |
| 3 | "SafeTD3 is too conservative because safety is expensive." | Mostly wrong. Greedy rarely violates on real days; the conservatism was an artefact of the solver (S2) and the reward (S1). |
| 4 | "Optuna will fix it." | Optuna tunes scalars; it cannot repair a structural credit-assignment problem (S1) or a biased solver (S2). It is still useful *after* the structure is right. |
| 5 | "The prof's QP-ablation rows A–D are four distinct arms." | They differ only in `c_violation`, which the QP arm never reads → identical runs (S7). |

---------------------------------------------------------------------------------------------------

## 3. Shortcomings catalogue

Format: **symptom → cause → evidence → fix → status / residual risk.**

### S1 — Sparse, delayed, global reward (the biggest single cause)
* **Symptom.** With the QP fixed but the original reward, SafeTD3 stays at idle level for 40
  episodes: cost 624, 50.7 % served **[run, arm B]**.
* **Cause.** The unmet-energy term (`c_unmet · deficit`) is charged only at departure/horizon end,
  for 54 slots, folded into one scalar. A charging decision at 09:00 is penalised at 17:00; with
  γ = 0.99 over 192 steps the critic must propagate that signal through ~100 steps while the
  immediate price cost of charging is *right now*. Result: "don't charge" looks locally optimal.
* **Fix.** `reward_mode="dense"`: per-step credit `c_unmet · Δ min(SoC, target) · cap` instead of the
  departure penalty. Return-equivalent up to a **policy-independent constant** (Σ over sessions of
  c_unmet·(target − min(SoC_arrival, target))·cap), so the optimal policy is unchanged; only the
  credit assignment improves. `FBase.analytic_reward` mirrors it so the model-based term is
  consistent.
* **Evidence.** Arm C (dense): unmet 160–260 kWh vs 650–800 for the sparse arm; cost 281 on
  validation at ep 9 **[run]**.
* **Status.** Fixed. *Theory note for the paper:* potential-based shaping (Ng, Harada, Russell 1999)
  preserves optimal policies; our dense reward is exactly a potential difference Φ(s') − Φ(s) with
  Φ = c_unmet·Σ min(SoC, tgt)·cap, so the policy-invariance result applies directly.

### S2 — QP solver unconverged and biased toward "do less"
* **Symptom.** `safe_v2g_greedy` and SafeTD3 both lost far more charging to the safety layer than
  the exact projection does.
* **Cause.** Slack-augmented QP: slack columns carry a 300× linear penalty next to actions of
  order 1e-2 p.u. → badly conditioned; fixed-iteration ADMM (120) returns a point that is *not* the
  projection. On real steps it removed **33 %** of requested charging; the exact OSQP solution of the
  same QP removes **12.8 %**. 1000 iterations were needed to converge.
* **Fix.** Hard projection with **no slack columns**, small ρ (0.1); matches the exact solution to
  0.05 kW in 120 its. Plus (S3) a relaxation for rows that are infeasible at idle. Unit test:
  `test_hard_first_matches_exact`.
* **Status.** Fixed for the training/eval path. The legacy path is kept behind `--legacy-qp` to
  reproduce old numbers.

### S3 — After conformal calibration, idle is infeasible on ~30 % of samples
* **Symptom.** Training was ~10× slower than expected and the QP kept hitting the slack path.
* **Cause.** The conformal margin (≈ 32 kVA) is comparable to the headroom of the small hub
  branches (limits ≈ 73–100 kVA). After calibration, `h_j < 0` for some thermal rows at idle on
  ~30 % of samples ⇒ the hard feasible set is *empty* ⇒ old code used the badly conditioned slack
  path.
* **Fix.** Rows with `h_j < 0` are relaxed to `h_j := 0` ("never make a row worse than idling").
  Idle is then always feasible ⇒ the hard QP always has a solution.
* **Residual risk / paper note.** This is a **design decision**, not a free lunch: when the margin
  eats all headroom the controller is prevented from *increasing* flow on that row but cannot repair a
  pre-existing violation. State this explicitly in the method section; consider a margin-capping
  rule `margin_j ≤ α · headroom_j` as an ablation.

### S4 — Critic target used the wrong constraint context
* **Symptom.** (Bias, not a crash.) Target action `a' = π_t(s')` was projected with the *current*
  step's box/exogenous injections.
* **Fix.** The buffer now stores the next-step context (`p_exo_n, q_exo_n, lo_n, hi_n, vpw_n, occ_n,
  cap_n, price_n`); `exact_next_ctx=True` uses it. (`--legacy-qp` restores the old behaviour.)

### S5 — Solver performance and numerical tolerance
* ADMM tolerance 1e-7 is unreachable in float32 on O(1e-2) p.u. quantities ⇒ every call ran the full
  iteration budget. Now 2e-6 for float32.
* Dense `(m + 2n)`-row implicit-gradient solve replaced by an **active-set-compressed** solve
  (pinned box coordinates dropped; only active polytope rows ≈ <30): mathematically identical,
  ~100× cheaper (tests pass: finite-difference gradient check).
* ADMM now applies the box block of C = [G; I] **implicitly** (no materialised identity rows).
* Row generation: solve with the 48 tightest rows, verify against all rows, enlarge if violated.
* torch with >1 thread **thrashes** on the many tiny batched solves on small CPUs
  (measured 1.3 s vs 0.075 s per update). `--threads 1`.

### S6 — Evaluation protocol was optimistic (winner's curse)
* The summary reported `evals[-1]` of evaluations drawn with *different* seeds each time, and
  "best of N" selection over 3-episode evals is biased upward.
* **Fix.** Fixed validation seeds (90000+) for selection; best checkpoint kept; **last vs best both
  re-evaluated on fresh seeds (7 000 000+)** that were never used for selection
  (`results[...]["final"]`).
* **Evidence.** Arm C: validation-best (ep 9) = 281 / 86.5 %; on fresh seeds the *best* checkpoint
  scores 314 / 83.2 % and the *last* 360 / 78.2 % **[run]**. The gap between 281 and 314 is the
  winner's curse in numbers.

### S7 — The professor's ablation rows A–D are identical
* The four rows differ only in `c_violation`, used only by the soft-penalty arm. With the QP active
  the value is never read ⇒ identical trajectories. Do not report them as distinct arms.

### S8 — On unstressed real days greedy is ~optimal and safe
* **Evidence [run].** Caltech 2019-10-18: uncontrolled = tou_greedy = 279 cost, 86.6 % served;
  violation 8.9e-3 p.u. total (tiny); `v2g_greedy` 288 (worse). ToU shifting buys nothing because
  dwell times are short relative to the cheap-price window, so "urgent" fires immediately.
* **Consequence.** No learning-based method can show a large win here; the safety layer has nothing
  to prevent. This is why the paper needs the **stress benchmark** (§5) — it is not "cheating", it is
  the regime in which the problem is non-trivial (high EV penetration, constrained feeders).

### S9 — Single-day training, no train/val/test split
* All runs train and evaluate on the *same calendar day* (different stochastic seeds). Results
  are overfit to one day's session pattern. Needed: date-based split (`optuna_search.py` already
  trains on one day and scores validation seeds; the final test day must stay untouched).

### S10 — Late-training instability (open)
* **Symptom.** After reaching greedy-level in ~10 episodes the policy oscillates: validation cost
  281 → 407 → 295 → 351 **[run, arm C]**.
* **Hypotheses (to be separated by ablation, not asserted):** (i) critic over-estimation on the
  rarely visited high-charge region; (ii) actor LR too high for a 108-d tanh head; (iii) the
  model-based term injecting a biased gradient (its effect looked small: arms C and E, λ_mb=0,
  tracked each other closely in early episodes — *unfinished*); (iv) exploration noise on 108 dims
  breaking coordinated schedules; (v) non-stationarity from re-calibrating the conformal margin every
  5 episodes (the *constraint set itself moves under the critic*). **(v) is now the demonstrated
  dominant cause — see S14.**
* **Mitigation implemented.** Greedy-anchored actor: `L_a = −(Q + λ_mb·MB)/|Q|_mean + w_bc ‖π(s) −
  a_greedy‖²` (TD3+BC, Fujimoto & Gu 2021) with *greedy demonstrations* in warmup, actor
  pre-training on them, and annealed `w_bc`. Result in §6.

### S11 — QP output left small residual violations
* Under maximum-charge stress the fixed-iteration QP left 0.0116 p.u. summed thermal overload over 7
  steps. Added `_restore_feasibility`: shrink toward idle by the largest θ ∈ [0,1] that makes all
  linear rows hold (idle feasible + convexity ⇒ stays in the box). Residual after the fix: 6.8e-5
  p.u. (the AC-vs-linear gap and the 1e-4 floor). Unit test included.
* **Paper note.** The guarantee is *with respect to the linearised model*; AC violations are
  measured, not proved.

### S14 — A single global conformal margin collapses the safe set (the cause of the "wobble")
* **Symptom.** Policy reaches greedy quality while the margin is 0 (episodes ≤ 9, `n_calib = 0`),
  then collapses to idle immediately after the first calibration (episode 10 → 11: cost ≈ 225 → ≈ 650,
  unmet 80 → 560 kWh), in *both* stress arms, even with the greedy anchor
  (`docs/study/legacy_margin_collapse/`). In the unstressed run the same mechanism shows up as the
  oscillation 281 → 407 → 295 → 351 (calibration is repeated every 5 episodes, so the constraint
  set moves under the critic).
* **Cause.** `ResidualTightening` used one scalar per family: the 99 %-quantile over samples of
  **max over all branches** of |s_true − s_pred|. The maximum is attained on the heavily loaded trunk
  (loss-driven error). Measured **[run]**: legacy global margin = **45.8 kVA**; thermal limits of the
  hub branches = **73 / 99.7 / 127.8 kVA** → the margin removes 63 % of the smallest headroom;
  with the relaxation of S3 the row becomes "do not increase flow".
  Per-branch one-sided 99 % margins on the same data: **7.0 / 0 / 6.0 kVA** on the three smallest
  branches (median over branches 3.0, max 41 kVA on the trunk where the limit is 4.5 MVA).
* **Fix.** Row-wise conformal margins: per-bus (two-sided) and per-branch (one-sided, because only an
  *under-estimated* flow endangers the limit), finite-sample level ⌈(n+1)(1−α)⌉/n, optional
  Bonferroni (α/m) for joint coverage, and a cap at 35 % of the row's limit so calibration noise can
  never empty a row. Legacy behaviour behind `--legacy-margin`.
* **Why this matters scientifically.** It is the first place where the *world-model ↔ safety-layer
  coupling* (the paper's claim) is quantitative: the world-model error does not just improve
  prediction, it sets how much of the feeder's headroom the controller is allowed to use. A
  model-accuracy metric that matters is therefore **error relative to row headroom**, not MAE.
* **Open.** Per-row margins give marginal (per-row) coverage, not joint coverage; with Bonferroni
  the margins grow ~log m. Report both. Exchangeability between calibration (on-policy buffer) and
  deployment (new policy, new day) is an assumption that must be tested.

### S12 — Oracle and baseline caveats
* `oracle_lp` (perfect foresight) fixes Q = 0, ignores the DoD degradation term and the
  willingness/discomfort term, and uses the same linearised rows as the QP. It is an **upper
  reference**, not a strict bound for the true environment (the env has SoC-dependent efficiency).
  Do not write "optimal"; write "perfect-foresight reference".

### S13 — Things we did *not* do (be upfront)
* 123-bus scale-up removed from scope at the professor's request (and the earlier "matrix explosion"
  remains unresolved there; do not claim scalability to 123 buses).
* JPL and communication-delay/loss experiments deferred by the professor.
* Federated results on real data not re-validated after the fixes above.
* No theory section yet (§8 proposes what *is* provable).

---------------------------------------------------------------------------------------------------

## 4. What was added in this iteration (and why)

| Component | Where | Why |
|---|---|---|
| Dense service reward | `EnvConfig.reward_mode`, `FBase.analytic_reward`, `--reward-mode` | S1 |
| Hard-projection QP, relaxation, row generation, feasibility certificate | `safety/qp_layer.py` | S2, S3, S11 |
| Next-step constraint context in buffer + critic target | `agents/replay.py`, `agents/trainer.py`, `agents/safe_td3.py` | S4 |
| QP-intervention logging (mean/median/P90/95/99, % modified, % heavy, cut fraction) | `trainer.EpisodeStats` | professor's plan |
| Fixed-seed validation, best-vs-last on fresh seeds | `trainer.train_site` | S6 |
| Greedy demonstrations + BC pre-training + TD3+BC actor loss | `trainer`, `safe_td3` (`--warmup-mode greedy --bc-pretrain N --bc-weight w --bc-final w2`) | S10 |
| Perfect-foresight oracle LP (voltage+thermal rows) | `agents/oracle.py` (`oracle_lp` rule) | headroom reference |
| Stress benchmark presets | `sample.py --stress / --stress-ev / --stress-cap / --stress-load` | S8 |
| Row-wise conformal margins (+cap, finite-sample level, Bonferroni option) | `safety/qp_layer.py::ResidualTightening`, `--legacy-margin` | S14 |
| Optuna search script | `scripts/optuna_search.py` | professor's request |

---------------------------------------------------------------------------------------------------

## 5. Benchmarks

### 5.1 Why a stress benchmark
At normal ratings greedy serves 86.6 % with ~zero violations, so a "safe" learner has no job. We
scale the charger power and tighten the hub capacity so that *plug-and-charge physically violates the
feeder*. Measured **[run]** (2 stochastic episodes, Caltech 2019-10-18, greedy rules; `s` = summed
thermal overload in p.u.; voltage never violates — the 33-bus feeder is voltage-strong at this
loading, which is itself a finding):

| load | EV power × | hub cap (kW) | uncontrolled cost / served | thermal overload `s` |
|---|---|---|---|---|
| 0.75 | 1 | 150 | 279 / 87 % | 0.010 |
| 0.75 | 1 | 100 | 279 / 87 % | 0.038 |
| 0.75 | 2 | 150 | 194 / 95 % | 0.145 |
| **0.75** | **2** | **100** | **194 / 95 %** | **0.246** ← *stress preset* |
| 0.75 | 3 | 100 | 172 / 97 % | 0.535 |
| 1.00 | 2 | 100 | 194 / 95 % | 0.138 |

### 5.2 Stress reference table (2× power, 100 kW hub) **[run, 2 episodes]**
| method | cost | served | violation (p.u.) |
|---|---|---|---|
| uncontrolled | 194.1 | 95.0 % | 0.221 |
| tou_greedy | 194.8 | 94.8 % | 0.221 |
| v2g_greedy | 218.4 | 93.1 % | 0.274 |
| safe_v2g_greedy (greedy + QP) | 243.2 | 90.5 % | 0 |
| **oracle_lp (perfect foresight)** | **211.6** | **93.1 %** | **0** |

**Reading.** The safe-but-myopic rule pays ≈ 32 cost units and ≈ 2.6 pp of service to be safe; the
oracle shows that ≈ 12 % of that is recoverable by *allocating scarce capacity intelligently*
(who gets power when, based on deadlines and price). That recoverable gap is the learner's
opportunity and the paper's central quantitative claim: **"fraction of the safe-greedy → oracle gap
closed by SafeTD3"**.

---------------------------------------------------------------------------------------------------

## 6. Results so far

### 6.1 Unstressed Caltech day, 40 episodes **[run]**
| arm | what | last (fresh seeds) | best (fresh seeds) |
|---|---|---|---|
| greedy refs | uncontrolled / tou_greedy | 279 / 86.6 % | — |
| safe_v2g_greedy | rule + QP | 300–313 / 84–85 % | — |
| B | QP fix, **sparse** reward | 624 / 50.7 % | 624 / 50.7 % |
| C | QP fix + **dense** reward | 360 / 78.2 % | 314 / 83.2 % (ep 9) |

Validation trajectory of C (3 fixed seeds): ep 9: 281/86.5 %, 14: 407/74 %, 19: 329/82 %, 24: 295/85 %,
29: 313/83 %, 34: 341/80 %, 39: 351/79 %. QP intervention on the last policy: modified 33 % of
steps, mean removed charging 8.8 %, zero violations.

### 6.2 Stress runs on one day (2019-10-18, 2x charger power, 100 kW hub cap), 40 episodes **[run]**
Cost = total_cost (energy + unmet penalty + degradation); fresh evaluation seeds; all rows zero violations unless noted.
Evidence: `docs/study/S1_stress_bc.json`, `S2_stress_bc.json`.

| policy | cost | served |
|---|---|---|
| uncontrolled / tou_greedy (no safety layer) | 194 | 95 % (but overloads branches by 0.22-0.29 p.u.) |
| safe_v2g_greedy (rule + QP) | 245-272 | 88-90 % |
| oracle_lp (tight margins, hindsight) | 213.7 | 93.1 % |
| **S1** SafeTD3, no BC anchor — best (ep 9) | **209.5** | 93.4 % |
| S1 — last (ep 39) | 279.7 | 85.9 % |
| **S2** SafeTD3 + greedy anchor (bc 1.0 -> 0.2) — best (ep 9) | 260.4 | 89.1 % |
| S2 — last (ep 39) | 267.1 | 88.6 % |

Reading it honestly:
* The best checkpoint (ep 9) is oracle-level (209.5 vs 213.7; the "oracle" is not a true lower bound because
  of its margins/convex approximation, see S12). Training then **degrades** (S1: 207 -> 279 over ep 9-39).
* The BC anchor made the late curve flatter (S2: 210 -> 255) but did not remove the degradation and cost
  performance at the best point. So the anchor is **not** a free win; it stays an ablation arm, not a default.
* Unexplained: why quality peaks at ep ~9 and decays. Hypotheses (not yet tested): noise annealing, critic
  drift under the dense shaped reward, replay dominated by late near-greedy data. Open item in section 12.

### 6.3 Where is there headroom at all? (`docs/study/headroom_oracle_vs_safe_uncontrolled.json`) **[run]**
Eight real Caltech days, normal and stress. `safe_uncontrolled` = plug-and-charge + QP (no learning).
On 14 of 16 cases it is within ~1-3 % of the oracle; on the two cases where the unprotected policy overloads
the feeder (2018-09-26 and 2021-09-13 stress), the QP costs ~6-8 % more energy but removes all violations
(0.625 and 0.31 p.u. overload -> 0).
**Consequence for the paper:** on real, energy-only-tariff ACN days a learned policy has almost nothing to
gain over "charge immediately + safety layer". Claiming otherwise would not survive a reviewer who runs that
baseline. This is why we introduced the scenarios where temporal coupling matters.

### 6.4 Demand-charge scenario (single day 2019-10-18, 1.5 $/kW of day-peak site power) **[run]**
safe_uncontrolled 438.3, EDF cap 1.0 / 0.6 / 0.45 / 0.3 = 456.5 / 414.4 / **386.8** / 433.0 (cap 0.3 serves only
79.7 %), oracle 379.8. A fixed cap tuned in hindsight on the test day recovers ~90 % of the possible gain. A
fair comparison must tune the cap on training days only (next section).

### 6.5 Multi-day protocol (`--multiday N`) **[run, 60 episodes, 1 seed]**
20 weekday dates evenly spread over 2018-2021, split 60/20/20 train/val/test with a fixed RNG. One SafeTD3
policy is trained on train days (each episode samples a day), checkpoints are chosen on val days, and the
report is on test days never seen. Baselines on the same test days: tou_greedy, safe_uncontrolled,
safe_tou_greedy, EDF-cap (cap tuned on **train** days only), oracle_lp. Arms: M1 (BC anchor), M2 (no anchor);
both with demand charge 1.5, stress, dense reward, lambda_mb = 0 (the analytic model-based term has no
demand-charge component, so it would mislead the actor).
Held-out test days (4 days, fresh seeds; cost = total_cost incl. demand charge; evidence `docs/study/M1_multiday.json`, `M2_multiday.json`):

| policy | cost | served | violation |
|---|---|---|---|
| tou_greedy (no safety layer) | 326.7 | 83.7 % | 0.088 p.u. |
| safe_uncontrolled | 313.3 | 83.9 % | 0 |
| safe_tou_greedy | 306.1 | 83.4 % | 0 |
| **EDF cap 0.6 (cap tuned on train days)** | **266.1** | 84.0 % | 0 |
| oracle_lp (hindsight) | 246.1 | 82.4 % | 0 |
| SafeTD3 M1 (BC anchor) best / last | 383.8 / 443.0 | 73.8 / 60.0 % | 0 |
| SafeTD3 M2 (no anchor) best / last | 313.5 / 395.2 | 83.9 / 73.3 % | 0 |

**Verdict (negative, and we should say so in the paper unless it changes):** after 60 episodes SafeTD3's best
checkpoint (M2) merely reproduces plug-and-charge + QP (313.5 vs 313.3) and does not discover the
peak-shaving behaviour that a one-parameter tuned heuristic already exploits (266). The tuned heuristic
closes ~70 % of the gap between safe_uncontrolled and the oracle. The anchor (M1) hurt. Last-checkpoint
quality is again much worse than best-checkpoint quality (S10 persists across 3 independent setups).
Training returns are very noisy across sampled days (cost 175-980 per episode, mostly day-to-day variance), so 60 episodes
on 12 train days is far too small a budget to judge; the GPU multi-seed run (>= 500 episodes, 5 seeds) is what
decides. Caveats: 1 seed, 4 test days, hyper-parameters untuned.
**What we do NOT claim:** that SafeTD3 beats tuned heuristics. What is established: the safety layer keeps
violations at 0 on every policy including bad ones, and learning has to be evaluated against the tuned-EDF baseline.

### 6.6 Learning headroom with NO learning (20 weekdays, stress + demand 1.5) **[run]** — `docs/study/headroom_multiday.json`, `scripts/headroom_multiday.py`
| policy | mean cost | served |
|---|---|---|
| safe_uncontrolled | 471.4 | — |
| EDF cap, best FIXED cap (0.6) | 400.5 | 83.0 % |
| EDF cap, best cap chosen PER DAY (clairvoyant, 1 scalar) | 382.4 | — |
| oracle_lp (hindsight) | 379.5 | 82.3 % |

Two facts follow. (i) The oracle's whole advantage over a tuned fixed cap is only **5.2 %**; a single scalar chosen per
day recovers essentially all of it (382.4 vs 379.5), so the optimal policy on this benchmark is "EDF with a
day-dependent cap". (ii) Consequently any learned policy can improve on a tuned EDF by at most ~5 % here, and
that ceiling is unreachable online (the per-day cap is chosen with hindsight). A learner that merely matches the tuned
heuristic is therefore the realistic good outcome; claiming a big performance win on this benchmark would be wrong.

### 6.7 Safe residual RL around the tuned EDF (R1) — fixes the late-degradation problem **[run, 1 seed]** — `docs/study/R1_residual_edf.json`
Pre-QP action = clip(base_EDF(cap tuned on train days) + 0.3·pi(s)), then the QP; critic sees the post-QP action; base
is also fed into the critic target and the model-based term. Same 60-episode, 4 held-out-day protocol as 6.5:

| policy | held-out cost (mean +/- day std) | served |
|---|---|---|
| EDF cap 0.6 | 266.1 +/- 133.9 | 84.0 % |
| oracle_lp | 246.1 +/- 135.1 | 82.4 % |
| SafeTD3 plain, M2 best / last | 313.5 / 395.2 | 83.9 / 73.3 % |
| **SafeTD3 residual R1, last / best (ep 14)** | **282.9 / 287.9** | 83.7 / 84.0 % |

* Last ~ best: the late-degradation of S10 disappears (the residual bound caps how far a drifting critic can
  push the policy). Cost 313-443 -> 283-288 and zero violations.
* But R1 is still ~6 % *worse* than the tuned EDF it is built on (283 vs 266). Hypotheses (untested): exploration
  noise of 0.25 on the residual, the QP margin shaving the base, critic noise from a 4-day val set. Day-to-day std (134)
  is far larger than the differences, so paired per-day comparisons and >= 5 seeds are mandatory.

### 6.8 Review of the external Optuna/Antigravity report (12 trials, `docs/study/antigravity_optuna_best.json`)
The report claims "SafeTD3 291.89 outperforms safe greedy (300-313)". We do not accept this as stated:
the 291.89 is the **validation** cost of the training day 2019-10-18 (unstressed, no demand charge), whereas the
table next to it lists **held-out stress+demand test days** (baselines 437-548): different benchmarks. On that same
unstressed day plain uncontrolled charging costs ~279 (86.6 % served), i.e. better than 291.9. The best of 12 trials
is also a winner's-curse estimate (S6), and the trial's validation curve (286-295 from ep 10) is indeed *stable*, which is
the useful finding. Its "Timid Agent" explanation (the agent stops charging to avoid demand charge / QP cuts)
is plausible, consistent with our S1 (delayed unmet penalty) but not proven; "formally prove" is unsupported. We
re-test these parameters on the held-out protocol (R2, R3 below).
### 6.9 Residual RL, 3 seeds + variants (held-out days, cost relative to the train-tuned EDF cap) **[run]**
Seeds change both the network init and the train/val/test split, so compare each run with its own EDF row.
Evidence: `docs/study/R1_*, R1s1_*, R1s2_*, R3_*, R4_*`.

| run | EDF (tuned) | oracle | SafeTD3 last | SafeTD3 best |
|---|---|---|---|---|
| R1 seed 0 (res 0.3) | 266.1 | 246.1 | 282.9 (+6.3 %) | 287.9 (+8.2 %) |
| R1s1 seed 1 | 410.6 | 393.2 | 428.5 (+4.4 %) | 419.3 (+2.1 %) |
| R1s2 seed 2 | 449.5 | 438.6 | 487.9 (+8.5 %) | 485.6 (+8.0 %) |
| R3 (res 0.3 + Optuna params, no BC) | 266.1 | 246.1 | 291.1 (+9.4 %) | 288.9 (+8.6 %) |
| R4 (res 0.3, noise 0.10 -> 0.03) | 266.1 | 246.1 | 287.6 (+8.1 %) | 287.1 (+7.9 %) |

**Result:** over three seeds residual SafeTD3 is consistently **2-9 % worse than the tuned EDF** (mean last-checkpoint
+6.4 %), always safe (0 violations), and always much better than the unprotected/plain arms. Lower exploration noise
(R4) and the Optuna parameters (R3) do **not** change this, so "noise" and "optimiser settings" are rejected as the
cause of the gap. Because the policy starts exactly at the base (zero-initialised head), learning from 60 episodes of
on-policy data makes it *worse* than its starting point: the critic's signal is noisier than the base's
hand-tuned structure. This is the central open scientific problem and it is documented as such.
### 6.10 What bounds the residual's damage (R5, R6) and the Optuna parameters on plain SafeTD3 (R2) **[run, seed 0]**
| run | change | last | best |
|---|---|---|---|
| R5 | R1 with `--no-tighten` (no conformal margin) | 287.9 (+8.2 %) | 287.9 (+8.2 %) |
| **R6** | **residual scale 0.1** | **270.9 (+1.8 %)** | 273.8 (+2.9 %) |
| R2 | plain SafeTD3 (no residual) with the Optuna parameters | 375.5 (+41 %), 74.8 % served | 363.4 (+37 %) |

(+x % = relative to the tuned EDF cap 266.1 on the same 4 held-out days; zero violations in all three.)
* R5 = R1: the margin does **not** explain the gap (the conformal margin is not binding on these days).
* R6 vs R1: the cost is monotone in how far the policy may deviate (scale 0.1 / 0.3: +1.8 % / +6.3 %). At the
  limit scale -> 0 the policy equals the EDF base. **In this budget the learned correction has negative value.**
* R2: the external Optuna parameters, applied to the plain (non-residual) learner and judged on held-out days, give
  363-376, worse than our own plain run M2 (313.5 best) and far from the EDF (266.1). Optuna on the *training day*
  therefore did not transfer to *other days* — over-fitting to one date, as the protocol warned.

<!--R23_RESULTS-->


---------------------------------------------------------------------------------------------------


### 6.11 Cross-site transfer and federated pilot (Caltech <-> JPL, normal and stress) **[run, seed 0]** — `docs/study/transfer_*.json`, `federated_*.json`, `scripts/cross_site_fed.py`
* Caltech-trained residual SafeTD3 in-domain: +0.9 % (normal) / +1.8 % (stress) vs the tuned EDF.
* **A tuned cap does not transfer to JPL** (best cap 0.45-0.6 at Caltech, 1.0 at JPL): Caltech cap costs JPL +35-45 % vs QP-wrapped plain charging; the residual inherits that (Prop. 3) and only recovers 6-7 % in stress.
* Safety-wrapped plain charging is within ~6 % of the oracle at JPL (and below the margin-tightened oracle in the stressed arms); raw JPL-tuned EDF violates the feeder (0.03-0.15 p.u.).
* Tariff shift (SCE stand-in -> synthetic CAISO-like) changes levels, not the ordering of policies.
* Federated pilot (8 rounds x 2 local episodes, locally tuned bases): FedProx ~ local-only within +/-0.5 %; no federation benefit measurable yet; zero violations; JPL stress: EDF base 794.7 vs safe-uncontrolled 687.5 (gate should refuse the residual).
Caveats: one seed, 4 test days/site, tiny training budget; real CAISO LMP and historical commercial tariffs still missing (see docs/DATASETS_REQUIREMENTS.md).

### 6.12 New proposal
`docs/proposal_v2.pdf` (source `docs/proposal_v2/proposal_v2.tex`) re-positions the work: certified-safe dispatch + headroom-gated learning + validation-gated residual + exact recursive tree decomposition (theory only, lemma checked numerically by `scripts/poc_tree_decomposition.py`). Negative PoCs recorded: day-level cap forecasting (`scripts/poc_cap_selection.py`) and price-blind online peak shaving (`oa_peak`).

## 7. Ablation plan (decides what the paper may claim)

Principle: **every component must earn its place.** If an ablation shows a component does not help,
drop it or reframe the contribution.

| ID | Arm | Question |
|---|---|---|
| A0 | safe_v2g_greedy, tou_greedy, uncontrolled, **oracle_lp**, MPC-with-forecast (todo) | reference ladder |
| A1 | SafeTD3, sparse reward | does shaping matter? (expected: yes, S1) |
| A2 | + dense reward | baseline learner |
| A3 | + greedy anchor (BC) | does the anchor stop the oscillation / beat safe-greedy? |
| A4 | A3 − QP (soft penalty) / Lagrangian (PPO-Lag or CPO) | is the *hard* layer worth it vs standard safe RL? |
| A5 | A3 − model-based term (λ_mb = 0) | does the world model help the *policy* (not only the margin)? |
| A6 | A3 − conformal tightening; legacy global vs row-wise vs Bonferroni margins | cost of safety margins vs violation rate; **demonstrate S14** |
| A7 | residual world model vs F_base only vs black-box | is the residual necessary? |
| A8 | legacy slack-ADMM QP | quantify S2 on the stress benchmark |
| A9 | federated vs local (communication experiments last) | personalisation value |

Protocol: 5 seeds × ≥3 train days; model selection on validation *days*; report test days only once;
mean ± 95 % CI (bootstrap over seeds×days); paired comparison to safe-greedy; QP-intervention table;
violation table (voltage and thermal separately).

---------------------------------------------------------------------------------------------------

## 8. What can honestly be *proved* (so the paper doesn't over-claim)

1. **Feasibility under the model.** Idle ∈ feasible set after the h ← max(h, 0) relaxation; hence the
   hard QP is feasible; the output of `_restore_feasibility` satisfies every linear row exactly.
2. **Conformal safety.** If the linearisation error on constrained quantities is exchangeable between
   calibration and deployment, the tightened rows imply true (AC) feasibility with probability ≥ 1 − α
   (standard split-conformal result; Vovk et al.; Angelopoulos & Bates). Exchangeability is an
   assumption: say so, and test under distribution shift (other days/sites).
3. **Shaping invariance.** The dense reward is a potential-based shaping term ⇒ same optimal
   policies (Ng et al. 1999).
4. **Gradient exactness.** The implicit gradient is exact at an optimum of the QP with non-degenerate
   active set; our solver is approximate, so *quantify* the bias (finite-difference check exists).
5. **Cost of safety.** A bound of the form J(π_safe) ≥ J(π_unsafe) − L·(margin + ε_solver) for an
   L-Lipschitz value function — plausible but not yet derived; do not claim until it is.

Non-claims: no formal safety *during exploration* beyond the projection; no guarantee for AC
violations; no scalability beyond 33 buses.

---------------------------------------------------------------------------------------------------

## 9. Roadmap to an A*-calibre submission (what reviewers will ask)

1. Multi-seed, multi-day, date-split evaluation (professor's GPU run) — **blocking**.
2. Stress benchmark as a headline table + sweep over stress level (curve of violation vs service vs
   method) — the figure that sells the paper.
3. Baselines: MPC with forecast error, Lagrangian/CPO-style safe RL, safe-greedy, oracle.
4. The ablation grid in §7.
5. One theorem-grade statement (§8.2 or a tight version of §8.5) + empirical coverage of the
   conformal margin (calibration plot).
6. Hyper-parameter fairness: same Optuna budget for every learned baseline.
7. Federated section only after the single-site claim is solid.
8. Candidate venues by claim strength: strong single-site stress result ⇒ NeurIPS/ICML workshops
   (climate / safe RL), IEEE Trans. Smart Grid, e-Energy/BuildSys; main-track A* needs items 1–5 *and*
   a clear algorithmic delta over "safety layer + TD3".

---------------------------------------------------------------------------------------------------

### 9.1 Paper strategy given the evidence (honest, in decreasing order of what the data already supports)
1. **Certified-safe V2G dispatch on real data**: row-wise split-conformal margins for LinDistFlow->AC mismatch (finite-sample coverage),
   feasibility restoration with an exact certificate (S11), and the empirical result that unprotected greedy overloads the feeder
   (0.09-0.31 p.u.) while every QP-wrapped policy has 0.  Provable pieces: coverage of the margin, feasibility after restoration, potential-based shaping invariance.
2. **Benchmark + headroom methodology (a Datasets & Benchmarks / applied-energy contribution)**: real ACN-Data days, SCE TOU tariff,
   demand charge, stress, multi-day train/val/test protocol, *and the "learning-headroom test"*: before claiming RL, compare to a tuned
   heuristic and a hindsight oracle. We show the headroom is ~5 %, which is itself a publishable, reviewer-proof insight and explains
   why many RL-for-EV papers beat only untuned baselines.
3. **Negative result on safe RL**: a differentiable-QP actor-critic (plain or residual) does not beat a one-parameter EDF; the failure
   mode (late degradation / critic noise) is characterised and bounded by residual RL. Keep this honest in the writeup.
4. Not yet supported: any claim that SafeTD3 outperforms tuned heuristics; any federated/JPL claim (not run, per the professor).

### 9.2 Experiments that could still change the picture (priority order, GPU, >=5 seeds, paired by day)
a. Scenario with real *uncertainty-driven* headroom: price/arrival forecast error, V2G with an assumed battery distribution (ACN has none), or a
   much larger fleet (123-bus matrix growth only matters then).  Re-run the headroom test FIRST; only continue if oracle-vs-tuned-heuristic > ~15 %.
b. Learn the **one scalar** (the cap, or a small vector: cap + slope) with a contextual bandit / TD3 over a 2-3 d action space instead of 108-d;
   the clairvoyant per-day cap reaches the oracle (§6.6), so this is the right low-variance formulation and the QP stays as the certificate.
c. Critic quality: n-step returns, larger replay, 500+ episodes, distributional critic; measure Q-error against Monte-Carlo returns.
d. Lagrangian / CPO / shielded-PPO baselines to make the safety comparison fair (currently only raw greedy is the unsafe baseline).

## 10. Literature anchors (verify bibliographic details before citing)

* Safe exploration via a safety layer: Dalal et al. 2018. Differentiable QP layers: Amos & Kolter
  (OptNet), 2017. Constrained policy optimisation: Achiam et al. 2017; Lagrangian/RCPO: Tessler et al.
  2018; Safety Gym: Ray et al. 2019.
* TD3: Fujimoto et al. 2018; TD3+BC: Fujimoto & Gu 2021; residual policy learning: Silver et al. /
  Johannink et al. 2018–19.
* Model-based RL: Dyna (Sutton 1991), MBPO (Janner et al. 2019).
* Reward shaping invariance: Ng, Harada, Russell 1999.
* Conformal prediction: Vovk et al.; Angelopoulos & Bates tutorial.
* OSQP/ADMM: Stellato et al. 2020 (OSQP), Boyd et al. 2011 (ADMM).
* Data/feeder: ACN-Data (Lee, Li, Low 2019); IEEE-33 (Baran & Wu 1989).
* Power-systems safe-RL surveys and EV applications: arXiv 2407.00304 (safe RL for modern power
  systems); NREL safe deep RL for active distribution MPC with EVs/DERs; arXiv 2110.10333
  (computationally efficient safe RL for power systems).

---------------------------------------------------------------------------------------------------

## 11. Reproduction

```bash
# unit tests (35+) and quick check
python sample.py --tests ; python sample.py --quicktest

# fixed pipeline, dense reward, unstressed
python sample.py --mode single --real-data caltech --acn-date 2019-10-18 --episodes 40 \
    --eval-every 5 --log-every 1 --threads 1 --device cpu --reward-mode dense

# stress benchmark + greedy anchor
python sample.py --mode single --real-data caltech --acn-date 2019-10-18 --episodes 40 \
    --stress --reward-mode dense --warmup-mode greedy --bc-pretrain 200 --bc-weight 1.0 --bc-final 0.2

# old pipeline (S2/S4)
python sample.py ... --legacy-qp

# hyper-parameters
python scripts/optuna_search.py --real-data caltech --train-date 2019-10-18 --trials 30 --device cuda
```

On a small CPU use `--threads 1` (S5). Training cost scales with the share of steps where the QP is
active: ≈1–5 min/episode on 2 cores when charging is heavy; GPU is far faster.

---------------------------------------------------------------------------------------------------

### 11.1 New flags (this iteration)
```
--demand 1.5          # $/kW of the day's peak site power (running-peak is in the observation)
--multiday 20         # train/val/test date protocol; tunes the EDF cap on train days only
--stress              # 2x charger power, 100 kW hub cap
--bc-weight 1.0 --bc-final 0.1 --warmup-mode greedy --bc-pretrain 300   # greedy anchor arm
--lambda-mb 0         # required for demand-charge arms
```
Full command: `scripts/run_multiday.sh <name> [extra flags]` (results in `results/study3/<name>/multiday_caltech_seed0.json`).
Runs are CPU-friendly (~0.5 s/update with `--threads 1`); the professor can scale `--episodes` and loop `--seed`.

### 6.13 Plan implementation (2026-10): MPC, per-hub caps, adaptive conformal, gate, tree-QP benchmark, drop-in data **[code + smoke/physics runs; the big GPU runs are NOT done]**
| item | code | what was run | result (honest) |
|---|---|---|---|
| Forecast-free rolling-horizon MPC | `baselines.py` (`mpc`, `mpc_r<k>`), `oracle.solve_oracle(known_only, reserve_steps)` | 1 stress day (2018-04-25) | **Worse than the tuned EDF**: mpc 345.5 (90% served), mpc_r8 317.9, mpc_r16 266.6 vs edf_cap_0.4 208.3 / oracle 214.0. Without an arrival forecast the LP under-reserves. 40-150 s per day. Treat as a weak baseline, not as evidence against MPC with a forecast. |
| Per-hub EDF caps | `edf_hub_<c1>_<c2>_<c3>`, `--hub-grid`, `headroom_multiday.py --hub-grid` (A2 kill test <5 %) | JPL stress, 3 days | per-hub triple was **worse** than the single cap (-33 %) -> A2 kill criterion triggered on this pilot; needs the 20-60 day run to be conclusive. |
| Adaptive conformal (ACI) under load shift | `pifsrl/safety/adaptive.py`, `scripts/adaptive_conformal_shift.py` | IEEE-33, calibrate at load 0.75, deploy at 1.25, 20 seeds x 4000 steps | fixed split-conformal margin coverage **8.1 %** (target 95 %; 95.0 % without a shift); ACI **95.0 %** (min 94.9 % over 20 seeds). Physics-only, LinDistFlow vs exact BFS; the strongest safety-layer result so far. Caveat: long-run coverage (Gibbs-Candes), transient miscoverage right after the shift is expected. |
| Statistics / deployment gate | `pifsrl/evalstats.py` (paired bootstrap, sign test, Hoeffding gate, IQM), wired into `sample.py --multiday` (`stats_vs_edf`) and `scripts/report_table.py` | unit tests (type-I error <= delta verified) | Plain Hoeffding is very conservative: certifying a $20/day gain with B=$300 needs ~1,350 days. Use bootstrap / empirical-Bernstein in the paper and say so. |
| Tree-decomposed safety projection | `scripts/bench_tree_qp.py` (consensus ADMM over blocks, coupling = 2 scalars per cut) | random radial trees n=33/123/400 | **Exact** when converged (gap ~1e-6..1e-5, zero violation) but plain consensus ADMM needs 30-400+ iterations and does **not** converge within 400 for many blocks (n=400, 30+ blocks); monolithic interior-point is milliseconds up to n=400. **No speed-up is demonstrated**; the claim must stay "exact decomposition + convergence is rho-sensitive". |
| Drop-in data layer | `datafiles.py`, `check_data.py`, templates in `data/templates/`, datasets.md Part A | 68 unit tests | Found and fixed a pandas timestamp-unit bug (np.interp on mixed ns/us int64) in `lmp_csv_price` and the load loader that silently returned a constant price; raw OASIS format (LMP_TYPE/MARKET_RUN_ID) now handled; `--lmp-analog` replay added. |
| Reference baseline at JPL | `sample.py`, `report_table.py`, `headroom_multiday.py` | JPL stress, 3-6 day pilot | **At JPL-stress `safe_uncontrolled` (873/day) beats the best violation-free EDF cap (1140/day)**: tight caps strand energy and pay the unmet penalty. The reference is now the better of {tuned cap, safe_uncontrolled} chosen on train days; headroom vs that reference is ~14 % (pilot; Caltech-stress was 5.2 %). Pilot only (3 train days): re-run with `--days 40+`. |
| Baseline fairness fix | `trainer.tune_edf_cap` | JPL stress | An unwrapped EDF cap that violates network limits looked cheap (and made the "oracle" look worse than it). The tuned EDF is now chosen among violation-free caps. At JPL stress the oracle LP can still exceed a heuristic's env cost because the LP optimises its own model (served fraction 68-84 %): the oracle is a bound for the LP model, not for the environment cost. |

## 12. Open questions for the team

1. Target venue and deadline? (decides how much theory vs. experiments.)
2. Is a stressed scenario acceptable as the *main* benchmark, with real-unstressed as a sanity check?
3. GPU budget (hours) → number of seeds/days we can afford.
4. Who owns the theory section?
5. Do we keep the federated component in this paper or split it into a second one?

### 12.1 Added after the multi-day run (priority order)
1. **Late-training degradation (S10)** is the real blocker: best checkpoint is fine, last is 20-40 % worse in every setup. Test in order: (a) constant low noise 0.1, (b) lr_actor 1e-4 + policy_delay 4, (c) larger replay / no late-greedy domination, (d) critic target clipping / Polyak 0.001, (e) evaluate Q-error vs Monte-Carlo return to see critic drift.
2. **Make the actor residual on the EDF-cap heuristic** (learn a correction around the best tuned heuristic, then QP). This guarantees "no worse than heuristic" at init and is the natural A*-style contribution (safe residual RL); needs an ablation vs. plain SafeTD3.
3. Longer/ multi-seed (>=500 episodes, 5 seeds, >=40 train days) on GPU; paired comparison on identical test days.
4. Honest paper framing if (1)-(3) do not beat tuned EDF: "certified-safe learning with a feasibility certificate and calibrated margins" (safety/conformal contribution) rather than a performance claim.
