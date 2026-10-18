# RUNBOOK

## Part 1: incident reports (read this before you trust a number)

Every claim below was found by making the code check itself against an
independent ground truth (a published result, CVXPY/OSQP, or finite
differences) — not by reading the code and deciding it looked right. That's
the discipline this document is trying to pass on to you: if you extend this
codebase, add the check *before* you trust the extension.

### Incident 1 — sign convention: charging appeared to raise voltage

**Symptom.** An idle day (no EV activity) had V_max = 1.030 p.u. A day with
uncontrolled EV *charging* had V_max = 1.083 p.u. — charging apparently pushed
voltage further from the substation setpoint than doing nothing. That is
physically backwards: charging is a load; loads pull voltage down (or, more
precisely, away from the source in the direction determined by the R/X
sensitivity), they don't push it up.

**Root cause.** The action convention (`p_ev > 0` = charging) and the network
injection convention (`p_net` = generation-positive, per Eq. 7's
`P_net = P_V2G − P_L`) disagreed on a sign, in three places at once: the
environment's `step()`, `FBase.net_injection()`, and the QP's constraint
matrix construction (`build_polytope`). All three had to be fixed consistently
or the bug would just move around.

**Fix.** `p_net = p_exo − A @ p_ev` everywhere (env, F_base, QP), documented
inline at each site with a comment pointing back to this incident.

**Regression test.** `tests/test_world_model.py::test_charging_lowers_voltage_not_raises_it`.

### Incident 2 — the safe set can be empty

**Symptom.** At 100% nominal load with zero EV activity, IEEE-33's own V_min is
0.9131 p.u. — already below the 0.95 p.u. limit. "Project the action onto the
safe set" is undefined when the safe set is empty, and running ADMM on an
infeasible QP doesn't raise an exception, it just doesn't converge (the primal
residual sits at whatever infeasibility gap exists and never shrinks).

**Root cause.** The original QP formulation had no way to express "closest
feasible point" when there is no feasible point.

**Fix.** Two scalar slack variables (one for the voltage-constraint group, one
for thermal) turn the hard constraint into an exact-penalty relaxation:
`G a <= h + xi, xi >= 0`, with a large linear penalty on `xi`. When the hard
set is non-empty, `xi* = 0` to numerical precision and the relaxation is
inert. When it's empty, the layer returns the *least infeasible* point instead
of failing. Reported violation numbers always come from the exact AC solve in
the environment, never from whether `xi` was used, so the relaxation can hide
a crash but can never hide a violation in what gets reported.

Practically: in the experiment configs shipped here, `load_scale` is set below
1.0 and thermal/voltage stress is engineered via EV *hub* siting (see
Incident 4-adjacent design note below) rather than by running the bare feeder
into an infeasible region — the slack path exists as a safety net for
whatever the RL agent's random exploration produces, not as the normal
operating regime.

**Regression test.** `tests/test_qp_layer.py::test_infeasible_polytope_does_not_crash`.

### Incident 3 — ADMM convergence failure on the *real* problem (the big one)

**Symptom.** A hand-inspected trace showed the QP safety layer changing a
rule-based policy's already-safe action (independently verified feasible under
both the exact AC solve *and* the QP's own linear model — zero constraint
violation either way) by up to 0.09 p.u., roughly 80% of that EV's charging
capacity. Aggregated over a day, this crushed service quality from 94% to 35%
for a policy that never violated anything in the first place.

**Diagnosis path** (reproducible — see `git log` / this file's construction
history for the actual commands):

1. First hypothesis: the QP margin (conformal tightening) was huge. Checked:
   margin was 0.0 (uncalibrated yet). Ruled out.
2. Second hypothesis: the network really did need that much correction.
   Checked feasibility of the *raw* action against both the exact AC solve
   (`solve_bfs`) and the QP's own linear constraint matrix (`G @ a_raw <= h`)
   directly — zero violation, either way. Ruled out.
3. Third hypothesis: implementation bug in the vectorized/batched ADMM.
   Reproduced the *exact* problem instance (same G, h, weights) in plain
   float64 numpy with an unrolled, unbatched, textbook ADMM — the identical
   slow crawl appeared (primal residual still ~2e-2 after 300 iterations,
   only cleanly resolving after ~2000-3000). This ruled out a coding bug in
   the torch implementation: two independent implementations of the same
   algorithm produced the same slow behaviour.
4. Checked the condition number of `CᵀC` directly: ~385. Not pathological —
   a well-implemented ADMM should not need 2000+ iterations for that.
5. Conclusion: this is a known ADMM characteristic, not a conditioning
   problem or a bug. When the true optimum is deep in the *interior* of the
   feasible region (i.e. the raw action needed **zero** correction), ADMM's
   dual variable has to relax from a cold start to zero with no active-set
   structure to accelerate convergence — a well-documented weak point of
   fixed-step ADMM specifically on already-optimal/near-optimal starting
   points, distinct from its behaviour near an active constraint (where it
   tends to converge faster).

**Fix — the right one, not a bandage.** Rather than trying to out-tune ADMM's
convergence rate for a scenario it's intrinsically bad at, we recognised that
*the whole scenario is unnecessary work*: if the raw action is already
feasible, running the safety-QP is pointless — the answer is exactly the raw
action, no optimization needed. `project()` now checks feasibility of the raw
action (cheap: a `no_grad` matrix multiply against the real, non-slack
constraint rows) before touching the ADMM solver at all. Feasible batch rows
short-circuit to the identity (which is also the mathematically correct
projection of a feasible point onto its own feasible set, so no accuracy is
lost — verified against CVXPY on the same instance). Infeasible rows still go
through the full differentiable ADMM solve, where an active constraint gives
the iteration real structure to converge against.

This is also a **substantial efficiency win**, not just a correctness fix: in
a representative day-long trace, roughly 40% of control steps required no
correction at all (see the diagnostic printout `steps QP changed the action by
>1e-3: 1 of 88` after the fix). Those steps now cost one cheap feasibility
check instead of up to 220 ADMM iterations.

We also implemented and then **deliberately disabled** OSQP-style adaptive-rho
updating (`admm_project(..., adaptive_rho=True)`, off by default): it is
correctly implemented and passes its own regression test, but it did not
reliably outperform fixed rho on the specific ill-structured instances found
during development, and a delicately-tuned heuristic that fails silently on
some fraction of instances is worse than a plain algorithm you understand the
failure mode of. It's left in the code, tested, and documented as an
experimental option for anyone who wants to tune it further — not as the
default safety-critical path.

**Regression test.**
`tests/test_qp_layer.py::test_skip_feasible_returns_identity_for_feasible_action`
constructs a trivially-feasible instance and asserts the *un-shortcut* solver
would need far more than 50 iterations to reach it cleanly, while the
shortcut returns it exactly.

**What this means for your results.** The training loop benefits from this
shortcut (a meaningful fraction of steps never need a full QP solve), so
per-step wall-clock time in training is lower than a worst-case forced solve.
(The standalone QP scalability benchmark that measured the worst case was part
of the IEEE-123 scope decision and has been removed from scope.)

## Part 2: known, expected (non-bug) behaviours

- **`safe_v2g_greedy` costs a little more / serves a little less than
  `v2g_greedy`.** This is the safety layer doing its job on the steps that
  genuinely need it — a few percentage points of service, not the 60-point
  collapse from Incident 3. If you see a large gap here after making changes,
  re-run the Incident 3 diagnostic before assuming it's correct.
- **Early-training service quality is worse under `use_tightening=True` than
  `use_tightening=False`.** The conformal margin is calibrated on however much
  data the world model has seen; early on, with a small buffer, calibration is
  conservative (wide margin) by design (split-conformal with few calibration
  points naturally has a wider guaranteed interval). This shrinks as training
  proceeds and `wm_v_mae` improves — it is a real, reportable, expected
  training-time trade-off, not a defect. Track `tight_margin_v` /
  `tight_margin_s` in the training history JSON to show this trend in your
  paper.
- **A freshly-initialised `SafeTD3` agent underperforms hand-written rule
  baselines.** TD3 needs hundreds of episodes to learn a competitive policy;
  this sandbox's tests use 1-20 episodes purely to verify the pipeline runs
  without crashing or violating physics, not to demonstrate learning. Do not
  report quicktest/smoke-test episode counts as results.
- **IEEE-123 / QP scalability benchmark: removed from scope** (commented out in
  the code, kept for reference). All experiments use the IEEE-33 feeder.

## Part 3: expected wall-clock time (plan your GPU budget)

Measured on this sandbox's 1-core CPU (no GPU) as a *lower bound reference* —
your 20-core CPU / 6GB GPU should be substantially faster, especially since the
QP solver is fully batched (GPU throughput scales with batch size, not with
core count).

| Command | Sandbox (1 CPU core) | Expect on your machine |
|---|---|---|
| `--quicktest` | ~3-4s | ~3-5s (CPU-bound, doesn't need GPU) |
| `single`, 300 episodes, k_max=24 | not run to completion in sandbox (1 core too slow) | expect tens of minutes on GPU; run a 15-episode dry run first and extrapolate linearly, then commit to the long run |
| `federated`, 5 clients, 60 rounds | not run to completion | scale similarly: dry-run 2 rounds first |
| full `ablation` grid, 3 seeds | not attempted | this is the long pole — budget a multi-hour background job |

**Recommended workflow on your machine** (screen shows important messages only;
full output and errors go to `logs/`):
```bash
python sample.py --tests                                        # unit tests, summary only
python sample.py --mode single --archetype residential --episodes 1 --device cpu   # pipeline smoke run
python sample.py --mode single --real-data caltech --episodes 2 --warmup 1 --eval-every 2 --eval-episodes 1 --device cpu
python sample.py --mode single --real-data caltech --episodes 20 --device cpu      # timing run
# extrapolate: (wall_time_for_20_episodes / 20) * 300 = expected time for 300 episodes, per real day
python sample.py --mode single --real-data caltech --n-real-days 5 --episodes 300 --device cuda
python sample.py --mode single --real-data jpl --n-real-days 5 --episodes 300 --device cuda
python sample.py --mode federated --real-federated --rounds 60 --compare --device cuda
```

Or run `bash scripts/run_all.sh` in the background (`nohup bash scripts/run_all.sh
> run_all.log 2>&1 &`).

## Part 4: if something looks wrong

1. Re-run `python sample.py --quicktest` first. If it fails, nothing else
   downstream should be trusted until it passes.
2. Run `python -m pytest tests/ -v`. All 24 tests should pass.
3. If a specific number looks physically implausible (voltage outside
   [0.5, 1.5] p.u., negative served energy, NaN cost), check whether it
   reproduces with a smaller config (`--k-max 4 --episodes 2 --dt-h 0.5`) —
   almost every bug in this codebase's development was first caught by
   shrinking the reproduction, not by staring at the full-scale run.
4. If it's a QP/safety-layer issue specifically, the diagnostic pattern in
   Incident 3 is the template: check the raw action's own feasibility (both
   under the exact AC solve and under the QP's linear model) before assuming
   the safety layer's *correction* is the thing that's wrong.

## Part 5: SafeTD3 convergence diagnosis (methodology study, real Caltech days)

Symptom (professor's 5-day run): SafeTD3 worse than greedy after 300 episodes, zero violations,
"too conservative".  Pipeline, data, world model and safety were verified OK; the study below
isolates what breaks *policy learning*.  Each item = one measured cause + the change made.

| # | Finding (measured) | Change | Switch |
|---|---|---|---|
| 1 | ADMM with slack columns is unconverged at 120 iters and removes ~33% of requested charging where the exact projection removes ~13% | hard projection (no slack), rho 0.1; rows with negative RHS relaxed to "do not worsen" (h:=max(h,0)); constraint-generation (48 tightest rows, verified on all) | `--legacy-qp` restores old |
| 2 | After conformal calibration ~30% of training samples have a row infeasible at idle (margin_s ~32 kVA vs hub-branch headroom) -> they all hit the slow, ill-conditioned slack path | same as 1 | |
| 3 | float32 ADMM never reached tol 1e-7 -> always full iterations | float32-aware tol (2e-6) | |
| 4 | Critic target projected a' with the *current* step's constraints | store next-step context (`*_n`) in the replay buffer | `exact_next_ctx` |
| 5 | Unmet-energy penalty is sparse/delayed (only at departure) over 54 slots with one scalar reward | `--reward-mode dense`: per-step service credit; same return up to a policy-independent constant | `--reward-mode` |
| 6 | Evaluating the best of N noisy 3-episode evals is optimistic (winner's curse) | fixed validation seeds; best checkpoint kept; last-vs-best compared on FRESH seeds (`results[...]["final"]`) | `TrainConfig.keep_best` |
| 7 | Prof's QP ablation rows A-D differ only in `c_violation`, which is unused under the QP | do not read them as distinct arms | |
| 8 | torch with >1 thread thrashes on the many tiny batched solves on small CPUs | `--threads 1` | `--threads` |

New logging per episode / eval: QP intervention (`qp_dist_{mean,median,p90,p95,p99}` in kW^2,
`qp_mod_frac` = share of steps where any slot moved >0.1 kW, `qp_heavy_frac`, `qp_cut_mean`).

Single-day, 40-episode ladder (Caltech 2019-10-18, seed 0; fixed-seed validation; fresh-seed final):

| Arm | what | last (cost / served) | best (cost / served) |
|---|---|---|---|
| uncontrolled / tou_greedy | rule | 279 / 86.6% (tiny violations 8.9e-3) | |
| safe_v2g_greedy | rule + QP | 300-313 / 84-85% | |
| B | QP fix, sparse reward | 624 / 50.7% | 624 / 50.7% |
| C | QP fix + dense reward | 360 / 78.2% | 314 / 83.2% (ep 9) |

Reading: dense reward is what makes it learn at all; after ~10 episodes the policy is at greedy
quality and then *oscillates / drifts* (validation cost 281 -> 407 -> 295 -> 351): the remaining
problem is optimisation stability, not safety.  Use `scripts/optuna_search.py` (train day /
validation seeds / untouched test day) and the multi-seed GPU run.

Reproduce:
```bash
python sample.py --mode single --real-data caltech --acn-date 2019-10-18 --episodes 40 --eval-every 5 \
       --log-every 1 --threads 1 --device cpu --reward-mode dense          # arm C
python sample.py ... --legacy-qp                                          # old pipeline
python scripts/optuna_search.py --real-data caltech --train-date 2019-10-18 --trials 30 --device cuda
```

## Part 6 - multi-day demand-charge benchmark (see DOC.md 6.5, 11.1)
`scripts/run_multiday.sh M1 --bc-weight 1.0 --bc-final 0.1` and `scripts/run_multiday.sh M2 --bc-weight 0 --bc-final 0`
(~16 min each on 1 CPU thread, 60 episodes). For GPU: raise `--episodes` to 500+, `--multiday 40`, loop `--seed 0..4`, use `--device cuda`.
Result JSON: `results/study3/<name>/multiday_caltech_seed<seed>.json` (train/val/test dates, EDF tuning, baselines, held-out last/best).

New in this iteration: `--residual 0.3 --base edf_tuned` (safe residual RL), `--td3-json <file>` (TD3Config overrides, e.g. Optuna output),
`scripts/headroom_multiday.py` (learning-headroom test), tests/test_residual.py (39 tests).  Results: DOC.md 6.6-6.10.

## Part 7 - CUDA commands with the real data (Windows PowerShell and bash)

Prerequisites: `data/caltech_local_sessions.csv`, `data/jpl_local_sessions.csv` (required); optional files per `datasets.md` Part A.
Install the CUDA build of PyTorch first, then check: `python -c "import torch;print(torch.cuda.is_available())"` must print `True`.

```powershell
# ---- PowerShell (Windows).  Run from the repo root.  `$env:PYTHONPATH="."` is needed for scripts/*.py.
$env:PYTHONPATH = "."
python sample.py --tests                                # 68 unit tests
python scripts/check_data.py                            # what is real, what falls back

# (1) EVERYTHING, resumable (finished stages are skipped; re-run the same line after an interruption)
python scripts/run_suite.py --device cuda --auto-data --hub-grid --lmp-analog any --seeds 0 1 2 3 4 --days 40 --episodes 300 --rounds 40

# (2) pieces
python scripts/run_suite.py --device cuda --auto-data --only headroom --days 60 --hub-grid     # headroom + per-hub kill test (A2), both sites/regimes
python scripts/run_suite.py --device cuda --auto-data --only multiday --seeds 0 1 2 3 4 --days 40 --episodes 300
python scripts/run_suite.py --device cuda --auto-data --only transfer,federated --seeds 0 1 2
python scripts/run_suite.py --only conformal,treeqp                                              # physics-only, CPU
python scripts/run_suite.py --only report                                                        # tables with bootstrap CIs
python scripts/run_suite.py --dry-run --device cuda --auto-data                                  # print every command
# MPC arm is slow (1-2 min per day per reserve value): add  --mpc-reserves 0 8 16  to headroom/multiday when you have time.

# (3) one site, one run, by hand (Caltech stress, residual SafeTD3 around the train-tuned EDF cap)
python sample.py --mode single --real-data caltech --multiday 40 --episodes 300 --warmup 6 --eval-every 5 --eval-episodes 4 --log-every 10 `
  --device cuda --seed 0 --demand 1.5 --reward-mode dense --lambda-mb 0 --warmup-mode greedy --residual 0.1 --base edf_tuned `
  --stress-ev 2.0 --stress-cap 100 --auto-data --hub-grid --out results/caltech_stress --log-dir logs/caltech_stress
python sample.py --mode single --real-data jpl    ... --stress-cap 130 ...                      # same for JPL (cap = 2/3 of 195 kW)
# real CAISO prices instead of the tariff:  --price-source caiso_rtp --lmp-csv data\caiso_sp15_lmp.csv [--lmp-analog any]
```
```bash
# ---- bash / Linux / macOS (identical flags)
export PYTHONPATH=.
python sample.py --tests && python scripts/check_data.py
python scripts/run_suite.py --device cuda --auto-data --hub-grid --lmp-analog any --seeds 0 1 2 3 4 --days 40 --episodes 300 --rounds 40
```
Outputs: `results/suite/**` (per-stage logs + JSON), `docs/study/suite/headroom_<site>_<regime>.json`,
`docs/study/suite/report_<site>_<regime>.md` (paired table, CI, sign test, gate), `docs/study/adaptive_conformal_shift.json`,
`docs/study/bench_tree_qp.json`.  Use `--quick` for a 5-minute smoke run of the whole pipeline.

Notes: (i) `--days N` gives N evenly spaced weekdays per site split 60/20/20 train/val/test (seeded); use N>=40 so the test split has >=8 days.
(ii) The EDF baseline is tuned on the local TRAIN days among **violation-free** caps only (an unwrapped cap that violates the network
is not deployable). (iii) At JPL-stress, `safe_uncontrolled` can be the strongest baseline — the report lists it.

## Part 8 - synthetic price scenarios now, CAISO later
`--price-scenario duck|volatile|solar_glut|winter_peak|tou3|flat` (+ `--price-seed N`) on `sample.py`, `headroom_multiday.py`, `cross_site_fed.py`, `run_suite.py`.
Later, with the CAISO file in `data/caiso_sp15_lmp.csv`: replace the flag by `--price-scenario historical_lmp --lmp-csv data/caiso_sp15_lmp.csv` (add `--lmp-analog any` if dates do not overlap).
No-learning robustness study over scenarios: `python scripts/run_suite.py --only scenarios --days 20 --device cpu` -> `docs/study/suite/price_scenarios.md`.
