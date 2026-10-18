# Physics-Informed Federated Safe RL for V2G Coordination

A working implementation of the Differentiable Residual-Dyna architecture
described in `dyna_v2_final.pdf`: an analytical LinDistFlow base model
(`F_base`) coupled to a learned residual (`N_theta`), a differentiable QP
hard-safety projection layer (implicit/OptNet-style gradients), willingness-aware
(VPW) bidding, and personalized federated learning (FedProx) across
Non-IID charging sites.

**Start here:** `python sample.py --tests` (unit tests) and
`python sample.py --mode single --archetype residential --episodes 1 --device cpu`
(a one-episode smoke run of the whole pipeline). `python sample.py --quicktest`
is a faster correctness pass over every stage (physics → env → world model →
safety layer → agent → federated round → VPW calibration).

**Output:** the screen shows only important messages (progress, results,
errors). Everything else — full output, warnings, tracebacks and the Intel MKL
lines — is written to `logs/<mode>_<timestamp>.log`. Add `--verbose` to see
everything on screen. **Scope:** IEEE-33 only; the IEEE-123 work and the QP
scalability benchmark were removed (commented out).

## Why you should read RUNBOOK.md before trusting any number

This codebase was built, then stress-tested against its own physics, and three
real bugs were found and fixed in that process — not typos, but things that
would have silently produced a plausible-looking but wrong paper:

1. **A sign-convention bug** where charging appeared to *raise* voltage.
2. **An empty-feasible-set problem**: the IEEE-33 feeder is already below the
   0.95 p.u. limit at 100% nominal load, so "project onto the safe set" is
   undefined until you decide what that means when the safe set is empty.
3. **A non-obvious ADMM convergence failure**: the safety-layer solver looked
   fine on a random synthetic test (converges in ~10 iterations) but silently
   failed to converge on the *real* problem structure (perturbing an
   already-safe action by up to 30%), for a subtle and specific reason.

`RUNBOOK.md` documents all three as incident reports: symptom → diagnosis →
root cause → fix → regression test. This matters for your submission: if a
reviewer or your professor pokes at the safety claims, you want to be the one
who already found and fixed the sharp edges, with a paper trail.

## Repository layout

```
pifsrl/
  physics/       feeder topology (IEEE-33 exact), exact BFS
                 AC power flow (ground truth), LinDistFlow (F_base's physics)
  env/           V2GEnv (gym-style), EV session generation (ACN-calibrated),
                 exogenous profiles (load/PV/price), VPW willingness model
  models/        F_base (analytical, exact Jacobian), N_theta (residual net),
                 WorldModel (training/eval/calibration)
  safety/        differentiable QP projection (batched ADMM + implicit grads),
                 conformal residual-tightening, slack relaxation for
                 infeasible instances
  agents/        SafeTD3 (actor-critic + QP + differentiable-model term),
                 replay buffer, rule-based baselines, training loop
  federated/     FedProx / FedAvg with shared-trunk / personal-head split
  experiments/   ablation runner (synthetic only), plotting/table generation
  runlog.py      logging layer (screen = important messages, file = everything)
tests/           unit tests: physics vs. published values, QP vs. CVXPY,
                 gradients vs. finite differences, federated split correctness,
                 real-data loader (skipped automatically if data/ CSVs are absent)
sample.py        main CLI entry point (--quicktest, --tests, --mode single/federated/ablation)
scripts/         shell scripts for the full multi-seed experiment grid
configs/         example YAML-equivalent Python configs for each experiment type
```

## Real data (read this first if you have ACN-Data access)

This codebase now runs on **real** EV charging sessions (Caltech & JPL, from ACN-Data),
real grid topology (IEEE-33, Baran & Wu 1989), and a real, currently-published SCE tariff
— not synthetic approximations. **See [`datasets.md`](datasets.md) for the full
provenance, citations, and an explicit list of what remains synthetic and why** (this is
the document to read before writing your paper's data section).

```bash
# Place your own ACN-Data CSVs (you already have access) at:
#   data/caltech_local_sessions.csv
#   data/jpl_local_sessions.csv

python sample.py --mode single --real-data caltech --episodes 300 --device cuda
```

If you don't have the CSVs, everything below still works with the synthetic (but
ACN-calibrated) session generator — just omit `--real-data`.

## Quickstart

```bash
pip install -r requirements.txt

# 0. Unit tests (screen: one summary line; details in logs/)
python sample.py --tests

# 1. One-episode smoke run of the full pipeline (synthetic archetype)
python sample.py --mode single --archetype residential --episodes 1 --device cpu

# 2. Real ACN-Data (CSVs in data/), short then longer
python sample.py --mode single --real-data caltech --episodes 2 --warmup 1 --eval-every 2 --eval-episodes 1 --device cpu
python sample.py --mode single --real-data caltech --n-real-days 5 --episodes 300 --device cuda
python sample.py --mode single --real-data jpl --n-real-days 5 --episodes 300 --device cuda

# 3. Real federated learning: Caltech + JPL as the two clients
python sample.py --mode federated --real-federated --rounds 2 --local-episodes 1 --device cpu
python sample.py --mode federated --real-federated --rounds 60 --compare --device cuda
```

See RUNBOOK.md for expected durations and `datasets.md` for data provenance.
`--mode ablation` is still synthetic-only (not wired to real data).

## What is real vs. synthetic — read `datasets.md` before writing your paper

- **Real:** EV sessions (ACN-Data Caltech/JPL CSVs), charger ratings and site
  capacity (Lee et al.), IEEE-33 topology (Baran & Wu 1989; validated to <0.2%),
  SCE TOU-D-PRIME tariff, and the willingness proxy signals for the real sites
  (derived from real timestamps).
- **Simulated (standard practice):** the power flow resulting from an action
  that was never taken historically.
- **Synthetic:** background (non-EV) grid load and PV; the office / residential /
  fleet-depot archetypes (no public per-session dataset exists); the willingness
  proxy for those archetypes.
- **Removed from scope:** IEEE-123 bus and the QP scalability benchmark.

## Citation-relevant equations

The DistFlow recursion (Eqs. 3–6 of the proposal) and their solution by
backward/forward sweep are standard and match Nagrath & Kothari's treatment of
radial distribution power flow; cite that alongside Baran & Wu (1989) for the
LinDistFlow linearisation (Eq. 7) if your department expects a textbook
reference in addition to the original papers.

## Update 2026-10: evidence-first pipeline (read RUNBOOK Part 7 and datasets.md Part A)
* Drop-in real data (tariffs, CAISO LMP, feeder load, battery models) with labelled fallbacks: `python scripts/check_data.py`, flag `--auto-data`.
* One resumable driver for both sites and both regimes on CUDA: `python scripts/run_suite.py --device cuda --auto-data` (`--quick` = smoke run).
* Baselines: train-tuned violation-free EDF cap, per-hub caps, forecast-free MPC, safe-uncontrolled, oracle LP; statistics: paired bootstrap, sign test, deployment gate (`pifsrl/evalstats.py`).
* Safety: row-wise conformal margins + adaptive conformal (`pifsrl/safety/adaptive.py`); tree-decomposition benchmark (`scripts/bench_tree_qp.py`).
* 68 unit tests: `python sample.py --tests`. Honest status of every claim: DOC.md 6.5-6.13.

## Update 2026-10-15: price module + findings + reproducibility
* Prices are a pluggable module (`pifsrl/env/pricing.py`): synthetic scenarios `duck|volatile|solar_glut|winter_peak|tou3|flat` now, `historical_lmp|historical_tariff` later — select with `--price-scenario`; no agent/baseline/safety code changes.
* `docs/FINDINGS.md` — every publishable finding with strength and caveats; `REPRODUCE.md` — claim → command → expected number; `docs/study/MANIFEST.json` — SHA-256 of data, code and evidence (`python scripts/make_manifest.py --verify`); `requirements-lock.txt` — exact versions.

* `REPRODUCE.md` — the complete ordered command list (setup, data placement, every experiment, tables, manifest verification).
