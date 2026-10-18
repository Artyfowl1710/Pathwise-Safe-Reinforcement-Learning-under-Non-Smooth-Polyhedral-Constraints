# REPRODUCE.md — how a third party checks every number

Principle: every claim in `docs/FINDINGS.md` has (i) one command, (ii) its expected output with a tolerance, (iii) the evidence file
we produced it with, (iv) a SHA-256 of that file in `docs/study/MANIFEST.json`. Nothing depends on a hidden state: synthetic prices
depend only on (scenario, date, price_seed); training depends only on (seed, data, package versions).

## 0. Environment (tested: Python 3.10–3.13, CPU & CUDA)
```bash
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e .                                        # installs package locally in editable mode
pip install -r requirements-lock.txt                    # or pip install -r requirements.txt for flexible versions
python sample.py --tests                                # unit tests incl. tests/test_reproducibility.py (bit-exact CPU re-runs)
python scripts/make_manifest.py --verify                # code/evidence hashes vs docs/study/MANIFEST.json
```
Determinism contract: on **CPU with the pinned versions, same seed => bit-identical results** (`tests/test_reproducibility.py` checks the
trained weights). On **CUDA** reductions are not bit-deterministic: expect the same *conclusions* and numbers within seed noise, not
identical digits. Report mean ± CI over seeds and days (`scripts/report_table.py`), never one run.

## 1. Data
* EV sessions: ACN-Data (Lee, Li, Low, e-Energy 2019), Caltech + JPL exports (`data/*_local_sessions.csv`; licence forbids redistribution —
  obtain them from ev.caltech.edu yourself; the manifest holds their SHA-256 so you can confirm you have the same files).
* Feeder: IEEE 33-bus (Baran & Wu 1989), coded in `pifsrl/physics/feeder.py`; branch ratings derived as 1.6x nominal flow (stated assumption).
* Prices: **synthetic scenarios by default** (`pifsrl/env/pricing.py`: duck, volatile, solar_glut, winter_peak, tou3, flat), selected with
  `--price-scenario`. Historical CAISO / tariff CSVs plug into the same interface (`--price-scenario historical_lmp|historical_tariff`)
  without touching any agent (verified by `tests/test_pricing.py`). Format: `datasets.md` Part A.

## 2. Claim -> command -> expected result
| # | claim | command | time | expected (tolerance) | evidence |
|---|---|---|---|---|---|
| C0 | one global conformal margin is unusable on a feeder with heterogeneous branch ratings; row-wise margins give the same per-branch coverage with a tiny margin | `python scripts/conformal_rowwise_vs_global.py` | <1 min | global margin 453 kVA (>10% of the limit on 30/32 branches, 3.9x the smallest limit); row-wise max 7.1% of limit; per-branch test coverage >=0.984 for both (alpha=0.01) | `docs/study/conformal_rowwise_vs_global.json` |
| C1 | fixed split-conformal voltage margin loses coverage under a load shift; adaptive conformal keeps it | `python scripts/adaptive_conformal_shift.py` | ~6 min CPU | at load x1.25: fixed 0.081±0.01, ACI 0.950±0.005 (target 0.95); no-shift fixed 0.950±0.01 | `docs/study/adaptive_conformal_shift.json` |
| C1' | same, miniature (deterministic) | `python -m pytest tests/test_reproducibility.py -q` | 20 s | passes | — |
| C2 | the lemma: cutting a radial feeder couples blocks through 3 scalars | `python scripts/poc_tree_decomposition.py` | <1 min | IEEE-33 27/27, random-123 80/80 cuts | stdout |
| C3 | exact decomposed projection vs monolithic | `python scripts/bench_tree_qp.py` | ~15 min | gap<=1e-4 when converged; no speed-up (see FINDINGS F6) | `docs/study/bench_tree_qp.json` |
| C4 | learning headroom after tuning the heuristic (Caltech, stress, demand 1.5, 20 days) | `python scripts/headroom_multiday.py --site caltech --days 20 --demand 1.5 --stress` | ~40 min | uncontrolled 471.4, best cap 400.5, oracle 379.5 (H=5.2%) | `docs/study/headroom_multiday.json` |
| C5 | residual SafeTD3 vs tuned EDF, held-out days | `python scripts/run_suite.py --only multiday --sites caltech --regimes stress --seeds 0 1 2` | hours | learner +2..9% worse than tuned EDF, 0 violations | `docs/study/R*.json` |
| C6 | cross-site transfer / federated | `python scripts/run_suite.py --only transfer,federated` | hours | Caltech cap +35..45% at JPL; FedProx ≈ local ±0.5% | `docs/study/transfer_*.json`, `federated_*.json` |
| C7 | price-scenario robustness of the headroom | `python scripts/run_suite.py --only scenarios --days 20` | ~1 h/site | `docs/study/suite/price_scenarios.md` | same |

## 3. Everything at once
`python scripts/run_suite.py --device cpu|cuda --price-scenario duck --seeds 0 1 2` (resumable; `--quick` for a smoke run; `--dry-run` lists commands).
After a run: `python scripts/make_manifest.py` regenerates hashes; commit `docs/study/MANIFEST.json` with the paper.

## 4. What a reviewer can break (we want you to)
* Change `--price-scenario` / `--price-seed`: conclusions about the safety layer must not change; cost conclusions may (that is C7).
* Change `--l-test`, `--gamma`, `--alpha`, `--two-sided` in C1.
* Run C4 with `--site jpl`: the reference baseline there is `safe_uncontrolled`, not a capped EDF.
