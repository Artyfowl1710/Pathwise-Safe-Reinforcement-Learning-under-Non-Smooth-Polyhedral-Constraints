# PAPER_COMMANDS.md — every command, in order, to regenerate everything used in the paper

Run from the repository root. PowerShell: replace `export PYTHONPATH=.` by `$env:PYTHONPATH="."` and `\` line-continuations by backticks.
All stages are resumable (a finished output file is skipped; add `--force` to recompute). Times are rough (CPU 2-core / RTX 3050).

## STEP 0 — environment (once)
```
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt                  # exact versions behind the reported numbers (install the CUDA build of torch for GPU)
export PYTHONPATH=.
python -c "import torch;print(torch.cuda.is_available())"     # True needed only for --device cuda
```

## STEP 1 — where the data goes
| file (put in `data/`) | needed? | what happens if absent |
|---|---|---|
| `caltech_local_sessions.csv`, `jpl_local_sessions.csv` (ACN-Data exports, ev.caltech.edu) | **required** | nothing runs |
| `tariff_caltech.csv`, `tariff_jpl.csv` | optional (real retail tariff + demand charge) | SCE-2026 stand-in, labelled |
| `caiso_sp15_lmp.csv` (raw OASIS format OK) | optional (real wholesale price) | synthetic scenario, labelled |
| `feeder_load.csv` | optional | load generator |
| `battery_models.csv` | optional (V2G what-if only) | no battery model |
Templates with the exact column headers: `data/templates/`. Full spec: `datasets.md` Part A.
```
python scripts/check_data.py            # prints OK / MISSING per file and the fallback in force
```

## STEP 2 — integrity (≈1 min)
```
python sample.py --tests                       # 68 unit tests, incl. bit-exact CPU re-runs and the miniature conformal claim
python sample.py --quicktest
python scripts/make_manifest.py --verify       # hashes of code + evidence vs docs/study/MANIFEST.json (your data hashes will differ only if your CSVs differ)
```

## STEP 3 — safety-layer results (physics only; no data, no GPU, no prices) — Findings F1–F3
```
python scripts/conformal_rowwise_vs_global.py                 # F2   (<1 min)  -> docs/study/conformal_rowwise_vs_global.json
python scripts/adaptive_conformal_shift.py                    # F1   (~6 min)  -> docs/study/adaptive_conformal_shift.json
python scripts/adaptive_conformal_shift.py --two-sided --out docs/study/adaptive_conformal_shift_twosided.json
python scripts/poc_tree_decomposition.py                      # F3a  (<1 min)  lemma: 27/27 and 80/80
python scripts/bench_tree_qp.py                               # F3b  (~15 min) -> docs/study/bench_tree_qp.json
```

## STEP 4 — MODE A: synthetic prices (what to run while CAISO/tariffs are not available)
Real sessions + IEEE-33 + synthetic price scenario. Pick ONE headline scenario (`duck`) and show robustness over the rest.
```
# 4a. no-learning headroom, both sites, both regimes, 40 days, per-hub test (A2)          (~2-3 h CPU)
python scripts/run_suite.py --only headroom --device cpu --price-scenario duck --days 40 --hub-grid
# 4b. price-scenario robustness (duck, volatile, solar_glut, winter_peak, flat)           (~6-8 h CPU)
python scripts/run_suite.py --only scenarios --device cpu --days 40
# 4c. residual SafeTD3 vs tuned baselines, 5 seeds, held-out days                         (GPU: ~10-20 h total)
python scripts/run_suite.py --only multiday --device cuda --price-scenario duck --days 40 --episodes 300 --seeds 0 1 2 3 4
# 4d. cross-site transfer + federated, both regimes, 3 seeds                              (GPU: ~6-10 h)
python scripts/run_suite.py --only transfer,federated --device cuda --price-scenario duck --episodes 300 --rounds 40 --seeds 0 1 2
# 4e. optional slow baseline: forecast-free MPC with reserve tuning (1-2 min per day per arm)
python scripts/run_suite.py --only headroom,multiday --device cuda --price-scenario duck --days 40 --mpc-reserves 0 8 16
```
(or all of 4a–4d in one resumable command:
`python scripts/run_suite.py --device cuda --price-scenario duck --days 40 --episodes 300 --rounds 40 --seeds 0 1 2 3 4 --hub-grid`)

## STEP 5 — MODE B: real prices (after the CSVs are in `data/`)
Same commands; only the price flag changes. Nothing in agents/baselines/safety changes.
```
python scripts/check_data.py
# real tariffs (demand charge from the tariff too):
python scripts/run_suite.py --device cuda --auto-data --demand-from-tariff --days 40 --episodes 300 --rounds 40 --seeds 0 1 2 3 4 --out results/suite_tariff
# real CAISO wholesale prices (exact dates if the file covers 2018-2021; otherwise labelled replay):
python scripts/run_suite.py --device cuda --auto-data --price-scenario historical_lmp --days 40 --episodes 300 --seeds 0 1 2 3 4 --out results/suite_caiso
python scripts/run_suite.py --device cuda --auto-data --price-scenario historical_lmp --lmp-analog any --days 40 --episodes 300 --seeds 0 1 2 3 4 --out results/suite_caiso_replay
```
Report each of A / tariff / CAISO / replay as a separate labelled column; never mix them in one table cell.
V2G what-if only: add `--battery-default` (assumed {40,60,75} kWh) or drop `data/battery_models.csv`.

## STEP 6 — tables, intervals, hashes
```
python scripts/run_suite.py --only report --days 40                  # docs/study/suite/report_<site>_<regime>.md  (paired bootstrap CI, sign test, gate, headroom)
python scripts/price_scenario_summary.py                              # docs/study/suite/price_scenarios.md
python scripts/make_manifest.py                                       # re-hash everything; commit docs/study/MANIFEST.json with the paper
```

## Map: paper item -> step
F1 adaptive conformal: 3 · F2 row-wise vs global margin: 3 · F3 tree lemma / decomposition: 3 · F4 headroom: 4a (+5) · F5 residual RL vs tuned EDF: 4c · F6 transfer/federated: 4d · F7 price-scenario invariance: 4b.


## Step 7 — differentiable twin + paper (added 2026-10-16)
```
PYTHONPATH=. python -m pytest -q tests/test_diffenv.py
PYTHONPATH=. python scripts/diff_env_poc.py --iters 200 --plan-iters 300      # synthetic days, ~10 min on 2 CPU cores -> docs/study/diff_env_poc.json
# REAL-DATA headroom with the FIXED script (must be rerun for JPL):
PYTHONPATH=. python scripts/headroom_multiday.py --site jpl --stress --days 20 --demand 1.5 --auto-data --out docs/study/headroom_jpl_stress_fixed.json
PYTHONPATH=. python scripts/make_paper_figs.py && (cd docs/paper && latexmk -pdf paper.tex)
```

## Step 8 — real tariff + real hourly CAISO prices from free sources (added 2026-10-16)
```
pip install acnportal tzdata openpyxl
# WEP workbook: LBNL "Wholesale Electricity Prices (WEP) tool" -> wep_download_data.xlsx  (CAISO hourly 2012-2025, NOT the SP15 node)
PYTHONPATH=. python scripts/build_real_price_files.py --wep /path/to/wep_download_data.xlsx    # writes data/tariff_{caltech,jpl}.csv, data/caiso_sp15_lmp.csv, data/PRICE_PROVENANCE.txt
PYTHONPATH=. python scripts/check_data.py
# headroom under (a) the real SCE TOU-EV-4 tariff incl. its demand charge (0.51 $/kW-day), (b) real hourly CAISO energy price + demand 1.5 $/kW-day
for s in caltech jpl; do
 PYTHONPATH=. python scripts/headroom_multiday.py --site $s --stress --days 20 --auto-data --demand-from-tariff --out docs/study/headroom_${s}_stress_tariff.json
 PYTHONPATH=. python scripts/headroom_multiday.py --site $s --stress --days 20 --demand 1.5 --auto-data --wep-lmp --out docs/study/headroom_${s}_stress_wep.json
done
# pathwise policy on real days (restoration variants: --restore-mode uniform|sens, --ste-beta 20):
PYTHONPATH=. python scripts/diff_env_poc.py --site caltech --stress 2.0 --train 30 --test 20 --ndays 50 --restore-mode sens --ste-beta 20 --out docs/study/diff_env_real_caltech_sens_ste.json
PYTHONPATH=. python scripts/td3_synth_stress.py --episodes 120 --seed 0        # critic-based baseline on the synthetic stress days
PYTHONPATH=. python scripts/conformal_normalised_shift.py; PYTHONPATH=. python scripts/conformal_rowwise_vs_global.py
```
Notes: the ICE 2019 workbook (daily on-peak block prices only) is NOT used (no intraday shape). The tariff is the March-2019 SCE TOU-EV-4 schedule from `acnportal`, applied to 2018-2021 and to both sites (assumption).

## Step 9 (2026-10-17): everything in one go on Windows/GPU
`powershell -ExecutionPolicy Bypass -File .\RUN_ALL.ps1`  (see docs/RUN_WINDOWS.md). Single script for the new experiment: `PYTHONPATH=. python scripts/wm_ablation.py --device cuda`.
