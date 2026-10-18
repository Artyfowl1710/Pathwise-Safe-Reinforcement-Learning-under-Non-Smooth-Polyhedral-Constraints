# Running everything on your Windows GPU machine (`RUN_ALL.ps1`)

## 0. What you need
* Windows 10/11, at least 8 GB RAM free (each real-data twin run peaks near 3.5 GB; the script runs them one at a time), NVIDIA GPU with a recent driver (>= 560 for the CUDA 12.8 wheels; `nvidia-smi` must work), ~15 GB free disk, internet for pip.
* Python **3.11, 3.12 or 3.13** (python.org, tick "Add to PATH"). Not 3.14 (pinned wheels untested).
* Unzip `pifsrl_v2g.zip`. Put your data into `data\`:
  * REQUIRED: `caltech_local_sessions.csv`, `jpl_local_sessions.csv` (the ACN exports).
  * Optional (each has a labelled fallback): `wep_download_data.xlsx` (the script builds `caiso_sp15_lmp.csv` from it), `tariff_caltech.csv`, `tariff_jpl.csv`, `feeder_load.csv`, `battery_models.csv` (templates are in `data\templates`).

## 1. Run
```
cd pifsrl_v2g
powershell -ExecutionPolicy Bypass -File .\RUN_ALL.ps1 -Profile Smoke      # 20-40 min plumbing check: do this first
powershell -ExecutionPolicy Bypass -File .\RUN_ALL.ps1                     # Standard (default), run it overnight
powershell -ExecutionPolicy Bypass -File .\RUN_ALL.ps1 -Profile Full       # bigger budgets
```
Useful switches: `-Resume <run folder>` (skip finished steps after a crash/reboot), `-Only "07,08"`, `-Skip "05"`, `-Force` (redo), `-AllowCpu` (otherwise it aborts if CUDA is missing), `-Promote` (also copy JSONs into `docs\study`), `-OutRoot D:\runs`.
The script keeps Windows from sleeping while it runs. A failing step is logged and the run continues.

## 2. Where the results go (the "safe space")
`%USERPROFILE%\PIFSRL_runs\<timestamp>_<profile>\` , outside the repo, so re-runs never overwrite earlier ones:
`results\*.json` (all outputs) · `logs\<step>.log` (full console of every step) · `status.json` (ok/fail + minutes per step) · `SHA256SUMS.txt` · `code_sha256.txt` · `00_environment.txt` (GPU, torch, versions) · `done\` (resume markers). The folder is set read-only at the end and zipped next to it as `<stamp>_<profile>.zip` -> **send me that zip**.

## 3. Steps and estimated time (Standard profile; ESTIMATES, not measured on your GPU)
| id | what | budget | runs on | est. time |
|---|---|---|---|---|
| setup | venv, pins, CUDA torch check | | | 5-15 min |
| 01-02 | data doctor, 68+ unit tests (incl. twin fidelity, GPU/CPU parity) | | CPU/GPU | 5-10 min |
| 04a-c | conformal: row-wise vs global, ACI under load shift, load-normalised score | | CPU | ~10-20 min |
| 05 | headroom, 2 sites x 3 price sources (SCE stand-in, TOU-EV-4, WEP) | 20 days | CPU (LP oracle) | 1-3 h |
| 06 | **congestion sweep on real Caltech**: ev-scale {1.5,2,3} x hub-cap {0.4,0.5,0.67} -> where does headroom H get large? | 10 days x 9 | CPU | 1-3 h |
| 07a-b | differentiable twin on synthetic days (fidelity, gradients, planning, pathwise x5 seeds, **projection-price test**) | 400 it, 5 seeds | GPU twin + CPU env | 0.5-1 h |
| 07 real | twin on real days: random split and **chronological 2019 split**, uniform and sens+STE, both sites (8 runs) | 5 seeds | GPU twin + CPU env | 2-5 h |
| 07pp | **projection-price closed-loop test (QP vs ray scaling), both sites** | 300 it x 3 seeds | GPU twin + CPU env | 0.5-1.5 h each |
| 07w | **your residual world model**: F_base vs F_base+N_theta vs black-box | 1500 steps x 3 seeds | GPU | 5-10 min |
| 08 | residual SafeTD3 on the synthetic stress days | 300 episodes x 3 seeds | GPU | 3-8 h |
| 08b | the proposal's 4 ablations: world model, safety layer, federated, willingness (VPW) | 150 episodes x 3 seeds | GPU | 3-8 h |
| 09 | (Full only) old grid: multiday learning + transfer + federated | 500 ep | GPU | +10-20 h |
| end | hashes, zip, read-only | | | 1 min |
**Total: Standard ~12-24 h, Full ~2-3 days, Smoke ~20-40 min.** The LP oracle, EDF/MPC evaluations, exact QP and AC solves are scipy/cvxpy/numpy and stay on the CPU by design, so they dominate wall-clock and a faster GPU will not shorten them. Only the torch parts (twin rollouts/training, SafeTD3, world model) use CUDA; the twin is float64, which consumer GPUs run slowly, so expect a modest gain over CPU there.

## 4. What I could and could not check
Tested here (CPU only): every Python script edit (a tiny run of the twin PoC with the new flags, the world-model ablation in full, the ablation runner), and the unit tests. **Not tested:** `RUN_ALL.ps1` itself (no PowerShell in my sandbox, so it was reviewed by hand but never executed), and any CUDA path. If `-Profile Smoke` fails, send me `logs\_runner.log` and the failing `logs\<step>.log`.

## 5. Not implemented (so the zip does not pretend)
Deadline-tightness (laxity-compression) and multi-lateral bus-placement sweeps; the four-way margin-scheme x policy cost table (conformal calibration vs RL, in one table); unrolled Dykstra projection inside the twin; a CVXPY solver switch/jitter for the tariff LP. The first three are suggested by your reviewers; they need code I have not written.

## Added in the last update
`07v` V2G discharge check (~10 min, CPU). `07pp` accepts `--init-bias` and `--restore-mode sens --ste-beta 20` if you want to repeat the stuck-seed experiments (all tried seeds stayed on the full-rate plateau here).
