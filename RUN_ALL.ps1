<#
.SYNOPSIS
  One-shot, resumable runner for the PIFSRL-V2G study (Windows PowerShell 5.1 or 7+, NVIDIA GPU).

.DESCRIPTION
  Put the data files in .\data first (see data\README.md), then:
      powershell -ExecutionPolicy Bypass -File .\RUN_ALL.ps1
  It creates a venv, installs a CUDA build of torch (verified), runs the checks, the tests and every
  experiment in a fixed order, and writes everything to a time-stamped "safe" folder OUTSIDE the repo
  (default: %USERPROFILE%\PIFSRL_runs\<stamp>): per-step logs, result JSONs, status.json, SHA-256 hashes,
  a zip, and the finished folder is marked read-only.  A failing step is logged and the run continues.
  Re-run with  -Resume <folder>  to skip steps that already finished.

  GPU-only means: torch work (twin rollouts, SafeTD3 training, federated suite) uses CUDA and the run
  ABORTS if CUDA is unavailable (override with -AllowCpu).  The LP oracle, EDF/MPC evaluations, the exact QP
  and the AC power-flow stay on the CPU by design (they are scipy/cvxpy/numpy) - the GPU does not speed them up.

.PARAMETER Profile   Standard (default, ~ overnight) | Full (bigger budgets, ~ 1-2 days) | Smoke (minutes, plumbing check only)
.PARAMETER Resume    path of an earlier run folder to continue
.PARAMETER Only      comma list of step ids (e.g. "05,06") ; .PARAMETER Skip likewise
.PARAMETER Promote   after success also copy result JSONs into .\docs\study (overwrites the files the paper figures read)
#>
[CmdletBinding()]
param(
  [ValidateSet("Smoke","Standard","Full")][string]$Profile = "Standard",
  [string]$OutRoot = (Join-Path $env:USERPROFILE "PIFSRL_runs"),
  [string]$Resume = "",
  [string]$Only = "", [string]$Skip = "",
  [switch]$AllowCpu, [switch]$Promote, [switch]$SkipInstall, [switch]$Force,
  [string]$Python = ""
)
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot; $env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"

# ------------------------------------------------------------------ budgets
$B = switch ($Profile) {
  "Smoke"    { @{ days=3;  sweepDays=3;  iters=10;  plan=10;  nseeds=1; train=6;  test=4;  ndays=10; episodes=6;   wmsteps=100;  abseeds=@(0); abepisodes=4;   tdseeds=@(0);       fedRounds=2;  suite=$false } }
  "Standard" { @{ days=20; sweepDays=10; iters=400; plan=600; nseeds=5; train=30; test=20; ndays=50; episodes=300; wmsteps=1500; abseeds=@(0,1,2); abepisodes=150; tdseeds=@(0,1,2);   fedRounds=30; suite=$false } }
  "Full"     { @{ days=40; sweepDays=20; iters=600; plan=800; nseeds=5; train=30; test=20; ndays=50; episodes=500; wmsteps=3000; abseeds=@(0,1,2,3,4); abepisodes=300; tdseeds=@(0,1,2,3,4); fedRounds=40; suite=$true  } }
}
$sites = @("caltech","jpl")

# ------------------------------------------------------------------ run folder
if ($Resume) { $RunDir = (Resolve-Path $Resume).Path; $resuming = $true }
else { $stamp = Get-Date -Format "yyyyMMdd_HHmmss"; $RunDir = Join-Path $OutRoot "${stamp}_$Profile"; $resuming = $false }
foreach ($d in "logs","results","done") { New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $d) | Out-Null }
$Log = Join-Path $RunDir "logs\_runner.log"
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m; Write-Host $l; Add-Content -LiteralPath $Log -Value $l }
Say "PIFSRL run folder: $RunDir  (profile=$Profile, resume=$resuming)"
try { Start-Transcript -Path (Join-Path $RunDir "logs\_transcript.txt") -Append | Out-Null } catch {}

# keep Windows awake while this script runs (sleep would silently kill an overnight run)
try { Add-Type -Namespace Win32 -Name Pwr -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'; [void][Win32.Pwr]::SetThreadExecutionState(0x80000001) } catch {}

# ------------------------------------------------------------------ python / venv / torch
$Venv = Join-Path $PSScriptRoot ".venv"; $Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
  $cands = @(); if ($Python) { $cands += ,@($Python) }
  $cands += ,@("py","-3.12"); $cands += ,@("py","-3.13"); $cands += ,@("py","-3.11"); $cands += ,@("python")
  $found = $null
  foreach ($c in $cands) {
    try {
      $rest = if ($c.Count -gt 1) { @($c[1..($c.Count-1)]) } else { @() }
      $v = & $c[0] @rest -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
      if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(11|12|13)$') { $found = $c; break }
    } catch {}
  }
  if (-not $found) { Say "ERROR: need Python 3.11-3.13 (3.14 is untested with the pinned wheels). Install from python.org, tick 'Add to PATH'."; exit 2 }
  Say "creating venv with: $($found -join ' ')"
  $rest = if ($found.Count -gt 1) { @($found[1..($found.Count-1)]) } else { @() }
  & $found[0] @rest -m venv $Venv
  if (-not (Test-Path $Py)) { Say "ERROR: venv creation failed"; exit 2 }
}
function PyOut([string[]]$a) { & $Py @a 2>&1 }
function CudaOK { $res = & $Py -c "import torch,sys;sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null; return ($res -eq "1") }

if (-not $SkipInstall) {
  Say "pip: upgrading pip and installing the pinned non-torch requirements"
  & $Py -m pip install --upgrade pip 2>&1 | Out-Null
  $req = Get-Content (Join-Path $PSScriptRoot "requirements-lock.txt") | Where-Object { $_ -and ($_ -notmatch '^\s*(torch|#)') }
  $tmp = Join-Path $RunDir "requirements_nontorch.txt"; Set-Content -LiteralPath $tmp -Value $req
  & $Py -m pip install -r $tmp 2>&1 | Tee-Object -FilePath (Join-Path $RunDir "logs\00_pip_requirements.log") | Out-Null
  if ($LASTEXITCODE -ne 0) { Say "WARNING: pip install of pinned requirements failed (see logs\00_pip_requirements.log); trying unpinned"
    & $Py -m pip install numpy scipy pandas cvxpy clarabel osqp scs matplotlib pytest tzdata 2>&1 | Out-Null }
  $haveCuda = CudaOK
  if (-not $haveCuda) {
    $pin = (Get-Content (Join-Path $PSScriptRoot "requirements-lock.txt") | Where-Object { $_ -match '^torch==' } | Select-Object -First 1)
    foreach ($tag in "cu128","cu126","cu124","cu121") {
      foreach ($spec in @($pin, "torch")) {
        if (-not $spec) { continue }
        Say "installing $spec from the PyTorch $tag index"
        & $Py -m pip install --force-reinstall --no-deps $spec --index-url "https://download.pytorch.org/whl/$tag" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0 -and (CudaOK)) { $haveCuda = $true; break }
      }
      if ($haveCuda) { break }
    }
    if (-not $haveCuda) { & $Py -m pip install torch 2>&1 | Out-Null; $haveCuda = CudaOK }   # last resort: default wheel
    # torch needs its own deps (filelock, sympy, networkx, jinja2, fsspec, typing-extensions); --no-deps skipped them
    & $Py -m pip install filelock sympy networkx jinja2 fsspec typing-extensions setuptools 2>&1 | Out-Null
  }
}
$cudaOk = CudaOK
$gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null) -join "; "
Say "GPU: $gpu ; torch.cuda.is_available() = $cudaOk"
if (-not $cudaOk -and -not $AllowCpu) { Say "ERROR: CUDA not available to torch. Update the NVIDIA driver (>= 560 for cu128) and re-run, or pass -AllowCpu."; exit 3 }
$Dev = if ($cudaOk) { "cuda" } else { "cpu" }
& $Py -c "import sys,platform,torch,numpy,scipy,cvxpy;print(sys.version);print(platform.platform());print('torch',torch.__version__,'cuda',torch.version.cuda,'numpy',numpy.__version__,'scipy',scipy.__version__,'cvxpy',cvxpy.__version__)" 2>&1 |
  Tee-Object -FilePath (Join-Path $RunDir "logs\00_environment.txt") | Out-Null
if ($gpu) { Add-Content (Join-Path $RunDir "logs\00_environment.txt") "GPU: $gpu" }

# ------------------------------------------------------------------ step machinery
$Status = [ordered]@{}; $Steps = New-Object System.Collections.Generic.List[object]
function Want($id) {
  if ($Only) { return (($Only -split ",") | ForEach-Object { $id.StartsWith($_.Trim()) }) -contains $true }
  if ($Skip -and ((($Skip -split ",") | ForEach-Object { $id.StartsWith($_.Trim()) }) -contains $true)) { return $false }
  return $true
}
function Q([string]$s) { if ($s -match '[\s"]') { return '"' + ($s -replace '"','\"') + '"' } else { return $s } }
function Step([string]$id, [string]$title, [string[]]$pyArgs, [string]$expect = "") {
  # $pyArgs: arguments to python; outputs referenced with {R}\file.json get the results folder
  if (-not (Want $id)) { return }
  $done = Join-Path $RunDir "done\$id.ok"
  if ((Test-Path $done) -and -not $Force) { Say "SKIP $id (already done)"; $Status[$id] = "done(previous)"; return }
  $args2 = $pyArgs | ForEach-Object { $_ -replace '\{R\}', (Join-Path $RunDir "results") }
  $lg = Join-Path $RunDir ("logs\{0}.log" -f $id)
  Say "START $id  $title"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $cmdline = '"{0}" -u {1} 1>> "{2}" 2>&1' -f $Py, (($args2 | ForEach-Object { Q $_ }) -join " "), $lg
  Set-Content -LiteralPath $lg -Value ("# " + $cmdline)
  cmd.exe /c ("`"" + $cmdline + "`"")
  $code = $LASTEXITCODE; $sw.Stop()
  $ok = ($code -eq 0)
  if ($ok -and $expect) { $ok = Test-Path (Join-Path $RunDir ("results\" + $expect)) }
  $Status[$id] = [ordered]@{ title=$title; ok=$ok; exit=$code; minutes=[math]::Round($sw.Elapsed.TotalMinutes,1) }
  if ($ok) { Set-Content -LiteralPath $done -Value (Get-Date -Format o); Say ("OK    $id  {0:N1} min" -f $sw.Elapsed.TotalMinutes) }
  else { Say ("FAIL  $id  exit=$code after {0:N1} min -> see logs\$id.log (run continues)" -f $sw.Elapsed.TotalMinutes); Get-Content $lg -Tail 6 | ForEach-Object { Say ("      | " + $_) } }
  ($Status | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath (Join-Path $RunDir "status.json")
}
$R = "{R}"

# ------------------------------------------------------------------ 01 data check, 02 tests
$dataOK = $true
foreach ($s in $sites) { if (-not (Test-Path ".\data\${s}_local_sessions.csv")) { Say "ERROR: data\${s}_local_sessions.csv is missing (required ACN export)"; $dataOK = $false } }
if (-not $dataOK) { Say "Put the two ACN CSVs into .\data and re-run."; exit 4 }
if ((Test-Path ".\data\wep_download_data.xlsx") -and -not (Test-Path ".\data\caiso_sp15_lmp.csv")) { Step "03a" "build price files from WEP xlsx" @("scripts\build_real_price_files.py","--wep","data\wep_download_data.xlsx") }
Step "01" "data doctor (requires both ACN CSVs)" @("scripts\check_data.py","--strict")
Step "02" "unit tests incl. twin fidelity + GPU/CPU parity" @("-m","pytest","-q","--no-header","tests")

# ------------------------------------------------------------------ 04 conformal (CPU, minutes)
Step "04a" "conformal: row-wise vs global margins"          @("scripts\conformal_rowwise_vs_global.py","--out","$R\conformal_rowwise_vs_global.json") "conformal_rowwise_vs_global.json"
Step "04b" "conformal: ACI under load shift"                @("scripts\adaptive_conformal_shift.py","--out","$R\adaptive_conformal_shift.json") "adaptive_conformal_shift.json"
Step "04c" "conformal: load-normalised score"               @("scripts\conformal_normalised_shift.py","--out","$R\conformal_normalised_shift.json") "conformal_normalised_shift.json"

# ------------------------------------------------------------------ 05 headroom, three price sources x two sites (CPU: LP oracle)
foreach ($s in $sites) {
  Step "05-$s-fixed"  "headroom $s stress, SCE TOU stand-in" @("scripts\headroom_multiday.py","--site",$s,"--stress","--days","$($B.days)","--demand","1.5","--auto-data","--out","$R\headroom_${s}_stress_fixed.json") "headroom_${s}_stress_fixed.json"
  Step "05-$s-tariff" "headroom $s stress, TOU-EV-4 tariff"  @("scripts\headroom_multiday.py","--site",$s,"--stress","--days","$($B.days)","--auto-data","--demand-from-tariff","--out","$R\headroom_${s}_stress_tariff.json") "headroom_${s}_stress_tariff.json"
  if (Test-Path ".\data\caiso_sp15_lmp.csv") {
    Step "05-$s-wep" "headroom $s stress, WEP CAISO hourly"   @("scripts\headroom_multiday.py","--site",$s,"--stress","--days","$($B.days)","--demand","1.5","--auto-data","--wep-lmp","--out","$R\headroom_${s}_stress_wep.json") "headroom_${s}_stress_wep.json"
  }
}

# ------------------------------------------------------------------ 06 congestion sweep on REAL Caltech sessions (answers: where does H become large?)
if ($Profile -ne "Smoke") { $evs = @(1.5,2.0,3.0); $caps = @(0.4,0.5,0.67) } else { $evs = @(2.0); $caps = @(0.5) }
foreach ($ev in $evs) { foreach ($cf in $caps) {
  $tag = ("ev{0}_cap{1}" -f $ev, $cf) -replace '\.','p'
  Step "06-caltech-$tag" "congestion sweep caltech ev-scale=$ev cap-frac=$cf" @("scripts\headroom_multiday.py","--site","caltech","--stress","--ev-scale","$ev","--cap-frac","$cf","--days","$($B.sweepDays)","--demand","1.5","--auto-data","--out","$R\sweep_caltech_$tag.json") "sweep_caltech_$tag.json"
} }

# ------------------------------------------------------------------ 07 differentiable twin (GPU for the rollouts)
Step "07a" "twin PoC, synthetic stress days (fidelity, grads, planning, pathwise x$($B.nseeds), projection price)" @("scripts\diff_env_poc.py","--device",$Dev,"--iters","$($B.iters)","--plan-iters","$($B.plan)","--nseeds","$($B.nseeds)","--out","$R\diff_env_synth.json") "diff_env_synth.json"
Step "07b" "twin PoC, synthetic, sens+STE"                @("scripts\diff_env_poc.py","--device",$Dev,"--iters","$($B.iters)","--plan-iters","$($B.plan)","--nseeds","$($B.nseeds)","--restore-mode","sens","--ste-beta","20","--out","$R\diff_env_synth_sens_ste.json") "diff_env_synth_sens_ste.json"
foreach ($s in $sites) {
  $common = @("scripts\diff_env_poc.py","--device",$Dev,"--site",$s,"--stress","2.0","--train","$($B.train)","--test","$($B.test)","--ndays","$($B.ndays)","--iters","$($B.iters)","--plan-iters","$($B.plan)","--nseeds","$($B.nseeds)")
  Step "07-$s-uniform"    "twin real $s random split, uniform scaling"  ($common + @("--out","$R\diff_env_real_${s}_uniform.json")) "diff_env_real_${s}_uniform.json"
  Step "07-$s-sens"       "twin real $s random split, sens+STE"         ($common + @("--restore-mode","sens","--ste-beta","20","--out","$R\diff_env_real_${s}_sens_ste.json")) "diff_env_real_${s}_sens_ste.json"
  Step "07-$s-chrono"     "twin real $s CHRONOLOGICAL 2019 split (pre-COVID), uniform" ($common + @("--chrono","2019","--out","$R\diff_env_real_${s}_chrono2019_uniform.json")) "diff_env_real_${s}_chrono2019_uniform.json"
  Step "07-$s-chrono-sens" "twin real $s CHRONOLOGICAL 2019 split, sens+STE" ($common + @("--chrono","2019","--restore-mode","sens","--ste-beta","20","--out","$R\diff_env_real_${s}_chrono2019_sens_ste.json")) "diff_env_real_${s}_chrono2019_sens_ste.json"
}

# ------------------------------------------------------------------ 07w residual world model (your F_base + N_theta) vs F_base-only vs black-box
foreach ($s in $sites) { Step "07pp-$s" "projection price, closed loop (QP vs ray scaling), $s" @("scripts\proj_price_closed_loop.py","--site",$s,"--iters","$($B.iters)","--nseeds","$($B.nseeds)","--out","$R\proj_price_closed_loop_$s.json") "proj_price_closed_loop_$s.json" }
Step "07v" "bidirectional V2G check: hindsight oracle gain and causal v2g_greedy (8 synthetic days)" @("scripts\v2g_discharge_check.py","--days","8","--out","$R\v2g_discharge_check.json") "v2g_discharge_check.json"
Step "07w" "world-model ablation (F_base only / +N_theta residual / black-box), 3 seeds" @("scripts\wm_ablation.py","--device",$Dev,"--steps","$($B.wmsteps)","--out","$R\wm_ablation.json") "wm_ablation.json"

Step "07w2" "world-model follow-up: SoC diagnosis + conformal margins (A5, one-step)" @("scripts\wm_ablation2.py","--steps","$($B.wmsteps)","--out","$R\wm_ablation2.json") "wm_ablation2.json"

# ------------------------------------------------------------------ 08 critic-based baseline on the synthetic high-headroom days (GPU)
foreach ($sd in $B.tdseeds) {
  Step "08-td3-s$sd" "residual SafeTD3 on synthetic stress days, seed $sd, $($B.episodes) episodes" @("scripts\td3_synth_stress.py","--device",$Dev,"--episodes","$($B.episodes)","--seed","$sd","--out","$R\td3_synth_stress_s$sd.json") "td3_synth_stress_s$sd.json"
}

# ------------------------------------------------------------------ 08b the proposal's four ablations (world model / safety layer / federated / VPW), SafeTD3 training on GPU
Step "08b" "proposal ablations (Sec 7.2): world_model, safety, federated, vpw" (@("-m","pifsrl.experiments.run_ablations","--ablation","all","--seeds") + @($B.abseeds | ForEach-Object { "$_" }) + @("--episodes","$($B.abepisodes)","--device",$Dev,"--out","$R\ablations"))

# ------------------------------------------------------------------ 09 old grid (only Full): learning on real days + federated, on GPU
if ($B.suite) {
  Step "09-suite" "run_suite.py (headroom/multiday/transfer/federated), GPU" @("scripts\run_suite.py","--device",$Dev,"--auto-data","--demand-from-tariff","--days","$($B.days)","--episodes","$($B.episodes)","--rounds","$($B.fedRounds)","--seeds","0","1","2","--out","$RunDir\suite")
}

# ------------------------------------------------------------------ finish: manifest, hashes, zip, read-only
Say "collecting artefacts"
Get-ChildItem -Recurse -File -Include *.py,*.ps1,*.txt -Path (Join-Path $PSScriptRoot "pifsrl"),(Join-Path $PSScriptRoot "scripts") -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLower(), $_.FullName.Substring($PSScriptRoot.Length+1) } | Set-Content -LiteralPath (Join-Path $RunDir "code_sha256.txt")
Copy-Item (Join-Path $PSScriptRoot "RUN_ALL.ps1") (Join-Path $RunDir "RUN_ALL.ps1.copy") -ErrorAction SilentlyContinue
$nOK = @($Status.GetEnumerator() | Where-Object { ($_.Value -is [string]) -or $_.Value.ok }).Count; $nAll = $Status.Count
Say "steps ok: $nOK / $nAll   (details: status.json; failed steps: logs\<id>.log)"
if ($Promote) { Copy-Item (Join-Path $RunDir "results\*.json") ".\docs\study\" -Force; Say "promoted result JSONs into docs\study" }
try { Stop-Transcript | Out-Null } catch {}
$sumFile = Join-Path $RunDir "SHA256SUMS.txt"
Get-ChildItem -Recurse -File $RunDir | Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
  ForEach-Object { "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLower(), $_.FullName.Substring($RunDir.Length+1) } | Set-Content -LiteralPath $sumFile
$zip = Join-Path $OutRoot ((Split-Path $RunDir -Leaf) + ".zip")
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $RunDir "*") -DestinationPath $zip -Force
Get-ChildItem -Recurse -File $RunDir | ForEach-Object { try { $_.IsReadOnly = $true } catch {} }
Write-Host "DONE. Results folder (read-only): $RunDir" -ForegroundColor Green
Write-Host "Zip to send back: $zip" -ForegroundColor Green
