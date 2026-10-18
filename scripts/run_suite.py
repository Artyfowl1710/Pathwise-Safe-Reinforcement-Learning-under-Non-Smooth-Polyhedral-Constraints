"""One-command, resumable experiment suite (Windows / Linux / macOS). Every stage writes its own log + JSON; a finished stage
(its output file exists) is skipped unless --force.  Nothing here needs a shell: only the Python interpreter running it.

  python scripts/run_suite.py --device cuda --auto-data                 # everything, both sites, both regimes
  python scripts/run_suite.py --device cuda --auto-data --sites caltech --only headroom,multiday
  python scripts/run_suite.py --dry-run                                 # print the commands only
Stages: tests, data, headroom, scenarios, multiday, transfer, federated, conformal, treeqp, report."""
import argparse, glob, os, shlex, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda"); ap.add_argument("--auto-data", action="store_true")
ap.add_argument("--sites", default="caltech,jpl"); ap.add_argument("--regimes", default="normal,stress")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--days", type=int, default=20)
ap.add_argument("--episodes", type=int, default=150, help="multiday training episodes per seed (GPU default; 60 was the CPU pilot)")
ap.add_argument("--residual", type=float, default=0.1); ap.add_argument("--demand", type=float, default=1.5)
ap.add_argument("--rounds", type=int, default=30); ap.add_argument("--lmp-analog", nargs="?", const="month", default="", choices=["", "month", "any"])
ap.add_argument("--battery-default", action="store_true"); ap.add_argument("--demand-from-tariff", action="store_true")
ap.add_argument("--hub-grid", action="store_true", help="include per-hub caps in headroom + multiday")
ap.add_argument("--mpc-reserves", type=int, nargs="*", default=[], help="e.g. 0 8 16 (slow: 1-2 min per day per arm)")
ap.add_argument("--only", default=""); ap.add_argument("--skip", default=""); ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--price-scenario", default="", help="synthetic scenario (duck|volatile|solar_glut|winter_peak|tou3|flat) or historical_lmp/historical_tariff; applied to every stage")
ap.add_argument("--price-seed", type=int, default=0)
ap.add_argument("--scenarios", default="duck,volatile,solar_glut,winter_peak,flat", help="stage `scenarios`: synthetic price scenarios for the no-learning headroom study")
ap.add_argument("--out", default="results/suite"); ap.add_argument("--quick", action="store_true", help="tiny smoke run (3 days, 6 episodes, 2 rounds)")
a = ap.parse_args()
if a.quick:
    a.days, a.episodes, a.rounds, a.seeds = 6, 6, 2, [0]
only = set(filter(None, a.only.split(","))); skip = set(filter(None, a.skip.split(",")))
sites, regimes = a.sites.split(","), a.regimes.split(",")
CAP = {"caltech": 100.0, "jpl": 130.0}                       # 2/3 of nameplate (Caltech 150 kW, JPL 195 kW)
PY = sys.executable
os.makedirs(os.path.join(ROOT, a.out), exist_ok=True); os.makedirs(os.path.join(ROOT, "docs/study/suite"), exist_ok=True)
env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
failed = []


def run(stage, name, cmd, out_file=None):
    if (only and stage not in only) or stage in skip:
        return
    if out_file and os.path.exists(os.path.join(ROOT, out_file)) and not a.force:
        print(f"[skip] {stage}/{name}: {out_file} exists"); return
    log = os.path.join(ROOT, a.out, f"{stage}_{name}.log")
    print(f"[run ] {stage}/{name}\n       {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if a.dry_run: return
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{'ok  ' if rc == 0 else 'FAIL'}] {stage}/{name} ({time.time() - t0:.0f}s) log: {log}", flush=True)
    if rc: failed.append(f"{stage}/{name}")


data_flags = (["--price-scenario", a.price_scenario, "--price-seed", str(a.price_seed)] if a.price_scenario else []) + (["--auto-data"] if a.auto_data else []) + (["--lmp-analog", a.lmp_analog] if a.lmp_analog else []) + \
             (["--battery-default"] if a.battery_default else []) + (["--demand-from-tariff"] if a.demand_from_tariff else [])

run("tests", "pytest", [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
run("data", "check", [PY, "scripts/check_data.py"])
for site in sites:
    for reg in regimes:
        st = ["--stress"] if reg == "stress" else []
        hf = f"docs/study/suite/headroom_{site}_{reg}.json"
        run("headroom", f"{site}_{reg}", [PY, "scripts/headroom_multiday.py", "--site", site, "--days", str(a.days), "--demand", str(a.demand),
            "--out", hf] + st + data_flags[:1] + data_flags[1:] +
            (["--hub-grid"] if a.hub_grid else []) + (["--mpc-reserves"] + [str(x) for x in a.mpc_reserves] if a.mpc_reserves else []), hf)
        for sd in a.seeds:
            od = f"{a.out}/multiday/{site}_{reg}"
            sf = f"{od}/multiday_{site}_seed{sd}.json"
            cmd = [PY, "sample.py", "--mode", "single", "--real-data", site, "--multiday", str(a.days), "--episodes", str(a.episodes),
                   "--warmup", "6", "--log-every", "10", "--eval-every", "5", "--eval-episodes", "4", "--device", a.device, "--seed", str(sd),
                   "--demand", str(a.demand), "--reward-mode", "dense", "--lambda-mb", "0", "--warmup-mode", "greedy",
                   "--residual", str(a.residual), "--base", "edf_tuned", "--out", od, "--log-dir", f"logs/suite/{site}_{reg}_s{sd}"] + data_flags
            if reg == "stress": cmd += ["--stress-ev", "2.0", "--stress-cap", str(CAP[site])]
            if a.hub_grid: cmd += ["--hub-grid"]
            if a.mpc_reserves: cmd += ["--mpc-reserves"] + [str(x) for x in a.mpc_reserves]
            run("multiday", f"{site}_{reg}_s{sd}", cmd, sf)
for reg in regimes:
    for task, extra in (("transfer", ["--episodes", str(a.episodes)]), ("federated", ["--rounds", str(a.rounds), "--algos", "fedprox,local"])):
        for sd in a.seeds:
            tag = f"{task}_{reg}_s{sd}"
            cmd = [PY, "scripts/cross_site_fed.py", "--task", task, "--regime", reg, "--seed", str(sd), "--device", a.device,
                   "--residual", str(a.residual), "--demand", str(a.demand), "--out", f"{a.out}/xsite", "--tag", tag] + extra + data_flags
            run(task, f"{reg}_s{sd}", cmd, f"{a.out}/xsite/{tag}.json")
for site in sites:                                          # price-scenario robustness (no learning): same real sessions, different synthetic price paths
    for reg in regimes:
        for sc in filter(None, a.scenarios.split(",")):
            hf = f"docs/study/suite/headroom_{site}_{reg}_{sc}.json"
            base_flags = [f for f in data_flags]
            if "--price-scenario" in base_flags:
                i = base_flags.index("--price-scenario"); del base_flags[i:i + 4]
            run("scenarios", f"{site}_{reg}_{sc}", [PY, "scripts/headroom_multiday.py", "--site", site, "--days", str(a.days), "--demand", str(a.demand), "--out", hf,
                "--price-scenario", sc] + (["--stress"] if reg == "stress" else []) + base_flags, hf)
run("scenarios", "summary", [PY, "scripts/price_scenario_summary.py", "--glob", "docs/study/suite/headroom_*_*_*.json", "--out", "docs/study/suite/price_scenarios.md"])
run("conformal", "shift", [PY, "scripts/adaptive_conformal_shift.py", "--seeds", "20", "--out", "docs/study/adaptive_conformal_shift.json"], "docs/study/adaptive_conformal_shift.json")
run("treeqp", "bench", [PY, "scripts/bench_tree_qp.py", "--sizes", "33", "123", "400", "--targets", "10", "30", "--seeds", "3", "--out", "docs/study/bench_tree_qp.json"], "docs/study/bench_tree_qp.json")
for site in sites:
    for reg in regimes:
        run("report", f"{site}_{reg}", [PY, "scripts/report_table.py", "--multiday", f"{a.out}/multiday/{site}_{reg}/multiday_{site}_seed*.json",
            "--headroom", f"docs/study/suite/headroom_{site}_{reg}.json", "--out", f"docs/study/suite/report_{site}_{reg}.md"])
print("\nFAILED stages:" if failed else "\nAll requested stages finished.", *failed)
sys.exit(1 if failed else 0)
