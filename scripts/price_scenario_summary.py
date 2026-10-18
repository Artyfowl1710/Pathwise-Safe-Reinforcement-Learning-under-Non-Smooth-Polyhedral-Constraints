"""Summarise the no-learning headroom study across synthetic price scenarios (files headroom_<site>_<regime>_<scenario>.json)."""
import argparse, glob, json, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--glob", default="docs/study/suite/headroom_*_*_*.json"); ap.add_argument("--out", default="docs/study/suite/price_scenarios.md")
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(a.glob)):
    m = re.match(r"headroom_(caltech|jpl)_(normal|stress)_(.+)\.json", os.path.basename(f))
    if not m: continue
    R = json.load(open(f)); C = lambda k: np.array([r[k][0] for r in R]); V = lambda k: np.array([r[k][2] for r in R])
    caps = [k for k in R[0] if k.startswith("edf_cap_") and V(k).mean() <= 1e-6] + (["safe_uncontrolled"] if V("safe_uncontrolled").mean() <= 1e-6 else [])   # an unprojected / violating rule is NOT a deployable baseline
    bf = min(caps, key=lambda k: C(k).mean()); o = C("oracle_lp").mean(); b = C(bf).mean()
    rows.append((m[1], m[2], m[3], len(R), C("safe_uncontrolled").mean(), bf, b, o, (b - o) / b))
L = ["| site | regime | price scenario | days | safe_uncontrolled | best heuristic | cost | oracle | headroom H |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.1f} | {r[5]} | {r[6]:.1f} | {r[7]:.1f} | {r[8]:.1%} |")
L.append("\nH = (J_best_heuristic - J_oracle)/J_best_heuristic. All prices are SYNTHETIC scenarios (provenance tag in each run); real sessions + IEEE-33 physics.")
txt = "\n".join(L); print(txt); os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w").write(txt + "\n")
