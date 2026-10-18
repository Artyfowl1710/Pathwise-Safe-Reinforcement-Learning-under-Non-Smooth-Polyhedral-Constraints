"""Per-day paired comparison table with uncertainty (plan item). Reads multiday_*.json files written by `sample.py --multiday`
(one per seed) and/or headroom_multiday*.json. Prints a Markdown table and writes it next to --out.
  python scripts/report_table.py --multiday results/**/multiday_caltech_seed*.json --ref edf --out docs/study/report_caltech.md
Columns: mean cost, paired difference vs the tuned EDF reference with a 95% day-block bootstrap CI, sign-test p, and IQM across
seeds for the learner. Seeds are pooled by averaging per-day costs across seeds FIRST (days are the independent unit)."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pifsrl.evalstats import paired_bootstrap, sign_test_p, hoeffding_gate, iqm

ap = argparse.ArgumentParser()
ap.add_argument("--multiday", nargs="*", default=[]); ap.add_argument("--headroom", default=None)
ap.add_argument("--out", default="docs/study/report.md"); ap.add_argument("--gate-delta", type=float, default=0.05)
a = ap.parse_args()
lines = []
files = sorted({f for p in a.multiday for f in glob.glob(p, recursive=True)})
if files:
    R = [json.load(open(f)) for f in files]
    te = R[0]["test"]; ref_k = R[0].get("ref_kind") or f"edf_cap_{R[0]['edf_best_cap']}"
    # per-method per-day cost, averaged over seeds (learner varies by seed; rule baselines are deterministic)
    meth = {}
    for r in R:
        for k, v in r["baselines"].items():
            meth.setdefault(k, []).append(v["episode_costs"])
        for tag in ("last", "best"):
            f = r.get("final", {}).get(tag)
            if f and "episode_costs" in f: meth.setdefault(f"SafeTD3_{tag}", []).append(f["episode_costs"])
    M = {k: np.mean(np.array(v), axis=0) for k, v in meth.items() if len({len(x) for x in v}) == 1}
    ref = M[ref_k]; B = float(M.get("safe_uncontrolled", ref).max())
    lines += [f"### Held-out days: {te}  (seeds: {len(R)}; reference = {ref_k})", "",
              f"| method | mean $/day | Δ vs ref ($/day, + = better) | 95% CI | sign p | gate (δ={a.gate_delta:.2f}) |", "|---|---|---|---|---|---|"]
    for k, c in sorted(M.items(), key=lambda kv: kv[1].mean()):
        d = ref - c; b = paired_bootstrap(d, 5000)
        g = hoeffding_gate(d, B, a.gate_delta)
        lines.append(f"| {k} | {c.mean():.1f} | {b['mean']:+.1f} | [{b['lo']:+.1f}, {b['hi']:+.1f}] | {sign_test_p(d):.3f} | {'DEPLOY' if g['deploy'] else 'reject'} (LB {g['lower_bound']:+.1f}) |")
    for tag in ("last", "best"):
        if f"SafeTD3_{tag}" in meth and len(meth[f"SafeTD3_{tag}"]) > 1:
            per_seed = [float(np.mean(x)) for x in meth[f"SafeTD3_{tag}"]]
            lines.append(f"\nSafeTD3_{tag}: per-seed mean cost {np.round(per_seed, 1).tolist()}, IQM {iqm(per_seed):.1f}")
    lines.append("\nGate = Hoeffding lower bound on mean improvement with a-priori per-day bound B=%.0f $ (max safe-uncontrolled day). With so few held-out days the gate cannot fire for any realistic effect: that is the point (n needed shown in JSON via evalstats.hoeffding_gate)." % B)
if a.headroom and os.path.exists(a.headroom):
    rows = json.load(open(a.headroom)); C = lambda k: np.array([r[k][0] for r in rows])
    caps = [k for k in rows[0] if k.startswith("edf_cap_")]
    caps = [k for k in caps if np.mean([r[k][2] for r in rows]) <= 1e-6] or caps          # violation-free caps only
    caps = caps + ["safe_uncontrolled"]                    # the wrapped uncontrolled rule is a legitimate fixed heuristic (strongest at JPL-stress)
    best_fixed = min(caps, key=lambda k: C(k).mean()); pd_ = np.min([C(k) for k in caps], axis=0)
    lines += ["", f"### Headroom ({len(rows)} days)", "", "| policy | mean $/day |", "|---|---|",
              f"| safe_uncontrolled | {C('safe_uncontrolled').mean():.1f} |", f"| best fixed heuristic ({best_fixed}) | {C(best_fixed).mean():.1f} |",
              f"| per-day best heuristic (clairvoyant) | {pd_.mean():.1f} |", f"| oracle LP | {C('oracle_lp').mean():.1f} |"]
    H = (C(best_fixed).mean() - C("oracle_lp").mean()) / C(best_fixed).mean()
    lines.append(f"\nHeadroom H = (J_tuned − J_oracle)/J_tuned = {H:.1%}  (gate for attempting RL: ≥15%).")
txt = "\n".join(lines); print(txt)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w").write(txt + "\n")
