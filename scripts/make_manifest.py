"""Reproducibility manifest: environment + SHA-256 of data, code and every evidence JSON, so a reader can verify that they are
running the same inputs and comparing against the same outputs.
  python scripts/make_manifest.py            # write docs/study/MANIFEST.json
  python scripts/make_manifest.py --verify   # recompute and report any file whose hash differs from the manifest"""
import argparse, glob, hashlib, json, os, platform, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); ap.add_argument("--path", default="docs/study/MANIFEST.json")
a = ap.parse_args(); os.chdir(ROOT)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def pkgs():
    out = {}
    for n in ("numpy", "scipy", "pandas", "torch", "cvxpy", "clarabel", "osqp", "matplotlib", "pytest"):
        try:
            import importlib.metadata as md; out[n] = md.version(n)
        except Exception:
            out[n] = None
    return out


groups = {"data": sorted(glob.glob("data/*.csv")) + sorted(glob.glob("data/templates/*.csv")),
          "code": sorted(glob.glob("pifsrl/**/*.py", recursive=True) + glob.glob("scripts/*.py") + ["sample.py"] + glob.glob("tests/*.py")),
          "evidence": sorted(glob.glob("docs/study/**/*.json", recursive=True))}
groups["evidence"] = [p for p in groups["evidence"] if os.path.basename(p) != "MANIFEST.json"]
cur = {g: {p: sha(p) for p in ps} for g, ps in groups.items()}
if a.verify:
    old = json.load(open(a.path))["files"]; bad = 0
    for g in cur:
        for p, h in cur[g].items():
            if old.get(g, {}).get(p) != h:
                bad += 1; print(f"[{'NEW' if p not in old.get(g, {}) else 'DIFF'}] {g}: {p}")
        for p in old.get(g, {}):
            if p not in cur[g]: bad += 1; print(f"[GONE] {g}: {p}")
    print("manifest OK" if not bad else f"{bad} difference(s) (data files you added/changed are expected to differ)"); sys.exit(1 if bad else 0)
man = {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs(),
       "note": "data/*.csv hashes identify the exact ACN-Data exports used; evidence hashes pin every number quoted in DOC.md/the proposal.",
       "files": cur}
os.makedirs(os.path.dirname(a.path), exist_ok=True); json.dump(man, open(a.path, "w"), indent=1)
print(f"wrote {a.path}: " + ", ".join(f"{g}={len(v)}" for g, v in cur.items()))
