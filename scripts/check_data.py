"""Data doctor: which files are present, are they valid, and which FALLBACK will the code use otherwise?
    python scripts/check_data.py            # report
    python scripts/check_data.py --strict   # exit 1 if a REQUIRED file (ACN CSVs) is missing/invalid
Exit code is 0 for missing OPTIONAL files: they only trigger a labelled fallback."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pifsrl.env.datafiles import FILES, load_tariff, load_profile_shape, load_battery_dist
from pifsrl.env.sessions import lmp_csv_price, list_acn_dates

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="data"); ap.add_argument("--strict", action="store_true")
a = ap.parse_args(); D = a.data_dir
rows, bad_required = [], False


def add(name, status, detail, fallback=""):
    rows.append((name, status, detail, fallback)); print(f"[{status:7s}] {name:34s} {detail}" + (f"\n          fallback: {fallback}" if fallback and status != "OK" else ""))


days = {}
for site in ("caltech", "jpl"):
    p = os.path.join(D, f"{site}_local_sessions.csv")
    if not os.path.exists(p):
        add(f"{site}_local_sessions.csv", "MISSING", "REQUIRED ACN-Data export", "none: the real-data runs cannot start"); bad_required = True; continue
    try:
        ds = list_acn_dates(p, site); days[site] = set(ds)
        add(f"{site}_local_sessions.csv", "OK", f"{len(ds)} days {ds[0]}..{ds[-1]}")
    except Exception as e:
        add(f"{site}_local_sessions.csv", "INVALID", str(e)[:100]); bad_required = True
if len(days) == 2:
    print(f"          days present at BOTH sites: {len(days['caltech'] & days['jpl'])}")

probe = ["2019-07-10", "2019-01-15", "2020-03-04", "2021-05-14"]
for site in ("caltech", "jpl"):
    p = os.path.join(D, FILES["tariff"].format(site=site))
    fb = "SCE TOU-D-PRIME 2026 residential STAND-IN (labelled in every result)"
    if not os.path.exists(p): add(f"tariff_{site}.csv", "MISSING", "optional historical retail tariff", fb); continue
    t = load_tariff(p, site)
    if t is None: add(f"tariff_{site}.csv", "INVALID", "columns/format wrong, see datasets.md", fb); continue
    cov = [d for d in probe if t.price(d, 96, 0.25) is not None]
    dem = [t.demand_per_day(d) for d in probe]
    add(f"tariff_{site}.csv", "OK" if len(cov) == len(probe) else "PARTIAL", f"covers {len(cov)}/{len(probe)} probe dates; demand $/kW-day {np.round([x for x in dem if x], 3).tolist()}", fb)
p = os.path.join(D, FILES["lmp"])
fb = "SYNTHETIC CAISO-like duck curve (labelled synthetic)"
if not os.path.exists(p): add("caiso_sp15_lmp.csv", "MISSING", "optional real CAISO SP15 LMP", fb)
else:
    cov = [d for d in probe if lmp_csv_price(p, 96, 0.25, d) is not None]
    add("caiso_sp15_lmp.csv", "OK" if len(cov) == len(probe) else "PARTIAL", f"covers {len(cov)}/{len(probe)} probe dates", fb)
p = os.path.join(D, FILES["load"]); fb = "archetype base-load generator"
if not os.path.exists(p): add("feeder_load.csv", "MISSING", "optional real feeder load time series", fb)
else:
    cov = [d for d in probe if load_profile_shape(p, 96, 0.25, d) is not None]
    add("feeder_load.csv", "OK" if len(cov) == len(probe) else "PARTIAL", f"covers {len(cov)}/{len(probe)} probe dates", fb)
p = os.path.join(D, FILES["battery"]); fb = "no battery assumption (cap=requested energy); add --battery-default for the ASSUMED {40,60,75} kWh what-if"
if not os.path.exists(p): add("battery_models.csv", "MISSING", "optional EV battery-capacity distribution (V2G what-if)", fb)
else:
    d = load_battery_dist(p)
    add("battery_models.csv", "OK" if d is not None else "INVALID", "" if d is None else f"{len(d[0])} models, mean {float((d[0]*d[1]).sum()):.1f} kWh", fb)
print("\nSUMMARY: " + ", ".join(f"{r[0]}={r[1]}" for r in rows))
sys.exit(1 if (a.strict and bad_required) else 0)
