"""PoC: can a day-level context choose the EDF cap better than ONE fixed cap?  (uses docs/study/headroom_multiday.json)
Leave-one-day-out kNN over the cost-vs-cap curves of the other days.  Features:
  exante : trailing-14-day mean daily energy (known before the day) + weekday flag
  expost : the day's own delivered energy (clairvoyant upper bound for any forecaster)"""
import json, datetime as dt, numpy as np, pandas as pd
rows = json.load(open("docs/study/headroom_multiday.json"))
caps = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
days = [r["date"] for r in rows]
C = np.array([[r[f"edf_cap_{c}"][0] for c in caps] for r in rows])          # day x cap
df = pd.read_csv("data/caltech_local_sessions.csv", usecols=["connectionTime", "kWhDelivered"])
t = pd.to_datetime(df["connectionTime"], format="%a, %d %b %Y %H:%M:%S GMT", utc=True).dt.tz_convert("America/Los_Angeles")
daily = df.groupby(t.dt.date.astype(str))["kWhDelivered"].sum()
def ante(d):
    d0 = dt.date.fromisoformat(d); prev = [daily.get((d0 - dt.timedelta(k)).isoformat(), np.nan) for k in range(1, 15)]
    return np.nanmean(prev) if np.isfinite(prev).any() else daily.mean()
F = {"exante": np.array([ante(d) for d in days]), "expost": np.array([daily.get(d, 0.0) for d in days])}
fixed = C.mean(0).argmin(); perday = C.min(1)
print(f"fixed best cap {caps[fixed]}: {C[:, fixed].mean():.1f} | per-day best cap (clairvoyant): {perday.mean():.1f} | oracle {np.mean([r['oracle_lp'][0] for r in rows]):.1f}")
res = {}
for name, f in F.items():
    z = (f - f.mean()) / (f.std() + 1e-9)
    for k in (3, 5, 8):
        tot = []
        for i in range(len(days)):
            tr = [j for j in range(len(days)) if j != i]
            nn = sorted(tr, key=lambda j: abs(z[j] - z[i]))[:k]
            c = C[nn].mean(0).argmin(); tot.append(C[i, c])
        res[(name, k)] = float(np.mean(tot)); print(f"  LOO kNN {name:7s} k={k}: {np.mean(tot):.1f}")
fl = [(C[[j for j in range(len(days)) if j != i]].mean(0).argmin()) for i in range(len(days))]
print(f"  LOO fixed-cap (tuned on the other 19 days): {np.mean([C[i, fl[i]] for i in range(len(days))]):.1f}")
corr = np.corrcoef(F['expost'], [caps[i] for i in C.argmin(1)])[0, 1]
print(f"  corr(day energy, best cap) = {corr:.2f};  best caps per day: {[caps[i] for i in C.argmin(1)]}")
