"""Load-normalised conformal score under load shift (answers the reviewer question "does physics normalisation remove ACI's margin inflation?").
Same stream as adaptive_conformal_shift.py.  Score s_t = max_bus (v_LDF - v_AC)+ ; observable load proxy g_t = total active load (p.u.) at step t.
Normalised score r_t = s_t / (g_t/g_nom)^k, margin_t = q * (g_t/g_nom)^k, with k either FIXED (2: I^2R) or FITTED on the calibration set (log-log OLS).
Reports (a) whole-stream and first-500-step coverage of the fixed-quantile margin, (b) ACI on the normalised score, (c) mean margin in physical units relative to the
unnormalised fixed margin, (d) the unnormalised ACI margin ratio for comparison.   PYTHONPATH=. python scripts/conformal_normalised_shift.py"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pifsrl.physics.feeder import ieee33
from pifsrl.physics.powerflow import solve_bfs, solve_lindistflow
from pifsrl.safety.adaptive import AdaptiveConformal


def stream(f, n, load, rng, hubs, ev_kw=40.0):
    sb = f.s_base_mva * 1e3
    lf = load * rng.uniform(0.85, 1.15, size=(n, 1))
    p = -f.p_load[None, :] * lf; q = -f.q_load[None, :] * lf
    ev = rng.uniform(0.0, 1.0, size=(n, len(hubs))) * ev_kw / sb
    for j, h in enumerate(hubs):
        p[:, h] -= ev[:, j]
    ex = solve_bfs(f, p, q).v; lin, _ = solve_lindistflow(f, p, q)
    return np.maximum(lin - ex, 0.0).max(axis=1), (-p).sum(axis=1)           # score, observable total load


def qhi(x, alpha):
    n = len(x); return float(np.quantile(x, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def run(f, hubs, a, l_test):
    keys = ["fixed_raw", "aci_raw", "fixed_k2", "aci_k2", "fixed_kfit", "aci_kfit"]
    R = {f"{k}_{w}": [] for k in keys for w in ("cov", "cov500")}
    R.update({"aci_raw_margin_ratio": [], "aci_k2_margin_ratio": [], "aci_kfit_margin_ratio": [], "fixed_k2_margin_ratio": [], "k_fit": []})
    for sd in range(a.seeds):
        rng = np.random.default_rng(sd)
        s_c, g_c = stream(f, a.n_cal, a.l_cal, rng, hubs); gn = float(np.median(g_c))
        s1 = np.empty(a.n_test); g1 = np.empty(a.n_test); ramp = a.n_test // 4
        for t0 in range(0, a.n_test, 200):
            fr = min(1.0, t0 / max(ramp, 1)); m = min(200, a.n_test - t0)
            s1[t0:t0 + 200], g1[t0:t0 + 200] = stream(f, m, a.l_cal + fr * (l_test - a.l_cal), rng, hubs)
        ok = s_c > 1e-12; k_fit = float(np.polyfit(np.log(g_c[ok] / gn), np.log(s_c[ok]), 1)[0]); R["k_fit"].append(k_fit)
        m_raw = qhi(s_c, a.alpha)
        # raw
        R["fixed_raw_cov"].append(float(np.mean(s1 <= m_raw))); R["fixed_raw_cov500"].append(float(np.mean(s1[:500] <= m_raw)))
        aci = AdaptiveConformal(a.alpha, a.gamma, window=1000, init_scores=s_c); mg = []
        for s in s1: mg.append(aci.margin()); aci.update(s)
        R["aci_raw_cov"].append(aci.coverage()); R["aci_raw_cov500"].append(1 - float(np.mean(aci.err_hist[:500]))); R["aci_raw_margin_ratio"].append(float(np.mean(mg[-1000:]) / m_raw))
        for tag, k in (("k2", 2.0), ("kfit", k_fit)):
            sc = lambda s, g: s / (g / gn) ** k
            r_c = sc(s_c, g_c); q = qhi(r_c, a.alpha)
            marg = q * (g1 / gn) ** k
            R[f"fixed_{tag}_cov"].append(float(np.mean(s1 <= marg))); R[f"fixed_{tag}_cov500"].append(float(np.mean(s1[:500] <= marg[:500])))
            aci = AdaptiveConformal(a.alpha, a.gamma, window=1000, init_scores=r_c); mg = []
            for s, g in zip(s1, g1):
                mg.append(aci.margin() * (g / gn) ** k); aci.update(sc(s, g))
            R[f"aci_{tag}_cov"].append(aci.coverage()); R[f"aci_{tag}_cov500"].append(1 - float(np.mean(aci.err_hist[:500])))
            R[f"aci_{tag}_margin_ratio"].append(float(np.mean(mg[-1000:]) / m_raw))
            if tag == "k2": R["fixed_k2_margin_ratio"].append(float(np.mean(marg[-1000:]) / m_raw))
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--l-cal", type=float, default=0.75); ap.add_argument("--l-test", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--n-cal", type=int, default=2000); ap.add_argument("--n-test", type=int, default=4000); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="docs/study/conformal_normalised_shift.json"); a = ap.parse_args()
    f = ieee33(); hubs = [f.n_bus - 1, f.n_bus - 8, f.n_bus // 2]; out = {"args": vars(a), "by_load": {}}
    for lt in a.l_test:
        R = run(f, hubs, a, lt); out["by_load"][str(lt)] = {k: float(np.mean(v)) for k, v in R.items()}; o = out["by_load"][str(lt)]
        print(f"L={lt}: k_fit={o['k_fit']:.2f} | whole-stream cov raw fixed {o['fixed_raw_cov']:.3f} k2 {o['fixed_k2_cov']:.3f} kfit {o['fixed_kfit_cov']:.3f} | ACI raw {o['aci_raw_cov']:.3f} k2 {o['aci_k2_cov']:.3f} kfit {o['aci_kfit_cov']:.3f}"
              f" | first500 fixed raw {o['fixed_raw_cov500']:.3f} k2 {o['fixed_k2_cov500']:.3f} kfit {o['fixed_kfit_cov500']:.3f} | margin/raw-fixed: ACI raw {o['aci_raw_margin_ratio']:.2f} k2 {o['aci_k2_margin_ratio']:.2f} kfit {o['aci_kfit_margin_ratio']:.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
