"""Physics-only test: does conformal voltage-margin coverage survive a LOAD SHIFT? (plan item: adaptive conformal)

Setup: IEEE-33. Score s = max over buses of the one-sided residual (v_LinDistFlow - v_AC)+  (LinDistFlow over-estimating
voltage is what makes a lower-voltage-bound constraint unsafe; we use |.| for the two-sided version via --two-sided).
Calibrate on load multiplier L_cal; deploy on a stream whose multiplier steps/ramps to L_test.
Compare: FIXED split-conformal margin vs ADAPTIVE (ACI). Hypothesis H: fixed coverage < 1-alpha-0.05 after the shift,
ACI long-run coverage within 0.02 of 1-alpha.  Kill criterion: if the fixed margin ALREADY covers (residual shift-insensitive)
the adaptive layer is unnecessary and we say so."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pifsrl.physics.feeder import ieee33
from pifsrl.physics.powerflow import solve_bfs, solve_lindistflow
from pifsrl.safety.adaptive import AdaptiveConformal


def scores(f, n, load, rng, hubs, two_sided=False, ev_kw=40.0):
    sb = f.s_base_mva * 1e3
    lf = load * rng.uniform(0.85, 1.15, size=(n, 1))                       # time-varying load level
    p = -f.p_load[None, :] * lf
    q = -f.q_load[None, :] * lf
    ev = rng.uniform(0.0, 1.0, size=(n, len(hubs))) * ev_kw / sb          # EV charging at hub buses
    for j, h in enumerate(hubs):
        p[:, h] -= ev[:, j]
    ex = solve_bfs(f, p, q).v
    lin, _ = solve_lindistflow(f, p, q)
    e = (lin - ex)
    e = np.abs(e) if two_sided else np.maximum(e, 0.0)
    return e.max(axis=1)


def one_shift(f, hubs, a, l_test, two_sided):
    res = {"no_shift_fixed": [], "fixed": [], "aci": [], "aci_transient500": [], "fixed_transient500": [], "aci_margin_ratio": []}
    for sd in range(a.seeds):
        rng = np.random.default_rng(sd)
        cal = scores(f, a.n_cal, a.l_cal, rng, hubs, two_sided)
        n = len(cal); lv = min(1.0, np.ceil((n + 1) * (1 - a.alpha)) / n)
        m_fix = np.quantile(cal, lv, method="higher")
        s0 = scores(f, a.n_test, a.l_cal, rng, hubs, two_sided)
        res["no_shift_fixed"].append(float(np.mean(s0 <= m_fix)))
        s1 = np.empty(a.n_test); ramp = a.n_test // 4
        for t0 in range(0, a.n_test, 200):
            frac = min(1.0, t0 / max(ramp, 1))
            s1[t0:t0 + 200] = scores(f, min(200, a.n_test - t0), a.l_cal + frac * (l_test - a.l_cal), rng, hubs, two_sided)
        res["fixed"].append(float(np.mean(s1 <= m_fix)))
        res["fixed_transient500"].append(float(np.mean(s1[:500] <= m_fix)))
        aci = AdaptiveConformal(a.alpha, a.gamma, window=1000, init_scores=cal)
        margins = []
        for s_ in s1:
            margins.append(aci.margin()); aci.update(s_)
        res["aci"].append(aci.coverage()); res["aci_transient500"].append(1.0 - float(np.mean(aci.err_hist[:500])))
        res["aci_margin_ratio"].append(float(np.mean(margins[-1000:]) / max(m_fix, 1e-12)))      # steady-state margin vs the fixed one
    return res


def env_info():
    import platform, numpy, scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--l-cal", type=float, default=0.75)
    ap.add_argument("--l-test", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--two-sided", action="store_true")
    ap.add_argument("--out", default="docs/study/adaptive_conformal_shift.json")
    a = ap.parse_args()
    f = ieee33()
    hubs = [f.n_bus - 1, f.n_bus - 8, f.n_bus // 2]
    out = {"args": vars(a), "env": env_info(), "target": 1 - a.alpha, "hub_buses": hubs, "by_load": {}}
    print(f"target coverage {1 - a.alpha:.2f}; calibrated at load x{a.l_cal}; 20-seed means (min over seeds in brackets)")
    print("l_test | fixed (transient 500) | ACI (transient 500) | ACI/fixed margin")
    for lt in a.l_test:
        r = one_shift(f, hubs, a, lt, a.two_sided)
        out["by_load"][str(lt)] = {"raw": r, **{k + "_mean": float(np.mean(v)) for k, v in r.items()}, **{k + "_min": float(np.min(v)) for k, v in r.items()}}
        o = out["by_load"][str(lt)]
        print(f"{lt:5.2f}  | {o['fixed_mean']:.3f} ({o['fixed_transient500_mean']:.3f}) | {o['aci_mean']:.3f} [{o['aci_min']:.3f}] ({o['aci_transient500_mean']:.3f}) | x{o['aci_margin_ratio_mean']:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
