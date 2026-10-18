"""Figures for docs/paper/paper.tex, drawn ONLY from docs/study/*.json (no hand-typed numbers).  PYTHONPATH=. python scripts/make_paper_figs.py"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
S, O = "docs/study/", "docs/paper/figs/"; C = ["#1f5f8b", "#d1495b", "#2e8b57", "#8a6d3b"]
# Fig 1: conformal coverage under load shift (whole stream = solid, first 500 steps = dashed)
A = json.load(open(S + "adaptive_conformal_shift.json")); L = A["args"]["l_test"]
g = lambda k: [np.mean(A["by_load"][str(l)]["raw"][k]) for l in L]
mr = g("aci_margin_ratio")
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.1))
ax[0].plot(L, g("fixed"), "o-", c=C[1], label="fixed, whole stream"); ax[0].plot(L, g("fixed_transient500"), "o--", c=C[1], alpha=.6, label="fixed, first 500 steps")
ax[0].plot(L, g("aci"), "s-", c=C[0], label="ACI, whole stream"); ax[0].plot(L, g("aci_transient500"), "s--", c=C[0], alpha=.6, label="ACI, first 500 steps"); ax[0].axhline(A["target"], ls=":", c="k", lw=.8)
ax[0].set_xlabel("deployment load (x nominal)"); ax[0].set_ylabel("coverage"); ax[0].set_ylim(0, 1.02); ax[0].legend(frameon=False, fontsize=5.5, loc="lower left")
ax[1].plot(L, mr, "s-", c=C[0], label="ACI margin / fixed margin"); ax[1].plot(L, [(l / A["args"]["l_cal"]) ** 2 for l in L], ":", c="k", label=r"$(L/L_{cal})^2$")
ax[1].set_xlabel("deployment load (x nominal)"); ax[1].set_ylabel("margin ratio"); ax[1].legend(frameon=False, fontsize=6); plt.tight_layout(); plt.savefig(O + "fig_conformal_shift.pdf"); plt.close()
# Fig 2: headroom H on real days (20 weekdays, stress, demand 1.5): what each admission rule for the reference gives
def H(f):
    R = json.load(open(f)); c = lambda k: np.array([r[k][0] for r in R]); v = lambda k: np.array([r[k][2] for r in R])
    caps = [k for k in R[0] if k.startswith("edf_cap_") and v(k).mean() <= 1e-6]; o = c("oracle_lp").mean()
    cap = min(c(k).mean() for k in caps); unp = c("uncontrolled").mean(); proj = c("safe_uncontrolled").mean()
    old = min(cap, unp); fin = min(cap, proj if v("safe_uncontrolled").mean() <= 1e-6 else 1e9)
    return dict(H_unprojected_admitted=(old - o) / old * 100, H_caps_only=(cap - o) / cap * 100, H_final=(fin - o) / fin * 100, viol_unprojected=float(v("uncontrolled").mean()),
                viol_projected=float(v("safe_uncontrolled").mean()), cost_cap=cap, cost_proj_unc=proj, cost_oracle=o, clairvoyant_best_heuristic=float(np.min([c(k) for k in caps + ["safe_uncontrolled"]], axis=0).mean()))
Hs = {"Caltech": H(S + "headroom_caltech_stress_fixed.json"), "JPL": H(S + "headroom_jpl_stress_fixed.json")}
fig, ax = plt.subplots(figsize=(4.6, 2.2)); x = np.arange(2); w = .26
for i, (k, lab, col) in enumerate((("H_unprojected_admitted", "unprojected rule admitted (old script)", C[3]), ("H_caps_only", "EDF caps only", C[1]), ("H_final", "violation-free incl. projected rule (final)", C[0]))):
    ax.bar(x + (i - 1) * w, [Hs[n][k] for n in Hs], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(list(Hs)); ax.set_ylabel("headroom H (%)"); ax.legend(frameon=False, fontsize=6); plt.tight_layout(); plt.savefig(O + "fig_headroom_gating.pdf"); plt.close()
json.dump(Hs, open(S + "headroom_gating_summary.json", "w"), indent=1)
rows, res = [], []
# Fig 3: pathwise policy vs references on synthetic-stress, real Caltech, real JPL (20 held-out days each; bars = mean cost, points = individual seeds)
def jl(f):
    return json.load(open(S + f)) if os.path.exists(S + f) else None
panels = [("Synthetic stress (H=33%)", [("diff_env_poc.json", "pathwise\n(uniform)"), ("diff_env_synth_sens_ste.json", "pathwise\n(sens+STE)")], "td3_synth_stress.json", None),
          ("Real Caltech (H=6%)", [("diff_env_real_caltech.json", "pathwise\n(uniform)"), ("diff_env_real_caltech_sens_ste.json", "pathwise\n(sens+STE)")], None, ["proj_price_closed_loop_caltech.json"]),
          ("Real JPL (H=0.7%)", [("diff_env_real_jpl.json", "pathwise\n(uniform)"), ("diff_env_real_jpl_sens_ste.json", "pathwise\n(sens+STE)")], None, ["proj_price_closed_loop_jpl.json"])]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, (title, runs, td3f, qpfs) in zip(axs, panels):
    labels, vals, pts, cols = [], [], [], []
    base = None
    for f, lab in runs:
        P = jl(f)
        if P is None or not P.get("Q4_learning"): continue
        base = base or P["baselines_test"]["mean"]
        m = [v["mean"] for v in P["Q4_learning"].values()]; labels.append(lab); vals.append(np.mean(m)); pts.append(m); cols.append(C[0])
    if base is None: ax.set_title(title + " (n/a)", fontsize=7); continue
    items = [("tuned\nref", base["ref"], C[1], [base["ref"]]), ("LP\noracle", base["oracle_lp"], "#555", [base["oracle_lp"]])]
    T = jl(td3f) if td3f else None
    if T: items.append(("residual\nTD3", T["last"]["total_cost"], C[2], [T["last"]["total_cost"]]))
    items += [(l, v, c, p_) for l, v, c, p_ in zip(labels, vals, cols, pts)]
    for qf in (qpfs or []):                                   # closed-loop runs of the same networks under the exact QP projection (Table tab:pp)
        Q = jl(qf)
        if Q is None: continue
        q = [v["qp"]["mean"] for v in Q["pathwise"].values()]; items.append(("pathwise\n(exact QP)", float(np.mean(q)), C[3 % len(C)], q)); break
    for i, (l, v, c, p_) in enumerate(items):
        ax.bar(i, v, color=c); ax.plot([i] * len(p_), p_, "k.", ms=3)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=5.5); ax.set_title(title, fontsize=7); ax.set_ylim(0.8 * min(i[1] for i in items), 1.12 * max(i[1] for i in items))
axs[0].set_ylabel("mean cost / day"); plt.tight_layout(); plt.savefig(O + "fig_diffenv_poc.pdf"); plt.close()
