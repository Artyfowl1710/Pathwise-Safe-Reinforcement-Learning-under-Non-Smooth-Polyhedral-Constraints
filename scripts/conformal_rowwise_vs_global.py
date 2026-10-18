"""Physics-only: why ONE global conformal margin is the wrong object for a feeder with heterogeneous branch ratings.
Scores: one-sided branch-flow under-estimate  e_b = max(S_AC,b - S_LinDistFlow,b, 0)  over random load/EV scenarios.
GLOBAL margin  = (1-alpha)-quantile of max_b e_b  (applied to every branch);   ROW-WISE margin_b = finite-sample (1-alpha)-quantile of e_b.
Reports, for each margin type, the fraction of each branch's thermal limit it consumes and the per-branch coverage."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pifsrl.physics.feeder import ieee33
from pifsrl.physics.powerflow import solve_bfs, solve_lindistflow

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4000); ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--load", type=float, default=1.0); ap.add_argument("--ev-kw", type=float, default=300.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="docs/study/conformal_rowwise_vs_global.json")
a = ap.parse_args()
f = ieee33(); rng = np.random.default_rng(a.seed); sb = f.s_base_mva * 1e3
n = a.n
lf = a.load * rng.uniform(0.6, 1.2, (n, 1))
p = -f.p_load[None, :] * lf; q = -f.q_load[None, :] * lf
hubs = rng.integers(0, f.n_bus, size=(n, 3))
for j in range(3):
    p[np.arange(n), hubs[:, j]] -= rng.uniform(0, 1, n) * a.ev_kw / sb
S_ac = solve_bfs(f, p, q).s_branch; _, S_lin = solve_lindistflow(f, p, q)
e = np.maximum(S_ac - S_lin, 0.0) * sb                                           # kVA, one-sided
cal, te = e[: n // 2], e[n // 2:]
lvl = min(1.0, np.ceil((len(cal) + 1) * (1 - a.alpha)) / len(cal))
m_glob = float(np.quantile(cal.max(1), 1 - a.alpha, method="higher"))
m_row = np.quantile(cal, lvl, axis=0, method="higher")
lim = f.s_max * sb
res = {"alpha": a.alpha, "global_margin_kva": m_glob, "rowwise_margin_kva_median": float(np.median(m_row)), "rowwise_margin_kva_max": float(m_row.max()),
       "limit_kva_min": float(lim.min()), "limit_kva_median": float(np.median(lim)),
       "global_margin_share_of_limit_max": float((m_glob / lim).max()), "global_margin_share_of_limit_median": float(np.median(m_glob / lim)),
       "rowwise_margin_share_of_limit_max": float((m_row / lim).max()), "rowwise_margin_share_of_limit_median": float(np.median(m_row / lim)),
       "n_branches_where_global_margin_exceeds_10pct_of_limit": int((m_glob / lim > 0.10).sum()),
       "n_branches_where_rowwise_margin_exceeds_10pct_of_limit": int((m_row / lim > 0.10).sum()),
       "rowwise_min_branch_coverage_test": float(np.mean(te <= m_row[None, :], axis=0).min()),
       "global_min_branch_coverage_test": float(np.mean(te <= m_glob, axis=0).min()), "n_branches": int(len(lim))}
# JOINT (all-rows-at-once) coverage -- the quantity a safety claim needs.  Three row-margin constructions:
m = len(lim)
lvl_b = min(1.0, np.ceil((len(cal) + 1) * (1 - a.alpha / m)) / len(cal)); m_bonf = np.quantile(cal, lvl_b, axis=0, method="higher")          # Bonferroni
q_norm = float(np.quantile((cal / lim[None, :]).max(1), lvl, method="higher")); m_norm = q_norm * lim                                       # rating-normalised max score
res.update({"joint_coverage_global": float(np.mean((te <= m_glob).all(1))), "joint_coverage_rowwise_independent": float(np.mean((te <= m_row[None, :]).all(1))),
            "joint_coverage_rowwise_bonferroni": float(np.mean((te <= m_bonf[None, :]).all(1))), "joint_coverage_rating_normalised_max": float(np.mean((te <= m_norm[None, :]).all(1))),
            "share_of_limit_max_bonferroni": float((m_bonf / lim).max()), "share_of_limit_max_normalised": float((m_norm / lim).max()), "share_of_limit_median_normalised": float(np.median(m_norm / lim)),
            "share_of_limit_median_bonferroni": float(np.median(m_bonf / lim))})
for k, v in res.items(): print(f"{k:58s} {v:.4g}" if isinstance(v, float) else f"{k:58s} {v}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
