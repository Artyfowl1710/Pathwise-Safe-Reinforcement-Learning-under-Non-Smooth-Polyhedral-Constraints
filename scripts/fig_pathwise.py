import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("docs/study/pgd_res_summary.json")); names = {"jpl_random": "JPL random", "jpl_chrono2019": "JPL 2019", "caltech_random": "Caltech random", "caltech_chrono2019": "Caltech 2019"}
fig, ax = plt.subplots(1, 4, figsize=(11, 3.1))
for a, (k, t) in zip(ax, names.items()):
    r = d[k]; of = json.load(open("docs/study/oracle_fixed_%s.json" % {"jpl_random":"jpl_random","jpl_chrono2019":"jpl_chrono","caltech_random":"caltech_random","caltech_chrono2019":"caltech_chrono"}[k]))["oracle_fixed_mean"]; v = [r["ref"], of] + [x["live_qp"] for x in r["runs"]]; c = ["#555", "#999", "#1b6ca8", "#1b6ca8", "#1b6ca8"]
    a.bar(range(5), v, color=c); lo = min(v) * 0.9; a.set_ylim(lo, max(v) * 1.03); a.set_xticks(range(5)); a.set_xticklabels(["ref", "LP corr.", "seed 0", "seed 1", "seed 2"], fontsize=7); a.set_title(t, fontsize=9); a.set_ylabel("$/day" if k == "jpl_random" else "")
    for i, x in enumerate(v): a.text(i, x, f"{x:.0f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout(); plt.savefig("docs/paper/figs/fig_pathwise_closedloop.pdf"); print("ok")
