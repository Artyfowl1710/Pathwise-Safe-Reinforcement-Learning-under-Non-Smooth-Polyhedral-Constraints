"""
Turn the JSON outputs of sample.py / run_ablations.py into the figures and
tables you'd actually put in a report: training curves, a baseline comparison
bar chart, and a LaTeX table matching the shape of Table 2/3 in the proposal.

Usage
-----
    python -m pifsrl.experiments.plotting --single results/single_residential_seed0.json
    python -m pifsrl.experiments.plotting --ablation-dir results/ablations
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def plot_training_curves(single_json: str, out_dir: str):
    d = _load(single_json)
    hist = d["history"]
    ep = [h["episode"] for h in hist]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(ep, [h["total_cost"] for h in hist], marker="o", ms=3)
    axes[0, 0].set_title("Episode cost (currency units)")
    axes[0, 0].set_xlabel("episode")

    v = [h["v_viol"] + h["s_viol"] for h in hist]
    axes[0, 1].semilogy(ep, np.maximum(v, 1e-8), marker="o", ms=3, color="crimson")
    axes[0, 1].set_title("Total constraint violation (log scale)")
    axes[0, 1].set_xlabel("episode")

    if "wm_v_mae" in hist[0]:
        wm = [h.get("wm_v_mae", np.nan) for h in hist]
        axes[1, 0].semilogy(ep, wm, marker="o", ms=3, color="darkgreen")
        axes[1, 0].set_title(r"World-model voltage MAE $|v_{\hat{}} - v_{true}|$ (p.u.)")
        axes[1, 0].set_xlabel("episode")

    served = [h.get("served_frac", np.nan) * 100 for h in hist]
    axes[1, 1].plot(ep, served, marker="o", ms=3, color="darkorange")
    axes[1, 1].set_title("Service level (% of requested energy delivered)")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].set_ylim(0, 105)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_baseline_comparison(single_json: str, out_dir: str):
    d = _load(single_json)
    baselines = d.get("baselines", {})
    if not baselines:
        print("no baselines found in", single_json)
        return
    names = list(baselines.keys()) + ["SafeTD3 (ours)"]
    costs = [baselines[k]["total_cost"] for k in baselines] + [d["evals"][-1]["total_cost"]]
    served = [baselines[k]["served_frac"] * 100 for k in baselines] + [d["evals"][-1]["served_frac"] * 100]
    viol = [baselines[k]["v_viol"] + baselines[k]["s_viol"] for k in baselines] + \
        [d["evals"][-1]["v_viol"] + d["evals"][-1]["s_viol"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for ax, vals, title in zip(axes, [costs, served, viol],
                              ["Total cost (lower is better)",
                               "Service level % (higher is better)",
                               "Total violation (lower is better, log)"]):
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[2].set_yscale("symlog", linthresh=1e-3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "baseline_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def make_latex_table(single_json: str, out_dir: str):
    d = _load(single_json)
    baselines = d.get("baselines", {})
    final = d["evals"][-1]
    rows = list(baselines.items()) + [("SafeTD3 (ours)", final)]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Cost & Served (\%) & Violation & $V_{\min}$ \\",
        r"\midrule",
    ]
    for name, r in rows:
        viol = r["v_viol"] + r["s_viol"]
        lines.append(f"{name.replace('_', ' ')} & {r['total_cost']:.1f} & "
                    f"{100*r['served_frac']:.1f} & {viol:.2e} & {r['v_min']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Baseline comparison. Violations measured on the exact AC power flow.}",
             r"\label{tab:results}", r"\end{table}"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {path}")

    # markdown too, for quick viewing / the README
    md = ["| Method | Cost | Served (%) | Violation | V_min |", "|---|---|---|---|---|"]
    for name, r in rows:
        viol = r["v_viol"] + r["s_viol"]
        md.append(f"| {name} | {r['total_cost']:.1f} | {100*r['served_frac']:.1f} | "
                 f"{viol:.2e} | {r['v_min']:.4f} |")
    path_md = os.path.join(out_dir, "results_table.md")
    with open(path_md, "w") as fh:
        fh.write("\n".join(md))
    print(f"Wrote {path_md}")


# DISABLED (professor's request): IEEE-123 bus work removed from scope. Code kept commented for reference.
# def plot_qp_benchmark(bench_json: str, out_dir: str):
#     d = _load(bench_json)
#     rows = d["rows"]
#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     for feeder in sorted(set(r["feeder"] for r in rows)):
#         sub = [r for r in rows if r["feeder"] == feeder]
#         sub.sort(key=lambda r: r["batch"])
#         ax.plot([r["batch"] for r in sub], [r["fwd_bwd_ms"] for r in sub],
#                marker="o", label=f"{feeder} (fwd+bwd)")
#         ax.plot([r["batch"] for r in sub], [r["forward_ms"] for r in sub],
#                marker="s", linestyle="--", label=f"{feeder} (fwd only)")
#     ax.set_xlabel("batch size (concurrent EV sessions)")
#     ax.set_ylabel("wall time (ms)")
#     ax.set_yscale("log")
#     ax.set_title("Section 4.1: QP scalability benchmark")
#     ax.legend(fontsize=8)
#     ax.grid(alpha=0.3)
#     fig.tight_layout()
#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, "qp_benchmark.png")
#     fig.savefig(path, dpi=150)
#     plt.close(fig)
#     print(f"Wrote {path}")
#     if "verdict" in d:
#         print("\nSection 4.1 decision:", d["verdict"].get("recommendation", ""))


def plot_ablation_bars(ablation_dir: str, out_dir: str):
    files = glob.glob(os.path.join(ablation_dir, "ablation_*.json"))
    for f in files:
        d = _load(f)
        summary = d.get("summary", {})
        if not summary:
            continue
        arms = list(summary.keys())
        metrics = [k for k in next(iter(summary.values())).keys()]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            means = [summary[a][m][0] for a in arms]
            stds = [summary[a][m][1] for a in arms]
            ax.bar(arms, means, yerr=stds, capsize=4, color=plt.cm.Set2(np.linspace(0, 1, len(arms))))
            ax.set_title(m, fontsize=9)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.grid(alpha=0.3, axis="y")
        name = os.path.splitext(os.path.basename(f))[0]
        fig.suptitle(name)
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=str, default=None)
    ap.add_argument("--ablation-dir", type=str, default=None)
    # ap.add_argument("--qp-bench", ...)  disabled: IEEE-123 / QP-benchmark out of scope
    ap.add_argument("--out", type=str, default="results/figures")
    args = ap.parse_args()

    if args.single:
        plot_training_curves(args.single, args.out)
        plot_baseline_comparison(args.single, args.out)
        make_latex_table(args.single, args.out)
    if args.ablation_dir:
        plot_ablation_bars(args.ablation_dir, args.out)
    if not any([args.single, args.ablation_dir]):
        print("Nothing to plot: pass --single and/or --ablation-dir")


if __name__ == "__main__":
    main()
