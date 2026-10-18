"""DISABLED (professor's request): this module was the Section 4.1 IEEE-33 vs IEEE-123
scalability / scope decision. IEEE-123 is out of scope, so the whole module is
commented out below (kept for reference)."""

# """
# Section 4.1: QP runtime benchmark and the primary-vs-stretch scale decision.
#
# This is deliberately a STANDALONE script with no dependency on the RL/world-
# model stack: it only needs the feeder's (R, X, M) matrices, exactly as the
# proposal specifies ("no need to wait on the rest of the physics engine").
# Run this FIRST, before committing to any 123-bus experiment -- that is the
# entire point of Section 4.1.
#
# Usage
# -----
#     python -m pifsrl.experiments.qp_benchmark --out results/qp_benchmark.json
#
# Decision rule implemented at the bottom (`decide_scope`): compares the measured
# 123-bus forward+backward time against a configurable control-loop period and
# prints/returns the primary/stretch recommendation from the proposal text.
# """
# from __future__ import annotations
#
# import argparse
# import json
# import time
# from dataclasses import asdict, dataclass
# from typing import Dict, List
#
# import numpy as np
# import torch
#
# from ..physics.feeder import ieee33, ieee123_like
# from ..safety.qp_layer import SafetyPolytope, admm_project, build_polytope
#
#
# @dataclass
# class BenchRow:
#     feeder: str
#     n_bus: int
#     n_branch: int
#     n_ev: int
#     n_constraints: int
#     batch: int
#     forward_ms: float
#     forward_std_ms: float
#     backward_ms: float
#     backward_std_ms: float
#     fwd_bwd_ms: float
#     device: str
#     warm_start: bool
#     admm_iters: float
#
#
# def _make_problem(feeder, n_ev: int, batch: int, device: str, seed: int = 0):
#     rng = np.random.default_rng(seed)
#     n = feeder.n_bus
#     buses = rng.choice(n, size=n_ev, replace=True)
#     A = np.zeros((n, n_ev))
#     A[buses, np.arange(n_ev)] = 1.0
#     p_exo = torch.tensor(-feeder.p_load[None, :] * rng.uniform(0.6, 1.0, size=(batch, 1)),
#                          dtype=torch.float32, device=device)
#     q_exo = torch.tensor(-feeder.q_load[None, :] * rng.uniform(0.6, 1.0, size=(batch, 1)),
#                          dtype=torch.float32, device=device)
#     lo = torch.full((batch, 2 * n_ev), -0.02, device=device)
#     hi = torch.full((batch, 2 * n_ev), 0.02, device=device)
#     R = torch.tensor(feeder.R, dtype=torch.float32, device=device)
#     X = torch.tensor(feeder.X, dtype=torch.float32, device=device)
#     M = torch.tensor(feeder.M, dtype=torch.float32, device=device)
#     At = torch.tensor(A, dtype=torch.float32, device=device)
#     s_max = torch.tensor(feeder.s_max, dtype=torch.float32, device=device)
#     poly = build_polytope(R, X, M, At, p_exo, q_exo, s_max, lo, hi,
#                           v_min=0.95, v_max=1.05, v0=1.03, n_cuts=6,
#                           allow_reactive=True, slack_penalty=300.0)
#     a_raw = torch.zeros(batch, 2 * n_ev, device=device, requires_grad=True)
#     return poly, a_raw, lo.shape[1] + poly.n_slack
#
#
# def bench_one(feeder, feeder_name: str, n_ev: int, batch: int, device: str,
#              admm_iters: int = 150, n_repeat: int = 8, warm_start: bool = False) -> BenchRow:
#     poly, a_raw, n_act = _make_problem(feeder, n_ev, batch, device)
#     n_real = a_raw.shape[1]
#     ns = poly.n_slack
#     if ns:
#         pad = torch.full((batch, ns), -300.0, device=device)
#         a_full = torch.cat([a_raw.detach(), pad], dim=1)
#     else:
#         a_full = a_raw.detach()
#     w = torch.ones_like(a_full)
#
#     # ---- forward-only timing (inference) ----
#     fwd_times = []
#     z0 = y0 = None
#     for r in range(n_repeat):
#         if device == "cuda":
#             torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         with torch.no_grad():
#             x, z, y, stats = admm_project(a_full, poly, w=w, iters=admm_iters,
#                                           z0=z0 if warm_start else None,
#                                           y0=y0 if warm_start else None)
#         if device == "cuda":
#             torch.cuda.synchronize()
#         fwd_times.append((time.perf_counter() - t0) * 1e3)
#         if warm_start:
#             z0, y0 = z, y
#
#     # ---- forward+backward timing (training step) ----
#     from ..safety.qp_layer import project
#     fb_times = []
#     for r in range(n_repeat):
#         a_raw2 = a_raw.detach().clone().requires_grad_(True)
#         if device == "cuda":
#             torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         out = project(a_raw2, poly, w=torch.ones_like(a_raw2), iters=admm_iters,
#                      skip_feasible=False)
#         loss = out.sum()
#         loss.backward()
#         if device == "cuda":
#             torch.cuda.synchronize()
#         fb_times.append((time.perf_counter() - t0) * 1e3)
#
#     n_constraints = poly.G.shape[1] + 2 * n_real
#     return BenchRow(feeder=feeder_name, n_bus=feeder.n_bus, n_branch=feeder.n_branch,
#                     n_ev=n_ev, n_constraints=n_constraints, batch=batch,
#                     forward_ms=float(np.mean(fwd_times)), forward_std_ms=float(np.std(fwd_times)),
#                     backward_ms=float(np.mean(fb_times) - np.mean(fwd_times)),
#                     backward_std_ms=float(np.std(fb_times)), fwd_bwd_ms=float(np.mean(fb_times)),
#                     device=device, warm_start=warm_start, admm_iters=stats["iters"])
#
#
# def run_benchmark(batches=(50, 200, 500), n_ev_frac: float = 0.5, admm_iters: int = 150,
#                   n_repeat: int = 8, seed: int = 0, device: str = "cpu") -> List[Dict]:
#     torch.manual_seed(seed)
#     feeders = {"ieee33": ieee33(), "ieee123": ieee123_like(seed=seed)}
#     rows = []
#     for name, feeder in feeders.items():
#         for batch in batches:
#             n_ev = max(4, int(feeder.n_bus * n_ev_frac))
#             row = bench_one(feeder, name, n_ev, batch, device, admm_iters, n_repeat)
#             rows.append(asdict(row))
#             print(f"  {name:9s} batch={batch:4d} n_ev={n_ev:3d} n_constr={row.n_constraints:4d} "
#                   f"| fwd {row.forward_ms:7.2f} ms | fwd+bwd {row.fwd_bwd_ms:7.2f} ms "
#                   f"| admm_iters {row.admm_iters:.0f}", flush=True)
#     return rows
#
#
# def decide_scope(rows: List[Dict], control_period_s: float = 60.0,
#                  target_batch: int = 200) -> Dict:
#     """Section 4.1 decision rule.
#
#     Compares 123-bus forward+backward time (at `target_batch`) against
#     `control_period_s` (the intended V2G dispatch interval).  Returns a
#     machine-readable verdict plus the human-readable text used in the report.
#     """
#     r123 = [r for r in rows if r["feeder"] == "ieee123" and r["batch"] == target_batch]
#     r33 = [r for r in rows if r["feeder"] == "ieee33" and r["batch"] == target_batch]
#     if not r123 or not r33:
#         return {"verdict": "insufficient_data"}
#     t123 = r123[0]["fwd_bwd_ms"] / 1000.0
#     t33 = r33[0]["fwd_bwd_ms"] / 1000.0
#     headroom = control_period_s / max(t123, 1e-9)
#     ok = t123 < control_period_s * 0.5           # require 2x headroom, not just <
#     verdict = {
#         "primary_scope": "ieee33",
#         "stretch_feasible": bool(ok),
#         "ieee33_fwd_bwd_s": t33,
#         "ieee123_fwd_bwd_s": t123,
#         "control_period_s": control_period_s,
#         "headroom_x": headroom,
#         "recommendation": (
#             f"123-bus forward+backward solve at batch={target_batch} takes "
#             f"{t123*1000:.1f} ms against a {control_period_s:.0f}s control period "
#             f"({headroom:.0f}x headroom) -> "
#             + ("proceed with 123-bus as a STRETCH target." if ok else
#                "123-bus is NOT within budget at this batch size; fall back to "
#                "batching the QP over EV subsets, or treat 123-bus as "
#                "evaluation-only / reduced-frequency, per Section 4.1's decision "
#                "rule. 33-bus remains the primary, committed deliverable "
#                "regardless of this outcome.")
#         ),
#     }
#     return verdict
#
#
# def main():
#     ap = argparse.ArgumentParser(description="Section 4.1 QP scalability benchmark")
#     ap.add_argument("--batches", type=int, nargs="+", default=[50, 200, 500])
#     ap.add_argument("--admm-iters", type=int, default=150)
#     ap.add_argument("--n-repeat", type=int, default=8)
#     ap.add_argument("--control-period-s", type=float, default=60.0,
#                     help="intended V2G dispatch interval, e.g. 60 for 1-min")
#     ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     ap.add_argument("--out", type=str, default="results/qp_benchmark.json")
#     ap.add_argument("--quicktest", action="store_true",
#                     help="tiny batches/iters, just to confirm the pipeline runs")
#     args = ap.parse_args()
#
#     if args.quicktest:
#         args.batches = [8, 16]
#         args.admm_iters = 30
#         args.n_repeat = 2
#
#     print(f"Device: {args.device}")
#     rows = run_benchmark(tuple(args.batches), admm_iters=args.admm_iters,
#                          n_repeat=args.n_repeat, device=args.device)
#     verdict = decide_scope(rows, control_period_s=args.control_period_s,
#                            target_batch=max(args.batches))
#     print("\n" + verdict.get("recommendation", "insufficient data for a verdict"))
#
#     import os
#     os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
#     with open(args.out, "w") as fh:
#         json.dump({"rows": rows, "verdict": verdict, "args": vars(args)}, fh, indent=2)
#     print(f"\nWrote {args.out}")
#
#
# if __name__ == "__main__":
#     main()
#
