"""Monolithic vs tree-DECOMPOSED safety projection (plan item: decomposed-solver benchmark; supports the IEEE-123 theory section).

Problem (P-only LinDistFlow on a random radial tree, n buses):  min_x 0.5||x-a||^2  s.t.  lo<=x<=hi,  v_i(x) >= vmin  for all i,
v_i = v0 - sum_{e in path(0->i)} r_e F_e,  F_e = sum of draws (base+x) in the subtree of e.
Decomposition: cut edges -> blocks; adjacent blocks are coupled ONLY through 2 scalars per cut (boundary voltage v_c, boundary flow F_c)
[tree lemma, verified in scripts/poc_tree_decomposition.py]. Consensus ADMM over blocks, each block a small QP (cvxpy, parametrised).
Reported: ADMM iterations to primal residual<tol, objective gap vs the monolithic solve, monolithic time, per-block solve time,
critical-path time = iters * max block time (what a parallel deployment would see). Honest caveat printed: wall time here is sequential."""
import argparse, json, os, sys, time
import numpy as np
import cvxpy as cp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def random_tree(n, rng, branch=0.3):
    parent = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        parent[i] = max(0, i - 1 - int(rng.geometric(1 - branch)) + 1) if rng.random() < 0.7 else rng.integers(0, i)
        parent[i] = min(parent[i], i - 1)
    r = rng.uniform(0.2, 1.0, n + 1) * 1e-3 * 4
    return parent, r


def path_sets(parent, n):
    anc = [[] for _ in range(n + 1)]                       # edges (indexed by child bus) on path 0->i
    for i in range(1, n + 1):
        anc[i] = anc[parent[i]] + [i]
    return anc


def mono_matrix(parent, r, n):
    anc = path_sets(parent, n)
    R = np.zeros((n + 1, n + 1))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            R[i, j] = sum(r[e] for e in set(anc[i]) & set(anc[j]))
    return R[1:, 1:]


def make_blocks(parent, n, target):
    ch = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        ch[parent[i]].append(i)
    size = np.zeros(n + 1, int); cuts = []
    for i in range(n, 0, -1):                              # parents have smaller index -> reverse order is post-order
        size[i] += 1
        if size[i] >= target and parent[i] != 0:
            cuts.append(i); size[parent[i]] += 0
        else:
            size[parent[i]] += size[i]
    cuts = sorted(set(cuts))
    root_of = np.zeros(n + 1, int)
    cutset = set(cuts)
    for i in range(1, n + 1):
        root_of[i] = i if i in cutset else root_of[parent[i]]
    root_of[0] = 0
    blocks = {}
    for i in range(1, n + 1):
        blocks.setdefault(root_of[i], []).append(i)
    return blocks, cuts


def run(n, target, seed, tol, max_it, rho, rho_v=1e4):
    rng = np.random.default_rng(seed)
    parent, r = random_tree(n, rng)
    base = rng.uniform(0.0, 1.0, n + 1) * 8.0 / n * 5; base[0] = 0.0       # p.u.-ish draws
    a = rng.uniform(0.5, 1.5, n + 1)[1:] * 20.0 / n; lo = np.zeros(n); hi = np.full(n, 3.0 * 20.0 / n)
    R = mono_matrix(parent, r, n); v0 = 1.0
    v_a = v0 - R @ (base[1:] + a); v_lo = v0 - R @ (base[1:] + lo)
    vmin = float(v_a.min() + 0.6 * (v_lo.min() - v_a.min()))
    # ---- monolithic
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - a)), [x >= lo, x <= hi, v0 - R @ (base[1:] + x) >= vmin])
    t0 = time.time(); prob.solve(solver=cp.CLARABEL); t_mono = time.time() - t0
    f_mono = prob.value; x_mono = x.value
    # ---- blocks
    blocks, cuts = make_blocks(parent, n, target)
    broots = sorted(blocks)                                  # 0 means the slack block (possibly empty of cut)
    if 0 not in blocks:
        blocks[0] = []; broots = sorted(blocks)
    # block structure: local buses S (list), root c (cut) ; children cuts attached at parent[c'] in S_parent
    bus2blk = {i: root_of for i, root_of in []}
    blk_of = np.zeros(n + 1, int)
    for k, S in blocks.items():
        for i in S: blk_of[i] = k
    kids = {k: [] for k in blocks}
    for c in cuts:
        kids[blk_of[parent[c]]].append(c)
    anc = path_sets(parent, n)
    subs = {}
    for k, S in blocks.items():
        if not S:
            subs[k] = None; continue
        idx = {b: j for j, b in enumerate(S)}
        m = len(S)
        # local shared-path resistance from block root; edges inside block (child bus in S, excluding the cut edge into root k)
        Rl = np.zeros((m, m))
        for i in S:
            ei = [e for e in anc[i] if e in idx and e != k] if k != 0 else [e for e in anc[i]]
            for j in S:
                ej = set([e for e in anc[j] if e in idx and e != k] if k != 0 else anc[j])
                Rl[idx[i], idx[j]] = sum(r[e] for e in set(ei) & ej)
        attach = [idx.get(parent[c], m) for c in kids[k]]          # index m == the slack bus (only for the slack block)
        nc = len(kids[k])
        xv = cp.Variable(m); vk = cp.Variable(); Fk = cp.Variable(); Fc = cp.Variable(nc) if nc else None; vc = cp.Variable(nc) if nc else None
        P_a = cp.Parameter(m); P_zv = cp.Parameter(); P_zF = cp.Parameter()
        P_zFc = cp.Parameter(nc) if nc else None; P_zvc = cp.Parameter(nc) if nc else None
        draw = base[S] + xv
        inj = draw
        cons = [xv >= lo[[b - 1 for b in S]], xv <= hi[[b - 1 for b in S]]]
        if nc:
            Aatt = np.zeros((m, nc))
            for q, ia in enumerate(attach):
                if ia < m: Aatt[ia, q] = 1.0
            inj = draw + Aatt @ Fc
        cons.append(Fk == cp.sum(inj))
        vloc = vk - Rl @ inj
        cons.append(vloc >= vmin)
        vloc_ext = cp.hstack([vloc, cp.reshape(vk, (1,), order='C')]) if nc else vloc
        if k == 0:
            cons.append(vk == v0)
        if nc:
            for q, c in enumerate(kids[k]):
                cons.append(vc[q] == vloc_ext[attach[q]] - r[c] * Fc[q])
        obj = 0.5 * cp.sum_squares(xv - a[[b - 1 for b in S]])
        pen = 0.5 * (rho_v * cp.square(vk - P_zv) + rho * cp.square(Fk - P_zF)) if k != 0 else 0.0
        if nc: pen += 0.5 * (rho * cp.sum_squares(Fc - P_zFc) + rho_v * cp.sum_squares(vc - P_zvc))
        subs[k] = dict(prob=cp.Problem(cp.Minimize(obj + pen), cons), xv=xv, vk=vk, Fk=Fk, Fc=Fc, vc=vc, S=S,
                       P=(P_zv, P_zF, P_zFc, P_zvc))
    # global z per cut c: (v_c, F_c); dual per copy
    zv = {c: v0 for c in cuts}; zF = {c: 0.0 for c in cuts}
    zv[0] = v0
    uown = {k: np.zeros(2) for k in blocks}; uch = {c: np.zeros(2) for c in cuts}
    x_it = np.zeros(n); tmax = 0.0; hist = []
    t_start = time.time()
    for it in range(1, max_it + 1):
        own = {}; chd = {}; tblk = []
        for k, s in subs.items():
            if s is None: continue
            pzv, pzF, pzFc, pzvc = s["P"]
            pzv.value = zv.get(k, v0) - uown[k][0]; pzF.value = zF.get(k, 0.0) - uown[k][1]
            if s["Fc"] is not None:
                pzFc.value = np.array([zF[c] - uch[c][1] for c in kids[k]]); pzvc.value = np.array([zv[c] - uch[c][0] for c in kids[k]])
            t0 = time.time(); s["prob"].solve(solver=cp.CLARABEL); tblk.append(time.time() - t0)
            own[k] = (float(s["vk"].value), float(s["Fk"].value))
            if s["Fc"] is not None:
                for q, c in enumerate(kids[k]): chd[c] = (float(s["vc"].value[q]), float(s["Fc"].value[q]))
            for j, b in enumerate(s["S"]): x_it[b - 1] = s["xv"].value[j]
        tmax = max(tmax, max(tblk))
        res = 0.0; old = {c: (zv[c], zF[c]) for c in cuts}
        for c in cuts:                                       # consensus between the child's own copy and parent's child-copy
            o = own[c]; p = chd[c]
            nv = 0.5 * ((o[0] + uown[c][0]) + (p[0] + uch[c][0])); nF = 0.5 * ((o[1] + uown[c][1]) + (p[1] + uch[c][1]))
            zv[c], zF[c] = nv, nF
            uown[c] += np.array([o[0] - nv, o[1] - nF]); uch[c] += np.array([p[0] - nv, p[1] - nF])
            res = max(res, abs(o[0] - p[0]) * 100.0, abs(o[1] - p[1]))
        hist.append(res)
        if res < tol: break
    t_dec = time.time() - t_start
    # evaluate decomposed solution on the TRUE monolithic constraints
    v_true = v0 - R @ (base[1:] + x_it)
    viol = float(max(0.0, vmin - v_true.min()))
    f_dec = 0.5 * float(np.sum((x_it - a) ** 2))
    return dict(n=n, n_blocks=len(blocks), block_target=target, iters=it, converged=bool(res < tol), mono_s=t_mono, dec_seq_s=t_dec,
                max_block_solve_s=tmax, critical_path_s=it * tmax, obj_mono=f_mono, obj_dec=f_dec,
                gap_rel=float((f_dec - f_mono) / max(abs(f_mono), 1e-12)), max_voltage_violation_pu=viol,
                mono_active=bool(v_true.min() < vmin + 1e-6), x_l2_err=float(np.linalg.norm(x_it - x_mono)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[33, 123, 400])
    ap.add_argument("--targets", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-it", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--rho-v", type=float, default=1e3)
    ap.add_argument("--out", default="docs/study/bench_tree_qp.json")
    a = ap.parse_args()
    rows = []
    for n in a.sizes:
        for tg in a.targets:
            if tg >= n: continue
            for sd in range(a.seeds):
                r = run(n, tg, sd, a.tol, a.max_it, a.rho, a.rho_v); rows.append(r)
                print(f"n={n:4d} blocks={r['n_blocks']:3d} it={r['iters']:3d} conv={r['converged']} gap={r['gap_rel']:+.2e} "
                      f"viol={r['max_voltage_violation_pu']:.1e} mono={r['mono_s']*1e3:.0f}ms crit={r['critical_path_s']*1e3:.0f}ms", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rows, open(a.out, "w"), indent=1)
    print("NOTE: critical_path = iterations x slowest block solve (parallel deployment). Sequential wall time is NOT a speedup claim.")


if __name__ == "__main__":
    main()
