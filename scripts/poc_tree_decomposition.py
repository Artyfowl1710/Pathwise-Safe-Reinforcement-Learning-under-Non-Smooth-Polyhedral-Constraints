"""PoC for the 123-bus plan: a radial LinDistFlow QP is *exactly* block-angular.
Lemma (checked numerically below): cut a radial feeder at any branch e_b; let S be the subtree below it.
 (i)  for every bus j OUTSIDE S, the voltage sensitivity R[i,j] (and X[i,j]) is the SAME for all i in S  -> the
      outside world acts on all voltages inside S through ONE scalar (the boundary voltage);
 (ii) for every branch e OUTSIDE the subtree edge set, the flow sensitivity M[e,j] is the SAME for all j in S
      -> S acts on the rest of the feeder through ONE aggregate injection (P_S, Q_S).
So each cut couples blocks through 3 scalars (v_b, P_S, Q_S) however large the subtree is."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pifsrl.physics.feeder import ieee33


def check(M, R, X):
    nE, n = M.shape; ok = 0; tot = 0; sizes = []
    for e in range(nE):
        S = np.where(M[e] > 0)[0]
        out = np.setdiff1d(np.arange(n), S)
        if len(S) < 2 or len(out) == 0: continue
        tot += 1
        i_ok = np.allclose(R[np.ix_(S, out)], R[S[0], out][None, :]) and np.allclose(X[np.ix_(S, out)], X[S[0], out][None, :])
        edges_in = np.where((M[:, S].sum(1) > 0))[0]            # edges whose downstream set meets S: ancestors + inside
        edges_out = np.setdiff1d(np.arange(nE), edges_in)
        anc = [d for d in edges_in if M[d, S].min() > 0]        # ancestors of the cut: all of S downstream
        inside = [d for d in range(nE) if d != e and M[d].any() and np.all(M[d] <= (M[e] > 0))]
        ii_ok = all(np.all(M[d, S] == M[d, S[0]]) for d in range(nE) if d not in inside)
        ok += int(i_ok and ii_ok); sizes.append(len(S))
    return ok, tot, sizes


f = ieee33()
ok, tot, sizes = check(f.M, f.R, f.X)
print(f"IEEE-33: lemma holds for {ok}/{tot} cut branches (subtree sizes 2..{max(sizes)})")
# random radial 123-bus tree with realistic branching (checks the lemma is topological, not specific to IEEE-33)
rng = np.random.default_rng(0); n = 122
parent = np.array([rng.integers(max(0, j - 6), j) if j > 0 else -1 for j in range(n)])
M = np.zeros((n, n))
for j in range(n):
    u = j
    while u >= 0:
        M[u, j] = 1; u = parent[u]
r = rng.uniform(.01, .1, n); x = rng.uniform(.01, .1, n)
R = M.T @ np.diag(r) @ M; X = M.T @ np.diag(x) @ M
ok, tot, sizes = check(M, R, X)
print(f"random 123-bus tree: lemma holds for {ok}/{tot} cut branches")
# size of the coupling problem: monolithic projection has n_var = 2*K; decomposed has sum(2*K_s) + 3*(#cuts) coupling scalars
for cuts in (2, 4, 8):
    print(f"  {cuts} cuts -> {3*cuts} coupling scalars (consensus variables) vs 2K action variables in a monolithic QP")
