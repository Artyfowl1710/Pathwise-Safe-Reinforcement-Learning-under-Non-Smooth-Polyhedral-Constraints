"""Differentiable-environment proof of concept.   PYTHONPATH=. python scripts/diff_env_poc.py --out docs/study/diff_env_poc.json

Four questions, in order (each is a pass/fail or a number, not a story):
 Q1 FIDELITY   does the torch twin (pifsrl/diffenv.py) reproduce V2GEnv's day cost for identical actions?
 Q2 GRADIENT   are its autograd gradients equal to finite differences?
 Q3 PLANNING   how close does gradient descent through the twin (perfect foresight) get to the LP oracle, and is the plan executable in V2GEnv?
 Q4 LEARNING   does pathwise policy-gradient (backprop through the twin, no critic) beat the train-tuned EDF cap on held-out days?
Data: SYNTHETIC sessions (the ACN CSVs are not in this sandbox); pass --site/--csv to rerun on real days (see REPRODUCE.md).
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode, make_components, tune_edf_cap
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=30); ap.add_argument("--test", type=int, default=20)
ap.add_argument("--iters", type=int, default=400); ap.add_argument("--plan-iters", type=int, default=600)
ap.add_argument("--stress", type=float, default=2.0, help="ev_load_scale (2.0 makes the hub constraint bind on the synthetic days)")
ap.add_argument("--demand", type=float, default=1.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--site", default="", choices=["", "caltech", "jpl"], help="real ACN sessions (needs data/<site>_local_sessions.csv); default = synthetic sessions")
ap.add_argument("--ndays", type=int, default=50, help="real mode: evenly spaced weekdays, split train/test at random (fixed seed)")
ap.add_argument("--ste-beta", type=float, default=0.0); ap.add_argument("--restore-mode", default="uniform", choices=["uniform", "sens"])
ap.add_argument("--device", default="cpu", help="cuda | cpu  (twin rollouts only; the env/LP/QP evaluations are CPU)")
ap.add_argument("--nseeds", type=int, default=3, help="independent pathwise training seeds")
ap.add_argument("--chrono", default="", help="real mode: 'YYYY' -> restrict to that year and split chronologically (first --train weekdays train, next --test test)")
ap.add_argument("--q5", type=int, default=1, help="projection-price test (ray-scaled reference; QP-projected learned policy)")
ap.add_argument("--out", default="docs/study/diff_env_poc.json")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(2)
if a.device == "cuda" and not torch.cuda.is_available(): sys.exit("--device cuda requested but torch.cuda.is_available() is False")
DEV = torch.device(a.device)

DATE = {}
if a.site:
    import datetime as dt
    from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
    csv = f"data/{a.site}_local_sessions.csv"
    alld = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
    if a.chrono:
        yd = [d for d in alld if d.startswith(a.chrono)]
        need = a.train + a.test
        if len(yd) < need: sys.exit(f"only {len(yd)} weekdays in {a.chrono}, need {need}")
        sel = [yd[i] for i in sorted(set(np.linspace(0, len(yd) - 1, need).astype(int)))]      # evenly spaced over the year, kept in time order
        DAYS_TR, DAYS_TE = sel[:a.train], sel[a.train:a.train + a.test]; days = sel
    else:
        days = [alld[i] for i in sorted(set(np.linspace(0, len(alld) - 1, a.ndays).astype(int)))]
        rs = np.random.default_rng(123).permutation(len(days)); te_idx = set(rs[:a.test].tolist())
        DAYS_TE = [days[i] for i in range(len(days)) if i in te_idx]; DAYS_TR = [days[i] for i in range(len(days)) if i not in te_idx][:a.train]
    ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
    ov["ev_load_scale"], ov["real_total_capacity_kw"] = a.stress, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * a.stress / 3, 1)
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=days[0],
                   reward_mode="dense", price_source="sce_tou_prime", demand_charge=a.demand, allow_discharge=False, allow_reactive=False, **ov)
else:
    ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", demand_charge=a.demand, allow_discharge=False, allow_reactive=False,
                   ev_load_scale=a.stress)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
cfg, S = env.cfg, env.s_base_kw
tr_seeds = list(range(1000, 1000 + (len(DAYS_TR) if a.site else a.train))); te_seeds = list(range(2000, 2000 + (len(DAYS_TE) if a.site else a.test)))
if a.site:
    DATE.update(dict(zip(tr_seeds, DAYS_TR))); DATE.update(dict(zip(te_seeds, DAYS_TE)))
t0 = time.time()
D_tr = [extract_day(env, fbase, s, date=DATE.get(s)) for s in tr_seeds]; D_te = [extract_day(env, fbase, s, date=DATE.get(s)) for s in te_seeds]
B_tr, B_te = pad_batch(D_tr, device=DEV), pad_batch(D_te, device=DEV)
print(f"extracted {len(D_tr)}+{len(D_te)} days in {time.time()-t0:.1f}s; K={env.k_max}", flush=True)
R = {"config": vars(a), "data": (f"real ACN {a.site}; train {DAYS_TR}; test {DAYS_TE}" if a.site else "synthetic sessions"), "restore_mode": a.restore_mode, "ste_beta": a.ste_beta}


class Replay:
    """Feeds a recorded (T,K) physical action sequence to V2GEnv; used for fidelity and plan execution."""
    def __init__(self, P): self.P, self.i = P, 0
    def __call__(self):
        p = self.P[min(self.i, len(self.P) - 1)]; self.i += 1; return p


def env_cost(seed, P):
    st = run_episode(env, Replay(P), buffer=None, seed=seed, collect=False, date=DATE.get(seed))
    return st


# ---------------- Q1 fidelity: random feasible-ish actions, twin vs env ----------------
class RandU(torch.nn.Module):
    def __init__(self, g): super().__init__(); self.g = g
    def forward(self, feat, t): return torch.rand(feat.shape[:2], generator=self.g, dtype=torch.float64)

g = torch.Generator().manual_seed(7)
tw = rollout_cost(B_te, RandU(g), cfg, S, cfg.demand_charge, cfg.c_unmet, return_traj=True)
errs, venv = [], []
for i, s in enumerate(te_seeds[:10]):
    st = env_cost(s, tw["p"][i].cpu().numpy())
    errs.append(abs(st["total_cost"] - float(tw["total"][i])) / max(abs(st["total_cost"]), 1e-9)); venv.append(st["v_viol"] + st["s_viol"])
R["Q1_fidelity"] = {"n_days": 10, "max_rel_err": float(max(errs)), "median_rel_err": float(np.median(errs)), "env_violation_sum": float(np.sum(venv)),
                    "note": "env total_cost includes any c_violation term; violation sum must be 0 for the comparison to be apples-to-apples"}
print("Q1", R["Q1_fidelity"], flush=True)

# ---------------- Q2 gradient check: autograd vs central finite differences ----------------
Z = (torch.randn(1, B_te["T"], B_te["K"], dtype=torch.float64) * 0.7).to(DEV)
one = pad_batch(D_te[:1], device=DEV)


class FixedZ(torch.nn.Module):
    def __init__(self, z): super().__init__(); self.z = torch.nn.Parameter(z)
    def forward(self, feat, t): return torch.sigmoid(self.z[:, t])

pol = FixedZ(Z.clone())
def f(): return rollout_cost(one, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=False)["total"].sum()
loss = f(); loss.backward(); g_auto = pol.z.grad.clone().flatten().cpu()
rng = np.random.default_rng(0); nz = torch.where(g_auto.abs() > 1e-9)[0].numpy(); idx = rng.choice(nz, 60, replace=False); rel = []
for j in idx:
    e = 1e-5; z0 = pol.z.data.clone().flatten()
    zp = z0.clone(); zp[j] += e; pol.z.data = zp.view_as(Z); fp = f().item()
    zm = z0.clone(); zm[j] -= e; pol.z.data = zm.view_as(Z); fm = f().item(); pol.z.data = z0.view_as(Z)
    fd = (fp - fm) / (2 * e); rel.append(abs(fd - g_auto[j].item()) / max(abs(fd), abs(g_auto[j].item()), 1e-6))
R["Q2_gradient"] = {"n_coords": 60, "median_rel_err": float(np.median(rel)), "frac_below_1e-4": float(np.mean(np.array(rel) < 1e-4)),
                    "note": "coordinates sampled among non-zero-gradient entries (empty slots have exactly zero gradient)", "frac_nonzero_grad": float((g_auto.abs() > 1e-9).float().mean())}
print("Q2", R["Q2_gradient"], flush=True)

# ---------------- baselines on the same days ----------------
def eval_rule(kind, seeds):
    out = []
    for s in seeds:
        pol_ = RuleBasedPolicy(kind, env, seed=s, fbase=fbase)
        st = run_episode(env, pol_, buffer=None, seed=s, collect=False, date=DATE.get(s), agent=(agent if kind.startswith("safe_") else None))
        out.append((st["total_cost"], st["served_frac"], st["v_viol"] + st["s_viol"]))
    return np.array(out)

caps = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]   # grid extended: 0.3 was the edge of the earlier grid
tab = {c: eval_rule(f"edf_cap_{c}", tr_seeds) for c in caps}
feas = [c for c in caps if tab[c][:, 2].mean() <= 1e-6] or caps
bestc = min(feas, key=lambda c: tab[c][:, 0].mean())
unc_tr = eval_rule("safe_uncontrolled", tr_seeds)
ref_kind = "safe_uncontrolled" if (unc_tr[:, 2].mean() <= 1e-6 and unc_tr[:, 0].mean() < tab[bestc][:, 0].mean()) else f"edf_cap_{bestc}"
base_te = eval_rule(ref_kind, te_seeds); unc_te = eval_rule("safe_uncontrolled", te_seeds); orc_te = eval_rule("oracle_lp", te_seeds)
tab_te = {c: eval_rule(f"edf_cap_{c}", te_seeds) for c in feas}
oracle_cap_day = np.min([tab_te[c][:, 0] for c in feas], axis=0)                 # clairvoyant per-day best violation-free cap
R["clairvoyant_cap_test_mean"] = float(oracle_cap_day.mean()); R["feasible_caps_train"] = feas; R["train_cap_costs"] = {str(c): float(tab[c][:, 0].mean()) for c in caps}
R["baselines_test"] = {"ref_kind": ref_kind, "ref": base_te[:, 0].tolist(), "safe_uncontrolled": unc_te[:, 0].tolist(), "oracle_lp": orc_te[:, 0].tolist(),
                       "ref_viol": float(base_te[:, 2].mean()), "unc_viol": float(unc_te[:, 2].mean()), "oracle_viol": float(orc_te[:, 2].mean()),
                       "mean": {"ref": float(base_te[:, 0].mean()), "safe_uncontrolled": float(unc_te[:, 0].mean()), "oracle_lp": float(orc_te[:, 0].mean())},
                       "headroom_H": float((base_te[:, 0].mean() - orc_te[:, 0].mean()) / base_te[:, 0].mean())}
print("baselines", {k: v for k, v in R["baselines_test"].items() if k not in ("ref", "safe_uncontrolled", "oracle_lp")}, flush=True)

# ---------------- Q3 gradient-based planning with perfect foresight (open loop, per-day z) ----------------
Zp = torch.zeros(B_te["B"], B_te["T"], B_te["K"], dtype=torch.float64, device=DEV, requires_grad=True)
class OpenLoop(torch.nn.Module):
    def forward(self, feat, t): return torch.sigmoid(Zp[:, t])
opt = torch.optim.Adam([Zp], lr=0.08)
for it in range(a.plan_iters):
    opt.zero_grad(); c = rollout_cost(B_te, OpenLoop(), cfg, S, cfg.demand_charge, cfg.c_unmet, restore=True, restore_mode=a.restore_mode, ste_beta=a.ste_beta)["total"]
    c.mean().backward(); opt.step()
tp = rollout_cost(B_te, OpenLoop(), cfg, S, cfg.demand_charge, cfg.c_unmet, restore=True, return_traj=True, restore_mode=a.restore_mode, ste_beta=a.ste_beta)
envp = [env_cost(s, tp["p"][i].detach().cpu().numpy()) for i, s in enumerate(te_seeds)]
R["Q3_planning"] = {"twin_cost_mean": float(tp["total"].mean()), "env_cost_mean": float(np.mean([e["total_cost"] for e in envp])),
                    "env_viol": float(np.mean([e["v_viol"] + e["s_viol"] for e in envp])), "served_frac": float(np.mean([e["served_frac"] for e in envp])),
                    "oracle_lp_mean": float(orc_te[:, 0].mean()), "ref_mean": float(base_te[:, 0].mean())}
print("Q3", R["Q3_planning"], flush=True)

# ---------------- Q4 pathwise policy gradient (causal MLP, no critic) ----------------
class MLP(torch.nn.Module):
    def __init__(self, d=10, h=48):
        super().__init__(); self.n = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
        torch.nn.init.constant_(self.n[-1].bias, 1.0)           # start near "charge at full rate" = safe_uncontrolled
    def forward(self, feat, t): return torch.sigmoid(self.n(feat).squeeze(-1))

learned = {}
for tag, restore, sd in [(f"pathwise_seed{i}", True, i) for i in range(a.nseeds)]:
    torch.manual_seed(100 + sd); pol = MLP().to(DEV); opt = torch.optim.Adam(pol.parameters(), lr=3e-3); hist = []
    for it in range(a.iters):
        opt.zero_grad(); c = rollout_cost(B_tr, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=restore, restore_mode=a.restore_mode, ste_beta=a.ste_beta)["total"]
        c.mean().backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0); opt.step(); hist.append(float(c.mean()))
        if it % 50 == 0: print(f"  [{tag}] it {it} train twin cost {hist[-1]:.1f}", flush=True)
    with torch.no_grad():
        tw = rollout_cost(B_te, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=restore, return_traj=True, restore_mode=a.restore_mode, ste_beta=a.ste_beta)
    ev = [env_cost(s, tw["p"][i].cpu().numpy()) for i, s in enumerate(te_seeds)]
    costs = np.array([e["total_cost"] for e in ev])
    learned[tag] = {"env_costs": costs.tolist(), "mean": float(costs.mean()), "viol": float(np.mean([e["v_viol"] + e["s_viol"] for e in ev])),
                    "served": float(np.mean([e["served_frac"] for e in ev])), "train_curve": hist[::25]}
    if a.q5:     # B: same trained policy, raw actions (no restoration) projected by the EXACT QP safety layer inside V2GEnv
        with torch.no_grad():
            raw = rollout_cost(B_te, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=False, return_traj=True)["p"].cpu().numpy()
        evq = [run_episode(env, Replay(raw[i]), buffer=None, seed=s_, collect=False, date=DATE.get(s_), agent=agent) for i, s_ in enumerate(te_seeds)]
        learned[tag]["qp_projected_eval"] = {"mean": float(np.mean([e["total_cost"] for e in evq])), "viol": float(np.mean([e["v_viol"] + e["s_viol"] for e in evq])),
                                              "served": float(np.mean([e["served_frac"] for e in evq])), "note": "INVALID as a test of QP projection: open-loop replay of actions computed on the unprojected twin trajectory (seeds disagree 736 vs 1039 on JPL). Use scripts/proj_price_closed_loop.py instead."}
    d = costs - base_te[:, 0]; rng = np.random.default_rng(1)
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    learned[tag]["vs_ref_mean_diff"] = float(d.mean()); learned[tag]["vs_ref_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    learned[tag]["vs_ref_rel"] = float(d.mean() / base_te[:, 0].mean())
    print("Q4", tag, {k: v for k, v in learned[tag].items() if k not in ("env_costs", "train_curve")}, flush=True)
R["Q4_learning"] = learned
if a.q5:     # A: degrade the reference -- projected-uncontrolled evaluated with the twin's ray-scaling instead of the exact QP
    class One(torch.nn.Module):
        def forward(self, feat, t): return torch.ones(feat.shape[:2], dtype=torch.float64, device=feat.device)
    q5 = {}
    for mode in ("uniform", "sens"):
        with torch.no_grad():
            tr_ = rollout_cost(B_te, One(), cfg, S, cfg.demand_charge, cfg.c_unmet, restore=True, return_traj=True, restore_mode=mode)
        ev = [env_cost(s_, tr_["p"][i].cpu().numpy()) for i, s_ in enumerate(te_seeds)]
        q5[mode] = {"mean": float(np.mean([e["total_cost"] for e in ev])), "viol": float(np.mean([e["v_viol"] + e["s_viol"] for e in ev])),
                    "served": float(np.mean([e["served_frac"] for e in ev]))}
    q5["qp_projected_uncontrolled_mean"] = float(unc_te[:, 0].mean()); q5["ref_mean"] = float(base_te[:, 0].mean())
    R["Q5_projection_price"] = q5; print("Q5", q5, flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(R, open(a.out, "w"), indent=1)
print("saved", a.out)
