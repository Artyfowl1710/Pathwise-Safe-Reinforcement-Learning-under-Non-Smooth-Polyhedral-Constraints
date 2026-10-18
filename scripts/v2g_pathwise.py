"""Upgrade 2: pathwise (critic-free) policy learning through the BIDIRECTIONAL differentiable twin.
Policy: u = clip(base_fraction(tuned non-discharging reference) + beta * tanh(MLP(x)), -1, 1); zero-initialised head, so training starts AT the reference.
Evaluation: every number is a SIMULATOR (V2GEnv, exact AC, exact QP safety layer) cost.  Compared on the same held-out days with
  (a) the best non-discharging reference tuned on the training days, (b) safe_v2g_greedy, (c) hindsight LP oracle WITH discharge, (d) hindsight LP WITHOUT discharge.
    PYTHONPATH=. python scripts/v2g_pathwise.py --site synthetic --out docs/study/v2g_pathwise_synth.json
    PYTHONPATH=. python scripts/v2g_pathwise.py --site jpl --chrono 2019 --prev docs/study/diff_env_real_jpl_chrono2019.json --out docs/study/v2g_pathwise_jpl_2019.json"""
import argparse, datetime as dt, json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
warnings.filterwarnings("ignore")
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.agents import oracle as O
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost_v2g, base_fraction

ap = argparse.ArgumentParser(); ap.add_argument("--site", default="synthetic"); ap.add_argument("--chrono", default="2019"); ap.add_argument("--prev", default="")
ap.add_argument("--ntrain", type=int, default=30); ap.add_argument("--ntest", type=int, default=8); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--nseeds", type=int, default=3)
ap.add_argument("--stress", type=float, default=1.0); ap.add_argument("--demand", type=float, default=0.0); ap.add_argument("--beta-res", type=float, default=1.0); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--small", action="store_true", help="5-feature policy (price level/rank, SoC, deficit, urgency) instead of all 11"); ap.add_argument("--opt", default="lsgd", choices=["lsgd", "adam"]); ap.add_argument("--pgd-steps", type=int, default=10); ap.add_argument("--skip-oracle", action="store_true"); ap.add_argument("--out", required=True); a = ap.parse_args()
torch.set_num_threads(2)
if a.site == "synthetic":
    ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=True, allow_reactive=False, ev_load_scale=a.stress, demand_charge=a.demand)
    tr_s = list(range(1000, 1000 + a.ntrain)); te_s = list(range(2000, 2000 + a.ntest)); DATE = {s: None for s in tr_s + te_s}
else:
    from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
    csv = f"data/{a.site}_local_sessions.csv"
    alld = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
    yd = [d for d in alld if d.startswith(a.chrono)]; need_ = a.ntrain + a.ntest
    sel = [yd[i] for i in sorted(set(np.linspace(0, len(yd) - 1, need_).astype(int)))]; DTR, DTE = sel[:a.ntrain], sel[a.ntrain:a.ntrain + a.ntest]
    ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
    ov["ev_load_scale"], ov["real_total_capacity_kw"] = 2.0, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * 2.0 / 3, 1)
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=DTR[0], reward_mode="dense",
                   price_source="sce_tou_prime", demand_charge=1.5, allow_discharge=True, allow_reactive=False, **ov)
    tr_s = list(range(1000, 1000 + len(DTR))); te_s = list(range(2000, 2000 + len(DTE))); DATE = {**dict(zip(tr_s, DTR)), **dict(zip(te_s, DTE))}
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
cfg, S = env.cfg, env.s_base_kw
print("env ready; train", len(tr_s), "test", len(te_s), flush=True)


def live_rule(kind, seeds, project=True):
    r = [run_episode(env, RuleBasedPolicy(kind, env, seed=1, fbase=fbase), buffer=None, seed=s, collect=False, date=DATE[s], agent=(agent if kind.startswith("safe_") else None)) for s in seeds]
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r])), "served": float(np.mean([x["served_frac"] for x in r])), "per_day": [x["total_cost"] for x in r]}


R = {"config": vars(a), "test_days": [DATE[s] for s in te_s]}
# ---- (a) tuned non-discharging reference: best of a cap grid on the TRAINING days, then evaluated on the test days
cands = ["safe_uncontrolled", "safe_edf_cap_1.0", "safe_edf_cap_0.8", "safe_edf_cap_0.6", "safe_edf_cap_0.5", "safe_edf_cap_0.4", "safe_tou_greedy"]
if a.prev:
    pv = json.load(open(a.prev))["baselines_test"]; rk = pv["ref_kind"]; cands = [("safe_" + rk) if not rk.startswith("safe_") else rk]
tune = {c: live_rule(c, tr_s)["mean"] for c in cands}; best = min(tune, key=tune.get); R["ref_tuning_train"] = tune; R["ref_kind"] = best
R["reference"] = live_rule(best, te_s); print("reference", best, round(R["reference"]["mean"], 2), flush=True)
R["safe_v2g_greedy"] = live_rule("safe_v2g_greedy", te_s); print("safe_v2g_greedy", round(R["safe_v2g_greedy"]["mean"], 2), flush=True)
BASE = "full" if best == "safe_uncontrolled" else float(best.split("_")[-1])
# ---- (c,d) hindsight LP with and without discharge (derated efficiency, tie-break), executed in the simulator
if not a.skip_oracle:
    orig = O.solve_oracle
    for name, kw in (("oracle_lp_discharge", dict(tie_break=1e-6, eta_derate=0.0132)), ("oracle_lp_nodischarge", dict(tie_break=1e-6, eta_derate=0.0132, no_discharge=True))):
        O.solve_oracle = lambda env_, fb_, _k=kw, **x: orig(env_, fb_, **_k, **x)
        try: R[name] = live_rule("oracle_lp", te_s)
        finally: O.solve_oracle = orig
        print(name, round(R[name]["mean"], 2), flush=True)
    json.dump(R, open(a.out, "w"), indent=1)
# ---- pathwise training on the bidirectional twin
D_tr = [extract_day(env, fbase, s, date=DATE[s]) for s in tr_s]; B_tr = pad_batch(D_tr); BUD = None if BASE == "full" else BASE * cfg.real_total_capacity_kw / S if a.site != "synthetic" else (None if BASE == "full" else BASE * float(B_tr["rate"].sum(2).max()))
if a.site == "synthetic" and BASE != "full": BUD = BASE * float(B_tr["rate"].sum(2).max(1).values.mean())


class ResMLP(torch.nn.Module):
    def __init__(self, d=11, h=48):
        super().__init__(); self.cols = [1, 2, 3, 4, 6] if a.small else list(range(11)); d = len(self.cols); self.n = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
        torch.nn.init.normal_(self.n[-1].weight, 0.0, 0.02); torch.nn.init.zeros_(self.n[-1].bias)   # small RANDOM head: an exactly-zero head puts every unwanted slot at p=0, where clamp_min has zero gradient on both sides
    def forward(self, feat, t): return torch.tanh(self.n(feat[..., self.cols]).squeeze(-1))


class LivePolicy:
    def __init__(self, day, pol):
        self.b, self.pol = pad_batch([day]), pol; self.site_prev = 0.0; self.peak = 0.0; self.last = None
        b = self.b; self.capn = float(b["rate"].sum(2).max(1).values.clamp_min(1e-9)); self.price = b["price"]; self.pmaxn = float(self.price.max())
        self.prank = (self.price.unsqueeze(2) >= self.price.unsqueeze(1)).double().mean(2)
    def __call__(self):
        b, t = self.b, env.t; K = b["K"]
        if self.last is not None: self.site_prev = float(self.last[:K].sum()); self.peak = max(self.peak, self.site_prev)
        lo, hi = env.action_bounds(); lo, hi = torch.as_tensor(lo[:K]).unsqueeze(0), torch.as_tensor(hi[:K]).unsqueeze(0)
        soc = torch.as_tensor(env.soc[:K], dtype=torch.float64).unsqueeze(0); occ = b["occ"][:, t]; cap_pu_h = b["cap_kwh"][:, t].clamp_min(1e-6) / S
        same = (b["sid"][:, t:, :] == b["sid"][:, t:t + 1, :]) & (b["occ"][:, t:, :] > .5); tl = torch.cumprod(same.double(), 1).sum(1)
        need = (b["tgt"][:, t] - soc).clamp_min(0) * cap_pu_h / cfg.eta_ch; t_left = (tl * cfg.dt_h).clamp_min(cfg.dt_h); T = b["T"]
        feat = torch.stack([occ, (self.price[:, t] / self.pmaxn).unsqueeze(1).expand(1, K), self.prank[:, t].unsqueeze(1).expand(1, K), soc, (b["tgt"][:, t] - soc).clamp_min(0), tl / T,
                            (need / (b["rate"][:, t].clamp_min(1e-9) * t_left)).clamp(0, 2) * occ, b["rate"][:, t] / b["rate"].max().clamp_min(1e-9),
                            torch.full((1, K), self.site_prev / self.capn, dtype=torch.float64), torch.full((1, K), self.peak / self.capn, dtype=torch.float64),
                            (-lo) / b["rate"][:, t].clamp_min(1e-9)], 2)
        with torch.no_grad():
            delta = self.pol(feat, t) if self.pol is not None else torch.zeros(1, K, dtype=torch.float64)
            u = (base_fraction(BASE, hi, need, tl, occ, BUD) + a.beta_res * delta).clamp(-1, 1)
            p = torch.where(u >= 0, u * hi, u * (-lo))
        return p[0].numpy()


_orig = env.step
def run_live(day, seed, pol):
    lp = LivePolicy(day, pol)
    def step(act, **kw): lp.last = np.asarray(act, dtype=np.float64).copy(); return _orig(act, **kw)
    env.step = step
    try: st = run_episode(env, lp, buffer=None, seed=seed, collect=False, date=DATE[seed], agent=agent)
    finally: env.step = _orig
    return st


def evaluate(pol):
    r = [run_live(extract_day(env, fbase, s, date=DATE[s]), s, pol) for s in te_s]
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r])), "served": float(np.mean([x["served_frac"] for x in r])), "per_day": [x["total_cost"] for x in r]}


R["start_policy"] = evaluate(None); print("start policy (delta=0) live:", round(R["start_policy"]["mean"], 2), "reference:", round(R["reference"]["mean"], 2), flush=True)
refd = np.array(R["reference"]["per_day"]); R["pathwise"] = {}
for sd in range(a.nseeds):
    torch.manual_seed(100 + sd); pol = ResMLP(); hist = []; t0 = time.time(); eta = 1e-2
    F = lambda: rollout_cost_v2g(B_tr, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, base=BASE, beta_res=a.beta_res, budget_pu=BUD, pgd_steps=a.pgd_steps, smooth_ind=True)["total"].mean()
    # normalised-gradient descent with backtracking: the pathwise gradient is exact locally (checked against finite differences) but the cost is stiff, so Adam steps overshoot
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    for it in range(a.iters):
        if a.opt == 'adam':
            opt.zero_grad(); f0 = F(); f0.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0); opt.step(); hist.append(float(f0.detach()))
            if it % 25 == 0: print(f'  seed{sd} it {it} twin train {hist[-1]:.1f} ({time.time()-t0:.0f}s)', flush=True)
            continue
        for q in pol.parameters(): q.grad = None
        f0 = F(); f0.backward(); gs = [q.grad.clone() for q in pol.parameters()]; gn = torch.sqrt(sum((g ** 2).sum() for g in gs)).item() + 1e-12; th = [q.detach().clone() for q in pol.parameters()]; f0v = float(f0.detach())
        ok = False
        for _ in range(8):
            with torch.no_grad():
                for q, t_, g in zip(pol.parameters(), th, gs): q.copy_(t_ - eta * g / gn)
                f1 = float(F())
            if f1 < f0v: ok = True; eta = min(eta * 1.5, 0.2); break
            eta *= 0.5
        if not ok:
            with torch.no_grad():
                for q, t_ in zip(pol.parameters(), th): q.copy_(t_)
            f1 = f0v
        hist.append(f1 if ok else f0v)
        if it % 25 == 0: print(f"  seed{sd} it {it} twin train {hist[-1]:.1f} eta {eta:.4f} ({time.time()-t0:.0f}s)", flush=True)
    ev = evaluate(pol); d = np.array(ev["per_day"]) - refd; rng = np.random.default_rng(1); bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    ev["vs_ref"] = {"diff_mean": float(d.mean()), "rel": float(d.mean() / refd.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "days_better": int((d < 0).sum())}
    ev["train_start"], ev["train_end"] = hist[0], hist[-1]; R["pathwise"][f"seed{sd}"] = ev
    print(f"seed{sd}: train {hist[0]:.1f}->{hist[-1]:.1f} | live {ev['mean']:.1f} served {ev['served']:.3f} viol {ev['viol']:.4f} | vs ref {ev['vs_ref']}", flush=True)
    json.dump(R, open(a.out, "w"), indent=1)
print("saved", a.out)
