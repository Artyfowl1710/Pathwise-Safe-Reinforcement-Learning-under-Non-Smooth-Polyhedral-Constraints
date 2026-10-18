"""Upgrade 1: unrolled orthogonal-projection restoration (project_pgd) + residual-around-reference policy, evaluated CLOSED-LOOP under the exact QP.
Train in the twin: u = clip(base_fraction(ref rule) + beta*tanh(MLP(x))), restoration = project_pgd (10 unrolled orthogonal row projections + ray-scaling certificate).
Evaluate the trained network live in V2GEnv under (a) the exact QP safety layer, (b) the same pgd restoration as in training.
The tuned reference and LP-oracle per-day costs are read from the previous result file for the same split (--prev).
    PYTHONPATH=. python scripts/pgd_residual_closed_loop.py --site jpl --prev docs/study/diff_env_real_jpl_q5_random.json --out docs/study/pgd_res_jpl_random.json
    PYTHONPATH=. python scripts/pgd_residual_closed_loop.py --site caltech --chrono 2019 --prev docs/study/diff_env_real_caltech_chrono2019.json --out docs/study/pgd_res_caltech_chrono2019.json"""
import argparse, datetime as dt, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost, project_pgd, base_fraction

ap = argparse.ArgumentParser(); ap.add_argument("--site", default="jpl"); ap.add_argument("--train", type=int, default=30); ap.add_argument("--test", type=int, default=20)
ap.add_argument("--ndays", type=int, default=50); ap.add_argument("--chrono", default=""); ap.add_argument("--iters", type=int, default=300); ap.add_argument("--nseeds", type=int, default=3)
ap.add_argument("--stress", type=float, default=2.0); ap.add_argument("--demand", type=float, default=1.5); ap.add_argument("--beta-res", type=float, default=0.5)
ap.add_argument("--pgd-steps", type=int, default=10); ap.add_argument("--ste-beta", type=float, default=0.0); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--prev", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
torch.set_num_threads(2)
csv = f"data/{a.site}_local_sessions.csv"
alld = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
if a.chrono:
    yd = [d for d in alld if d.startswith(a.chrono)]; need_ = a.train + a.test
    sel = [yd[i] for i in sorted(set(np.linspace(0, len(yd) - 1, need_).astype(int)))]; DTR, DTE = sel[:a.train], sel[a.train:a.train + a.test]; days = sel
else:
    days = [alld[i] for i in sorted(set(np.linspace(0, len(alld) - 1, a.ndays).astype(int)))]
    rs = np.random.default_rng(123).permutation(len(days)); te_idx = set(rs[:a.test].tolist())
    DTE = [days[i] for i in range(len(days)) if i in te_idx]; DTR = [days[i] for i in range(len(days)) if i not in te_idx][:a.train]
prev = json.load(open(a.prev)); BT = prev["baselines_test"]; ref_kind = BT["ref_kind"]; ref_day = np.array(BT["ref"]); orc_day = np.array(BT["oracle_lp"])
BASE = "full" if ref_kind == "safe_uncontrolled" else float(ref_kind.split("_")[-1])
ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
ov["ev_load_scale"], ov["real_total_capacity_kw"] = a.stress, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * a.stress / 3, 1)
ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=days[0], reward_mode="dense",
               price_source="sce_tou_prime", demand_charge=a.demand, allow_discharge=False, allow_reactive=False, **ov)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
cfg, S = env.cfg, env.s_base_kw
BUD = None if BASE == "full" else BASE * cfg.real_total_capacity_kw / S
tr_seeds, te_seeds = list(range(1000, 1000 + len(DTR))), list(range(2000, 2000 + len(DTE)))
DATE = {**dict(zip(tr_seeds, DTR)), **dict(zip(te_seeds, DTE))}
D_tr = [extract_day(env, fbase, s, date=DATE[s]) for s in tr_seeds]; D_te = [extract_day(env, fbase, s, date=DATE[s]) for s in te_seeds]
B_tr, B_te = pad_batch(D_tr), pad_batch(D_te); print("days", len(D_tr), len(D_te), "base", BASE, flush=True)
KW = dict(restore=True, restore_mode="pgd", ste_beta=a.ste_beta, base=BASE, beta_res=a.beta_res, budget_pu=BUD, pgd_steps=a.pgd_steps)


class ResMLP(torch.nn.Module):
    """delta = tanh(net(x)); last layer initialised to zero, so the policy STARTS AT the reference rule."""
    def __init__(self, d=10, h=48):
        super().__init__(); self.n = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
        torch.nn.init.zeros_(self.n[-1].weight); torch.nn.init.zeros_(self.n[-1].bias)
    def forward(self, feat, t): return torch.tanh(self.n(feat).squeeze(-1))


class LivePolicy:
    def __init__(self, day, pol, mode):
        self.b, self.pol, self.mode = pad_batch([day]), pol, mode; self.site_prev = 0.0; self.peak = 0.0; self.last = None
        b = self.b; self.capn = float(b["rate"].sum(2).max(1).values.clamp_min(1e-9)); self.price = b["price"]; self.pmaxn = float(self.price.max())
        self.prank = (self.price.unsqueeze(2) >= self.price.unsqueeze(1)).double().mean(2)
    def __call__(self):
        b, t = self.b, env.t; K = b["K"]
        if self.last is not None: self.site_prev = float(self.last[:K].sum()); self.peak = max(self.peak, self.site_prev)
        soc = torch.as_tensor(env.soc[:K], dtype=torch.float64).unsqueeze(0); occ = b["occ"][:, t]; cap_pu_h = b["cap_kwh"][:, t].clamp_min(1e-6) / S
        head = (cfg.soc_max - soc).clamp_min(0) * cap_pu_h / (cfg.eta_ch * cfg.dt_h); pmax = torch.minimum(b["rate"][:, t], head) * occ
        same = (b["sid"][:, t:, :] == b["sid"][:, t:t + 1, :]) & (b["occ"][:, t:, :] > .5); tl = torch.cumprod(same.double(), 1).sum(1)
        need = (b["tgt"][:, t] - soc).clamp_min(0) * cap_pu_h / cfg.eta_ch; t_left = (tl * cfg.dt_h).clamp_min(cfg.dt_h); T = b["T"]
        feat = torch.stack([occ, (self.price[:, t] / self.pmaxn).unsqueeze(1).expand(1, K), self.prank[:, t].unsqueeze(1).expand(1, K), soc, (b["tgt"][:, t] - soc).clamp_min(0), tl / T,
                            (need / (b["rate"][:, t].clamp_min(1e-9) * t_left)).clamp(0, 2) * occ, b["rate"][:, t] / b["rate"].max().clamp_min(1e-9),
                            torch.full((1, K), self.site_prev / self.capn, dtype=torch.float64), torch.full((1, K), self.peak / self.capn, dtype=torch.float64)], 2)
        with torch.no_grad():
            delta = self.pol(feat, t) if self.pol is not None else torch.zeros(1, K, dtype=torch.float64)
            u = (base_fraction(BASE, pmax, need, tl, occ, BUD) + a.beta_res * delta).clamp(0, 1); p = u * pmax
            if self.mode == "pgd": p, _ = project_pgd(p, b["G"][:, t].double(), b["h"][:, t].double(), torch.zeros_like(pmax), pmax, n_steps=a.pgd_steps)
        return p[0].numpy()


_orig = env.step
def run_live(day, seed, pol, mode, project):
    lp = LivePolicy(day, pol, mode)
    def step(act, **kw): lp.last = np.asarray(act, dtype=np.float64).copy(); return _orig(act, **kw)
    env.step = step
    try: st = run_episode(env, lp, buffer=None, seed=seed, collect=False, date=DATE[seed], agent=(agent if project else None))
    finally: env.step = _orig
    return st

def evaluate(pol, mode, project):
    r = [run_live(D, s, pol, mode, project) for D, s in zip(D_te, te_seeds)]
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r])), "served": float(np.mean([x["served_frac"] for x in r])), "per_day": [x["total_cost"] for x in r]}

def vs_ref(per_day):
    d = np.array(per_day) - ref_day; rng = np.random.default_rng(1); bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return {"diff_mean": float(d.mean()), "rel": float(d.mean() / ref_day.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "days_better": int((d < 0).sum())}

R = {"config": vars(a), "ref_kind": ref_kind, "ref_mean": float(ref_day.mean()), "oracle_lp_mean": float(orc_day.mean()), "test_days": DTE, "train_days": DTR}
R["start_policy_qp"] = evaluate(None, "qp", True)                       # delta = 0 : the reference rule itself, executed live (sanity: should equal ref_mean)
print("base rule live under QP:", round(R["start_policy_qp"]["mean"], 1), "tuned ref:", round(ref_day.mean(), 1), flush=True)
R["pathwise"] = {}
for sd in range(a.nseeds):
    torch.manual_seed(100 + sd); pol = ResMLP(); opt = torch.optim.Adam(pol.parameters(), lr=a.lr); hist = []; t0 = time.time()
    for it in range(a.iters):
        opt.zero_grad(); c = rollout_cost(B_tr, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, **KW)["total"]; c.mean().backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0); opt.step(); hist.append(float(c.mean()))
        if it % 50 == 0: print(f"  seed{sd} it {it} twin train cost {hist[-1]:.1f} ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad(): twin_te = float(rollout_cost(B_te, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, **KW)["total"].mean())
    out = {"twin_test_cost": twin_te, "train_start": hist[0], "train_end": hist[-1], "qp": evaluate(pol, "qp", True), "pgd": evaluate(pol, "pgd", False)}
    out["qp"]["vs_ref"] = vs_ref(out["qp"]["per_day"]); R["pathwise"][f"seed{sd}"] = out
    print(f"seed{sd}: train {hist[0]:.1f}->{hist[-1]:.1f} | twin test {twin_te:.1f} | live pgd {out['pgd']['mean']:.1f} | live QP {out['qp']['mean']:.1f} (served {out['qp']['served']:.3f}, viol {out['qp']['viol']}) vs ref {out['qp']['vs_ref']}", flush=True)
    json.dump(R, open(a.out, "w"), indent=1)
print("saved", a.out)
