"""Closed-loop projection-price test (replaces the invalid open-loop replay in diff_env_poc Q5-B).
Train the pathwise MLP in the twin exactly as diff_env_poc does (ray-scaling restoration), then run the SAME network CLOSED-LOOP in V2GEnv under three projections:
  qp   : raw action u*pmax from the live env state -> exact QP safety layer (what the tuned reference uses)
  ray  : raw action -> uniform ray scaling on the day's (G,h) rows  (must equal the twin cost; this validates the live-state features)
  none : raw action, no projection (violations counted)
Also the tuned reference's rule (u=1) under qp and under ray.
    PYTHONPATH=. python scripts/proj_price_closed_loop.py --site jpl --out docs/study/proj_price_closed_loop_jpl.json
"""
import argparse, datetime as dt, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates, ACN_SITE_INFO
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost, scale_to_rows, scale_to_rows_sens

ap = argparse.ArgumentParser(); ap.add_argument("--site", default="jpl"); ap.add_argument("--train", type=int, default=30); ap.add_argument("--test", type=int, default=20)
ap.add_argument("--ndays", type=int, default=50); ap.add_argument("--iters", type=int, default=300); ap.add_argument("--nseeds", type=int, default=3); ap.add_argument("--stress", type=float, default=2.0)
ap.add_argument("--demand", type=float, default=1.5); ap.add_argument("--init-bias", type=float, default=1.0, help="final-layer bias of the policy (sigmoid(bias)=initial action fraction); 1.0 = start near full rate (original), -1.0 = start inside the feasible set");
ap.add_argument("--restore-mode", default="uniform", choices=["uniform", "sens"]); ap.add_argument("--ste-beta", type=float, default=0.0); ap.add_argument("--out", default="docs/study/proj_price_closed_loop.json"); a = ap.parse_args()
torch.set_num_threads(2)
csv = f"data/{a.site}_local_sessions.csv"
alld = [d for d in list_acn_dates(csv, a.site) if dt.date.fromisoformat(d).weekday() < 5]
days = [alld[i] for i in sorted(set(np.linspace(0, len(alld) - 1, a.ndays).astype(int)))]
rs = np.random.default_rng(123).permutation(len(days)); te_idx = set(rs[:a.test].tolist())
DTE = [days[i] for i in range(len(days)) if i in te_idx]; DTR = [days[i] for i in range(len(days)) if i not in te_idx][:a.train]
ov = recommended_env_overrides_for_acn(a.site); ov["k_max"] = 54
ov["ev_load_scale"], ov["real_total_capacity_kw"] = a.stress, round(ACN_SITE_INFO[a.site]["total_capacity_kw"] * a.stress / 3, 1)
ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=0, sessions_source="acn_csv", acn_csv_file=csv, acn_site=a.site, acn_date=days[0], reward_mode="dense",
               price_source="sce_tou_prime", demand_charge=a.demand, allow_discharge=False, allow_reactive=False, **ov)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu")
cfg, S = env.cfg, env.s_base_kw
tr_seeds, te_seeds = list(range(1000, 1000 + len(DTR))), list(range(2000, 2000 + len(DTE)))
DATE = {**dict(zip(tr_seeds, DTR)), **dict(zip(te_seeds, DTE))}
D_tr = [extract_day(env, fbase, s, date=DATE[s]) for s in tr_seeds]; D_te = [extract_day(env, fbase, s, date=DATE[s]) for s in te_seeds]
B_tr, B_te = pad_batch(D_tr), pad_batch(D_te); print("days extracted", len(D_tr), len(D_te), flush=True)


class MLP(torch.nn.Module):
    def __init__(self, d=10, h=48):
        super().__init__(); self.n = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
        torch.nn.init.constant_(self.n[-1].bias, a.init_bias)
    def forward(self, feat, t): return torch.sigmoid(self.n(feat).squeeze(-1))


class LivePolicy:
    """Called by run_episode as policy(); rebuilds the twin's features from the LIVE env state, returns the raw (unprojected) action in p.u."""
    def __init__(self, day, pol, mode):
        self.b, self.pol, self.mode = pad_batch([day]), pol, mode; self.site_prev = 0.0; self.peak = 0.0; self.last = None
        b = self.b; self.capn = float(b["rate"].sum(2).max(1).values.clamp_min(1e-9)); self.price = b["price"]; self.pmaxn = float(self.price.max())
        self.prank = (self.price.unsqueeze(2) >= self.price.unsqueeze(1)).double().mean(2)
    def __call__(self):
        b, t = self.b, env.t; K = b["K"]
        if self.last is not None:                                 # applied power of the previous step (captured from env.step)
            self.site_prev = float(self.last[:K].sum()); self.peak = max(self.peak, self.site_prev)
        soc = torch.as_tensor(env.soc[:K], dtype=torch.float64).unsqueeze(0); occ = b["occ"][:, t]
        cap_pu_h = b["cap_kwh"][:, t].clamp_min(1e-6) / S
        head = (cfg.soc_max - soc).clamp_min(0) * cap_pu_h / (cfg.eta_ch * cfg.dt_h); pmax = torch.minimum(b["rate"][:, t], head) * occ
        same = (b["sid"][:, t:, :] == b["sid"][:, t:t + 1, :]) & (b["occ"][:, t:, :] > .5); tl = torch.cumprod(same.double(), 1).sum(1)
        need = (b["tgt"][:, t] - soc).clamp_min(0) * cap_pu_h / cfg.eta_ch; t_left = (tl * cfg.dt_h).clamp_min(cfg.dt_h)
        T = b["T"]
        feat = torch.stack([occ, (self.price[:, t] / self.pmaxn).unsqueeze(1).expand(1, K), self.prank[:, t].unsqueeze(1).expand(1, K), soc, (b["tgt"][:, t] - soc).clamp_min(0), tl / T,
                            (need / (b["rate"][:, t].clamp_min(1e-9) * t_left)).clamp(0, 2) * occ, b["rate"][:, t] / b["rate"].max().clamp_min(1e-9),
                            torch.full((1, K), self.site_prev / self.capn, dtype=torch.float64), torch.full((1, K), self.peak / self.capn, dtype=torch.float64)], 2)
        with torch.no_grad():
            p = self.pol(feat, t) * pmax if self.pol is not None else pmax
            if self.mode == "ray":                                  # the SAME restoration the twin used in training
                Gt, ht = b["G"][:, t].double(), b["h"][:, t].double()
                p = scale_to_rows_sens(p, Gt, ht) if a.restore_mode == "sens" else scale_to_rows(p, Gt, ht)[0]
        return p[0].numpy()


_orig = env.step
def run_live(day, seed, pol, mode, project):
    lp = LivePolicy(day, pol, mode)
    def step(a, **kw): lp.last = np.asarray(a, dtype=np.float64).copy(); return _orig(a, **kw)
    env.step = step
    try: st = run_episode(env, lp, buffer=None, seed=seed, collect=False, date=DATE[seed], agent=(agent if project else None))
    finally: env.step = _orig
    return st


def evaluate(pol, mode, project):
    r = [run_live(D, s, pol, mode, project) for D, s in zip(D_te, te_seeds)]
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r])), "served": float(np.mean([x["served_frac"] for x in r])),
            "per_day": [x["total_cost"] for x in r]}


R = {"config": vars(a), "test_days": DTE, "train_days": DTR}
t0 = time.time()
R["rule_u1"] = {"qp": evaluate(None, "qp", True), "ray": evaluate(None, "ray", False)}                       # u = 1: the 'safe_uncontrolled' rule
print("rule u=1:", {k: (round(v["mean"], 1), round(v["served"], 3), v["viol"]) for k, v in R["rule_u1"].items()}, f"({time.time()-t0:.0f}s)", flush=True)
R["twin_ray_u1"] = float(rollout_cost(B_te, type("P", (torch.nn.Module,), {"forward": lambda s, f, t: torch.ones(f.shape[:2], dtype=torch.float64)})(), cfg, S, cfg.demand_charge, cfg.c_unmet)["total"].mean())
R["pathwise"] = {}
for sd in range(a.nseeds):
    torch.manual_seed(100 + sd); pol = MLP(); opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    for it in range(a.iters):
        opt.zero_grad(); c = rollout_cost(B_tr, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=True, restore_mode=a.restore_mode, ste_beta=a.ste_beta)["total"]; c.mean().backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0); opt.step()
        if it % 100 == 0: print(f"  seed{sd} it {it} twin train cost {float(c.mean()):.1f}", flush=True)
    with torch.no_grad(): twin_te = float(rollout_cost(B_te, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, restore=True, restore_mode=a.restore_mode, ste_beta=a.ste_beta)["total"].mean())
    out = {"twin_test_cost": twin_te, "qp": evaluate(pol, "qp", True), "ray": evaluate(pol, "ray", False)}
    R["pathwise"][f"seed{sd}"] = out
    print(f"seed{sd}: twin {twin_te:.1f} | live ray {out['ray']['mean']:.1f} (served {out['ray']['served']:.3f}) | live QP {out['qp']['mean']:.1f} (served {out['qp']['served']:.3f}, viol {out['qp']['viol']})", flush=True)
    json.dump(R, open(a.out, "w"), indent=1)
print("saved", a.out)
