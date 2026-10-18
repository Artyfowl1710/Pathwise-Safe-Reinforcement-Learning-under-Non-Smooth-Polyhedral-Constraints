"""Cimmino-projection residual pathwise policy on the SYNTHETIC stress days (same footing as the TD3 comparison): 30 training days (seeds 1000-1029), 20 held-out days (seeds 2000-2019).
Reference: EDF cap tuned on the training days.  Evaluation live in V2GEnv through the exact QP.
    PYTHONPATH=. python scripts/pgd_residual_synth.py --out docs/study/pgd_res_synth.json"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings("ignore")
import numpy as np, torch
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.trainer import run_episode, make_components
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig
from pifsrl.agents import oracle as O
from pifsrl.diffenv import extract_day, pad_batch, rollout_cost, base_fraction, project_pgd
ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=150); ap.add_argument("--nseeds", type=int, default=3); ap.add_argument("--beta-res", type=float, default=0.5); ap.add_argument("--out", required=True); a = ap.parse_args()
torch.set_num_threads(2)
ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", demand_charge=1.5, allow_discharge=False, allow_reactive=False, ev_load_scale=2.0)
env, fbase, _, agent = make_components(ec, TD3Config(use_tightening=False, qp_iters=120), WMConfig(), "cpu"); cfg, S = env.cfg, env.s_base_kw
tr_s, te_s = list(range(1000, 1030)), list(range(2000, 2020))
def live(kind, seeds):
    r = [run_episode(env, RuleBasedPolicy(kind, env, seed=1, fbase=fbase), buffer=None, seed=s, collect=False, agent=(agent if kind.startswith("safe_") else None)) for s in seeds]
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "per_day": [x["total_cost"] for x in r], "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r]))}
tune = {c: live(f"safe_edf_cap_{c}", tr_s)["mean"] for c in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)}; cap = min(tune, key=tune.get); BASE = float(cap)
R = {"tune": tune, "cap": cap, "reference": live(f"safe_edf_cap_{cap}", te_s)}; print("cap", cap, "ref", R["reference"]["mean"], flush=True)
orig = O.solve_oracle; O.solve_oracle = lambda e_, f_, **x: orig(e_, f_, tie_break=1e-6, eta_derate=0.0132, no_discharge=True, **x)
R["oracle_fixed"] = live("oracle_lp", te_s); O.solve_oracle = orig; print("oracle fixed", R["oracle_fixed"]["mean"], flush=True)
D_tr = [extract_day(env, fbase, s) for s in tr_s]; B_tr = pad_batch(D_tr)
class ResMLP(torch.nn.Module):
    def __init__(s, d=10, h=48):
        super().__init__(); s.n = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
        torch.nn.init.zeros_(s.n[-1].weight); torch.nn.init.zeros_(s.n[-1].bias)
    def forward(s, f, t): return torch.tanh(s.n(f).squeeze(-1))
class Live:
    def __init__(s, day, pol): s.b, s.pol = pad_batch([day]), pol; s.sp = 0.0; s.pk = 0.0; s.last = None; b = s.b; s.capn = float(b["rate"].sum(2).max(1).values.clamp_min(1e-9)); s.price = b["price"]; s.pm = float(s.price.max()); s.pr = (s.price.unsqueeze(2) >= s.price.unsqueeze(1)).double().mean(2)
    def __call__(s):
        b, t = s.b, env.t; K = b["K"]
        if s.last is not None: s.sp = float(s.last[:K].sum()); s.pk = max(s.pk, s.sp)
        soc = torch.as_tensor(env.soc[:K], dtype=torch.float64).unsqueeze(0); occ = b["occ"][:, t]; cph = b["cap_kwh"][:, t].clamp_min(1e-6) / S
        head = (cfg.soc_max - soc).clamp_min(0) * cph / (cfg.eta_ch * cfg.dt_h); pmax = torch.minimum(b["rate"][:, t], head) * occ
        same = (b["sid"][:, t:, :] == b["sid"][:, t:t + 1, :]) & (b["occ"][:, t:, :] > .5); tl = torch.cumprod(same.double(), 1).sum(1)
        need = (b["tgt"][:, t] - soc).clamp_min(0) * cph / cfg.eta_ch; tlh = (tl * cfg.dt_h).clamp_min(cfg.dt_h); T = b["T"]
        feat = torch.stack([occ, (s.price[:, t] / s.pm).unsqueeze(1).expand(1, K), s.pr[:, t].unsqueeze(1).expand(1, K), soc, (b["tgt"][:, t] - soc).clamp_min(0), tl / T, (need / (b["rate"][:, t].clamp_min(1e-9) * tlh)).clamp(0, 2) * occ, b["rate"][:, t] / b["rate"].max().clamp_min(1e-9), torch.full((1, K), s.sp / s.capn, dtype=torch.float64), torch.full((1, K), s.pk / s.capn, dtype=torch.float64)], 2)
        with torch.no_grad():
            d = s.pol(feat, t) if s.pol is not None else torch.zeros(1, K, dtype=torch.float64)
            u = (base_fraction(BASE, pmax, need, tl, occ, BASE * pmax.sum(1, keepdim=True)) + a.beta_res * d).clamp(0, 1); p = u * pmax
        return p[0].numpy()
_o = env.step
def evalp(pol):
    r = []
    for sd in te_s:
        lp = Live(extract_day(env, fbase, sd), pol)
        def step(act, _lp=lp, **kw): _lp.last = np.asarray(act, dtype=np.float64).copy(); return _o(act, **kw)
        env.step = step
        try: r.append(run_episode(env, lp, buffer=None, seed=sd, collect=False, agent=agent))
        finally: env.step = _o
    return {"mean": float(np.mean([x["total_cost"] for x in r])), "per_day": [x["total_cost"] for x in r], "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r]))}
R["start"] = evalp(None); print("start policy", R["start"]["mean"], flush=True); R["pathwise"] = {}
KW = dict(restore=True, restore_mode="pgd", base=BASE, beta_res=a.beta_res, budget_pu="dyn", pgd_steps=10); ref = np.array(R["reference"]["per_day"])
for sd in range(a.nseeds):
    torch.manual_seed(100 + sd); pol = ResMLP(); opt = torch.optim.Adam(pol.parameters(), lr=3e-3); h = []
    for it in range(a.iters):
        opt.zero_grad(); c = rollout_cost(B_tr, pol, cfg, S, cfg.demand_charge, cfg.c_unmet, **KW)["total"]; c.mean().backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0); opt.step(); h.append(float(c.mean().detach()))
    ev = evalp(pol); d = np.array(ev["per_day"]) - ref; ev["rel"] = float(d.mean() / ref.mean()); ev["days_better"] = int((d < 0).sum()); ev["train"] = (h[0], h[-1]); R["pathwise"][f"seed{sd}"] = ev
    print(f"seed{sd}: train {h[0]:.1f}->{h[-1]:.1f} live {ev['mean']:.1f} rel {ev['rel']:.3f} better {ev['days_better']} viol {ev['viol']:.5f}", flush=True); json.dump(R, open(a.out, "w"), indent=1)
