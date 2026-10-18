"""Follow-up to wm_ablation.py: (1) SoC-regression diagnosis, (2) conformal-margin effect of N_theta (proposal ablation A5, one-step, no policy).
    PYTHONPATH=. python scripts/wm_ablation2.py --out docs/study/wm_ablation2.json
Margins here are EMPIRICAL (1-alpha)-quantiles of the held-out per-transition error, split by day into calibration (first half of held-out days) and test; coverage on the test half is reported.
  e_v = max_i |v_true - v_hat| (p.u.) ;  e_s = max_b (s_true - s_hat)_+ / s_max_b  (rating-normalised, one-sided: under-prediction is the unsafe direction).
SoC diagnosis: errors are reported (a) over all occupied slots, (b) only over slots that stay occupied into t+1 and do not depart in the step ('continuous').
Variants: fbase_only; residual (as in the library); residual_nosoc (dsoc forced to 0); residual_slotsoc (library residual for v,S + a per-slot SHARED MLP (soc,p,cap)->dsoc trained on continuous slots).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from pifsrl.env.v2g_env import EnvConfig
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents.replay import ReplayBuffer
from pifsrl.agents.trainer import make_components, run_episode, buffer_dims
from pifsrl.agents.safe_td3 import TD3Config
from pifsrl.models.world_model import WMConfig, WorldModel

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--stress", type=float, default=2.0); ap.add_argument("--n", type=int, default=2880); ap.add_argument("--out", default="docs/study/wm_ablation2.json"); a = ap.parse_args()
ec = EnvConfig(feeder="ieee33", horizon=96, dt_h=0.25, seed=0, reward_mode="dense", allow_discharge=False, allow_reactive=False, ev_load_scale=a.stress)
env, fbase, _, _ = make_components(ec, TD3Config(use_tightening=False), WMConfig(), "cpu")
kinds = ["edf_cap_1.0", "edf_cap_0.6", "edf_cap_0.4", "uncontrolled"]


def collect(seeds):
    buf = ReplayBuffer(200000, buffer_dims(env), "cpu")
    for i, s in enumerate(seeds): run_episode(env, RuleBasedPolicy(kinds[i % 4], env, seed=s, fbase=fbase), buffer=buf, seed=s, collect=True)
    return buf.all()


tr, te = collect(range(1000, 1030)), collect(range(2000, 2010))
smax = fbase.s_max.float()
def cont(b): return ((b["occ"] > .5) & (b["occ_n"] > .5) & (b["departing"] < .5)).float()
print("train/test", len(tr["soc"]), len(te["soc"]), "frac occupied slots continuous:", float(cont(te).sum() / (te["occ"] > .5).float().sum()), flush=True)


def pred(wm, b, slot=None, nosoc=False):
    o = wm.predict(b["act"], b["p_exo"], b["q_exo"], b["soc"], b["cap"], b["occ"], b["price"])
    soc = o["soc_hat"]
    if nosoc: soc = torch.where(b["occ"] > .5, o["soc_next"], torch.zeros_like(soc))
    if slot is not None: soc = (o["soc_next"] + slot(torch.stack([b["soc"], b["act"], b["cap"], b["occ"]], -1)).squeeze(-1) * 0.05).clamp(0, 1) * b["occ"]
    return o["v_hat"], o["s_hat"], soc


def metrics(v, s, soc, b, cal_mask):
    m_all = (b["occ"] > .5).float(); m_c = cont(b)
    ae = (soc - b["soc_next"]).abs()
    ev = (v - b["v_true"]).abs().max(1).values; es = ((b["s_true"] - s).clamp_min(0) / smax).max(1).values
    n = len(ev); h = n // 2; out = {"soc_mae_all": float((ae * m_all).sum() / m_all.sum()), "soc_mae_continuous": float((ae * m_c).sum() / m_c.sum().clamp_min(1)),
                                  "v_mae": float((v - b["v_true"]).abs().mean()), "v_max_ae": float((v - b["v_true"]).abs().max())}
    for al in (0.05, 0.01):
        qv = float(torch.quantile(ev[:h], 1 - al)); qs = float(torch.quantile(es[:h], 1 - al))
        out[f"margin_v_{int(al*100)}"] = qv; out[f"margin_s_{int(al*100)}"] = qs
        out[f"cover_v_{int(al*100)}_test_half"] = float((ev[h:] <= qv).float().mean()); out[f"cover_s_{int(al*100)}_test_half"] = float((es[h:] <= qs).float().mean())
    return out


rows = []
for variant in ("fbase_only", "residual", "residual_nosoc", "residual_slotsoc", "blackbox"):
    for sd in (a.seeds if variant != "fbase_only" else a.seeds[:1]):
        torch.manual_seed(sd); rng = np.random.default_rng(sd)
        mode = {"fbase_only": "fbase_only", "blackbox": "blackbox"}.get(variant, "residual")
        wm = WorldModel(fbase, env.feeder, WMConfig(mode=mode), allow_reactive=False)
        slot = None
        if variant == "residual_slotsoc":
            slot = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)); so = torch.optim.AdamW(slot.parameters(), 1e-3)
        idx_all = np.arange(len(tr["soc"]))[:a.n]
        for it in range(a.steps if variant != "fbase_only" else 0):
            ix = rng.choice(idx_all, 256); b = {k: v[ix] for k, v in tr.items()}
            if variant == "residual_nosoc" or variant == "residual_slotsoc":
                wm.cfg.lambda_soc = 0.0                              # grid residual only
            wm.update(b)
            if slot is not None:
                o = wm.predict(b["act"], b["p_exo"], b["q_exo"], b["soc"], b["cap"], b["occ"], b["price"], detach_residual=True)
                x = torch.stack([b["soc"], b["act"], b["cap"], b["occ"]], -1); pr = (o["soc_next"].detach() + slot(x).squeeze(-1) * 0.05).clamp(0, 1)
                mk = cont(b); loss = (((pr - b["soc_next"]) ** 2) * mk).sum() / mk.sum().clamp_min(1); so.zero_grad(); loss.backward(); so.step()
        with torch.no_grad():
            v, s, soc = pred(wm, te, slot=slot, nosoc=(variant == "residual_nosoc"))
            r = metrics(v, s, soc, te, None)
        r.update(variant=variant, seed=sd); rows.append(r)
        print(variant, sd, {k: round(x, 5) for k, x in r.items() if k in ("soc_mae_all", "soc_mae_continuous", "v_max_ae", "margin_v_1", "margin_s_1")}, flush=True)
json.dump({"config": vars(a), "rows": rows}, open(a.out, "w"), indent=1); print("saved", a.out)
