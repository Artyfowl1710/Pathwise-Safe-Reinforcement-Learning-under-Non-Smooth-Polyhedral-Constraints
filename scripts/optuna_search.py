"""Optuna hyper-parameter search for SafeTD3 on REAL ACN days.

Protocol (one research question: "which optimiser settings make SafeTD3 converge?"):
  * TRAIN days   : the agent trains on --train-date(s)  (one at a time per trial)
  * VALIDATION   : the objective is the mean fixed-seed validation cost of the LAST --avg-evals
                   evaluations (so a lucky checkpoint cannot win) + 1e3 * any violation
  * TEST         : never touched here -- use `sample.py --acn-date <test day>` with the chosen params.
Pruning: median pruner on the validation cost after each evaluation.

    python scripts/optuna_search.py --real-data caltech --train-date 2019-10-18 \
        --trials 30 --episodes 40 --device cuda --storage sqlite:///results/optuna.db

Search space (kept small on purpose; everything else is fixed by the diagnosis):
  lr_actor, lr_critic, batch_size, expl_noise (start), expl_noise_final, lambda_mb, policy_delay,
  target_noise, tau.  reward_mode is fixed to "dense" (see RUNBOOK 'SafeTD3 diagnosis').
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import optuna


def build(args, trial_params, seed):
    from pifsrl.env.v2g_env import EnvConfig
    from pifsrl.env.sessions import recommended_env_overrides_for_acn
    from pifsrl.models.world_model import WMConfig
    from pifsrl.agents.safe_td3 import TD3Config
    from pifsrl.agents.trainer import TrainConfig
    csv = args.acn_csv or f"data/{args.real_data}_local_sessions.csv"
    ov = recommended_env_overrides_for_acn(args.real_data)
    if args.stress:
        ov["ev_load_scale"], ov["real_total_capacity_kw"] = 2.0, 100.0
    ec = EnvConfig(feeder="ieee33", horizon=192, dt_h=0.25, seed=seed, sessions_source="acn_csv",
                   acn_csv_file=csv, acn_site=args.real_data, acn_date=args.train_date,
                   price_source="sce_tou_prime", reward_mode="dense", **ov)
    tc = TD3Config(**trial_params)
    wc = WMConfig(hidden=128, mode="residual")
    trc = TrainConfig(episodes=args.episodes, warmup_episodes=args.warmup, device=args.device,
                      eval_every=args.eval_every, eval_episodes=args.eval_episodes, seed=seed,
                      keep_best=False, final_test_episodes=0, log_every=10 ** 6,
                      warmup_mode="greedy" if tc.bc_weight > 0 else "random",
                      bc_pretrain_steps=200 if tc.bc_weight > 0 else 0)
    return ec, tc, wc, trc


def objective(trial, args):
    import torch
    from pifsrl.agents import trainer
    params = dict(
        lr_actor=trial.suggest_float("lr_actor", 3e-5, 1e-3, log=True),
        lr_critic=trial.suggest_float("lr_critic", 1e-4, 1e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        expl_noise=trial.suggest_float("expl_noise", 0.10, 0.30),
        expl_noise_final=trial.suggest_float("expl_noise_final", 0.01, 0.08),
        lambda_mb=trial.suggest_categorical("lambda_mb", [0.0, 0.1, 0.5]),
        policy_delay=trial.suggest_categorical("policy_delay", [2, 4]),
        target_noise=trial.suggest_float("target_noise", 0.05, 0.3),
        tau=trial.suggest_float("tau", 0.002, 0.02, log=True),
        bc_weight=trial.suggest_float("bc_weight", 0.0, 2.0),
        bc_weight_final=trial.suggest_float("bc_weight_final", 0.0, 0.5),
    )
    ec, tc, wc, trc = build(args, params, seed=args.seed)
    # evaluate incrementally so the pruner can act: run in chunks of eval_every episodes via callback
    seen = {"n": 0}

    def cb(rec):                      # called each episode by train_site(log_fn=...)
        pass

    out = trainer.train_site(ec, tc, wc, trc, verbose=False, log_fn=cb)
    ev = out["evals"]
    for i, e in enumerate(ev):
        trial.report(e["total_cost"], i)
    last = ev[-args.avg_evals:]
    score = float(np.mean([e["total_cost"] + 1e3 * e.get("any_violation", 0.0) for e in last]))
    trial.set_user_attr("val_costs", [round(e["total_cost"], 1) for e in ev])
    trial.set_user_attr("val_served", [round(e["served_frac"], 3) for e in ev])
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-data", choices=["caltech", "jpl"], default="caltech")
    ap.add_argument("--acn-csv", default=None)
    ap.add_argument("--train-date", default="2019-10-18")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--avg-evals", type=int, default=3)
    ap.add_argument("--stress", action="store_true", help="2x charger power, 100 kW hub capacity")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study", default="safetd3")
    ap.add_argument("--out", default="results/optuna_best.json")
    a = ap.parse_args()
    study = optuna.create_study(study_name=a.study, storage=a.storage, load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study.optimize(lambda t: objective(t, a), n_trials=a.trials)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"best_value": study.best_value, "best_params": study.best_params}, open(a.out, "w"), indent=2)
    print("best:", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
