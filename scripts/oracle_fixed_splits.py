"""Corrected perfect-foresight oracle (no discharge atom, tie-break, derated efficiency) on the exact held-out days of a pgd_residual_closed_loop split.
    PYTHONPATH=. python scripts/oracle_fixed_splits.py --site jpl --prev docs/study/diff_env_real_jpl_q5_random.json --out docs/study/oracle_fixed_jpl_random.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
src = open("scripts/pgd_residual_closed_loop.py").read().split("class ResMLP")[0]
exec(src)
from pifsrl.agents.baselines import RuleBasedPolicy
from pifsrl.agents import oracle as O
orig = O.solve_oracle; res = {}
O.solve_oracle = lambda e_, f_, **x: orig(e_, f_, tie_break=1e-6, eta_derate=0.0132, no_discharge=True, **x)
r = [run_episode(env, RuleBasedPolicy("oracle_lp", env, seed=1, fbase=fbase), buffer=None, seed=s, collect=False, date=DATE[s], agent=None) for s in te_seeds]
res = {"oracle_fixed_mean": float(np.mean([x["total_cost"] for x in r])), "per_day": [x["total_cost"] for x in r], "viol": float(np.mean([x["v_viol"] + x["s_viol"] for x in r])), "ref_mean": float(ref_day.mean()), "old_oracle_mean": float(orc_day.mean())}
print(res["oracle_fixed_mean"], res["ref_mean"], res["old_oracle_mean"]); json.dump(res, open(a.out, "w"), indent=1)
