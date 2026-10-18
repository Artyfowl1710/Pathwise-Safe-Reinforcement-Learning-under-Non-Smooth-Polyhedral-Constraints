# Literature review and gap analysis (searched 15–16 Oct 2026; arXiv/Google-indexed only — not exhaustive)

Verified by opening the page this session: PI-TD3 (2510.12335), EV2Gym (2404.01849), TL-MAPPO (2604.03278), ACoFi (2604.18482), conformal sMPC for charging hubs (2504.00685),
real-world safe-RL/MPC/TreeC trial (2408.07435), Differentiable Power-Flow Optimization (2603.28203), GridFM conformal layer (2602.07995). Other references (OptNet, cvxpylayers, Dalal, Gibbs–Candès,
Henderson, Agarwal, ACN-Data, Ng et al.) are cited from memory/search snippets — check bibliographic details before submission.

| Area | Closest work | What it does | Implication for us |
|---|---|---|---|
| Differentiable-model RL for EV + voltage | **PI-TD3** (Orfanoudakis et al.) | actor trained by backprop through K-step known power-flow+SoC rollouts, 34/123-bus, vs TD3/SAC/PPO/oracle MPC | **Our differentiable-env idea is not novel.** Differentiators must be: tuned violation-free reference + headroom, conformal-calibrated margins, real-session multi-site transfer, rigorous stats |
| EV simulators | EV2Gym, ACN-Sim, SustainGym | EV2Gym has no power flow; finds MPC best on profit/satisfaction | A feeder-aware differentiable environment is a real gap for a *benchmark/dataset-track* contribution, if released cleanly with tuned baselines |
| Lagrangian safe MARL | TL-MAPPO (33-bus, 4 EVCS) | expectation-level safety; baselines only MAPPO/MATD3/MASAC | We should compare against a Lagrangian/CMDP baseline too |
| Conformal in control | ACoFi, conformal sMPC (charging hubs, 0.8 % gain), GridFM stratified CP | thresholds / forecast intervals / surrogate screening | No work found calibrating the LinDistFlow→AC voltage error of an action projection under load shift (not proof of absence) |
| RL vs tuned control | Ruddick et al. real-world trial | RBC≈MPC≈TreeC, RL +25 % | supports our negative result; cite as convergent evidence |
| Differentiable AC power flow | Öz et al. (DPF) | speed/batching of PF solves | not control; we use LinDistFlow rows, simpler |
| Decentralised EV charging | ADMM/dual methods (Gan–Low et al.) | radial-feeder distributed protocols | our tree lemma is a modest, known-style result; position as engineering lemma, not theory contribution |

## What to do more (ordered by value per effort)
1. **Recompute JPL headroom** with the fixed script on real data (blocks every JPL claim). 
2. **Real-data pathwise experiment**: run `diff_env_poc.py` logic on Caltech/JPL days (add `--site`); this is the single result that decides whether the paper's headline is "learning helps when headroom exists".
3. **Baselines reviewers will demand**: TD3/SAC/PPO on the same days, Lagrangian-PPO, forecast-aware MPC (arrival forecast), PI-TD3 reimplementation (it is the direct competitor).
4. **Model-mismatch test**: train through the LinDistFlow twin, evaluate in an unbalanced three-phase model (OpenDSS/pandapower); report violations with/without conformal margins (this is where F1/F2 pay off).
5. **Load-conditional conformal scores** to remove the 1.9–4.6× margin inflation.
6. **Scale/stats**: 40–60 test days, 5 seeds, third ACN site, real tariffs/CAISO, IQM + paired bootstrap.
7. **Venue**: with 1–5 done, a Q1 energy/power-systems journal (IEEE TSG / Applied Energy / Electric Power Systems Research) or e-Energy/BuildSys is plausible; a NeurIPS Datasets & Benchmarks track only if the differentiable feeder environment is released as a benchmark with tuned baselines. A*-main-track ML is not realistic on current evidence.
