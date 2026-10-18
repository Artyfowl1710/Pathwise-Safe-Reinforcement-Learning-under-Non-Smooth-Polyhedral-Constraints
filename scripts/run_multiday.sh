#!/bin/bash
# multi-day, demand-charge, stress benchmark: train 60% / val 20% / test 20% days; EDF cap tuned on train days only
# usage: scripts/run_multiday.sh <name> [extra sample.py flags]   (flags after the defaults override them)
cd "$(dirname "$0")/.."
n=$1; shift
mkdir -p logs/study3 results/study3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 sample.py --threads 1 --mode single --real-data caltech --multiday 20 --episodes 60 --warmup 6 --log-every 5 --eval-every 5 --eval-episodes 4 --device cpu --seed 0 --stress --demand 1.5 --reward-mode dense --lambda-mb 0 --warmup-mode greedy --bc-pretrain 300 --out results/study3/$n --log-dir logs/study3/$n "$@" > logs/study3/$n.out 2>&1
