#!/usr/bin/env bash
# Full experiment grid for the Physics-Informed Federated Safe RL V2G project.
#
# Pattern: quicktest -> unit tests -> a short
# dry run to measure wall-clock time per episode -> the real multi-seed runs.
# Safe to background: `nohup bash scripts/run_all.sh > run_all.log 2>&1 &`
#
# Override any of these via environment variables, e.g.:
#   DEVICE=cuda EPISODES=300 SEEDS="0 1 2 3 4" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
EPISODES="${EPISODES:-300}"
ABLATION_EPISODES="${ABLATION_EPISODES:-200}"
FED_ROUNDS="${FED_ROUNDS:-60}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-results}"

mkdir -p "$OUT" "$OUT/ablations" "$OUT/figures"

echo "=================================================================="
echo "STEP 0: quicktest (must pass before anything else runs)"
echo "=================================================================="
python sample.py --quicktest

echo "=================================================================="
echo "STEP 0.5: unit tests"
echo "=================================================================="
python -m pytest tests/ -q

# (STEP 1, the Section 4.1 QP benchmark / IEEE-123 scope decision, was removed: 123-bus is out of scope)

echo "=================================================================="
echo "STEP 2: dry run (15 episodes) to estimate wall-clock time"
echo "=================================================================="
START=$(date +%s)
python sample.py --mode single --archetype residential --episodes 15 \
    --device "$DEVICE" --seed 999 --out /tmp/dryrun
END=$(date +%s)
PER_EP=$(( (END - START) / 15 ))
EST_TOTAL=$(( PER_EP * EPISODES ))
echo "Dry run: ${PER_EP}s/episode -> estimated ${EST_TOTAL}s (~$((EST_TOTAL/60)) min) for a full ${EPISODES}-episode run"

echo "=================================================================="
echo "STEP 3: single-site training, multiple seeds, multiple archetypes"
echo "=================================================================="
for ARCH in residential workplace_caltech fleet_depot; do
    for SEED in $SEEDS; do
        echo "--- archetype=$ARCH seed=$SEED ---"
        python sample.py --mode single --archetype "$ARCH" --episodes "$EPISODES" \
            --device "$DEVICE" --seed "$SEED" --out "$OUT"
    done
done

echo "=================================================================="
echo "STEP 3.5: REAL ACN-Data runs (only if data/*.csv are present -- see datasets.md)"
echo "=================================================================="
if [ -f "data/caltech_local_sessions.csv" ]; then
    echo "--- real Caltech, multi-day (real equivalent of multi-seed) ---"
    python sample.py --mode single --real-data caltech --n-real-days "${N_REAL_DAYS:-5}" \
        --episodes "$EPISODES" --device "$DEVICE" --seed 0 --out "$OUT"
else
    echo "data/caltech_local_sessions.csv not found -- skipping real-data runs. "
    echo "See datasets.md section 5 to place your own copy."
fi
if [ -f "data/jpl_local_sessions.csv" ]; then
    echo "--- real JPL, multi-day ---"
    python sample.py --mode single --real-data jpl --n-real-days "${N_REAL_DAYS:-5}" \
        --episodes "$EPISODES" --device "$DEVICE" --seed 0 --out "$OUT"
fi
if [ -f "data/caltech_local_sessions.csv" ] && [ -f "data/jpl_local_sessions.csv" ]; then
    echo "--- real federated: Caltech + JPL, FedProx vs FedAvg ---"
    python sample.py --mode federated --real-federated --rounds "$FED_ROUNDS" --compare \
        --device "$DEVICE" --out "$OUT"
fi

echo "=================================================================="
echo "STEP 4: federated learning, FedProx vs FedAvg vs local-only"
echo "=================================================================="
python sample.py --mode federated --rounds "$FED_ROUNDS" --compare \
    --device "$DEVICE" --out "$OUT"

echo "=================================================================="
echo "STEP 5: Section 7.2 ablations (4 ablations x seeds)"
echo "=================================================================="
python sample.py --mode ablation --episodes "$ABLATION_EPISODES" \
    --seeds $SEEDS --device "$DEVICE" --out "$OUT/ablations"

echo "=================================================================="
echo "STEP 6: figures and tables"
echo "=================================================================="
for f in "$OUT"/single_*.json; do
    [ -e "$f" ] || continue
    python -m pifsrl.experiments.plotting --single "$f" --out "$OUT/figures"
done
python -m pifsrl.experiments.plotting --ablation-dir "$OUT/ablations" --out "$OUT/figures"

echo "=================================================================="
echo "ALL DONE. Results in $OUT/, figures in $OUT/figures/"
echo "=================================================================="
