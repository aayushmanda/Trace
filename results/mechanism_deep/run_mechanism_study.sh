#!/usr/bin/env bash
# Full mechanism-deep study: depths x seeds, GPU, real step budget.
# Settings (LR, STEPS, ...) are chosen from the D=4 learning-rate diagnostic
# recorded in results/mechanism_deep/diagnostics/ -- see SUMMARY.md.
set -uo pipefail

REPO=/home/hariguru/aayus/trace
source /home/hariguru/aayus/.venv/bin/activate
cd "$REPO"

LR="${LR:?set LR}"
STEPS="${STEPS:?set STEPS}"
TRAIN_SIZE="${TRAIN_SIZE:-10000}"
VAL_SIZE="${VAL_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_ROOT="${OUT_ROOT:-$REPO/results/mechanism_deep/runs}"
DEPTHS="${DEPTHS:-2 4 6 8}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DEVICES=(cuda:0 cuda:1 cuda:2 cuda:3)
CHECKPOINTS="${CHECKPOINTS:?set CHECKPOINTS (space separated steps)}"

MAXJOBS="${MAXJOBS:-6}"
running=0
i=0
for depth in $DEPTHS; do
  for seed in $SEEDS; do
    dev="${DEVICES[$((i % 4))]}"
    i=$((i+1))
    out="$OUT_ROOT/D${depth}_seed${seed}"
    mkdir -p "$out"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup python -m src outcome-local \
      --config configs/experiments/mechanism_deep.yaml \
      --depth "$depth" --seed "$seed" \
      --steps "$STEPS" --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
      --batch-size "$BATCH_SIZE" --lr "$LR" --grad-clip 1.0 \
      --checkpoints $CHECKPOINTS \
      --device "$dev" \
      --output "$out" > "$out/log.txt" 2>&1 &
    echo "launched depth=$depth seed=$seed dev=$dev pid=$! -> $out"
    running=$((running+1))
    if [ "$running" -ge "$MAXJOBS" ]; then
      wait -n
      running=$((running-1))
    fi
  done
done
wait
echo ALL_RUNS_DONE
