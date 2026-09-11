#!/usr/bin/env bash
# Interpolate between outcome-only and full process supervision.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT=results/induced_rule/fraction.csv
for TF in 0.02 0.05 0.15 0.50 1.00; do
  for SEED in 2001 2002; do
    $PY -m src induced \
      --depth 4 --condition both --trace-fraction $TF --seed $SEED \
      --train-size 30000 --val-size 800 --probe-size 300 \
      --checkpoints 0 500 2000 4000 --device cuda:0 --out $OUT
  done
done
echo "FRACTION SWEEP DONE"
