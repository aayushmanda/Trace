#!/usr/bin/env bash
# Exponent test: does credit scale as eps_rule^(D-1) in the model's own rule geometry?
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT=results/induced_rule/depth.csv
for D in 3 4 5 6; do
  for SEED in 2001 2002 2003; do
    echo "=== depth=$D seed=$SEED ==="
    $PY -m src induced \
      --depth $D --condition both --seed $SEED \
      --train-size 100000 --val-size 2000 --probe-size 400 \
      --checkpoints 0 250 500 1000 2000 4000 6000 \
      --device cuda:2 --out $OUT
  done
done
echo "DEPTH SWEEP DONE"
