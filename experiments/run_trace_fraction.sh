#!/usr/bin/env bash
# The decisive test.  Interpolate between outcome-only and full process
# supervision while keeping BOTH readout queries in distribution, and ask where
# the trained model sits relative to the near-mixing ball the credit bound
# assumes, and whether its induced kernels compose.
set -u
cd /home/hariguru/aayus/trace
OUT=results/induced_rule_fraction.csv
for TF in 0.02 0.05 0.15 0.50 1.00; do
  for SEED in 2001 2002; do
    uv run python experiments/induced_rule.py \
      --depth 4 --condition both --trace-fraction $TF --seed $SEED \
      --train-size 30000 --val-size 800 --probe-size 300 \
      --checkpoints 0 500 2000 4000 --device cuda:0 --out $OUT
  done
done
echo "FRACTION SWEEP DONE"
