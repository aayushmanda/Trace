#!/usr/bin/env bash
# Trajectory: eps_rule, delta_comp and accuracy over training, by supervision.
set -u
cd /home/hariguru/aayus/trace
OUT=results/induced_rule_traj.csv
for COND in both outcome process; do
  for SEED in 2001 2002 2003; do
    echo "=== cond=$COND seed=$SEED ==="
    uv run python experiments/induced_rule.py \
      --depth 4 --condition $COND --seed $SEED \
      --train-size 100000 --val-size 2000 --probe-size 400 \
      --checkpoints 0 250 500 1000 2000 4000 6000 \
      --device cuda:3 --out $OUT
  done
done
echo "TRAJECTORY DONE"
