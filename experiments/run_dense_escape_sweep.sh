#!/usr/bin/env bash
# Dense early-checkpoint sweep to resolve the mixing-escape trajectory that
# the original depth_replication schedule (0,100,500,1000,2000) skips over.
# See /home/hariguru/.claude/plans/how-to-fix-this-replicated-gadget.md.
set -eu
cd "$(dirname "$0")/.."
OUT=results/executor_comparison/dense_escape
CKPTS="0 5 10 15 20 25 30 40 50 65 80 100 130 160 200"
for DEPTH in 2 4 6; do
  for SEED in 42 43 44; do
    D="$OUT/depth${DEPTH}_seed${SEED}"
    if [ -d "$D" ] && [ "$(ls -A "$D" 2>/dev/null)" ]; then
      echo "skip depth=$DEPTH seed=$SEED (exists)"
      continue
    fi
    echo "=== depth=$DEPTH seed=$SEED ==="
    uv run python -m src executor \
      --config configs/experiments/executor_comparison.yaml \
      --output "$D" \
      --depth "$DEPTH" --seed "$SEED" --steps 200 \
      --checkpoints $CKPTS \
      --device cuda:0 --no-compile
  done
done
echo "DENSE ESCAPE SWEEP DONE"
