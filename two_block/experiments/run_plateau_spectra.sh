#!/usr/bin/env bash
# Appendix G.1: outcome training at n = 8, init_std = 0.02, seed 42, with checkpoints every 1000 updates, then the exact
# Jacobian degree split at the answer position (16 Gaussian directions) at initialization and 1k, 3k, 6k, 8k updates.
#   bash two_block/experiments/run_plateau_spectra.sh         (set CUDA_VISIBLE_DEVICES to pick a GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/paper/exact_spectra/plateau_s42}
mkdir -p "$OUT" logs/plateau_s42
TRACE_TQDM=0 uv run python two_block/counting.py task=count word_len=8 mod=2 seed=42 init_std=0.02 steps=8000 \
  modes=outcome save_every=1000 save_dir=ckpt/n8_block_s42 tag=s42 > logs/plateau_s42/train.log 2>&1
uv run python two_block/experiments/spectra.py exact site=answer dirs=16 init_std=0.02 init_seed=42 > "$OUT/init.txt"
for k in 1000 3000 6000 8000; do
  uv run python two_block/experiments/spectra.py exact site=answer dirs=16 \
    ckpt=ckpt/n8_block_s42/s42_outcome_step$k.pt > "$OUT/o$k.txt"
done
