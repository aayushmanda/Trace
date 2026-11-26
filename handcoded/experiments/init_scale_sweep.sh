#!/usr/bin/env bash
# Table 10(c): initialization-scale sweep, step and outcome from one seeded theta_0 (8k updates, seeds 42-44).
# Writes results/seeded/rnd_n{N}_K2_e{EPS}_s{SEED}.json; existing files are skipped.
#   bash handcoded/experiments/init_scale_sweep.sh            (set CUDA_VISIBLE_DEVICES to pick a GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/seeded logs/seeded
for eps in 0.1 0.5; do for n in 8 16; do for s in 42 43 44; do
  out=results/seeded/rnd_n${n}_K2_e${eps}_s${s}.json
  TRACE_TQDM=0 uv run python handcoded/experiments/seeded_sweep.py out=$out task=count word_len=$n mod=2 \
    init=random init_std=$eps seed=$s steps=8000 > logs/seeded/rnd_n${n}_e${eps}_s${s}.log 2>&1 &
done; done; done
wait
