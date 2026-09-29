#!/usr/bin/env bash
# Reliability sweep for counting. Run from anywhere.
cd "$(dirname "$0")/.."

mkdir -p logs/rho_sweep

for corrupt in scatter; do
  for seed in 42 ; do
    for rho in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      python handcoded/lettertrace.py \
        task=count \
        mod=2 \
        alphabet=4 \
        word_len=8 \
        layout=stream \
        n_blocks=2 \
        rho=$rho \
        corrupt=$corrupt \
        steps=8000 \
        seed=$seed \
        | tee logs/rho_sweep/${corrupt}_rho${rho}_seed${seed}.txt
    done
  done
done
