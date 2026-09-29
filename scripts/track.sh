#!/usr/bin/env bash
# Spectra of saved counting checkpoints. Run from anywhere.
cd "$(dirname "$0")/.."

mkdir -p logs/track
shopt -s nullglob
outcome_ckpts=(ckpt/n8/*outcome_step*.pt)
process_ckpts=(ckpt/n8/*process_step*.pt)
if ((${#outcome_ckpts[@]} == 0 && ${#process_ckpts[@]} == 0)); then
  echo "no checkpoints in ckpt/n8/. Train first, e.g.:" >&2
  echo "  uv run python handcoded/lettertrace.py task=count word_len=8 mod=2 save_every=500 save_dir=ckpt/n8" >&2
  exit 1
fi
for f in "${outcome_ckpts[@]}"; do
  uv run python handcoded/spectrum.py task=count word_len=8 mod=2 site=answer ckpt="$f" \
    out=logs/track/$(basename "$f" .pt).json
done
for f in "${process_ckpts[@]}"; do
  uv run python handcoded/spectrum.py task=count word_len=8 mod=2 site=state ckpt="$f" \
    out=logs/track/$(basename "$f" .pt)_state.json
done
uv run python handcoded/figures.py spectra logs/track/*.json
