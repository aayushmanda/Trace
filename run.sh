
# mkdir -p logs

# for task in count ; do
#     for n in 16; do
#         for S in 2; do
#             python handcoded/lettertrace.py \
#                 task=$task \
#                 word_len=$n \
#                 mod=3 \
#                 alphabet=10 \
#                 layout=block \
#                 steps=20000 \
#                 seed=$S \
#                 n_blocks=4

#         done
#     done
# done


# for n in 2 4 6 8 9 10; do
#   python handcoded/spectrum.py task=count word_len=$n mod=2 max_exact=10000000 \
#     out=logs/spec_count_n$n.json
# done

mkdir -p logs/track
shopt -s nullglob
outcome_ckpts=(ckpt/n8/*outcome_step*.pt)
process_ckpts=(ckpt/n8/*process_step*.pt)
if ((${#outcome_ckpts[@]} == 0 && ${#process_ckpts[@]} == 0)); then
  echo "no checkpoints in ckpt/n8/. Train first, e.g.:" >&2
  echo "  python handcoded/lettertrace.py task=count word_len=8 mod=2 save_every=500 save_dir=ckpt/n8" >&2
  exit 1
fi
for f in "${outcome_ckpts[@]}"; do
  python handcoded/spectrum.py task=count word_len=8 mod=2 site=answer ckpt=$f \
    out=logs/track/$(basename $f .pt).json
done
for f in "${process_ckpts[@]}"; do
  python handcoded/spectrum.py task=count word_len=8 mod=2 site=state ckpt=$f \
    out=logs/track/$(basename $f .pt)_state.json
done
python handcoded/spectra_table.py logs/track/*.json
