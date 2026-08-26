mkdir -p results/depth

for D in 4 8 12 16 20; do
  uv run sweep_ratio.py \
    --task boolean_circuit_${D} \
    --rhos 0.30 0.50 0.60 0.70 0.80 0.90 1.00 \
    --seeds 2001 2002 2003 2004 2005 \
    --checkpoints 4000 8000 12000\
    --train-size 100000 \
    --val-size 2000 \
    --batch-size 128 \
    --layers 2 \
    --output results/depth/boolean_D${D}.csv
done