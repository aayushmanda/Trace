# mkdir -p results/depth

# for D in 4 8 12 16 20; do
#   uv run sweep_ratio.py \
#     --task boolean_circuit_${D} \
#     --rhos 0.30 0.50 0.60 0.70 0.80 0.90 1.00 \
#     --seeds 2001 2002 2003 2004 2005 \
#     --checkpoints 4000 8000 12000\
#     --train-size 100000 \
#     --val-size 2000 \
#     --batch-size 128 \
#     --layers 2 \
#     --output results/depth/boolean_D${D}.csv
# done

for SEED in 2001 2002 2003 2004 2005; do
  python validate_claims.py \
    --task boolean_circuit_8 \
    --rhos 0.5 0.6 0.7 0.8 0.9 1.0 \
    --steps 8000 \
    --train-size 100000 \
    --val-size 2000 \
    --margin-size 512 \
    --seed $SEED \
    --probe-base-rho 0.8 \
    --probe-step 8000 \
    --probe-rhos 0.0 0.25 0.5 0.75 1.0 \
    --probe-batch-size 128 \
    --probe-lr 1e-4 \
    --out results/claim_validation_seed${SEED}.csv
done