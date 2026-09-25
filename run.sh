for N in 20000 100000; do for L in 2 6; do
    TRACE_COMPILE=0 python -m src reliability --task boolean_circuit_8 \
    --rhos 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seeds 2001 2002 2003 --checkpoints 8000 \
    --train-size $N --val-size 1000 --train-seed 501 --val-seed 101 \
    --ratio-seed 777 --batch-seed 12345 --batch-size 128 --eval-batch-size 128 \
    --include-outcome --no-compile --embedding 128 --heads 4 --layers $L \
    --dropout 0 --lr 0.0003 --weight-decay 0 --grad-clip 1 --bf16 \
    --output results/paper/boolean_reliability_N${N}_L${L}.csv --device cuda
done; done