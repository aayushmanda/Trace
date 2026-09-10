#!/usr/bin/env bash
# Queue the resubmission bridge on GPU 2/3 without flooding the busy A100s.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p results/revision logs/revision Paper/figures results/revision/induced_ckpts

running=0
wait_slot() {
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 20
  done
}

i=0
for D in 2 4 6 8; do
  for COND in outcome process; do
    for SEED in 2001 2002 2003; do
      wait_slot
      GPU=$((2 + i % 2))
      i=$((i + 1))
      MID=$((D / 2)); [ "$MID" -lt 1 ] && MID=1
      LOG=logs/revision/induced_${COND}_D${D}_s${SEED}.log
      echo "induced $COND D=$D seed=$SEED GPU=$GPU"
      CUDA_VISIBLE_DEVICES=$GPU $PY -m src induced \
        --config configs/experiments/induced_rule.yaml \
        --depth "$D" --condition "$COND" --seed "$SEED" \
        --train-size 30000 --val-size 800 --probe-size 300 \
        --probe-steps 1 "$MID" "$D" --with-pullback \
        --checkpoints 0 500 2000 4000 \
        --ckpt-dir results/revision/induced_ckpts \
        --device cuda:0 \
        --out results/revision/induced_rule.csv \
        > "$LOG" 2>&1 &
    done
  done
done
wait

echo "length generalization"
CUDA_VISIBLE_DEVICES=2 $PY -m src length \
  --config configs/experiments/length_generalization.yaml \
  --device cuda:0 --out results/revision/length_generalization.csv \
  > logs/revision/length_generalization.log 2>&1

echo "m_min histograms"
CUDA_VISIBLE_DEVICES=3 $PY -m src margins \
  --config configs/experiments/margin_histograms.yaml \
  --device cuda:0 --out results/revision/mmin_histograms.csv \
  > logs/revision/mmin_histograms.log 2>&1

echo "architecture calibration (10-seed protocol)"
$PY -m src architecture plan \
  --config configs/experiments/architecture_controls.yaml \
  --output results/architecture_controls_n10 \
  --rates 0.002 0.001 0.0005 0.0002 \
  --confirmation-seeds 42 43 44 45 46 47 48 49 50 51 \
  > logs/revision/arch_plan.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 $PY -m src architecture calibrate \
  --config configs/experiments/architecture_controls.yaml \
  --output results/architecture_controls_n10 \
  --devices cuda:0 cuda:1 \
  --calibration-steps 2000 \
  > logs/revision/arch_calibrate.log 2>&1

$PY -m src analyze
echo "REVISION BRIDGE DONE"
