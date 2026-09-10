# Leftover GPU commands (week 2 freeze)

At freeze time GPU 2/3 had spare memory but **~100% utilization** (other jobs). Do not kill them. Compile off. Prefer physical 2 or 3.

```bash
PY=/home/hariguru/aayus/.venv/bin/python
cd /home/hariguru/aayus/trace

# E2 calibration ONLY (40 jobs). Stop if cards fill. Not 80-job confirm.
$PY -m src architecture calibrate --config configs/experiments/e2_architecture.yaml \
  --output results/paper_revision_v2/e2_architecture --devices cuda:2 cuda:3 --no-compile

# After selected_rates.json exists:
# $PY -m src architecture confirm --config configs/experiments/e2_architecture.yaml \
#   --output results/paper_revision_v2/e2_architecture --devices cuda:2 cuda:3 --no-compile

# E1 five-condition confirmation (not Table 1 until complete):
# $PY -m src supervision --config configs/experiments/e1_five_condition.yaml \
#   --seeds 2001 2002 2003 2004 2005 --steps 8000 --train-size 20000 --val-size 1000 \
#   --batch-size 128 --device cuda:3 --no-compile \
#   --output results/paper_revision_v2/e1_five_condition/confirm.csv

# E3 confirmation readout (one depth/seed; skip_readout false):
# $PY -m src induced --config configs/experiments/induced_rule.yaml --depth 8 \
#   --condition both --seed 2001 --with-pullback --device cuda:3 --no-compile \
#   --out results/paper_revision_v2/e3_mask_trace/confirm.csv
```
