# How to run Trace (paper experiments)

Audience: research engineer reproducing the Boolean-circuit results. From the repo root. Training prints tqdm progress bars; set `TRACE_TQDM=0` to silence them.

## Environment

Python 3.12+. Either:

```bash
conda activate aayus
# or
uv sync
```

This box: 4× A100 80GB. Prefer **GPU 2 or 3** when 0/1 are occupied. Do not kill other jobs.

```bash
export CUDA_VISIBLE_DEVICES=2   # then --device cuda:0
# or
python -m src … --device cuda:2
```

If `CUDA_VISIBLE_DEVICES` is set, scripts treat `cuda:0` as that physical GPU. Default device selection already prefers `cuda:2` then `cuda:3`. Cap concurrent jobs (`MAX_JOBS=2` in `experiments/run_revision_bridge.sh`).

---

## Handcoded tutorial notebook

Package: `handcoded/` (`gates`, `tokenizer`, `models`, `data`, `generate`, `train`, `eval`, `animate`). Paper hyperparameters: `configs/handcoded.yaml`. Smoke: `configs/handcoded_smoke.yaml` (or `SMOKE = True` in the notebook).

```bash
conda activate aayus   # or: uv sync && source .venv/bin/activate
cd /path/to/trace
# Jupyter kernel = that env's python
jupyter notebook handcoded/handcoded_executors.ipynb
# or VS Code / Cursor: select the env kernel, run all
```

Device: the first cell prefers `cuda:2` if ≥3 GPUs exist, else `cuda`, else CPU. Plots use `src.plot_style.apply_style()` (seaborn-like grey grid, DejaVu Sans) — do not copy rcParams into scripts.

`handcoded_utils.py` is a deprecated import shim; new code should `import handcoded`.

---

## Two stacks (do not mix forwards)

**GPT** — character-token LayerNorm Transformer (`src.models.gpt.GPTModel`). Boolean circuits as character strings. Induced-rule, length, m_min.

**Handcoded** — semantic-token executors (`handcoded/` package): one token per 4-bit state / gate. Local tables, composition, architecture × supervision. Not the same architecture as GPT.

Shared: seed, YAML under `configs/experiments/`, CSV writers, device, progress bars.

Entry point: `python -m src <command>`. Equivalent paper scripts still exist under `experiments/` if you need a one-off flag.

---

## Smoke (minutes) vs full paper

**Smoke** (one small process GPT, 2 train steps):

```bash
python -m src smoke
# same as:
python -m src induced --config configs/experiments/smoke.yaml --depth 2 --condition process --seed 2001
```

Handcoded smoke (CPU or one GPU; empty output dir required):

```bash
python -m src executor --config configs/experiments/executor_comparison.yaml \
  --output results/executor_comparison/smoke --steps 2 --train-size 64 --test-size 16 \
  --probe-size 8 --batch-size 16 --checkpoints 0 2 --device cpu --threads 1
```

**Full paper:** sections below in the run order at the end. Wall time is GPU-days, not minutes. Use `experiments/run_revision_bridge.sh` to queue GPT induced + length + m_min on GPU 2/3 (`MAX_JOBS=2`).

---

## GPT: induced-rule / ε_rule / pullback

Config: `configs/experiments/induced_rule.yaml`  
D ∈ {2,4,6,8}, outcome and process, seeds 2001–2003. Probe steps 1, ⌊D/2⌋, D. Pullback on. Discard readouts with `state_on_set_mass < 0.5` at analysis time.

One cell:

```bash
python -m src induced --config configs/experiments/induced_rule.yaml \
  --depth 4 --condition process --seed 2001 --with-pullback --device cuda:2
```

Grid (or run the shell script):

```bash
for D in 2 4 6 8; do
  for COND in outcome process; do
    for SEED in 2001 2002 2003; do
      python -m src induced --config configs/experiments/induced_rule.yaml \
        --depth $D --condition $COND --seed $SEED --with-pullback --device cuda:2
    done
  done
done
```

CSV appends to `results/revision/induced_rule.csv`. Checkpoints: `results/revision/induced_ckpts/`. Logs if using the shell: `logs/revision/induced_*.log`.

Standalone pullback (same GPT stack, default D=4):

```bash
python -m src pullback --depth 4 --seed 2001 --device cuda:2 --out results/pullback.csv
```

---

## Handcoded: induced-rule analogue / ε_rule / pullback (gradient cosine)

Config: `configs/experiments/handcoded_induced.yaml` (same keys as `executor_comparison.yaml`).  
Local gold-prefix tables, `epsilon_rule`, composition TV, mixed-format gradient cosine vs true-rule and random controls. Output directory must be empty.

```bash
python -m src executor --config configs/experiments/handcoded_induced.yaml \
  --output results/executor_comparison/induced --device cuda:2
```

Depth replication D=2,4,6,8 (5 seeds): `configs/experiments/executor_depths.yaml`

```bash
python -m src executor-depths --config configs/experiments/executor_depths.yaml
```

Writes `results/executor_comparison/depth_replication/metrics.csv` and per-run logs next to it. Default worker device in that launcher is CPU; override by editing the subprocess `--device` if you want GPU.

Figures from a comparison run: `results/executor_comparison/<run>/*.pdf` plus `metrics.csv`, `report.md`, `checkpoints/`.

---

## Architecture × supervision (handcoded, 10 seeds, LR grid, success >95%)

Config: `configs/experiments/architecture_controls.yaml`  
Output **`results/architecture_controls_n10`** (do not reuse `results/architecture_controls/` if `protocol.json` already exists). Rates `2e-3, 1e-3, 5e-4, 2e-4`. Confirmation seeds 42–51. Report fraction of seeds with selected-checkpoint **test** answer accuracy **> 95%**.

```bash
python -m src architecture plan --config configs/experiments/architecture_controls.yaml
python -m src architecture calibrate --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3
python -m src architecture confirm --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3
python -m src architecture summarize --config configs/experiments/architecture_controls.yaml
```

Per-run logs: `results/architecture_controls_n10/{calibration,confirmation}/*/run.log`. Summary: `success_fraction.csv` and `success_fraction.json` in that directory.

---

## Length generalization (GPT)

Config: `configs/experiments/length_generalization.yaml`  
Train D=8, eval 8/10/12/16, outcome and process, seeds 2001–2003.

```bash
python -m src length --config configs/experiments/length_generalization.yaml --device cuda:2
```

CSV: `results/revision/length_generalization.csv`. Checkpoints: `results/revision/length_ckpts/`.

---

## m_min histograms at ρ = 0.80 (GPT)

Config: `configs/experiments/margin_histograms.yaml`  
Task `boolean_circuit_8`. Proposition 1 / greedy flips; not a “phase transition” claim.

```bash
python -m src margins --config configs/experiments/margin_histograms.yaml --device cuda:2
```

CSV: `results/revision/mmin_histograms.csv`.

---

## Figures

After CSVs exist:

```bash
python -m src analyze
# GPT induced-rule tables + Paper/figures/induced_rule.pdf (fixed paths):
python experiments/analyze_induced.py
```

`python -m src analyze` writes `Paper/figures/revision_induced_rule.pdf`, `revision_length_generalization.pdf`, `revision_mmin.pdf` from `results/revision/*.csv`, and prints architecture success fractions if `results/architecture_controls_n10/success_fraction.csv` exists.

`Paper/` is local (often gitignored). Create `Paper/figures` if missing.

---

## Outputs (where things go)

| What | Path |
|---|---|
| GPT paper CSVs | `results/revision/` |
| GPT induced / length checkpoints | `results/revision/induced_ckpts/`, `results/revision/length_ckpts/` |
| GPT queue logs | `logs/revision/` |
| Architecture (n=10) | `results/architecture_controls_n10/` |
| Handcoded executor runs | `results/executor_comparison/` |
| Figures | `Paper/figures/` |
| Pullback-only CSV | `results/pullback.csv` |

---

## Tests

```bash
TRACE_TQDM=0 python -m unittest tests.test_plumbing tests.test_handcoded experiments.test_executor_comparison experiments.test_revision_bridge
# or
TRACE_TQDM=0 python -m unittest discover -s tests -v
```

`experiments/test_revision_bridge.py` is heavier than `tests/test_plumbing.py`.

---

## Other GPT diagnostics (not the revision grid)

```bash
python -m src supervision --tasks boolean_circuit_4 --seeds 2001 --steps 100 --train-size 1000
python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001
```

Claim map for remaining `experiments/*.py`: [experiments/README.md](experiments/README.md). Tutorial notebook: `handcoded/handcoded_executors.ipynb`.
