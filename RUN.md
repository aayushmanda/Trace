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

**Compile is off by default.** Training may opt in with YAML `compile: true`, `TRACE_COMPILE=1`, or `--compile`. That compiles a **wrapper** used only for the train loss; `generate`, probes, induced-rule, and pullback always call the eager module (same parameters). Do not replace `model.forward` with `torch.compile`. `--no-compile` forces eager. Tests never compile. Architecture-controls `run_one` stays eager (no `compile: true` in that YAML).

```bash
python -m src executor --config configs/experiments/executor_comparison.yaml --compile   # train compiled
python -m src executor --no-compile   # override YAML compile: true
```

---

## Two stacks (do not mix forwards)

**GPT** — character-token LayerNorm Transformer (`src.models.gpt.GPTModel`). Boolean circuits as character strings. Used for Table 1 / Figs 1–2 / LoRA, and for **optional extras** (induced-rule, length, \(m_{\min}\)) that are **not** paper Figure 4.

**Handcoded** — semantic-token executors (`handcoded/` package): one token per 4-bit state / gate. **Paper §7 / Figs 4–5 / Table 7** are this stack (one-block width 96), not GPT and not the oracle-shaped Table 4 nets.

Shared: seed, YAML under `configs/experiments/`, CSV writers, device, progress bars.

Entry point: `python -m src <command>`. Equivalent paper scripts still exist under `experiments/` if you need a one-off flag.

---

## Paper §7: handcoded semantic-token readout (Figs 4–5, Table 7)

This is the completed five-seed study in the compiled PDF. Protocol: one causal block, width 96, 4 heads, 192-unit ReLU MLP, no LN/dropout, \(D\in\{2,4,6\}\), seeds 42–46, 10k train circuits, 2k updates, process / outcome / mixed. Builder refuses incomplete seed×depth coverage.

```bash
# Single comparison run (empty output dir required):
python -m src executor --config configs/experiments/executor_comparison.yaml \
  --output results/executor_comparison/smoke --steps 2 --train-size 64 --test-size 16 \
  --probe-size 8 --batch-size 16 --checkpoints 0 2 --device cpu --threads 1

# Depth replication (paper: D=2,4,6 × seeds 42–46). Default workers are CPU.
python -m src executor-depths --config configs/experiments/executor_depths.yaml

# Verified figures + Table 7 TeX (run after replication is complete):
python experiments/build_executor_results.py \
  --input results/executor_comparison/depth_replication --paper Paper
```

Writes:

- `Paper/figures/trained_executor_bridge.pdf` — Figure 4
- `Paper/figures/trained_executor_depths.pdf` — Figure 5
- `Paper/data/trained_executor_rows.tex` — Table 7 rows (`\input` from `trained_model_protocol.tex`)
- `Paper/data/trained_executor_numbers.tex` — §7 macros (`\input` from `trained_model_results.tex`)
- paste copy: `Paper/data/trained_executor_table.md`

Existing complete run: `results/executor_comparison/depth_replication/` (`metrics.csv`, per-seed `report.md`). Do not treat GPT `induced_rule.py` output as this figure.

The YAML may list depth 8; the **manuscript study is \(D\in\{2,4,6\}\)** as in `protocol.json`. Do not pool with character-token GPT readouts.

---

## Handcoded tutorial notebook

Package: `handcoded/` (`gates`, `tokenizer`, `models`, `data`, `generate`, `train`, `eval`, `animate`). Paper hyperparameters: `configs/handcoded.yaml`. Smoke: `configs/handcoded_smoke.yaml` (or `SMOKE = True` in the notebook).

```bash
conda activate aayus   # or: uv sync && source .venv/bin/activate
cd /path/to/trace
jupyter notebook handcoded/handcoded_executors.ipynb
```

Device: the first cell prefers `cuda:2` if ≥3 GPUs exist, else `cuda`, else CPU. Plots use `src.plot_style.apply_style()`. `handcoded_utils.py` is a deprecated import shim; new code should `import handcoded`.

---

## Mechanistic revision extras (run in this order)

These are **not** paper Figure 4 / Table 7 (that is the semantic-token §7 study above). Do **not** add five-condition / LoRA / reliability sweeps. Prefer `--device cuda:2` or `cuda:3`. Compile stays **off** (`compile: false` in these YAMLs). One `(depth, seed)` call at a time; do not launch the full grid from this file.

Discard GPT readouts with `state_on_set_mass < 0.5`. Mixed-format checkpoints use `condition: both` / `trace_fraction: 0.5`.

### 1. Exact serialized-gradient pullback (highest priority)

Full-vocabulary autoregressive CE on the serialized continuation (same mask as training). Reports \(\cos(-\nabla_\theta L_{\mathrm{out}}^{\mathrm{LM}}, J^\top W_{\mathrm{rule}})\), random controls, relative gradient error. Grid: \(D\in\{2,4,6,8\}\), seeds 2001–2005.

```bash
# Smoke (no full readout grid):
python -m src pullback --config configs/experiments/pullback.yaml \
  --depth 2 --seed 2001 --train-size 64 --val-size 16 --probe-size 8 \
  --checkpoints 0 --n-layer 1 --n-embd 32 --batch-size 8 --device cpu \
  --out results/revision/smoke_pullback.csv --no-compile

# Full (one job; GPU hours):
python -m src pullback --config configs/experiments/pullback.yaml \
  --depth 4 --seed 2001 --device cuda:2 --no-compile
# Repeat depths 2,4,6,8 × seeds 2001–2005.
```

Also on mixed-format induced checkpoints: `--with-pullback` (same LM objective).

### 2. Step-dependent induced kernels

For each position \(t=1\ldots D\) read \(\widehat P^{(t)}_g\) (several gate backgrounds). Composition is \(e_{s_0}^\top \widehat P^{(1)}_{g_1}\cdots\widehat P^{(D)}_{g_D}\) vs the model's direct-answer law — **not** one table reused at every depth. CSV columns: `eps_rule_hat` per `probe_step`, `eps_step_std`, `table_step_tv`, `background_eps_std`, `delta_comp_tv`.

```bash
python -m src induced --config configs/experiments/induced_rule.yaml \
  --depth 4 --condition both --seed 2001 --with-pullback --device cuda:2 --no-compile
# Grid: D ∈ {2,3,4,6,8}, seeds 2001–2005, mixed format.
# Tiny: python -m src smoke
```

### 3. Split-verdict depth table (central figure)

Same runs: \(D\in\{2,3,4,6,8\}\). Predicted exponent \(D-1\), fitted exponent in the \(\varepsilon\le 1/2\) ball (`refit_in_ball`), \(\delta_{\mathrm{comp}}\), exact-LM gradient cosine.

```bash
python -m src split-verdict --config configs/experiments/split_verdict.yaml
# or: python experiments/split_verdict.py
```

Writes `results/revision/split_verdict.csv` and `Paper/figures/split_verdict.pdf` via `apply_style()`. Needs induced (and optional pullback) CSVs from steps 1–2.

### 4. Architecture × supervision 2×2

PDF **Table 6** is still the historical **one-seed** notebook. Do not overwrite those cells. YAML has `compile: false`; `run_one` is eager. Devices default to GPU 2/3.

**Job counts (do not start the 80-job confirm in a plumbing pass):** calibrate = 2 architectures × 4 LRs × 2 clips × 2 formats = **32** jobs; confirm after `selected_rates.json` = 2 × 10 seeds × 2 clips × 2 formats = **80** jobs.

```bash
python -m src architecture plan --config configs/experiments/architecture_controls.yaml
# After plan only, optional 1-run smoke (CPU):
python -m src architecture single --output /tmp/arch_smoke --architecture process --mode process \
  --seed 1 --lr 0.0005 --clip 1 --stage calibrate --steps 1 \
  --train-size 8 --val-size 4 --test-size 4 --batch-size 4 --device cpu --no-compile

# 10-seed confirm *after* calibrate finishes (GPU hours; cuda:2/3):
python -m src architecture calibrate --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3 --no-compile
python -m src architecture confirm --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3 --no-compile
python -m src architecture summarize --config configs/experiments/architecture_controls.yaml
```

Output: `results/architecture_controls_n10/`. Existing leftover plan file: `results/architecture_controls/protocol.json`.

### 5. Shared trainable kernel executor

One tensor \(A\in\mathbb{R}^{M\times K\times K}\), \(P_g=\mathrm{softmax}(A_g)\). Same parameters for outcome compose vs process local CE. Init near uniform; vary \(\varepsilon\), \(D\). Projected true-rule gradient, escape time, success vs \(D\). Cheap CPU/GPU.

```bash
python -m src escape --smoke
python -m src escape --config configs/experiments/escape_time.yaml --device cpu
# or: python experiments/escape_time_law.py --shared --smoke
```

CSV: `results/revision/shared_kernel.csv`.

### 6. Length + \(m_{\min}\) (AFTER 1–4; do not run the full grid now)

Train \(D=8\), eval 8/10/12/16; gold-path \(m_{\min}\) at \(\rho=0.80\). No figure/table number in `main.pdf`. Keep the scripts; run only after pullback / induced / split-verdict / architecture calibration.

```bash
python -m src length --config configs/experiments/length_generalization.yaml --device cuda:2 --no-compile
python -m src margins --config configs/experiments/margin_histograms.yaml --device cuda:2 --no-compile
python -m src analyze   # Paper/figures/revision_*.pdf if CSVs exist
```

---

## Smoke

```bash
TRACE_TQDM=0 python -m unittest tests.test_plumbing tests.test_revision_bridge
python -m src escape --smoke
python -m src architecture plan --config configs/experiments/architecture_controls.yaml
python -m src smoke   # tiny mixed-format induced D=2; still a full 16×52 readout — use GPU if needed
```

Handcoded smoke is the short `python -m src executor …` command in the §7 section above.

---

## Outputs

| What | Path |
|---|---|
| §7 depth replication | `results/executor_comparison/depth_replication/` |
| Fig 4 / 5 + Table 7 TeX | `Paper/figures/trained_executor_*.pdf`, `Paper/data/trained_executor_*.tex` |
| Architecture plan (not Table 6) | `results/architecture_controls/protocol.json` |
| Architecture n=10 (not run) | `results/architecture_controls_n10/` |
| Split verdict | `results/revision/split_verdict.csv`, `Paper/figures/split_verdict.pdf` |
| Exact-LM pullback | `results/revision/pullback.csv` |
| Shared kernel | `results/revision/shared_kernel.csv` |
| GPT extras | `results/revision/` |
| `Paper/` | often gitignored; local manuscript |

---

## Tests

```bash
TRACE_TQDM=0 python -m unittest tests.test_plumbing tests.test_handcoded tests.test_executor_comparison tests.test_revision_bridge
# or
TRACE_TQDM=0 python -m unittest discover -s tests -v
```

Claim map: [experiments/README.md](experiments/README.md). Manuscript map: [Paper/EXPERIMENT_MAP.md](Paper/EXPERIMENT_MAP.md).
