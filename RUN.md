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

## Architecture × supervision (Table 6 is still one seed)

PDF **Table 6** (`tab:2x2`) is the historical **one-seed** reachability notebook (process/outcome 61.7→38.1%; process/process 100%; outcome/outcome diverged). **Do not replace those cells** with unrun 10-seed numbers.

The planned multi-seed LR grid is `experiments/architecture_controls.py` → `results/architecture_controls_n10/` (rates \(2\times10^{-3},10^{-3},5\times10^{-4},2\times10^{-4}\), confirmation seeds 42–51, success = test answer accuracy \(>95\%\)). That directory is empty; only `results/architecture_controls/protocol.json` (`plan`) exists. Calibration / confirm have not produced a table.

```bash
python -m src architecture plan --config configs/experiments/architecture_controls.yaml
python -m src architecture calibrate --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3
python -m src architecture confirm --config configs/experiments/architecture_controls.yaml --devices cuda:2 cuda:3
python -m src architecture summarize --config configs/experiments/architecture_controls.yaml
```

---

## Optional extra / not in this PDF

These implement `app:regime-spec` on **character-token GPT**, plus length and \(m_{\min}\). They are revision extras. They are **not** Figure 4 / Table 7. `experiments/run_revision_bridge.sh` queues this GPT grid, not the semantic-token protocol.

### GPT induced-rule / ε_rule / pullback

Config: `configs/experiments/induced_rule.yaml`  
D ∈ {2,4,6,8}, outcome and process, seeds 2001–2003. Discard readouts with `state_on_set_mass < 0.5`.

```bash
python -m src induced --config configs/experiments/induced_rule.yaml \
  --depth 4 --condition process --seed 2001 --with-pullback --device cuda:2
python -m src pullback --depth 4 --seed 2001 --device cuda:2 --out results/pullback.csv
python experiments/analyze_induced.py   # Paper/figures/induced_rule.pdf (not \includegraphics'd in main.pdf)
```

CSV: `results/revision/induced_rule.csv`.

### Length generalization (GPT)

Train D=8, eval 8/10/12/16. No figure/table number in `main.pdf`.

```bash
python -m src length --config configs/experiments/length_generalization.yaml --device cuda:2
```

### \(m_{\min}\) histograms at ρ = 0.80 (GPT)

Proposition 1 / greedy flips. Histograms are not in this PDF.

```bash
python -m src margins --config configs/experiments/margin_histograms.yaml --device cuda:2
```

`python -m src analyze` writes `Paper/figures/revision_*.pdf` from `results/revision/*.csv` if present.

---

## Smoke

```bash
python -m src smoke
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
