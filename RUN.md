# How to run Trace (paper experiments)

Audience: research engineer reproducing the paper's results. From the repo root. Training prints tqdm progress bars; set `TRACE_TQDM=0` to silence them.

## Environment

Python 3.12+. Either:

```bash
conda activate aayus
# or
uv sync
```

This box: 4× A100 80GB, shared with other users. Prefer **GPU 2 or 3** when 0/1 are occupied. Do not kill other jobs.

```bash
export CUDA_VISIBLE_DEVICES=2   # then --device cuda:0
# or
python -m src <command> --device cuda:2
```

Compile is off by default in every shipped config (`compile: false`). Do not turn it on for these runs.

---

## Two stacks

**GPT** — character-token Transformer (`src.models.gpt.GPTModel`). Boolean circuits as character strings. This is the stack behind Table 1, Figures 1–2/Table 3, and the Proposition 8 frontier check. Tiny synthetic-token model; not a HuggingFace LM.

**Pretrained LoRA** — rank-8 Peft on `AutoModelForCausalLM` (`python -m src lora`). Same Boolean-circuit serialization as GPT, but a real HF tokenizer. Default backbone is SmolLM2-135M; this is the only place a HuggingFace model id is swappable.

**Handcoded** — semantic-token executors (`handcoded/` package): one token per 4-bit state / gate, hand-constructed attention + ReLU weights. Used only by the tutorial notebook (`handcoded/handcoded_executors.ipynb`) and to numerically sanity-check the realizability construction in `Paper/appendix_architectures.tex` (Appendix A). Not the source of any current table or figure.

Entry point for training commands: `python -m src <command>`. Figure builders and closed-form/numerical checks (no training): `python experiments/run.py <command>` (see [experiments/README.md](experiments/README.md)).

---

## Table 1: five-condition supervision comparison

```bash
# Smoke (seconds, not a paper number):
python -m src supervision --config configs/experiments/e1_five_condition.yaml

# Table 1 itself, with every seed's row preserved:
python -m src supervision --config configs/experiments/e1_five_condition_rerun.yaml
```

`e1_five_condition_rerun.yaml` promotes the pilot config's frozen settings unchanged: `boolean_circuit_8`, five conditions (outcome / answer_first / filler / process / corrupted), seeds 2001–2005, 8000 steps, train 20000 / val 1000, batch 128, embedding 128 / 4 heads / 2 layers. Output: `results/paper/e1_five_condition_rerun/table1_rerun.csv` (one row per seed × condition) plus a sibling `_persist.json`.

---

## Main-paper figures and trace reliability

Rebuild all four main figures from existing canonical rows (no training):

```bash
uv run python -m src.experiments.plot_main_paper
```

This produces the combined reliability/corruption-structure figure, credit
schematic, compact gradient-alignment plot, and depth-control plot. Source hashes and moved-table
locations are tracked by LaTeX label in the artifact manifest.


`rho` is the probability a training example gets a valid trace; the terminal answer stays correct regardless. Metrics: `answer_accuracy` (rollout), `exact_trace_accuracy` (full trace), `trace_step_accuracy` (per-step correctness on the free-running trace). Each run writes a CSV and a sibling `*_persist.json`.

```bash
# Smoke:
python -m src reliability --task boolean_circuit_2 --rhos 0.8 --seeds 2001 \
  --checkpoints 2 --train-size 32 --val-size 8 --batch-size 8 \
  --output results/smoke.csv --device cpu

# Canonical protocol (GPU hours; use a fresh output path for a rerun):
TRACE_COMPILE=0 python -m src reliability \
  --task boolean_circuit_8 \
  --rhos 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --seeds 2001 2002 2003 2004 2005 --checkpoints 8000 \
  --train-size 20000 --val-size 1000 \
  --train-seed 501 --val-seed 101 --ratio-seed 777 --batch-seed 12345 \
  --batch-size 128 --eval-batch-size 128 --include-outcome --no-compile \
  --embedding 128 --heads 4 --layers 2 --dropout 0 \
  --lr 0.0003 --weight-decay 0 --grad-clip 1 --bf16 \
  --output results/paper/boolean_reliability_canonical_rerun.csv \
  --device cuda
```

`Paper/figures/reliability_structure.pdf` (built by `python -m src.experiments.plot_main_paper`) and the Transformer overlay in `Paper/figures/noise_threshold.pdf` read from `results/paper/boolean_reliability_canonical.csv`. The saved sweep has 60 rows and clean-process accuracy 82.20±6.76%, agreeing closely with the five-condition result 82.00±7.69%. Replication keeps the four data/assignment/minibatch seeds fixed. The runner refuses to overwrite an existing CSV, so the command above uses a separate rerun path. Figure-only rebuild commands and source hashes are in [the artifact manifest](results/paper/ARTIFACT_MANIFEST.md). The earlier 100k sweep is archival only.

The compilation robustness check uses the same command with `TRACE_COMPILE=1`, `--compile` instead of `--no-compile`, and a fresh output path such as `results/paper/boolean_reliability_compile_rerun.csv`. Its archived source is `results/my_reliability_sweep_comp.csv` (clean endpoint 82.86±7.17%); it supports the appendix robustness note while the eager sweep remains canonical.

**Depth × reliability** (does the reliability frontier predicted by Proposition 8 move with composition depth `D`?): rerun the same command for `--task boolean_circuit_{2,4,8}` with `--rhos 0.5 0.7 0.8 0.9 1.0 --seeds 2001 2002 2003 --checkpoints 8000`, output under `results/paper/depth_reliability/boolean_circuit_{D}.csv`.

---

## Pretrained LoRA (SmolLM2-135M)

```bash
python -m src lora --help
python -m src lora --config configs/experiments/lora_transfer.yaml
```

Trains outcome / answer-first / process-`rho` completions on a rank-8 LoRA adapter and reports free-generation accuracy (answer, exact trace, exact state rollout), plus a 16-way local-state scorer on the same HF tokenizer. Startup aborts if `encode(prefix)+encode(state) != encode(prefix+state)` at the `>` boundary — common on some BPE/SentencePiece tokenizers unless special tokens are mapped.

**Can swap the backbone.** Pass another `AutoModelForCausalLM` id with `--model` (e.g. `Qwen/Qwen2.5-0.5B`). Do not download a large checkpoint unless intended; the default 135M run is the paper setting. Output: `results/paper/smollm/smollm_trace.csv`, with raw per-seed generations, adapters, and local-margin CSVs archived under `results/paper/smollm/artifacts/`.

**Cannot swap into GPT.** Do not pass a HF model id into `supervision`/`reliability`; those are the tiny synthetic-token `GPTModel`, not `AutoModelForCausalLM`.

---

## Everything else: `python experiments/run.py <command>`

No training; these rebuild figures or check closed-form/numerical claims directly.

```bash
python experiments/run.py --help
python experiments/run.py noise-threshold        # Figure 3 / Tables 5-6 — App. I reliability threshold
python experiments/run.py family-sampling         # App. I frontier under family-first gate sampling
python experiments/run.py tabular-sampling        # App. I: 1/K vs 1/2 under uniform vs family gate sampling
python experiments/run.py clean-convergence       # Corollary 5: GD trajectory converging to T_g
python experiments/run.py credit-suppression      # Figure App.C: credit identities, numerically confirmed
python experiments/run.py prop8-frontier          # Proposition 8: measured vs predicted rho_c(a,D,beta)
python experiments/run.py fraction-vs-amount      # fixed-rho, varying-N grid
python experiments/run.py plot-paper-figures      # rebuild figures/{boolean_reliability,architectures}.pdf
```

Full command-to-claim map: [experiments/README.md](experiments/README.md).

---

## Handcoded tutorial notebook

Package: `handcoded/` (`gates`, `tokenizer`, `models`, `data`, `generate`, `train`, `eval`, `animate`). Hyperparameters: `configs/handcoded.yaml`.

```bash
conda activate aayus   # or: uv sync && source .venv/bin/activate
jupyter notebook handcoded/handcoded_executors.ipynb
```

Device: the first cell prefers `cuda:2` if ≥3 GPUs exist, else `cuda`, else CPU.

---

## Outputs

| What | Path |
|---|---|
| Table 1 rerun | `results/paper/e1_five_condition_rerun/table1_rerun.csv` |
| Reliability sweeps (Figures 1-2 / Table 3) | `results/reliability_sweeps/` |
| Depth × reliability | `results/paper/depth_reliability/` |
| LoRA adaptation study | `results/paper/smollm/` |
| App. I noise threshold / family sampling / tabular sampling | `results/noise_threshold/`, `results/fraction_vs_amount/` |
| Credit-identity numerics | `results/credit_suppression/` |
| Clean-convergence numerics | `results/clean_convergence/` |
| Proposition 8 frontier | `results/prop8_frontier/` |
| `Paper/` | The manuscript; not tracked by this repo's git history |
