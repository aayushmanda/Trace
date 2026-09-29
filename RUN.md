# How to run Trace (paper experiments)

Audience: research engineer reproducing the paper's results. From the repo root. Training prints tqdm progress bars; set `TRACE_TQDM=0` to silence them.

## Environment

Python 3.12+. From the repo root:

```bash
uv sync
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```

`uv sync` writes `.venv` from `uv.lock`. Later commands in this file are `uv run python ...`, or the same command after `source .venv/bin/activate`.

This box: 4× A100 80GB, shared with other users. Prefer **GPU 2 or 3** when 0/1 are occupied. Do not kill other jobs.

```bash
export CUDA_VISIBLE_DEVICES=2   # then --device cuda:0
# or
uv run python -m src <command> --device cuda:2
```

Compile is off by default in every shipped config (`compile: false`). Do not turn it on for these runs.

---

## Two stacks

**GPT** — character-token Transformer (`src.gpt.GPTModel`). Boolean circuits, the register machine, and the state machine. This is the stack behind the Boolean and register tables. Tiny synthetic-token model; not a HuggingFace LM.

**Pretrained LoRA** — rank-8 Peft on `AutoModelForCausalLM` (`python -m src lora`). Same Boolean-circuit serialization as GPT, but a real HF tokenizer. Default backbone is SmolLM2-135M; this is the only place a HuggingFace model id is swappable.

**Handcoded** — the counting Transformer in `handcoded/lettertrace.py`. One architecture, process loss and outcome loss, plus the two programs written into the weights by hand. Spectra, figures, and probes live in the same folder.

Entry point for the GPT and LoRA runs: `python -m src <command>`.

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

## Boolean reliability

Counting figures come from `handcoded/lettertrace.py`. The Boolean-8 sweep is the character-token model.

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

The saved Boolean-8 sweep is `results/paper/boolean_reliability_canonical.csv`: 60 rows, clean-process accuracy 82.20±6.76%, close to the five-condition result 82.00±7.69%. Replication keeps the four data/assignment/minibatch seeds fixed. The runner refuses to overwrite an existing CSV, so the command above uses a separate rerun path. The earlier 100k sweep is archival only.

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

## Gradient alignment

```bash
python -m src align --help
```

Cosine between the Boolean-8 local transition gradient and the process and outcome gradients.

---

## Counting notebook

Open `handcoded/lettertrace_train.ipynb` with the `.venv` kernel from `uv sync`. It trains process and outcome from one initialization and writes `handcoded/animations/lettertrace_train.gif`. Pin a free GPU in the first cell. Do not use GPU 1.

Counting from the command line, and the spectrum of a saved checkpoint:

```bash
uv run python handcoded/lettertrace.py task=count word_len=8 mod=2 save_every=500 save_dir=ckpt/n8
uv run python handcoded/spectrum.py task=count word_len=8 mod=2 site=answer ckpt=ckpt/n8/<name>.pt out=logs/<name>.json
uv run python handcoded/figures.py spectra logs/*.json
```

---

## Outputs

| What | Path |
|---|---|
| Table 1 rerun | `results/paper/e1_five_condition_rerun/table1_rerun.csv` |
| Reliability sweeps (Figures 1-2 / Table 3) | `results/reliability_sweeps/` |
| Depth × reliability | `results/paper/depth_reliability/` |
| LoRA adaptation study | `results/paper/smollm/` |
| `Paper/` | The manuscript |
