# How to run

Python 3.12+, from the repo root. `uv sync` creates `.venv` from `uv.lock`. Training shows tqdm bars; `TRACE_TQDM=0`
silences them. On a shared machine, pick a GPU with `CUDA_VISIBLE_DEVICES`. All outputs go to `results/`, which is not
tracked.

## Counting (handcoded/)

```bash
# Step and outcome training from one random initialization, plus the two exact executors (count modulo K).
uv run python handcoded/lettertrace.py task=count word_len=16 mod=2 steps=8000
uv run python handcoded/lettertrace.py task=first word_len=12 steps=8000          # first-occurrence control
uv run python handcoded/lettertrace.py task=count word_len=1000 handcoded_only=True # executors at n = 1000

# Outcome plateau (Theorem 3): exit time tau(0.1) from the initialization ball, n = 4..16, zero readout.
REGIME_ZERO_READOUT=1 uv run python handcoded/regime_both.py 0.002

# Step learning for the gate head (Theorem 4): joint gradient flow from randomly scaled hand-coded directions.
uv run python handcoded/joint_gate.py --n 4 6 8 --eps 0.3 0.2 0.1 --seeds 0 1 2 --out results/joint_gate/main.json
# Theorem 4 audit: threshold, derivative identities and reduced flow vs the full softmax model; reduced-ODE table.
uv run python handcoded/theory_checks/theorem4_checks.py
uv run python handcoded/reduced_ode.py                       # K = 2 and gate readout on I - R for K = 3, 4, 6
uv run python handcoded/reduced_ode.py --K 3 4 6 --gate full --out results/reduced_ode/runs_fullgate.json
uv run python handcoded/make_ode_table.py   # prints the LaTeX of tab:J-ode
# Gate head inside the full network: seeded runs.
uv run python handcoded/gate_steering.py --n 4 6 8 --start A --seeds 0 1 2 --out results/gate_steering/main.json
```

`lettertrace.py` takes `key=value` settings; the defaults and their meaning are at the top of the file.
`handcoded/lettertrace_train.ipynb` trains both losses from one initialization and animates them against the executors
(run it from the repo root; it writes to `handcoded/animations/`).

## Beyond counting (src/)

```bash
# Boolean-8 and the register machine on the character-token GPT (GPT-2L).
uv run python -m src supervision --config configs/experiments/boolean8_gpt.yaml
uv run python -m src supervision --config configs/experiments/register16_gpt.yaml

# Rank-8 LoRA on Boolean-8: SmolLM2-135M, then Qwen2.5-0.5B, then the 8x outcome budget.
uv run python -m src lora --config configs/experiments/lora_transfer.yaml
uv run python -m src lora --config configs/experiments/lora_transfer.yaml --model Qwen/Qwen2.5-0.5B \
    --output results/paper/qwen05/main.csv --run-dir results/paper/qwen05/artifacts
uv run python -m src lora --config configs/experiments/lora_transfer.yaml --model Qwen/Qwen2.5-0.5B \
    --conditions outcome --steps 6400 --output results/paper/qwen05/outcome_6400.csv \
    --run-dir results/paper/qwen05/artifacts_outcome_6400

# BIG-Bench Hard tracking shuffled objects, Qwen2.5-0.5B LoRA (K = 3, 5, 7; seeds 2001-2003).
uv run python -m src shuffled zeroshot --out results/shuffled/zeroshot_qwen05.json
for K in 3 5 7; do for c in outcome process; do for s in 2001 2002 2003; do
  uv run python -m src shuffled train --K $K --condition $c --seed $s --out results/shuffled/bbh/bbh_K${K}_${c}_s${s}.json
done; done; done
```

`data/bbh/` holds the official BIG-Bench Hard items, unchanged. Training items are generated from their scenes and
item pools; the real items are held out for testing.
