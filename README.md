# Trace Supervision Experiments

This repository studies when intermediate reasoning traces teach an autoregressive Transformer to execute a multi-step computation even when the prompt already determines the final answer.

The central intervention is **trace reliability** \(\rho\in[0,1]\): the prompt and terminal answer are kept fixed while the validity of intermediate supervision is changed.

The main paper-facing experiments cover:

- random finite-state machines,
- two-register modular machines,
- reversible Boolean circuits,
- outcome-only, answer-first, valid-process, and corrupted-process controls,
- reliability sweeps over \(\rho\),
- trace-level versus step-level corruption,
- training-time and mechanism diagnostics.

## Repository structure

```text
Trace/
├── README.md
├── compare_supervision.py
├── experiment.py
├── mechanism_diagnostics.py
├── save_data.py
├── sweep_ratio.py
├── trace_vs_step_corruption.py
├── config.py
├── pyproject.toml
├── requirements.txt
├── results/
└── src/
    ├── model.py
    ├── tokenizer.py
    ├── dataclass.py
    ├── registry.py
    ├── task.py
    ├── state_machine_tasks.py
    ├── sequential_tasks.py
    ├── boolean_circuit_tasks.py
    └── hard_word_index_tasks.py
```

## Installation

Using `uv`:

```bash
uv sync
```

or:

```bash
uv pip install torch numpy matplotlib tqdm
```

Check CUDA:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Core data construction

Every sampled problem contains:

```text
prompt
correct_trace
wrong_trace
gold_answer
```

A valid process example is:

```text
PROMPT CORRECT_TRACE : GOLD
```

A corrupted process example is:

```text
PROMPT WRONG_TRACE : GOLD
```

The final answer remains correct in both cases. For controlled \(\rho\) sweeps, the same underlying prompt pool and validation set are reused; only intermediate trace validity changes.

## Available tasks

Tasks are registered in:

```python
from src.registry import TASKS
```

Main sequential tasks include:

```text
state_machine_2
state_machine_4
state_machine_8
state_machine_12
state_machine_16
state_machine_20

register_machine_2
register_machine_4
register_machine_8
register_machine_12
register_machine_16
register_machine_20

boolean_circuit_4
boolean_circuit_8
boolean_circuit_12
boolean_circuit_16
boolean_circuit_20
```

Additional modular-program, stack-machine, and word-index tasks are also registered.

## Supervision conditions

### Outcome-only

```text
PROMPT : GOLD
```

The model receives no intermediate state supervision.

### Process

```text
PROMPT CORRECT_TRACE : GOLD
```

The valid trace occurs before the answer and can be reused autoregressively.

### Corrupted process

```text
PROMPT WRONG_TRACE : GOLD
```

The trace is locally invalid while the terminal answer remains correct.

### Answer-first

The correct answer is generated before the valid trace. This separates trace content from causal availability before answer generation.

## Matched supervision comparison

Example:

```bash
uv run compare_supervision.py \
  --tasks state_machine_12 state_machine_16 state_machine_20 \
  --modes outcome answer_first process corrupted \
  --seeds 2001 2002 2003 2004 2005 \
  --train-size 100000 \
  --val-size 2000 \
  --steps 8000 \
  --layers 2
```

This is the main control experiment for distinguishing valid pre-answer process supervision from matched alternatives.

## Reliability sweep

`rho` is the fraction/probability of examples receiving a valid trace.

The main sweep is implemented in:

```text
sweep_ratio.py
```

Boolean example:

```bash
uv run sweep_ratio.py \
  --task boolean_circuit_8 \
  --rhos 0.30 0.50 0.60 0.70 0.80 0.85 0.90 0.95 1.00 \
  --seeds 2001 2002 2003 2004 2005 \
  --checkpoints 1000 2000 4000 6000 8000 \
  --train-size 100000 \
  --val-size 2000 \
  --batch-size 128
```

State-machine boundary example:

```bash
uv run sweep_ratio.py \
  --task state_machine_16 \
  --rhos 0.80 0.82 0.84 0.85 0.86 0.87 0.88 0.90 1.00 \
  --seeds 2001 2002 2003 2004 2005 \
  --checkpoints 1000 2000 4000 6000 8000 12000 16000 \
  --train-size 100000 \
  --val-size 2000 \
  --batch-size 128
```

## Trace-level versus step-level corruption

The standard reliability sweep uses **trace-level corruption**. One Bernoulli variable is shared across the complete trajectory:

\[
Z\sim\mathrm{Bernoulli}(\rho),\qquad Z_1=\cdots=Z_D=Z.
\]

Thus an example is either fully valid or fully corrupted.

The step-level intervention independently samples validity at each transition:

\[
Z_t\sim\mathrm{Bernoulli}(\rho),\qquad t=1,\ldots,D.
\]

The two schemes match the marginal expected fraction of valid local transitions but differ in cross-transition correlation.

### Important construction detail

Step-level traces are generated **sequentially**. If a previous transition was corrupted, a later clean transition applies the correct operation to the current mixed state rather than jumping back to the canonical clean trajectory.

For transition \(t\):

\[
s_t = \Phi(s_{t-1},u_t)\quad\text{if }Z_t=1,
\]

while a corrupted transition emits a successor explicitly different from \(\Phi(s_{t-1},u_t)\).

Do not implement step-level corruption by naively splicing pre-generated clean and corrupted trace strings position-wise; after one corrupted transition the source state has changed.

### Run trace-vs-step corruption

State machine:

```bash
uv run trace_vs_step_corruption.py \
  --task state_machine_16 \
  --modes trace step \
  --rhos 0.80 0.82 0.84 0.86 0.88 0.90 1.00 \
  --seeds 2001 2002 2003 2004 2005 \
  --checkpoints 2000 4000 6000 8000 12000 16000 \
  --train-size 100000 \
  --val-size 2000 \
  --batch-size 128
```

Register machine:

```bash
uv run trace_vs_step_corruption.py \
  --task register_machine_16 \
  --modes trace step \
  --rhos 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 \
  --seeds 2001 2002 2003 2004 2005 \
  --checkpoints 1000 2000 4000 6000 8000
```

Boolean circuit:

```bash
uv run trace_vs_step_corruption.py \
  --task boolean_circuit_8 \
  --modes trace step \
  --rhos 0.30 0.50 0.60 0.70 0.80 0.85 0.90 0.95 1.00 \
  --seeds 2001 2002 2003 2004 2005 \
  --checkpoints 1000 2000 4000 6000 8000
```

The trace-vs-step CSV should record:

```text
task
corruption_mode
rho
realized_valid_step_rate
realized_clean_trace_rate
seed
step
loss
answer_accuracy
exact_trace_accuracy
trace_step_accuracy
colon_rate
```

### Interpretation

This intervention tests whether changing supervision correlation changes finite-budget acquisition while approximately matching the expected amount of locally valid supervision.

The paper should claim matched **marginal local validity**, not automatically identical Transformer mean gradients. Autoregressive prefixes differ across the two schemes, so empirical gradient means and variances should be measured rather than assumed equal.

## Mechanism diagnostics

`mechanism_diagnostics.py` measures quantities such as:

- answer accuracy,
- exact-trace accuracy,
- free-running transition accuracy,
- first-error depth,
- teacher-forced full-step accuracy,
- teacher-forced state accuracy,
- teacher-forced state-token accuracy,
- teacher-forced state NLL,
- predicted exact-trace probability,
- clean-state gradient norm,
- training-gradient norm,
- clean/corrupt gradient cosine,
- projected gradient mean,
- projected gradient variance,
- between-component variance,
- clean-state diagnostic loss.

Example:

```bash
uv run mechanism_diagnostics.py \
  --tasks boolean_circuit_8 \
  --rhos 0.30 0.50 0.80 0.85 0.90 0.95 1.00 \
  --seeds 2001 2002 2003 2004 2005
```


```bash
uv run lora.py \
  --rhos 0.0 0.2 0.4 0.5 0.6 0.8 1.0 \
  --seeds 2001 2002 2003 \
  --train-size 12000 \
  --steps 800 \
  --include-answer-first \
  --overwrite
```


A global gradient cosine should not by itself be interpreted as evidence of semantic acquisition because formatting, delimiters, positional structure, and token marginals can dominate it.

## Same-trajectory checkpoint experiment

To study finite-time acquisition, train each `(rho, seed)` once to a long horizon and evaluate checkpoints from the same training trajectory.

Recommended checkpoints:

```text
1000 2000 4000 6000 8000 10000 12000 16000
```

This supports analysis of:

- acquisition/hitting time,
- seed dependence,
- boundary movement with update budget,
- local-state acquisition before exact rollout,
- answer accuracy tracking complete execution.

Do not independently retrain a new model for each checkpoint.

## Batch-size experiment

To test finite-batch effects, hold all other settings fixed and vary:

```text
B = 64
B = 128
B = 256
B = 512
```

Recommended outputs:

```text
batch_size
rho
seed
checkpoint
answer_accuracy
exact_trace_accuracy
trace_step_accuracy
```

## Metrics

### Answer accuracy

Fraction of held-out prompts for which the generated terminal answer equals the gold answer.

### Exact-trace accuracy

Fraction of held-out prompts for which the entire free-running generated trace exactly matches the canonical valid trace.

### Trace-step accuracy

Fraction of individual generated transitions that match the canonical corresponding transition.

### Teacher-forced state accuracy

Accuracy of the next state when the preceding correct trace is supplied. This separates local transition competence from free-running rollout survival.

### First-error depth

Index of the first incorrect transition in the generated trace.

## Reproducibility

Keep randomness sources separate:

```text
train seed       -> underlying training instances
validation seed  -> held-out validation instances
ratio seed       -> clean/corrupted assignment
corruption seed  -> corrupted successor construction
batch seed       -> minibatch order
model seed       -> model initialization
```

For controlled comparisons:

- keep the prompt pool fixed,
- keep validation fixed,
- keep architecture fixed,
- keep optimizer and update budget fixed,
- keep minibatch order fixed where possible,
- vary only the intended intervention.

Reliability assignments should be nested across \(\rho\) when studying acquisition-boundary movement.


## Results

Outputs are written under:

```text
results/
```

Keep raw per-seed CSV files rather than only averaged results.

Typical result files include:

```text
*_phase_*.csv
*_trace_vs_step_*.csv
*_mechanism_*.csv
```

## Recommended paper experiment set

The strongest experimental package includes:

1. matched supervision controls,
2. multi-seed reliability sweeps,
3. local-state versus exact-rollout measurements,
4. same-trajectory checkpoint sweeps,
5. trace-level versus step-level corruption,
6. batch-size intervention,
7. early progress/noise diagnostics predicting later acquisition,
8. corruption-geometry intervention.

## Claim boundary

The experiments are designed to support the finite-budget claim:

> Sufficiently reliable trace-first supervision can teach an autoregressive Transformer a reusable local executor that matched outcome-only, answer-first, and corrupted supervision does not acquire within the same finite training budget.

The repository does **not** by itself establish:

- a universal critical reliability,
- optimizer-independent phase behavior,
- asymptotic failure below a positive threshold,
- outcome-only impossibility,
- transfer to natural-language reasoning,
- identical gradient means under trace-level and step-level corruption.

