# Trace Supervision Experiments

This repository contains a small synthetic Transformer setup for studying how the fraction of correct reasoning traces affects task performance.

The main experiment uses the `word_index` task. A model receives a word and a queried character and must generate a reasoning trace followed by the correct index.

Example:

```text
abcdefghij;f 0a 1b 2c 3d 4e 5f 6g 7h 8i 9j : 5
```

The training data can contain either correct traces or corrupted traces, while the final answer remains correct. The parameter $\rho$ denotes the fraction of training examples containing correct traces.

## Project Structure

```text
trace/
├── config.py
├── model.py
├── tokenizer.py
├── dataclass.py
├── task.py
├── registry.py
├── save_data.py
├── train.ipynb
├── think.py
└── word_index/
    ├── word_index_rho_0.txt
    ├── word_index_rho_10.txt
    ├── word_index_rho_20.txt
    ├── ...
    └── word_index_rho_100.txt
```

## Files

### `config.py`

Stores experiment and model hyperparameters such as:

```python
TASK_NAME = "word_index"

RHO_VALUES = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
]

MODEL_SEEDS = [1000, 1500, 2200, 3000, 4000]

DATASET_SIZE = 50000

BATCH_SIZE = 256
STEPS = 6000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-5
WARMUP_STEPS = 600
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.05

VAL_SEED = 99999
VAL_SIZE = 1000
VAL_BATCH_SIZE = 256

DATA_SEED = 100

USE_COMPILE = True
SAVE_MODELS = False
```

### `model.py`

Contains the decoder-only Transformer model.

### `tokenizer.py`

Contains the character-level tokenizer.

### `dataclass.py`

Contains shared dataclasses such as `Instance` and `Task`.

### `task.py`

Contains task samplers and trace construction.

### `registry.py`

Contains the task registry, for example:

```python
TASKS["word_index"]
```

### `save_data.py`

Creates fixed training files at different correct-trace ratios.

### `train.ipynb`

Loads a selected training file, trains the model, evaluates on held-out validation data, and can run sweeps over $\rho$ and model seeds.

## Word Index Task

Each example has the form:

```text
prompt trace : answer
```

Example:

```text
abcdefghij;f 0a 1b 2c 3d 4e 5f 6g 7h 8i 9j : 5
```

The prompt is:

```text
abcdefghij;f
```

The correct trace is:

```text
0a 1b 2c 3d 4e 5f 6g 7h 8i 9j
```

The final answer is:

```text
5
```

Wrong traces can contain corrupted characters, scrambled indices, or both. The final answer remains correct even when the trace is wrong.

## Correct Trace Ratio

$\rho$ is the fraction of training examples with correct traces.

For 50,000 training examples:

```text
ρ = 0.0  ->     0 correct + 50000 wrong
ρ = 0.1  ->  5000 correct + 45000 wrong
ρ = 0.5  -> 25000 correct + 25000 wrong
ρ = 0.9  -> 45000 correct +  5000 wrong
ρ = 1.0  -> 50000 correct +     0 wrong
```

All ratio files should use the same underlying prompt pool so changing $\rho$ changes trace correctness rather than the task distribution.

## Generate Training Files

```python
from registry import TASKS
from save_data import save_mixed_trace_file

DATASET_SIZE = 50000

CORRECT_RATIOS = [
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

task_name = "word_index"

instances = [
    TASKS[task_name].sample()
    for _ in range(DATASET_SIZE)
]

for rho in CORRECT_RATIOS:
    save_mixed_trace_file(
        instances,
        correct_ratio=rho,
        output_file=f"{task_name}/{task_name}_rho_{int(rho*100)}.txt",
        seed=100,
    )
```

This creates:

```text
word_index/word_index_rho_0.txt
word_index/word_index_rho_10.txt
word_index/word_index_rho_20.txt
...
word_index/word_index_rho_100.txt
```

## Train One Model

Choose a ratio:

```python
rho = 0.5
data_file = f"{TASK_NAME}/{TASK_NAME}_rho_{int(rho*100)}.txt"
```

The file is loaded and tensorized once. Minibatches are then sampled from this fixed dataset during training.

For controlled comparisons, reset the model seed before constructing each model:

```python
random.seed(model_seed)
torch.manual_seed(model_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(model_seed)
```

## Validation

Validation data is generated separately from training.

For `word_index`, validation excludes any underlying word that appears in training:

```python
if word in train_words:
    continue
```

Default validation size:

```python
VAL_SIZE = 1000
```

Validation is run once after training.

Two quantities are recorded.

### Accuracy

Fraction of held-out examples for which the final predicted index is correct.

### Separator Rate

Fraction of generated continuations that contain the `:` separator between the trace and final answer.

Example:

```text
0a 1b 2c ... : 5
```

A separator rate of 100% means the model consistently generates the expected answer delimiter. It does not mean the final answer is correct.

## Sweep Over rho

Typical sweep:

```python
RHO_VALUES = [
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]
```

For each value:

1. Load the corresponding fixed training file.
2. Reset the model initialization seed.
3. Train for the same number of optimization steps.
4. Evaluate on the same held-out validation set.
5. Record validation accuracy and separator rate.

## Multiple Model Seeds

To measure sensitivity to initialization:

```python
MODEL_SEEDS = [
    1000,
    1500,
    2200,
    3000,
    4000
]
```

Each seed trains a fresh model from scratch on the same data.

For each $\rho$, report the mean validation accuracy and standard deviation across seeds.

## Plot Accuracy vs. rho

```python
plt.figure(figsize=(8, 5))

plt.errorbar(
    rho_values,
    mean_acc,
    yerr=std_acc,
    marker="o",
    capsize=4
)

plt.xlabel(r"Correct Trace Ratio $\rho$")
plt.ylabel("Validation Accuracy (%)")
plt.title(r"Validation Accuracy vs. $\rho$ Across Model Seeds")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Current Experimental Observation

Preliminary runs show validation accuracy staying near chance for low values of $\rho$, followed by a sharp increase over a narrow range of correct-trace ratios and saturation near the task ceiling.

This should currently be described as empirical transition-like behavior rather than a confirmed phase transition.

Important follow-up controls include:

- multiple model initialization seeds,
- multiple trace-mixture seeds,
- denser values of $\rho$ around the transition,
- different batch sizes,
- different model depths,
- different trace corruption schemes,
- answer-only loss versus full trace supervision.

## Performance

Useful speed settings include:

```python
torch.set_float32_matmul_precision("high")
```

BF16 autocast when supported:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

Fused AdamW:

```python
torch.optim.AdamW(..., fused=True)
```

Optional compilation:

```python
model = torch.compile(model, mode="reduce-overhead")
```

The training tensors can be moved to GPU once before training to avoid repeated CPU-to-GPU transfers.

Validation should be performed only once after training when only final performance is needed.

## Reproducibility

Keep different sources of randomness separate:

```text
DATA_SEED   -> underlying generated dataset
mix seed    -> which prompts receive correct versus wrong traces
model seed  -> model initialization and training randomness
VAL_SEED    -> held-out validation set
```

For a controlled $\rho$ sweep, keep the dataset and validation set fixed while changing only trace correctness.

## Environment

Typical dependencies:

```text
Python 3.12
PyTorch
NumPy
Matplotlib
tqdm
Jupyter
```

Example installation with `uv`:

```bash
uv pip install torch numpy matplotlib tqdm ipykernel
```
