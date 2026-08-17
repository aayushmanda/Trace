# Empirical Analysis of Phase Transitions in Transformer Reasoning

An empirical framework designed to analyze critical phase transitions ($\rho_c$) and reasoning behavior in decoder-only Transformers trained on synthetic tasks with varying correct-to-corrupted reasoning trace ratios $\rho$[cite: 32, 35].

## Repository Architecture

| Module | Core Functionality |
| :--- | :--- |
| `analysis.py` | Fits logistic curves $\text{acc}(\rho) = a_{\text{lo}} + \frac{a_{\text{hi}} - a_{\text{lo}}}{1 + \exp(-(\rho - \rho_c)/\tau)}$ to measure critical transition ratios $\rho_c$ and transition widths $\tau$[cite: 32]. Evaluates scaling exponents $\rho_c \sim B^\alpha$ against Signal-to-Noise Ratio (SNR) theory ($\alpha = -0.500$)[cite: 32]. |
| `config.py` | Houses frozen hierarchical configuration trees (`ModelConfig`, `OptimConfig`, `DataConfig`, `EvalConfig`) to ensure self-describing runs and reproducible experiment grids[cite: 33]. |
| `data.py` | Efficiently handles compact on-disk binary pools (uint8 tokens, int16 boundaries), trace ratio mixing, and held-out evaluation sets[cite: 34]. |
| `engine.py` | Defines a causal GPT model, AdamW optimization with cosine learning rate decay, and integrated evaluation probes[cite: 35]. |
| `sweeps.py` | Orchestrates resumable sweeps across 5 experimental regimes (`phase`, `family`, `ablation`, `batch`, `tokens`) backed by a JSONL result store[cite: 36]. |
| `tasks.py` | Manages `CharTokenizer`, trace rendering modes (`correct_think`, `wrong_think`, `no_think`), and synthetic task generators[cite: 37]. |

## Evaluation & Measurement Probes

* **Free Generation (`free_acc`)**: Measures end-to-end performance by sampling directly from bare prompts up to the newline token[cite: 32, 35].
* **Answer Forcing (`forced_acc` / `forced_wrong_acc`)**: Supplies clean or corrupted traces in context to isolate downstream answer computation from trace generation errors[cite: 33, 35].
* **Teacher-Forced Probes (`answer_nll` / `answer_tf_acc`)**: Measures cross-entropy loss and token argmax matching on gold answer tokens in a single forward pass[cite: 33, 35].

## Synthetic Task Suite

* **`word_index`**: Reports 0-based position of a queried letter in a given word[cite: 37].
* **`sort_letters`**: Outputs selection sort steps on a string of distinct letters[cite: 37].
* **`multiply`**: Calculates 2-digit multiplication via partial products[cite: 37].
* **`count_char`**: Tracks a running tally of a queried character in a word[cite: 37].

## Quickstart

### Running Experimental Sweeps
Execute resumable sweeps across target regimes (`phase`, `family`, `ablation`, `batch`, `tokens`):
```bash
python sweeps.py --regime phase

## Data Model & Storage Architecture

The framework relies on a binary data pipeline optimized for trace mixing, zero-copy loading, and scalable token indexing.

### On-Disk Binary Layout
To eliminate text-parsing overhead during high-throughput training sweeps, task samples are pre-rendered into compact binary pools on disk[cite: 34]:

* **Token Buffer (`tokens.bin`)**: Flat `uint8` array storing tokenized character sequences sequentially using `CharTokenizer`[cite: 34, 37].
* **Boundary Index (`boundaries.bin`)**: Array of `int16` tuples `(start_idx, prompt_len, trace_len, answer_len)` defining slice offsets for prompts, reasoning traces, and target answers within the token buffer[cite: 34].

### Trace Rendering Modes & Mixing
Data loaders dynamically construct contexts by blending three explicit trace rendering modes at sampling time according to the mixture ratio $\rho \in [0, 1]$[cite: 34, 37]:

| Rendering Mode | Description | Example Context Layout |
| :--- | :--- | :--- |
| `correct_think` | Valid step-by-step reasoning trace ending in the ground-truth answer. | `[Prompt] [Correct Reasoning] -> [Answer]` |
| `wrong_think` | Corrupted or perturbed reasoning trace ending in the ground-truth answer. | `[Prompt] [Corrupted Reasoning] -> [Answer]` |
| `no_think` | Bare prompt mapped directly to the target answer, skipping step-by-step traces. | `[Prompt] -> [Answer]` |

### Data Entities & Schema

```python
# Task Entity (tasks.py)
@dataclass
class Task:
    name: str                # Task identifier string
    chars: str               # Tokenizer alphabet string
    block_size: int          # Maximum sequence context length (e.g., 128)
    max_new_tokens: int      # Sampling generation limit (e.g., 16)
    sample: Callable         # Data generator yielding (prompt, trace, answer)
    chance_acc: float        # Baseline accuracy under uniform random guessing
    ceiling_acc: float       # Theoretical maximum task performance
    description: str         # Human-readable task description
    answer_pattern: str      # Regex pattern to parse generated answers
    bayes_prob: Callable     # Task-specific analytical Bayes baseline function

# Data Configuration (config.py)
@dataclass
class DataConfig:
    task_name: str           # Target task identifier
    rho: float               # Correct-to-corrupted trace ratio rho in [0, 1]
    num_samples: int         # Total dataset size
    val_samples: int         # Size of held-out validation pool
    seed: int                # Data generation random seed
    data_dir: str            # Root path for binary pool storage