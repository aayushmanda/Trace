"""Every number that defines a run, in one place.

Configuration is a tree of frozen dataclasses rather than module-level globals,
for three reasons that matter once results go into a paper:

  1. a run's config can be *recorded next to its metrics*, so a results file is
     self-describing and re-analysable months later;
  2. two configs can differ in one field (`replace(cfg, optim=...)`) without a
     sweep script mutating global state that other code already imported;
  3. nothing downstream can silently pick up a hyperparameter it was never
     handed -- `model.py` cannot read `n_layer` out of the air.

The sweep grids at the bottom are the *defaults* for the experiment builders in
sweeps.py; every one of them is overridable from the CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. Vocabulary and context length come from the Task, not here."""

    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.05


@dataclass(frozen=True)
class OptimConfig:
    """AdamW + cosine schedule with linear warmup."""

    steps: int = 6000
    batch_size: int = 256
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 600
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    """Pool size and the seeds that separate train from held-out data.

    `n_samples` is deliberately equal to `steps * batch_size` at the reference
    batch size, so a run at the reference setting never recycles a sample.
    """

    n_samples: int = 1_536_000
    data_seed: int = 100
    val_seed: int = 999_999
    val_examples: int = 1000
    cache_dir: str = "dataset_cache"


@dataclass(frozen=True)
class EvalConfig:
    """Which measurements to take on the trained model.

    `free` is the end-to-end metric (generate from the bare prompt); `forced`
    is answer forcing -- the clean trace is supplied in context and only the
    answer is produced, which removes the trace-generation and token-budget
    confounds from the number. `forced_wrong` repeats it with a corrupted trace
    and is the direct test of whether the model reads the trace at all.
    """

    batch_size: int = 256
    free: bool = True
    forced: bool = True
    forced_wrong: bool = True
    teacher_forced: bool = True  # answer NLL + argmax match, one forward pass


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: str = "results"
    figures_dir: str = "figures"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**d["model"]),
            optim=OptimConfig(**d["optim"]),
            data=DataConfig(**d["data"]),
            evaluation=EvalConfig(**d["evaluation"]),
            device=d.get("device", "cpu"),
            results_dir=d.get("results_dir", "results"),
            figures_dir=d.get("figures_dir", "figures"),
        )


# ---------------------------------------------------------------------------
# Sweep grids (defaults for sweeps.py; all overridable from the CLI)
# ---------------------------------------------------------------------------

RATIOS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0)
SEEDS: Tuple[int, ...] = (2200, 1337, 2026, 2003, 10)

TASKS_TO_TEST: Tuple[str, ...] = ("word_index", "count_char", "sort_letters", "multiply")
DEFAULT_TASK: str = "word_index"

BATCH_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512)
# The batch sweeps are many more runs than the ratio sweep, so they use fewer
# seeds; the token sweep is the most expensive of all (steps scale as 1/B).
BATCH_SWEEP_SEEDS: Tuple[int, ...] = SEEDS[:1]
TOKEN_SWEEP_SEEDS: Tuple[int, ...] = SEEDS[:1]

REFERENCE_BATCH_SIZE: int = 256
# Fixed-exposure budget for the token sweep: n_steps(B) = SAMPLE_BUDGET / B.
SAMPLE_BUDGET: int = REFERENCE_BATCH_SIZE * OptimConfig.steps  # 1,536,000


# ---------------------------------------------------------------------------
# CLI plumbing -- one flag set shared by every entry point
# ---------------------------------------------------------------------------


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags that override the config tree. Defaults are the dataclass defaults."""
    g = parser.add_argument_group("run configuration")
    g.add_argument("--steps", type=int, default=OptimConfig.steps)
    g.add_argument("--batch", type=int, default=OptimConfig.batch_size)
    g.add_argument("--lr", type=float, default=OptimConfig.learning_rate)
    g.add_argument("--n-layer", type=int, default=ModelConfig.n_layer)
    g.add_argument("--n-embd", type=int, default=ModelConfig.n_embd)
    g.add_argument("--n-head", type=int, default=ModelConfig.n_head)
    g.add_argument("--samples", type=int, default=DataConfig.n_samples)
    g.add_argument("--data-seed", type=int, default=DataConfig.data_seed)
    g.add_argument("--val-seed", type=int, default=DataConfig.val_seed)
    g.add_argument("--val-examples", type=int, default=DataConfig.val_examples)
    g.add_argument("--cache-dir", default=DataConfig.cache_dir)
    g.add_argument("--results-dir", default=Config.results_dir)
    g.add_argument("--figures-dir", default=Config.figures_dir)
    g.add_argument("--device", default=None)
    g.add_argument(
        "--fast-eval",
        action="store_true",
        help="skip the answer-forcing probes; report the end-to-end metric only",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    base = Config()
    return replace(
        base,
        model=ModelConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=base.model.dropout,
        ),
        optim=replace(
            base.optim, steps=args.steps, batch_size=args.batch, learning_rate=args.lr
        ),
        data=replace(
            base.data,
            n_samples=args.samples,
            data_seed=args.data_seed,
            val_seed=args.val_seed,
            val_examples=args.val_examples,
            cache_dir=args.cache_dir,
        ),
        evaluation=replace(
            base.evaluation,
            forced=not args.fast_eval,
            forced_wrong=not args.fast_eval,
            teacher_forced=not args.fast_eval,
        ),
        device=args.device or base.device,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
