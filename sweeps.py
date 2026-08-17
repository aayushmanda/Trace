"""What to run, in what order, and how not to run it twice.

A sweep is a list of `RunSpec`s -- one trained model each -- and an executor
that turns them into records in an append-only store. Everything the five
experiments differ by is which grid of specs they build:

    phase     the ratio axis on one task                      (rho x seed)
    family    the same axis on every task                      (task x rho x seed)
    ablation  supervision conditions A / B / C                 (cond x rho x seed)
    batch     batch size with steps held FIXED                 (B x rho x seed)
    tokens    batch size with steps scaled as budget/B         (B x rho x seed)

Two properties make long sweeps survivable, and both come from the store:

  * **Resumable.** Every finished run is appended to `results/<exp>.jsonl` with
    a uid derived from its spec and config. A re-launched sweep skips what is
    already there, so an interrupted 220-run grid costs only what it had left.
  * **Shared.** Runs are looked up across *all* stores before being trained, so
    the B=256 column of a batch sweep is not retrained if the ratio sweep on the
    same task already produced it.

Pools are loaded once per (task, variant) and mixed once per ratio; models are
freed between runs. The only thing that scales with the length of the sweep is
the results file.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

import config as C
from config import Config
from data import Pool, build_eval_set, load_pools, mix_pools
from engine import evaluate, train
from tasks import get_task

# Ablation condition -> (supervision variant, label used in tables and plots).
CONDITIONS: Dict[str, Tuple[str, str]] = {
    "A": ("full", "L_full (full supervision)"),
    "B": ("answer_only", "L_ans (answer-only loss)"),
    "C": ("direct", "L_direct (no trace)"),
}
VARIANT_TO_CONDITION = {v: c for c, (v, _) in CONDITIONS.items()}


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One trained model. Everything that changes its outcome is a field here."""

    task: str
    variant: str  # full | answer_only | direct
    ratio: Optional[float]  # None for `direct`: no correct/wrong axis exists
    seed: int
    steps: int
    batch_size: int
    n_samples: int
    data_seed: int
    config_hash: str

    @property
    def condition(self) -> str:
        return VARIANT_TO_CONDITION[self.variant]

    @property
    def pool_key(self) -> Tuple:
        return (self.task, self.variant, self.n_samples, self.data_seed)

    def uid(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def label(self) -> str:
        rho = "  --  " if self.ratio is None else f"{self.ratio * 100:5.1f}%"
        return (
            f"{self.task:<12} {self.condition}/{self.variant:<11} rho={rho} "
            f"seed={self.seed:<5} B={self.batch_size:<4} steps={self.steps}"
        )


def config_hash(cfg: Config) -> str:
    """Fingerprint of everything in the config that a RunSpec does not carry.

    Steps, batch size, sample count and data seed live on the spec, so they are
    excluded here; what remains is architecture, optimizer, held-out set and
    which measurements are taken. Two runs with the same uid are the same
    experiment.
    """
    payload = {
        "model": asdict(cfg.model),
        "optim": {
            k: v
            for k, v in asdict(cfg.optim).items()
            if k not in ("steps", "batch_size")
        },
        "val": {"n": cfg.data.val_examples, "seed": cfg.data.val_seed},
        "eval": asdict(cfg.evaluation),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------


class ResultStore:
    """Append-only JSONL of finished runs, plus a tidy CSV mirror."""

    FIELDS = [
        "experiment",
        "task",
        "condition",
        "variant",
        "ratio",
        "seed",
        "steps",
        "batch_size",
        "n_samples",
        "data_seed",
        "free_acc",
        "forced_acc",
        "forced_wrong_acc",
        "answer_nll",
        "answer_tf_acc",
        "no_answer_rate",
        "truncation_rate",
        "train_loss",
        "seconds",
        "uid",
        "config_hash",
    ]

    def __init__(self, experiment: str, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self.experiment = experiment
        self.results_dir = results_dir
        self.path = os.path.join(results_dir, f"{experiment}.jsonl")
        self.records: List[dict] = read_records(self.path)
        self._own = {r["uid"] for r in self.records}

    def has(self, spec: RunSpec) -> bool:
        return spec.uid() in self._own

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        self._own.add(record["uid"])

    def to_csv(self) -> str:
        import csv

        path = os.path.join(self.results_dir, f"{self.experiment}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        return path


def read_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_all_records(results_dir: str) -> List[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        records.extend(read_records(path))
    return records


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def _free(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


def execute(specs: Sequence[RunSpec], cfg: Config, experiment: str) -> ResultStore:
    """Trains and evaluates every spec that is not already on disk."""
    store = ResultStore(experiment, cfg.results_dir)
    elsewhere = {r["uid"]: r for r in read_all_records(cfg.results_dir)}

    # Sort so that consecutive runs share a pool, then a mix: pools are loaded
    # once per (task, variant) and mixed once per ratio.
    ordered = sorted(
        specs,
        key=lambda s: (s.task, s.variant, s.n_samples, s.data_seed, s.ratio or 0.0, s.batch_size, s.steps, s.seed),
    )

    todo = [s for s in ordered if not store.has(s)]
    print(
        f"\n=== {experiment}: {len(ordered)} runs "
        f"({len(ordered) - len(todo)} already in {store.path}) ===\n"
    )

    pools: Optional[Dict[str, Pool]] = None
    pool_key = mix_key = None
    train_pool: Optional[Pool] = None
    instances = None
    eval_key = None

    started, done = time.time(), 0
    for spec in todo:
        # 1. reuse an identical run from another experiment's store if there is one
        shared = elsewhere.get(spec.uid())
        if shared is not None:
            record = dict(shared, experiment=experiment, reused_from=shared.get("experiment"))
            store.append(record)
            print(f"  [reuse] {spec.label()}  <- {shared.get('experiment')}")
            continue

        task = get_task(spec.task)

        # 2. held-out set (cached on disk, shared by every run on this task)
        if eval_key != spec.task:
            instances = build_eval_set(
                task, cfg.data.val_examples, cfg.data.val_seed, cfg.data.cache_dir
            )
            eval_key = spec.task

        # 3. pools: generated once, then read from disk
        if pool_key != spec.pool_key:
            _free(train_pool, pools)
            train_pool, pools, mix_key = None, None, None
            pools = load_pools(
                task,
                variant=spec.variant,
                n_samples=spec.n_samples,
                data_seed=spec.data_seed,
                cache_dir=cfg.data.cache_dir,
                device=cfg.device,
            )
            pool_key = spec.pool_key

        # 4. the ratio mix (no-op for the direct condition)
        if mix_key != (spec.pool_key, spec.ratio):
            _free(train_pool)
            train_pool = (
                pools["direct"]
                if spec.variant == "direct"
                else mix_pools(pools["correct"], pools["wrong"], spec.ratio)
            )
            mix_key = (spec.pool_key, spec.ratio)

        # 5. train, measure, record
        t0 = time.time()
        model, train_loss = train(
            train_pool, task, cfg, spec.seed, steps=spec.steps, batch_size=spec.batch_size
        )
        metrics = evaluate(
            model, task, instances, cfg, trace_in_context=(spec.variant != "direct")
        )
        metrics.train_loss = train_loss
        _free(model)

        record = {
            "experiment": experiment,
            "condition": spec.condition,
            **asdict(spec),
            **metrics.as_dict(),
            "seconds": round(time.time() - t0, 1),
            "uid": spec.uid(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        store.append(record)
        elsewhere[spec.uid()] = record

        done += 1
        eta = (time.time() - started) / done * (len(todo) - done)
        print(
            f"  [{done:>3}/{len(todo)}] {spec.label()}  "
            f"free={_pct(metrics.free_acc)} forced={_pct(metrics.forced_acc)} "
            f"forced_wrong={_pct(metrics.forced_wrong_acc)} nll={_num(metrics.answer_nll)}  "
            f"({record['seconds']:.0f}s, eta {_fmt_hms(eta)})"
        )

    _free(train_pool, pools)
    path = store.to_csv()
    print(f"\n[store] {len(store.records)} records -> {store.path} and {path}")
    return store


def _pct(x: Optional[float]) -> str:
    return "  --  " if x is None else f"{x * 100:5.1f}%"


def _num(x: Optional[float]) -> str:
    return " -- " if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


@dataclass
class GridOptions:
    """The axes an experiment sweeps over. Defaults come from config.py."""

    tasks: Sequence[str] = C.TASKS_TO_TEST
    ratios: Sequence[float] = C.RATIOS
    seeds: Sequence[int] = C.SEEDS
    batch_sizes: Sequence[int] = C.BATCH_SIZES
    conditions: Sequence[str] = ("A", "B", "C")
    budget: int = C.SAMPLE_BUDGET


def _spec(cfg: Config, chash: str, **kw) -> RunSpec:
    base = dict(
        variant="full",
        steps=cfg.optim.steps,
        batch_size=cfg.optim.batch_size,
        n_samples=cfg.data.n_samples,
        data_seed=cfg.data.data_seed,
        config_hash=chash,
    )
    return RunSpec(**{**base, **kw})


def grid_ratio(cfg: Config, opts: GridOptions) -> List[RunSpec]:
    """The ratio axis, on one task (`phase`) or on all of them (`family`)."""
    chash = config_hash(cfg)
    return [
        _spec(cfg, chash, task=t, ratio=r, seed=s)
        for t in opts.tasks
        for r in opts.ratios
        for s in opts.seeds
    ]


def grid_ablation(cfg: Config, opts: GridOptions) -> List[RunSpec]:
    """Conditions A and B over the ratio axis; C is a single baseline per seed.

    C has no ratio axis on purpose: dropping the trace makes a correct-trace and
    a wrong-trace example byte-identical, so rho has nothing left to vary.
    """
    chash = config_hash(cfg)
    specs: List[RunSpec] = []
    for task in opts.tasks:
        for cond in opts.conditions:
            variant, _ = CONDITIONS[cond]
            if cond == "C":
                specs += [
                    _spec(cfg, chash, task=task, variant=variant, ratio=None, seed=s)
                    for s in opts.seeds
                ]
            else:
                specs += [
                    _spec(cfg, chash, task=task, variant=variant, ratio=r, seed=s)
                    for r in opts.ratios
                    for s in opts.seeds
                ]
    return specs


def grid_batch(cfg: Config, opts: GridOptions) -> List[RunSpec]:
    """Batch size with steps held FIXED -- the rho_c(B) ~ B^-1/2 test."""
    chash = config_hash(cfg)
    return [
        _spec(cfg, chash, task=t, ratio=r, seed=s, batch_size=b)
        for t in opts.tasks
        for b in opts.batch_sizes
        for r in opts.ratios
        for s in opts.seeds
    ]


def grid_tokens(cfg: Config, opts: GridOptions) -> List[RunSpec]:
    """Batch size at a fixed sample budget: steps(B) = budget / B.

    Holding total exposure constant isolates minibatch gradient variance from
    cumulative compute, which is what the SNR account of rho_c(B) is about.
    """
    chash = config_hash(cfg)
    return [
        _spec(cfg, chash, task=t, ratio=r, seed=s, batch_size=b, steps=int(opts.budget / b))
        for t in opts.tasks
        for b in opts.batch_sizes
        for r in opts.ratios
        for s in opts.seeds
    ]


EXPERIMENTS = {
    "phase": grid_ratio,
    "family": grid_ratio,
    "ablation": grid_ablation,
    "batch": grid_batch,
    "tokens": grid_tokens,
}

# Per-experiment defaults for the axes the CLI did not pin down.
DEFAULT_OPTIONS = {
    "phase": dict(tasks=(C.DEFAULT_TASK,), seeds=C.SEEDS),
    "family": dict(tasks=C.TASKS_TO_TEST, seeds=C.SEEDS),
    "ablation": dict(tasks=C.TASKS_TO_TEST, seeds=C.SEEDS),
    "batch": dict(tasks=(C.DEFAULT_TASK,), seeds=C.BATCH_SWEEP_SEEDS),
    "tokens": dict(tasks=(C.DEFAULT_TASK,), seeds=C.TOKEN_SWEEP_SEEDS),
}


def build_specs(experiment: str, cfg: Config, opts: GridOptions) -> List[RunSpec]:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment!r}; have {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[experiment](cfg, opts)


def run(experiment: str, cfg: Config, opts: GridOptions) -> ResultStore:
    specs = build_specs(experiment, cfg, opts)
    print_plan(experiment, cfg, opts, specs)
    return execute(specs, cfg, experiment)


def print_plan(experiment: str, cfg: Config, opts: GridOptions, specs: Sequence[RunSpec]) -> None:
    from engine import GPT

    tasks = sorted({s.task for s in specs})
    params = {t: GPT.for_task(get_task(t), cfg).n_params() for t in tasks}
    print("\n" + "=" * 78)
    print(f" EXPERIMENT: {experiment}   ({len(specs)} runs)")
    print("=" * 78)
    for t in tasks:
        task = get_task(t)
        print(
            f"   {task.name:<13} {task.description}\n"
            f"   {'':<13} vocab={task.tokenizer.vocab_size} block={task.block_size} "
            f"params={params[t]:,} chance={task.chance_acc:.4f} ceiling={task.ceiling_acc:.4f}"
        )
    print(
        f"   conditions={sorted({s.condition for s in specs})}  "
        f"batch_sizes={sorted({s.batch_size for s in specs})}  "
        f"seeds={list(opts.seeds)}"
    )
    print(
        f"   samples={cfg.data.n_samples:,}  device={cfg.device}  "
        f"cache={cfg.data.cache_dir}  results={cfg.results_dir}"
    )
    print("=" * 78)
