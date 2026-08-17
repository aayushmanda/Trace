"""Dataset generation, the on-disk cache, ratio mixing and batching.

Everything here is task-agnostic: a Task supplies the sampler and the
tokenizer, this module supplies the loss mask, the cache, the ratio mix and the
batcher.

GENERATE ONCE, THEN READ FROM DISK. Pools are expensive to sample and must be
byte-identical across every run that claims to share data, so they are written
to `cache_dir` and never regenerated once present. A pool is stored compactly
as the padded token rows (uint8) plus the index at which the loss mask turns on
(int16) -- ~66 bytes per example instead of the ~380 that materialised x/y/mask
tensors would take -- and the training tensors are rebuilt *per batch* on the
GPU. A 1.5M-sample pool is therefore ~100 MB of device memory rather than
~2.3 GB, which is what makes holding correct and wrong pools resident at once
practical.

Cache files are keyed by task, variant, seed, block size AND a fingerprint of
the sampler's output, so editing a sampler invalidates its cache instead of
silently reusing stale data. A cache of N samples also serves any run that asks
for fewer, because rows are sampled i.i.d.: the first n rows of a big pool are
a valid pool of size n.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tasks import Instance, Task, get_task

CACHE_FORMAT = 2

# The three ablation conditions differ in what the sequence contains and where
# the loss mask starts:
#   full        -- prompt + trace + answer, loss on trace AND answer  (L_full)
#   answer_only -- prompt + trace + answer, loss on answer only       (L_ans)
#   direct      -- prompt + answer, no trace at all                   (L_direct)
VARIANTS: Tuple[str, ...] = ("full", "answer_only", "direct")

# "direct" drops the trace, so a correct-trace and a wrong-trace example become
# identical and there is nothing left for the ratio axis to vary.
POOL_NAMES: Dict[str, Tuple[str, ...]] = {
    "full": ("correct", "wrong"),
    "answer_only": ("correct", "wrong"),
    "direct": ("direct",),
}

RENDER_MODE = {"correct": "correct_think", "wrong": "wrong_think", "direct": "no_think"}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_row(task: Task, prompt: str, answer: str, supervise: str = "full"):
    """Encodes one example to (padded token row, mask boundary, truncated).

    supervise="full"        -> loss on every token after the prompt
    supervise="answer_only" -> loss only after the final ':'; the trace is still
                               fed as context but receives no gradient.
    """
    tok = task.tokenizer
    ids = tok.encode(prompt + answer + "\n")
    full = ids[: task.block_size]
    truncated = len(ids) > len(full)

    if supervise == "full":
        boundary_text = prompt
    elif supervise == "answer_only":
        head, sep, _ = answer.rpartition(":")
        boundary_text = prompt + head + sep if sep else prompt
    else:
        raise ValueError(f"unknown supervise mode {supervise!r}")

    boundary = min(len(tok.encode(boundary_text)), len(full))
    row = full + [tok.pad_id] * (task.block_size - len(full))
    return row, boundary, truncated


def expand(tokens: torch.Tensor, boundary: torch.Tensor, pad_id: int):
    """Rebuilds (x, y, mask) from the compact representation, on its device.

    The mask is on for every target position at or after the boundary that is
    not padding, which is exactly the definition of the two supervision
    conditions -- there is no second implementation of it anywhere.
    """
    tokens = tokens.long()
    boundary = boundary.long()

    x, y = tokens[:, :-1], tokens[:, 1:]
    positions = torch.arange(1, tokens.shape[1], device=tokens.device).unsqueeze(0)
    mask = (positions >= boundary.unsqueeze(1)) & (y != pad_id)
    return x.contiguous(), y.contiguous(), mask.float()


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@dataclass
class Pool:
    """A training pool in its compact form, plus the batcher that expands it."""

    tokens: torch.Tensor  # uint8 [N, block_size]
    boundary: torch.Tensor  # int16 [N]
    pad_id: int

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    def to(self, device) -> "Pool":
        return Pool(self.tokens.to(device), self.boundary.to(device), self.pad_id)

    def take(self, n: int) -> "Pool":
        return Pool(self.tokens[:n], self.boundary[:n], self.pad_id)

    def batch(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Uniform-with-replacement minibatch, expanded to (x, y, mask)."""
        idx = torch.randint(
            0, len(self), (batch_size,), device=self.device, generator=generator
        )
        return expand(self.tokens[idx], self.boundary[idx], self.pad_id)

    def nbytes(self) -> int:
        return self.tokens.numel() + 2 * self.boundary.numel()


def mix_pools(correct: Pool, wrong: Pool, ratio: float) -> Pool:
    """The ratio axis: the first `ratio` of the rows keep their correct trace.

    Both pools come from the *same* sampled instances, so row i holds the same
    problem under both trace regimes and the mix varies trace quality alone --
    never the problem distribution.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    if len(correct) != len(wrong):
        raise ValueError("correct and wrong pools must have the same length")

    n_correct = int(len(correct) * ratio)
    return Pool(
        torch.cat([correct.tokens[:n_correct], wrong.tokens[n_correct:]], dim=0),
        torch.cat([correct.boundary[:n_correct], wrong.boundary[n_correct:]], dim=0),
        correct.pad_id,
    )


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _cache_stem(task: Task, data_seed: int, variant: str) -> str:
    return (
        f"{task.name}_{variant}_seed{data_seed}_b{task.block_size}"
        f"_v{CACHE_FORMAT}_{task.fingerprint()}"
    )


def cache_path(task, n_samples: int, data_seed: int, variant: str, cache_dir: str) -> str:
    stem = _cache_stem(get_task(task), data_seed, variant)
    return os.path.join(cache_dir, f"{stem}_n{n_samples}.pt")


def find_cached(
    task, n_samples: int, data_seed: int, variant: str, cache_dir: str
) -> Optional[str]:
    """Smallest cached pool with at least n_samples rows, or None."""
    task = get_task(task)
    stem = _cache_stem(task, data_seed, variant)
    candidates = []
    for path in glob.glob(os.path.join(cache_dir, f"{stem}_n*.pt")):
        m = re.search(r"_n(\d+)\.pt$", path)
        if m and int(m.group(1)) >= n_samples:
            candidates.append((int(m.group(1)), path))
    return min(candidates)[1] if candidates else None


def _generate_blob(task: Task, n_samples: int, data_seed: int, variant: str, verbose: bool):
    """Samples every pool the variant needs, as compact CPU tensors."""
    names = POOL_NAMES[variant]
    # "direct" has no trace, so the supervision mask is the same either way.
    supervise = "full" if variant == "direct" else variant

    if verbose:
        print(
            f"[gen ] {task.name}/{variant}: {n_samples:,} samples "
            f"-- this happens once, then it is read from disk."
        )

    rng_state = random.getstate()
    random.seed(data_seed)

    tokens = {n: torch.empty((n_samples, task.block_size), dtype=torch.uint8) for n in names}
    bounds = {n: torch.empty(n_samples, dtype=torch.int16) for n in names}
    n_truncated = 0
    t0 = time.time()

    for i in range(n_samples):
        inst = task.sample()
        truncated = False
        for name in names:
            prompt, answer = task.render(inst, RENDER_MODE[name])
            row, boundary, trunc = encode_row(task, prompt, answer, supervise)
            tokens[name][i] = torch.tensor(row, dtype=torch.uint8)
            bounds[name][i] = boundary
            truncated = truncated or trunc
        n_truncated += int(truncated)

        if verbose and n_samples >= 200_000 and (i + 1) % 250_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"       {i + 1:,}/{n_samples:,} ({rate:,.0f}/s)")

    random.setstate(rng_state)

    if n_truncated:
        print(
            f"[WARN] {n_truncated:,}/{n_samples:,} examples exceeded "
            f"block_size={task.block_size} for task={task.name} and lost their "
            f"answer. Raise Task.block_size and delete the cache."
        )

    return {
        "pools": {n: {"tokens": tokens[n], "boundary": bounds[n]} for n in names},
        "meta": {
            "task": task.name,
            "variant": variant,
            "n_samples": n_samples,
            "data_seed": data_seed,
            "block_size": task.block_size,
            "vocab_size": task.tokenizer.vocab_size,
            "pad_id": task.tokenizer.pad_id,
            "fingerprint": task.fingerprint(),
            "n_truncated": n_truncated,
            "format": CACHE_FORMAT,
        },
    }


def _load_or_generate(task, n_samples, data_seed, variant, cache_dir, use_cache, verbose):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; have {VARIANTS}")
    if task.tokenizer.vocab_size > 255:
        raise ValueError("the uint8 cache needs vocab_size <= 255")

    if use_cache and cache_dir:
        hit = find_cached(task, n_samples, data_seed, variant, cache_dir)
        if hit:
            blob = torch.load(hit, map_location="cpu", weights_only=True)
            if verbose:
                have = blob["meta"]["n_samples"]
                extra = f", using first {n_samples:,}" if have > n_samples else ""
                print(f"[cache] {os.path.basename(hit)} ({have:,} samples{extra})")
            return blob

    blob = _generate_blob(task, n_samples, data_seed, variant, verbose)
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = cache_path(task, n_samples, data_seed, variant, cache_dir)
        tmp = path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)  # never leave a half-written cache behind
        if verbose:
            print(f"[cache] wrote {path} ({os.path.getsize(path) / 1e6:,.0f} MB)")
    return blob


def load_pools(
    task,
    variant: str = "full",
    n_samples: int = 1_536_000,
    data_seed: int = 100,
    cache_dir: str = "dataset_cache",
    device: str = "cpu",
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, Pool]:
    """{"correct": Pool, "wrong": Pool} for A/B, {"direct": Pool} for C.

    The pools are returned on `device` in their compact form; expansion into
    x/y/mask happens per batch.
    """
    task = get_task(task)
    blob = _load_or_generate(
        task, n_samples, data_seed, variant, cache_dir, use_cache, verbose
    )
    pad_id = blob["meta"]["pad_id"]
    return {
        name: Pool(
            pool["tokens"][:n_samples].to(device),
            pool["boundary"][:n_samples].to(device),
            pad_id,
        )
        for name, pool in blob["pools"].items()
    }


# ---------------------------------------------------------------------------
# Held-out evaluation set
# ---------------------------------------------------------------------------


def eval_set_path(task: Task, n: int, seed: int, cache_dir: str) -> str:
    return os.path.join(
        cache_dir, f"eval_{task.name}_seed{seed}_n{n}_{task.fingerprint()}.json"
    )


def build_eval_set(
    task,
    n: int = 1000,
    seed: int = 999_999,
    cache_dir: Optional[str] = "dataset_cache",
    verbose: bool = False,
) -> List[Instance]:
    """Held-out instances, deduplicated on the prompt and cached to disk.

    Both traces are kept for every instance, so the end-to-end metric and both
    answer-forcing probes are measured on *the same problems* -- differences
    between them are then differences in what the model was conditioned on,
    never differences in the sample.
    """
    task = get_task(task)
    path = eval_set_path(task, n, seed, cache_dir) if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            return [Instance(**d) for d in json.load(f)]

    rng_state = random.getstate()
    random.seed(seed)

    instances, seen = [], set()
    attempts, max_attempts = 0, n * 200
    while len(instances) < n and attempts < max_attempts:
        attempts += 1
        inst = task.sample()
        if inst.prompt in seen:
            continue
        seen.add(inst.prompt)
        instances.append(inst)

    random.setstate(rng_state)
    if len(instances) < n:
        print(
            f"[WARN] only {len(instances)} unique prompts available for "
            f"task={task.name} (asked for {n})."
        )

    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump([i.__dict__ for i in instances], f)
        os.replace(tmp, path)
        if verbose:
            print(f"[cache] wrote {path} ({len(instances)} instances)")
    return instances


def teacher_forcing_rows(task: Task, instances: Sequence[Instance], mode: str, device):
    """(tokens, boundary) for scoring the gold answer with the context supplied.

    The boundary is placed at the end of `task.context(inst, mode)`, i.e. the
    same prefix the generation-based probe is conditioned on, so the NLL and the
    argmax match cover the gold answer and its newline and nothing else.
    """
    gold_mode = {
        "free": "correct_think",
        "forced_correct": "correct_think",
        "forced_wrong": "wrong_think",
        "direct": "no_think",
    }[mode]

    rows, bounds = [], []
    tok = task.tokenizer
    for inst in instances:
        context = task.context(inst, mode)
        prompt, answer = task.render(inst, gold_mode)
        full = tok.encode(prompt + answer + "\n")[: task.block_size]
        rows.append(full + [tok.pad_id] * (task.block_size - len(full)))
        bounds.append(min(len(tok.encode(context)), len(full)))

    return (
        torch.tensor(rows, dtype=torch.uint8, device=device),
        torch.tensor(bounds, dtype=torch.int16, device=device),
    )


# ---------------------------------------------------------------------------
# Cache maintenance (used by `main.py prepare` / `main.py cache`)
# ---------------------------------------------------------------------------


def prepare_cache(
    task_names: Sequence[str],
    variants: Sequence[str],
    n_samples: int,
    data_seed: int,
    cache_dir: str,
    val_examples: int = 1000,
    val_seed: int = 999_999,
    force: bool = False,
) -> None:
    """Materialises every pool and eval set a sweep will ask for, once."""
    os.makedirs(cache_dir, exist_ok=True)

    for name in task_names:
        task = get_task(name)
        longest = task.max_example_len(n_probe=20_000)
        if longest > task.block_size:
            print(
                f"[SKIP] {task.name}: longest example {longest} > "
                f"block_size {task.block_size}; fix the task before caching."
            )
            continue

        build_eval_set(task, val_examples, val_seed, cache_dir, verbose=True)

        for variant in variants:
            path = cache_path(task, n_samples, data_seed, variant, cache_dir)
            hit = find_cached(task, n_samples, data_seed, variant, cache_dir)
            if hit and not force:
                print(f"[skip ] {task.name}/{variant}: covered by {os.path.basename(hit)}")
                continue
            if force and os.path.exists(path):
                os.remove(path)

            t0 = time.time()
            pools = load_pools(
                task,
                variant=variant,
                n_samples=n_samples,
                data_seed=data_seed,
                cache_dir=cache_dir,
                device="cpu",
            )
            del pools
            print(f"[done ] {task.name}/{variant} in {time.time() - t0:.0f}s\n")

    describe_cache(cache_dir)


def describe_cache(cache_dir: str) -> None:
    if not os.path.isdir(cache_dir):
        print(f"No cache directory at {cache_dir!r}")
        return
    entries = sorted(f for f in os.listdir(cache_dir) if f.endswith((".pt", ".json")))
    if not entries:
        print(f"Cache directory {cache_dir!r} is empty")
        return

    total = 0
    print(f"\n{'file':<66} | {'size':>9}")
    print("-" * 80)
    for name in entries:
        size = os.path.getsize(os.path.join(cache_dir, name))
        total += size
        print(f"{name:<66} | {size / 1e6:>6,.1f} MB")
    print("-" * 80)
    print(f"{'total':<66} | {total / 1e6:>6,.1f} MB\n")
