"""Dataset generation, the on-disk cache, ratio mixing and batching.

Everything here is task-agnostic: a Task supplies the sampler and the
tokenizer, this module supplies the loss mask, the cache, the ratio mix and the
batcher.

GENERATE ONCE, THEN READ FROM DISK. Pools are expensive to sample and must be
byte-identical across every run that claims to share data, so they are written
to `cache_dir` and never regenerated once present. Sampling lives in exactly
three functions -- `ensure_pool_cached`, `ensure_index_stream_cached` and
`build_eval_set` -- and every training path calls the loaders with
`require_cache=True`, which turns a missing file into a `CacheMiss` naming the
`prepare` command instead of quietly sampling 1.5M examples with a GPU idling
behind it. A pool is stored compactly
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

    def gather(self, idx: torch.Tensor):
        """The rows named by `idx`, expanded to (x, y, mask)."""
        idx = idx.to(self.device, dtype=torch.long)
        return expand(self.tokens[idx], self.boundary[idx], self.pad_id)

    def batch(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Uniform-with-replacement minibatch (the `iid` sampling mode)."""
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


def _save_atomically(obj, path: str, verbose: bool) -> str:
    """Writes to `path` via a temp file, so a half-written cache never appears."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    if verbose:
        print(f"[cache] wrote {path} ({os.path.getsize(path) / 1e6:,.0f} MB)")
    return path


class CacheMiss(FileNotFoundError):
    """Something a run needed was not on disk and the run refused to sample it.

    Raised only under `require_cache=True`, which is what every training path
    uses: sampling is a `prepare` step, never something a sweep does with a GPU
    sitting idle behind it.
    """


def _missing(what: str, command: str) -> CacheMiss:
    return CacheMiss(
        f"{what} is not on disk, and this run will not sample it while training.\n"
        f"       Generate it first with:\n\n           {command}\n\n"
        f"       (or pass --allow-runtime-generation to sample it inline.)"
    )


def _prepare_command(task, n_samples: int, data_seed: int, variant: str, cache_dir: str) -> str:
    return (
        f"python main.py prepare --tasks {get_task(task).name} --variants {variant} "
        f"--samples {n_samples} --data-seed {data_seed} --cache-dir {cache_dir}"
    )


def ensure_pool_cached(
    task,
    n_samples: int = 1_536_000,
    data_seed: int = 100,
    variant: str = "full",
    cache_dir: str = "dataset_cache",
    verbose: bool = True,
) -> str:
    """Guarantees the pool file exists on disk; returns its path.

    This is the ONLY place a training pool is ever sampled. Everything else
    reads the file it writes, so generation happens in one visible phase up
    front rather than in the middle of a sweep.
    """
    task = get_task(task)
    _check_variant(task, variant)
    hit = find_cached(task, n_samples, data_seed, variant, cache_dir)
    if hit:
        return hit
    blob = _generate_blob(task, n_samples, data_seed, variant, verbose)
    return _save_atomically(
        blob, cache_path(task, n_samples, data_seed, variant, cache_dir), verbose
    )


def _check_variant(task: Task, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; have {VARIANTS}")
    if task.tokenizer.vocab_size > 255:
        raise ValueError("the uint8 cache needs vocab_size <= 255")


def _load_or_generate(
    task, n_samples, data_seed, variant, cache_dir, use_cache, verbose, require_cache
):
    _check_variant(task, variant)

    if use_cache and cache_dir:
        hit = find_cached(task, n_samples, data_seed, variant, cache_dir)
        if hit is None and require_cache:
            raise _missing(
                f"the {task.name}/{variant} pool ({n_samples:,} samples, "
                f"data_seed={data_seed})",
                _prepare_command(task, n_samples, data_seed, variant, cache_dir),
            )
        if hit is None:
            hit = ensure_pool_cached(task, n_samples, data_seed, variant, cache_dir, verbose)
        blob = torch.load(hit, map_location="cpu", weights_only=True)
        if verbose:
            have = blob["meta"]["n_samples"]
            extra = f", using first {n_samples:,}" if have > n_samples else ""
            print(f"[read ] {os.path.basename(hit)} ({have:,} samples{extra})")
        return blob

    if require_cache:
        raise CacheMiss(
            "require_cache=True but caching is disabled (use_cache=False or no "
            "cache_dir); there is no file to read the pool from."
        )
    return _generate_blob(task, n_samples, data_seed, variant, verbose)


def load_pools(
    task,
    variant: str = "full",
    n_samples: int = 1_536_000,
    data_seed: int = 100,
    cache_dir: str = "dataset_cache",
    device: str = "cpu",
    use_cache: bool = True,
    verbose: bool = True,
    require_cache: bool = False,
) -> Dict[str, Pool]:
    """{"correct": Pool, "wrong": Pool} for A/B, {"direct": Pool} for C.

    The pools are returned on `device` in their compact form; expansion into
    x/y/mask happens per batch. With `require_cache=True` a missing file is an
    error rather than a licence to sample 1.5M examples mid-sweep.
    """
    task = get_task(task)
    blob = _load_or_generate(
        task, n_samples, data_seed, variant, cache_dir, use_cache, verbose, require_cache
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
# Training order: which examples each step sees, generated once and saved
# ---------------------------------------------------------------------------
#
# The pool says *what* the training data is; this says *in what order it is
# consumed*, and it is cached on the same terms. Drawing it fresh per run made
# the sample order a second, uncontrolled source of run-to-run variance on top
# of model initialisation. A saved stream removes it:
#
#   * two seeds at the same ratio now differ only in initialisation, so the
#     seed spread is optimization noise and nothing else;
#   * two ratios now see the *same problems in the same order* -- only the
#     trace quality of some rows differs -- which makes the ratio curve a
#     paired comparison and sharply reduces the noise in the fitted rho_c;
#   * two batch sizes consume the same stream, differently chunked, because a
#     shorter stream is a prefix of a longer one.
#
# The stream is a concatenation of independent shuffles of the pool, so every
# example is seen equally often (sampling without replacement within an epoch)
# rather than Poisson-many times.


def index_stream_path(n_samples: int, length: int, order_seed: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"order_N{n_samples}_seed{order_seed}_len{length}.pt")


def find_cached_stream(n_samples: int, length: int, order_seed: int, cache_dir: str):
    """Shortest saved stream that is at least `length` long, or None."""
    pattern = os.path.join(cache_dir, f"order_N{n_samples}_seed{order_seed}_len*.pt")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"_len(\d+)\.pt$", path)
        if m and int(m.group(1)) >= length:
            candidates.append((int(m.group(1)), path))
    return min(candidates)[1] if candidates else None


def _build_index_stream(n_samples: int, length: int, order_seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(order_seed)
    chunks, total = [], 0
    while total < length:
        chunks.append(torch.randperm(n_samples, generator=generator))
        total += n_samples
    return torch.cat(chunks)[:length].to(torch.int32)


def ensure_index_stream_cached(
    n_samples: int,
    length: int,
    order_seed: int,
    cache_dir: str = "dataset_cache",
    verbose: bool = True,
) -> str:
    """Guarantees the training order file exists; returns its path."""
    hit = find_cached_stream(n_samples, length, order_seed, cache_dir)
    if hit:
        return hit
    stream = _build_index_stream(n_samples, length, order_seed)
    return _save_atomically(
        stream, index_stream_path(n_samples, length, order_seed, cache_dir), verbose
    )


def load_index_stream(
    n_samples: int,
    length: int,
    order_seed: int,
    cache_dir: Optional[str] = "dataset_cache",
    device: str = "cpu",
    verbose: bool = True,
    require_cache: bool = False,
) -> torch.Tensor:
    """The saved training order: an int32 tensor of `length` row indices.

    A longer stream's prefix is the shorter stream, so one file serves every
    (steps, batch size) combination that fits inside it.
    """
    if cache_dir:
        hit = find_cached_stream(n_samples, length, order_seed, cache_dir)
        if hit is None and require_cache:
            raise _missing(
                f"the training order for N={n_samples:,}, order_seed={order_seed}, "
                f"length={length:,}",
                f"python main.py prepare --samples {n_samples} --order-seed {order_seed} "
                f"--cache-dir {cache_dir}",
            )
        if hit is None:
            hit = ensure_index_stream_cached(
                n_samples, length, order_seed, cache_dir, verbose
            )
        stream = torch.load(hit, map_location="cpu", weights_only=True)
        if verbose:
            print(f"[read ] {os.path.basename(hit)} (using first {length:,})")
        return stream[:length].to(device)

    if require_cache:
        raise CacheMiss(
            "require_cache=True but no cache_dir was given; there is no file to "
            "read the training order from."
        )
    return _build_index_stream(n_samples, length, order_seed).to(device)


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
    require_cache: bool = False,
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

    if require_cache:
        raise _missing(
            f"the held-out set for {task.name} (n={n}, val_seed={seed})",
            f"python main.py prepare --tasks {task.name} --val-examples {n} "
            f"--val-seed {seed} --cache-dir {cache_dir}",
        )

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
    order_length: int = 0,
    order_seed: int = 0,
    force: bool = False,
) -> None:
    """Materialises every pool, eval set and training order a sweep asks for."""
    os.makedirs(cache_dir, exist_ok=True)

    if order_length:
        ensure_index_stream_cached(n_samples, order_length, order_seed, cache_dir)

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
            ensure_pool_cached(task, n_samples, data_seed, variant, cache_dir)
            print(f"[done ] {task.name}/{variant} in {time.time() - t0:.0f}s\n")

    describe_cache(cache_dir)


def decode_pool_rows(task: Task, pool: Pool, idx: Sequence[int]):
    """[(full sequence, the part the loss is taken on)] for the named rows.

    What the model actually consumes at a given step, as text -- the check that
    a cached pool, a ratio mix and a saved training order compose into the
    examples you think they do.
    """
    tok = task.tokenizer
    out = []
    for i in idx:
        row = pool.tokens[int(i)].tolist()
        boundary = int(pool.boundary[int(i)])
        ids = [t for t in row if t != tok.pad_id]
        out.append((tok.decode(ids), tok.decode(ids[boundary:])))
    return out


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
