"""Training loop for the character-token GPT.
"""
import csv
from pathlib import Path

def append_rows(path, rows):
    """Append dict rows; divert to a sibling file if the header does not match."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if path.exists():
        with path.open(newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing is not None and existing != fields:
            path = path.with_name(f"{path.stem}__schema{len(fields)}{path.suffix}")
            print(f"schema differs from the existing file; writing {path}", flush=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path

def append_csv(path, row: dict):
    return append_rows(path, [row])

def write_csv(path, rows):
    """Write dict rows to `path`. Rows may have heterogeneous keys (e.g. one
    condition logging extra fields another doesn't); the header is the union
    of all keys, in first-seen order, and missing cells are written empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path

"""Thin YAML/JSON config loader. CLI kwargs override file keys."""
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    text = path.read_text()
    if path.suffix in {".json"}:
        import json
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return _simple_yaml(text)

def _simple_yaml(text):
    """Enough YAML for this repo's configs (nested maps, inline lists)."""
    import ast

    def coerce(raw):
        raw = raw.split("#", 1)[0].strip()
        if raw == "":
            return {}
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            if raw in {"true", "True"}:
                return True
            if raw in {"false", "False"}:
                return False
            return raw.strip("'\"")

    root = {}
    stack = [( -1, root)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = coerce(rest)
        parent[key.strip()] = value
        if value == {}:
            stack.append((indent, parent[key.strip()]))
    return root

def merge_dict(base, override):
    out = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out

def flatten_experiment(cfg):
    """Lift model/train/data blocks into one namespace for argparse-style code."""
    flat = {}
    for block in ("experiment", "model", "train", "data"):
        if isinstance(cfg.get(block), dict):
            flat.update(cfg[block])
    for key, value in cfg.items():
        if key not in {"experiment", "model", "train", "data"} and not isinstance(value, dict):
            flat[key] = value
    return flat

def load_experiment(path, cli=None):
    cfg = load_yaml(path) if path else {}
    flat = flatten_experiment(cfg)
    if cli:
        for key, value in vars(cli).items() if not isinstance(cli, dict) else cli.items():
            if key in {"config", "action"}:
                continue
            if value is not None:
                flat[key] = value
    return Namespace(**flat), cfg

"""Optional in-process DataParallel. Default remains one GPU.

Not torchrun/DDP: these nets are tiny; splitting a global batch across 2 GPUs is enough.
Probes, generate, and induced-rule stay on the eager module on the primary device.
"""
import argparse
import os
import warnings

import torch
from torch import nn

def parse_trace_devices():
    raw = os.environ.get("TRACE_DEVICES")
    if raw is None or raw.strip() == "":
        return None
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids or None

def distributed_enabled(explicit=None):
    """Off unless `--distributed` / YAML `distributed: true` / TRACE_DEVICES lists 2+ GPUs.

    TRACE_DISTRIBUTED=0/1 wins (same pattern as TRACE_COMPILE). `--no-distributed` wins
    over TRACE_DEVICES.
    """
    env = os.environ.get("TRACE_DISTRIBUTED")
    if env is not None and env != "":
        return env.strip().lower() not in {"0", "false", "off", "no"}
    if explicit is not None:
        return bool(explicit)
    ids = parse_trace_devices()
    return bool(ids) and len(ids) >= 2

def distributed_device_ids(enabled=None):
    """CUDA index list for DataParallel, or None for single-device.

    Does not grab all visible A100s: default is two cards, preferring physical 2 and 3.
    If CUDA_VISIBLE_DEVICES is set, indices are in that remapped space (cuda:0 is the first visible).
    """
    if not distributed_enabled(explicit=enabled):
        return None
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n < 2:
        return None
    env_ids = parse_trace_devices()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    remapped = visible is not None and visible != ""
    if remapped:
        want = len(env_ids) if env_ids and len(env_ids) >= 2 else 2
        ids = list(range(min(want, n)))
    elif env_ids:
        ids = [i for i in env_ids if 0 <= i < n]
    else:
        ids = [i for i in (2, 3) if i < n]
        if len(ids) < 2:
            ids = list(range(min(2, n)))
    if len(ids) < 2:
        warnings.warn(
            "distributed requested but fewer than 2 usable CUDA devices; staying single-GPU",
            stacklevel=2,
        )
        return None
    return ids

def align_device_for_distributed(device, enabled=None):
    """Primary device for DP (output_device). Unchanged when distributed is off."""
    ids = distributed_device_ids(enabled=enabled)
    if not ids:
        return device
    primary = f"cuda:{ids[0]}"
    if isinstance(device, torch.device):
        return torch.device(primary)
    return primary

def unwrap_model(model):
    """Eager nn.Module under DataParallel and/or torch.compile."""
    inner = model
    if isinstance(inner, nn.DataParallel):
        inner = inner.module
    orig = getattr(inner, "_orig_mod", None)
    if orig is not None:
        return orig
    return inner

def maybe_data_parallel(model, device=None, enabled=None):
    """Wrap `model` for the train loss only. Returns `model` unchanged if DP is off.

    Keep the original module for generate / probes. Optimizer should use the eager
    module's parameters (shared with the wrapper).

    `batch_size` in YAML/CLI is the **global** batch; DataParallel splits it across GPUs.
    Prefer a global batch divisible by the GPU count (e.g. 128 on 2 GPUs → 64 each).
    """
    ids = distributed_device_ids(enabled=enabled)
    if not ids:
        return model
    if device is not None and torch.device(device).type != "cuda":
        return model
    primary = torch.device(f"cuda:{ids[0]}")
    torch.cuda.set_device(ids[0])
    model.to(primary)
    wrapped = nn.DataParallel(model, device_ids=ids, output_device=ids[0])
    wrapped._trace_data_parallel = True
    return wrapped

def add_distributed_flags(parser, cfg=None, *, from_yaml=True):
    cfg = cfg or {}
    if from_yaml:
        default = bool(cfg.get("distributed", False))
    else:
        default = None
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=(
            "Train with torch.nn.DataParallel on two GPUs (prefer TRACE_DEVICES or cuda:2,3). "
            "Off by default; does not occupy all 4 A100s. Generate/probes stay on one GPU. "
            "batch_size is global. Not torchrun DDP."
        ),
    )

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch

# Only compile these; fixed / custom-attention executors stay eager.
_COMPILABLE = {"GPTModel", "LearnedOneLayerTransformer", "UnifiedExecutor"}

_explicit_compile = None
_explicit_bf16 = None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def default_device(prefer=("cuda:2", "cuda:3")):
    """Prefer free A100s. If CUDA_VISIBLE_DEVICES is already set, use cuda:0.

    TRACE_DEVICES=2,3 makes the first listed card the single-GPU default (still one GPU
    unless distributed is on).
    """
    if not torch.cuda.is_available():
        return "cpu"

    env_ids = parse_trace_devices()
    if env_ids:
        prefer = tuple(f"cuda:{i}" for i in env_ids)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != "":
        return "cuda:0"
    n = torch.cuda.device_count()
    for name in prefer:
        idx = int(name.split(":")[1])
        if idx < n:
            return name
    return "cuda:0"

def _in_tests():
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return any("unittest" in arg or "pytest" in arg for arg in sys.argv)

def _env_flag(name):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in {"0", "false", "off", "no"}

def configure_device(device, compile=None, bf16=None, distributed=None):
    """TF32 + cuDNN benchmark for the fast CUDA path. Not deterministic.

    Call once per process. `compile` / `bf16` store YAML/CLI overrides used by
    maybe_compile / autocast_context. TRACE_COMPILE / TRACE_BF16 env still win.
    If distributed, `device` is moved to the DataParallel primary (e.g. cuda:2).
    """
    global _explicit_compile, _explicit_bf16

    device = torch.device(align_device_for_distributed(device, enabled=distributed))
    if compile is not None:
        _explicit_compile = bool(compile)
    if bf16 is not None:
        _explicit_bf16 = bool(bf16)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return device

def maybe_high_precision(device, compile=None, bf16=None, distributed=None):
    """Alias for configure_device (historical name)."""
    return configure_device(device, compile=compile, bf16=bf16, distributed=distributed)

def add_compile_bf16_flags(parser, cfg=None, *, from_yaml=True):
    """`--compile` / `--bf16` / `--data-on-device`. YAML defaults; compile off if omitted.

    `from_yaml=False` leaves defaults as None so load_experiment can fill from YAML.
    """
    cfg = cfg or {}
    if from_yaml:
        compile_default = bool(cfg.get("compile", False))
        bf16_default = bool(cfg.get("bf16", True))
        data_default = bool(cfg.get("data_on_device", False))
    else:
        compile_default = None
        bf16_default = None
        data_default = None
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=compile_default,
        help="Compile a training wrapper only; generate/eval stay eager. Default off unless YAML compile: true.",
    )
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=bf16_default,
        help="CUDA bfloat16 autocast. Default on unless YAML bf16: false.",
    )
    parser.add_argument(
        "--data-on-device", action=argparse.BooleanOptionalAction, default=data_default,
        dest="data_on_device",
        help="Keep small ContinuationDataset tensors on GPU.",
    )
    add_distributed_flags(parser, cfg, from_yaml=from_yaml)

def compile_enabled(explicit=None, device=None):
    """Compile only if YAML/CLI `--compile`, TRACE_COMPILE=1, or configure_device(compile=True).

    Default is off. Tests never compile.
    """
    if _in_tests():
        return False
    env = _env_flag("TRACE_COMPILE")
    if env is not None:
        wanted = env
    elif explicit is not None:
        wanted = bool(explicit)
    elif _explicit_compile is not None:
        wanted = _explicit_compile
    else:
        wanted = False
    if not wanted:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True

def bf16_enabled(explicit=None, device=None):
    env = _env_flag("TRACE_BF16")
    if env is not None:
        wanted = env
    elif explicit is not None:
        wanted = bool(explicit)
    elif _explicit_bf16 is not None:
        wanted = _explicit_bf16
    else:
        wanted = True
    if not wanted:
        return False
    device = torch.device(device) if device is not None else torch.device("cpu")
    return device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def autocast_context(device, enabled=None):
    use = bf16_enabled(explicit=enabled, device=device)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use)

def maybe_compile(model, device, enabled=None):
    """Return an optional compiled *wrapper* for training. Never mutate `model.forward`.

    Keep `model` for generate / probes / induced-rule / pullback. Train with the
    return value (`train_model = maybe_compile(model, ...)`); it shares parameters.
    Replacing `forward` recaptures a Dynamo graph on every new generate length.

    If `model` is DataParallel, compile the replica (`model.module`) after the wrap
    and leave generate on the eager module.
    """
    from torch.nn import DataParallel

    parallel = isinstance(model, DataParallel)
    replica = model.module if parallel else model
    name = type(replica).__name__
    if name not in _COMPILABLE:
        return model
    if getattr(replica, "_trace_compiled", False) or getattr(model, "_trace_compiled", False):
        return model
    if not compile_enabled(explicit=enabled, device=device):
        return model
    try:
        compiled = torch.compile(replica, mode="default", fullgraph=False)
        compiled._trace_compiled = True
        if parallel:
            model.module = compiled
            return model
        return compiled
    except Exception as exc:
        warnings.warn(
            f"torch.compile failed for {name} ({exc}); continuing eager",
            stacklevel=2,
        )
        return model

def prepare_train_model(model, device, compile=None, distributed=None):
    """DataParallel (optional) then compile wrapper. Eager `model` stays for eval."""

    train_model = maybe_data_parallel(model, device, enabled=distributed)
    return maybe_compile(train_model, device, enabled=compile)

import torch

from src.gpt import GPTModel

def build_gpt(task, args, device):
    """Eager GPT. Callers that train compiled must keep this and `maybe_compile` it."""
    n_embd = getattr(args, "embedding", getattr(args, "n_embd", 128))
    n_head = getattr(args, "heads", getattr(args, "n_head", 4))
    n_layer = getattr(args, "layers", getattr(args, "n_layer", 2))
    dropout = getattr(args, "dropout", 0.0)
    tok = task.tokenizer
    return GPTModel(
        vocab_size=tok.vocab_size,
        block_size=task.block_size,
        pad_id=tok.pad_id,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

def make_adamw(params, lr, weight_decay=0.0, device=None):
    params = list(params)
    fused = torch.device(device).type == "cuda" if device is not None else any(p.is_cuda for p in params)
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def make_optimizer(model, args, device):
    lr = getattr(args, "lr", 3e-4)
    wd = getattr(args, "weight_decay", 0.0)
    return make_adamw(model.parameters(), lr, weight_decay=wd, device=device)

def make_loader(dataset, args, device, extra_seed=0):
    on_device = bool(getattr(args, "data_on_device", False)) and torch.device(device).type == "cuda"
    if on_device and hasattr(dataset, "to"):
        dataset = dataset.to(device)
        workers, pin = 0, False
    else:
        workers = getattr(args, "workers", 0)
        pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(getattr(args, "batch_seed", 12345) + extra_seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        generator=generator,
        pin_memory=pin,
    )

"""GPT training loop. A batch is (x, y, mask). Eval stays on the eager model."""
import os
import sys

import torch
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm as _tqdm

def _tqdm_disabled():
    if os.environ.get("TRACE_TQDM", "1") in {"0", "false", "False"}:
        return True
    return any("unittest" in arg or "pytest" in arg for arg in sys.argv)

def progress(iterable=None, **kwargs):
    """One tqdm helper. Quiet in unit tests or when TRACE_TQDM=0."""
    kwargs.setdefault("disable", _tqdm_disabled())
    return _tqdm(iterable, **kwargs)

def gpt_lm_loss(model, batch, device):
    x, y, mask = batch
    x = x.to(device, dtype=torch.long, non_blocking=True)
    y = y.to(device, dtype=torch.long, non_blocking=True)
    mask = mask.to(device, dtype=torch.float32, non_blocking=True)
    if x.ndim != 2 or y.shape != x.shape or mask.shape != x.shape:
        raise ValueError(f"GPT batch must be (B, T) triples, got {tuple(x.shape)}")
    # DataParallel cannot gather a 0-dim CE; compute the same masked mean on gathered logits.
    if isinstance(model, DataParallel):
        logits, _ = model(x)
        inner = model.module
        pad_id = getattr(getattr(inner, "_orig_mod", inner), "pad_id")
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
            ignore_index=pad_id,
        ).view_as(y)
        return (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
    _, loss = model(x, targets=y, mask=mask)
    return loss

def _maybe_autocast(device):
    return autocast_context(device)

def train_steps(model, loader, optimizer, device, n_steps, grad_clip=1.0, loss_fn=None, desc=None):
    """Run n_steps updates. loss_fn(model, batch) -> scalar; default is GPT CE."""
    iterator = iter(loader)
    model.train()
    loss = None
    step_loss = loss_fn or (lambda m, b: gpt_lm_loss(m, b, device))
    bar = progress(range(n_steps), desc=desc or "train", leave=False)
    for _ in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device):
            loss = step_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if loss is not None:
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return float(loss.detach())

def train_with_checkpoints(model, loader, optimizer, device, checkpoints, on_checkpoint,
                           grad_clip=1.0, loss_fn=None, desc=None, train_model=None):
    """on_checkpoint(step, model, loss) uses the eager `model` (generate/probes).

    Pass `train_model=prepare_train_model(model, ...)` so the loss path can be
    DataParallel and/or compiled. Checkpoints and generate use eager `model`.
    """
    runner = train_model if train_model is not None else model
    ckpts = sorted(set(checkpoints))
    iterator = iter(loader)
    step_loss = loss_fn or (lambda m, b: gpt_lm_loss(m, b, device))
    loss = None
    bar = progress(range(0, max(ckpts) + 1), desc=desc or "train", leave=False)
    for step in bar:
        if step in ckpts:
            on_checkpoint(step, model, loss)
            model.train()
            if runner is not model:
                runner.train()
        if step == max(ckpts):
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device):
            loss = step_loss(runner, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if loss is not None:
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return model
