"""Optional in-process DataParallel. Default remains one GPU.

Not torchrun/DDP: these nets are tiny; splitting a global batch across 2 GPUs is enough.
Probes, generate, and induced-rule stay on the eager module on the primary device.
"""
from __future__ import annotations

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
