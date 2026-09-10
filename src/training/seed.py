import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch

# Only compile these; fixed / custom-attention executors stay eager.
_COMPILABLE = {"GPTModel", "LearnedOneLayerTransformer"}

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
    from src.training.distributed import parse_trace_devices

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
    from src.training.distributed import align_device_for_distributed

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
    from src.training.distributed import add_distributed_flags
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
    from src.training.distributed import maybe_data_parallel

    train_model = maybe_data_parallel(model, device, enabled=distributed)
    return maybe_compile(train_model, device, enabled=compile)
