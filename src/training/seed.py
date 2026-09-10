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
    """Prefer free A100s. If CUDA_VISIBLE_DEVICES is already set, use cuda:0."""
    if not torch.cuda.is_available():
        return "cpu"
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


def configure_device(device, compile=None, bf16=None):
    """TF32 + cuDNN benchmark for the fast CUDA path. Not deterministic.

    Call once per process. `compile` / `bf16` store YAML/CLI overrides used by
    maybe_compile / autocast_context. TRACE_COMPILE / TRACE_BF16 env still win.
    """
    global _explicit_compile, _explicit_bf16
    device = torch.device(device)
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


def maybe_high_precision(device, compile=None, bf16=None):
    """Alias for configure_device (historical name)."""
    return configure_device(device, compile=compile, bf16=bf16)


def compile_enabled(explicit=None, device=None):
    """TRACE_COMPILE=1 is the CUDA default outside tests. Tests never compile."""
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
        wanted = True
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
    """Compile `forward` only (not greedy generate). Eager fallback on failure.

    First compiled call is slow (Inductor). Subsequent train steps reuse the graph
    when sequence length is fixed. Growing generate lengths can recapture graphs.
    """
    name = type(model).__name__
    if name not in _COMPILABLE:
        return model
    if getattr(model, "_trace_compiled", False):
        return model
    if not compile_enabled(explicit=enabled, device=device):
        return model
    try:
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
        model._trace_compiled = True
    except Exception as exc:
        warnings.warn(
            f"torch.compile failed for {name} ({exc}); continuing eager",
            stacklevel=2,
        )
    return model
