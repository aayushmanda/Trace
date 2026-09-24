"""Hyperparameters for the semantic-token tutorial."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fallbacks match configs/handcoded.yaml if the file is missing.
# These are LIVE: `configure()` rebinds them and every default in the package
# resolves them at call time, so a notebook can change the task shape in one place.
N_BITS = 4
N_GATES = None      # None = every gate string; an int = that many alias-free gates
DEPTH = 4
TRAIN_SIZE = 20_000
TEST_SIZE = 1_000
STEPS = 10_000
BATCH_SIZE = 128
LR = 2e-3
D_MODEL = 96
N_HEADS = 4
D_FF = 192
DATA_SEED = 123
TEST_SEED = 9_000
MODEL_SEED = 42
BATCH_SEED = 2_026

N_STATES = 2 ** N_BITS          # derived; recomputed by configure()

_DEFAULTS = {
    "N_BITS": N_BITS, "N_GATES": N_GATES, "DEPTH": DEPTH, "TRAIN_SIZE": TRAIN_SIZE, "TEST_SIZE": TEST_SIZE,
    "STEPS": STEPS, "BATCH_SIZE": BATCH_SIZE, "LR": LR, "D_MODEL": D_MODEL,
    "N_HEADS": N_HEADS, "D_FF": D_FF, "DATA_SEED": DATA_SEED, "TEST_SEED": TEST_SEED,
    "MODEL_SEED": MODEL_SEED, "BATCH_SEED": BATCH_SEED,
}


def configure(**overrides):
    """Set hyperparameters from a notebook, e.g. `configure(n_bits=2, depth=3)`.

    Names are case-insensitive. Derived values (`N_STATES`) are recomputed.
    Call this BEFORE building a tokenizer, circuits, or models -- objects already
    constructed keep the shape they were built with.
    """
    resolved = {key.upper(): value for key, value in overrides.items()}
    unknown = sorted(set(resolved) - set(_DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown setting(s) {unknown}; valid names are {sorted(_DEFAULTS)}"
        )
    n_gates = resolved.get("N_GATES", globals()["N_GATES"])
    if n_gates is not None and n_gates < 2:
        raise ValueError(f"n_gates must be None or at least 2 (got {n_gates})")
    n_bits = resolved.get("N_BITS", globals()["N_BITS"])
    if n_bits < 2:
        raise ValueError(f"n_bits must be at least 2 (got {n_bits}); "
                         "a 1-bit state space has no non-trivial gates")
    globals().update(resolved)
    globals()["N_STATES"] = 2 ** globals()["N_BITS"]
    return current()


def current():
    """The active settings, as a dict."""
    values = {key: globals()[key] for key in _DEFAULTS}
    values["N_STATES"] = globals()["N_STATES"]
    return values


def reset():
    """Restore the packaged defaults."""
    return configure(**_DEFAULTS)


def load_config(name="handcoded.yaml"):
    """Load `configs/<name>`. Nested maps are not required."""
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / "configs" / name if not path.exists() else path
        if not path.exists():
            path = ROOT / name
    text = path.read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        from src.training.config import load_yaml
        data = load_yaml(path)
    return data


def make_checkpoints(steps=None, animation_checkpoints=120):
    import numpy as np
    steps = STEPS if steps is None else steps
    uniform = np.linspace(0, steps, animation_checkpoints + 1).round().astype(int)
    early = [1, 2, 5, 10, 20, 25, 50, 75, 100, 250, 500]
    return sorted(set(uniform.tolist()) | {s for s in early if s <= steps})
