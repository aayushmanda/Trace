"""Hyperparameters for the semantic-token tutorial (paper vs smoke YAML)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fallbacks match configs/handcoded.yaml if the file is missing.
N_BITS = 4
N_STATES = 2 ** N_BITS
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


def load_config(name="handcoded.yaml"):
    """Load `configs/<name>` (paper or smoke). Nested maps are not required."""
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


def make_checkpoints(steps=STEPS, animation_checkpoints=120):
    import numpy as np
    uniform = np.linspace(0, steps, animation_checkpoints + 1).round().astype(int)
    early = [1, 2, 5, 10, 20, 25, 50, 75, 100, 250, 500]
    return sorted(set(uniform.tolist()) | {s for s in early if s <= steps})
