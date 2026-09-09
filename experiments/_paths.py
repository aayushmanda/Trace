"""Put the repository root and src/ on sys.path.

Every entry point in this directory imports this first, so scripts run the same
way from any working directory:

    python experiments/<name>.py            # from the repo root
    python <name>.py                        # from inside experiments/

It supports both import styles used in the codebase, `from src.model import ...`
and `from model import ...`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
