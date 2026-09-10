"""Semantic-token executors: re-export the `handcoded` package."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import handcoded as h  # noqa: E402

__all__ = ["h"]
