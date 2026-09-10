"""Deprecated: `import handcoded_utils`. Prefer `import handcoded` from the repo root."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from handcoded import *  # noqa: F401,F403
from handcoded import __all__  # noqa: F401
