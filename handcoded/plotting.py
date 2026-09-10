"""Figure style for the tutorial. Implementation: `src.plot_style.apply_style`."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.plot_style import STYLE, apply_style

__all__ = ["STYLE", "apply_style"]
