"""Put the repository root on sys.path so `import src...` works from any cwd."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", HERE, ROOT / "handcoded"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
