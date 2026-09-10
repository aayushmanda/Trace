"""Split-verdict depth table. Prefer: python -m src split-verdict ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["split-verdict", *sys.argv[1:]]))
