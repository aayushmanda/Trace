"""Architecture × supervision. Prefer: python -m src architecture ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["architecture", *sys.argv[1:]]))
