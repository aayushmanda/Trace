"""Shared-kernel / escape-time law. Prefer: python -m src escape ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["escape", *sys.argv[1:]]))
