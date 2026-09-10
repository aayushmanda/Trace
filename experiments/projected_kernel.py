"""Projected simplex kernel. Prefer: python -m src projected ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["projected", *sys.argv[1:]]))
