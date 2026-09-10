"""Induced-rule readout. Prefer: python -m src induced ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["induced", *sys.argv[1:]]))
