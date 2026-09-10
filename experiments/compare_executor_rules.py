"""Handcoded executor comparison. Prefer: python -m src executor ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["executor", *sys.argv[1:]]))
