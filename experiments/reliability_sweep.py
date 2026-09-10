"""Trace-reliability sweep. Prefer: python -m src reliability ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["reliability", *sys.argv[1:]]))
