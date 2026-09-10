"""Revision-bridge figures. Prefer: python -m src analyze"""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))
