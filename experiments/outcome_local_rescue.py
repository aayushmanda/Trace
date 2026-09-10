"""Outcome-architecture local-credit mechanism test. Prefer: python -m src outcome-local ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["outcome-local", *sys.argv[1:]]))
