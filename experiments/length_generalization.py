"""Train at D=8, evaluate longer circuits. Prefer: python -m src length ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["length", *sys.argv[1:]]))
