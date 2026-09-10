"""Five-condition supervision comparison. Prefer: python -m src supervision ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["supervision", *sys.argv[1:]]))
