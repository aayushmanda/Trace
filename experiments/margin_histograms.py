"""Gold-path m_min histograms. Prefer: python -m src margins ..."""
import sys

import _paths  # noqa: F401
from src.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["margins", *sys.argv[1:]]))
