"""Paper figure builders and closure checks. Public CLI: python experiments/run.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reject_extra_flags(argv, doc: str | None = None) -> None:
    """For commands with no argparse: honor --help, reject anything else."""
    argv = [] if argv is None else list(argv)
    if not argv:
        return
    if argv in (["-h"], ["--help"]):
        print(doc or "")
        raise SystemExit(0)
    raise SystemExit(f"unexpected arguments: {argv}")
