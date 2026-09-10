"""One tqdm helper. Quiet in unit tests or when TRACE_TQDM=0."""
import os
import sys

from tqdm.auto import tqdm as _tqdm


def tqdm_disabled():
    if os.environ.get("TRACE_TQDM", "1") in {"0", "false", "False"}:
        return True
    return any("unittest" in arg or "pytest" in arg for arg in sys.argv)


def progress(iterable=None, **kwargs):
    kwargs.setdefault("disable", tqdm_disabled())
    return _tqdm(iterable, **kwargs)
