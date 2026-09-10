import csv
from pathlib import Path


def append_rows(path, rows):
    """Append dict rows; divert to a sibling file if the header does not match."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if path.exists():
        with path.open(newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing is not None and existing != fields:
            path = path.with_name(f"{path.stem}__schema{len(fields)}{path.suffix}")
            print(f"schema differs from the existing file; writing {path}", flush=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path


def append_csv(path, row: dict):
    return append_rows(path, [row])


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
