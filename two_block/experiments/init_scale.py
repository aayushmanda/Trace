"""Single cell of the initialization-scale sweep (Table 7 / tab:controls, Appendix G.1).

Trains step ('process') and outcome supervision from one shared initialization theta_0 and writes
the held-out exact accuracy curves to a JSON file.

    uv run python two_block/experiments/init_scale.py \\
        out=results/seeded/rnd_n8_K2_e0.1_s42.json task=count word_len=8 mod=2 init=random init_std=0.1 seed=42
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

TWO_BLOCK_DIR = Path(__file__).resolve().parents[1]
if str(TWO_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_BLOCK_DIR))

import counting as lt  # noqa: E402


def run_seeded_cell(out_path: Path, overrides: dict[str, Any]) -> dict[str, Any] | None:
    """Run one paired step/outcome cell and save metrics to out_path (skips if already present)."""
    if out_path.exists():
        return None

    cfg = lt.configure(**overrides)
    train, shown, test = lt.make_corpus(cfg)
    theta_0 = lt.init_theta(cfg)

    record: dict[str, Any] = {
        "config": cfg.to_dict(exclude_device=True),
        "seeded_entries": sum(int((t != 0).sum()) for t in theta_0.values()),
        "total_entries": sum(t.numel() for t in theta_0.values()),
        "modes": {},
    }

    t0 = time.monotonic()
    for mode in ("process", "outcome"):
        _, history = lt.train_mode(mode, theta_0, train, shown, test, cfg=cfg)
        evals = [row for row in history if "exact" in row]
        record["modes"][mode] = {
            "final_exact": evals[-1]["exact"],
            "final_answer": evals[-1]["answer"],
            "solved_at": next((row["step"] for row in evals if row["exact"] >= 0.9), None),
            "curve": [(row["step"], row["exact"]) for row in evals],
        }
    record["seconds"] = time.monotonic() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(
        f"{out_path.name}: step {record['modes']['process']['final_exact']:.3f} "
        f"(solved {record['modes']['process']['solved_at']})  "
        f"outcome {record['modes']['outcome']['final_exact']:.3f}",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    out_path: Path | None = None
    filtered: list[str] = []
    for arg in args:
        if arg.startswith("out="):
            out_path = Path(arg.split("=", 1)[1])
        else:
            filtered.append(arg)
    if out_path is None:
        raise SystemExit("Usage: seeded_sweep.py out=path.json [key=value ...]")
    overrides = lt.parse_cli_overrides(filtered)
    run_seeded_cell(out_path, overrides)


if __name__ == "__main__":
    main()
