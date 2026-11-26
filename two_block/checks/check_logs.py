"""Audit counting-experiment claims from counting.py training logs (Tables 6-7: tab:gap, tab:controls).

Reads evaluation lines (`MODE step N loss L test exact E answer A`) from:
  - logs/must_run/tab3_count_n*_s*.log   (extended 32k horizon sweep)
  - logs/must_run/tab3_first_n*_s*.log   (first-occurrence control)
  - logs/remaining/depth_L{4,6}_*.log    (deeper 4- and 6-block models)
  - logs/must_run/out32k_K3_s*.log       (counting modulo 3 at n = 8)

    uv run python two_block/checks/check_logs.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATTERN = re.compile(r"^(process|outcome)\s+step\s+(\d+)\s+loss\s+\S+\s+test exact\s+([\d.]+)", re.M)


def parse_curves(log_path: Path) -> dict[str, dict[int, float]]:
    """Parse held-out exact accuracy (%) curves by mode from a counting.py log."""
    curves: dict[str, dict[int, float]] = {"process": {}, "outcome": {}}
    for mode, step_str, exact_str in EVAL_PATTERN.findall(log_path.read_text()):
        curves[mode][int(step_str)] = 100.0 * float(exact_str)
    return curves


def value_at(curve: dict[int, float], step_budget: int) -> float:
    return curve[max(s for s in curve if s <= step_budget)]


def reached_threshold(curve: dict[int, float], step_budget: int, threshold: float = 90.0) -> bool:
    return any(val >= threshold for step, val in curve.items() if step <= step_budget)


def min_max_span(values: list[float]) -> list[float]:
    return [round(min(values), 1), round(max(values), 1)]


def load_log_group(root: Path, pattern: str) -> dict[str, dict[str, dict[int, float]]]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No logs matched pattern {pattern!r} under {root}")
    return {path.stem: parse_curves(path) for path in paths}


def summarize_logs(root: Path = ROOT) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    # Table 6 (tab:gap): extended 32k horizon sweeps
    for word_len in (8, 12, 16, 24):
        runs = load_log_group(root, f"logs/must_run/tab3_count_n{word_len}_s*.log")
        summary[f"horizon n={word_len}"] = {
            "step_8k": min_max_span([value_at(c["process"], 8_000) for c in runs.values()]),
            "outcome_8k": min_max_span([value_at(c["outcome"], 8_000) for c in runs.values()]),
            "step_90_by_16k": sum(reached_threshold(c["process"], 16_000) for c in runs.values()),
            "outcome_90_by_32k": sum(reached_threshold(c["outcome"], 32_000) for c in runs.values()),
            "runs": len(runs),
        }

    # Table 7 (tab:controls): first-occurrence control
    first_runs = load_log_group(root, "logs/must_run/tab3_first_n*_s*.log")
    summary["first occurrence n=8-24"] = {
        "step_8k": min_max_span([value_at(c["process"], 8_000) for c in first_runs.values()]),
        "outcome_8k": min_max_span([value_at(c["outcome"], 8_000) for c in first_runs.values()]),
        "step_90_by_16k": sum(reached_threshold(c["process"], 16_000) for c in first_runs.values()),
        "outcome_90_by_16k": sum(reached_threshold(c["outcome"], 16_000) for c in first_runs.values()),
        "runs": len(first_runs),
    }

    # Table 7 (tab:controls): deeper 4- and 6-block models
    deep_count = load_log_group(root, "logs/remaining/depth_L*_count_n*_s*.log")
    summary["count mod 2, 4 or 6 blocks"] = {
        "step_final": min_max_span([value_at(c["process"], 8_000) for c in deep_count.values()]),
        "outcome_final": min_max_span([value_at(c["outcome"], 8_000) for c in deep_count.values()]),
        "step_above_80": sum(value_at(c["process"], 8_000) > 80.0 for c in deep_count.values()),
        "runs": len(deep_count),
    }

    deep_first = load_log_group(root, "logs/remaining/depth_L*_first_n*_s*.log")
    summary["first occurrence, 4 or 6 blocks"] = {
        "step_final": min_max_span([value_at(c["process"], 8_000) for c in deep_first.values()]),
        "outcome_final": min_max_span([value_at(c["outcome"], 8_000) for c in deep_first.values()]),
        "runs": len(deep_first),
    }

    # Counting modulo 3 at n = 8
    mod3_runs = load_log_group(root, "logs/must_run/out32k_K3_s*.log")
    summary["mod 3, n=8, outcome at 32k"] = {
        name: round(value_at(c["outcome"], 32_000), 1) for name, c in mod3_runs.items()
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit counting.py log claims for Tables 6-7.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/paper/log_claims.json")
    args = parser.parse_args(argv)

    summary = summarize_logs(args.root)
    for label, stats in summary.items():
        print(f"{label:34s} {stats}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
