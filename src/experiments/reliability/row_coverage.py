"""Row coverage N_{a,s} of a training corpus, and the sample slack beta_N it implies.

body.tex's Theorem 3(b) charges a finite-sample budget

    beta_N = sqrt( (2 / N_min) * log(K^2 M / delta) ),    N_min = min_{a,s} N_{a,s},

against the row margin, so the threshold a task faces depends on how evenly its
sampler covers the K*M state-operation cells, not only on how corrupted its
traces are.  Section 4.1 reads the two canonical corpora this way.  This script
counts the displayed (operation, source state) pairs in the corpus a run would
actually train on and reports N_min, the 10th percentile, the median and the
resulting beta_N.  No training and no model are involved.

Usage:
    uv run python -m src.experiments.reliability.row_coverage \
        --output results/paper/row_coverage.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from src.data.registry import TASKS
from src.data.sample import generate_unique


def _rows_boolean(inst):
    """Boolean circuit: prompt 'i<bits>;u<gates>', steps '<gate>><bits>'."""
    state = inst.prompt.split(";")[0][1:]
    for token in inst.correct_trace.split():
        operation, successor = token.split(">")
        yield operation, state
        state = successor


def _rows_register(inst):
    """Register machine: prompt 'x<nn>;y<nn>;u<instructions>', steps '<i><nn><nn>'."""
    fields = inst.prompt.split(";")
    state = (fields[0][1:], fields[1][1:])
    for token in inst.correct_trace.split():
        yield token[0], state
        state = (token[1:3], token[3:5])


#: task -> (state count K, operation count M, row walker, steps per instance)
CORPORA = {
    "boolean_circuit_8": (16, 52, _rows_boolean, 8),
    "register_machine_16": (17 * 17, 5, _rows_register, 16),
}


def beta_n(n_min: int, n_states: int, n_operations: int, delta: float) -> float:
    """Theorem 3(b)'s sample slack."""
    return float(np.sqrt(2.0 / n_min * np.log(n_states**2 * n_operations / delta)))


def coverage(task_name, n_states, n_operations, walker, steps, train_size, train_seed, delta):
    instances = generate_unique(TASKS[task_name], train_size, train_seed)
    counts: collections.Counter = collections.Counter()
    for instance in instances:
        for operation, state in walker(instance):
            counts[(operation, state)] += 1
    cells = n_states * n_operations
    # unvisited cells count as zero, not as absent
    values = np.array(list(counts.values()) + [0] * (cells - len(counts)))
    return {
        "task": task_name,
        "train_size": train_size,
        "train_seed": train_seed,
        "steps_per_instance": steps,
        "K": n_states,
        "M": n_operations,
        "cells": cells,
        "occupied_cells": len(counts),
        "transitions": int(values.sum()),
        "N_min": int(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "N_max": int(values.max()),
        "delta": delta,
        "beta_N": beta_n(int(values.min()), n_states, n_operations, delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("results/paper/row_coverage.json"))
    args = parser.parse_args()

    report = {
        name: coverage(name, *spec, args.train_size, args.train_seed, args.delta)
        for name, spec in CORPORA.items()
    }
    for row in report.values():
        print(
            f"{row['task']:22s} K={row['K']:4d} M={row['M']:3d} "
            f"cells={row['cells']:5d} transitions={row['transitions']:,} "
            f"N_min={row['N_min']:4d} p10={row['p10']:.0f} median={row['median']:.0f} "
            f"beta_N={row['beta_N']:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
