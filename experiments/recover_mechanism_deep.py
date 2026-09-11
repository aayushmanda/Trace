"""Recover the depth x seed mechanism study from run logs, and fit escape times.

The runs of results/mechanism_deep/ completed training but crashed in the CSV
writer (a fieldnames schema bug) before metrics.csv was written.  Each
checkpoint's summary had already been printed to log.txt as a JSON line, so the
measurements survive.  This script parses them and reports two things:

  table     final-checkpoint held-out answer accuracy per depth and condition,
            which is the five-seed depth table;
  escape    the number of updates each condition needs to first cross a fixed
            answer-accuracy threshold, which is the depth-graded quantity the
            plateau table cannot show.

Writes results/mechanism_deep/recovered.csv and recovered_summary.json.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import _paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "mechanism_deep" / "runs"
OUT = ROOT / "results" / "mechanism_deep"
CONDITIONS = ("outcome", "outcome_local", "process")
THRESHOLDS = (0.10, 0.25, 0.50)


def _json_objects(text: str, marker: str):
    """Yield JSON objects starting at `marker`, brace-matched.

    tqdm writes progress with carriage returns, so a checkpoint's JSON is
    embedded inside a long progress line rather than on one of its own.
    """
    start = text.find(marker)
    while start != -1:
        depth, i, in_str, esc = 0, start, False, False
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
            i += 1
        start = text.find(marker, start + 1)


def parse_run(path: Path):
    """Checkpoints as {step: row}, plus any patch diagnostics."""
    text = path.read_text(errors="replace")
    checkpoints, patch = {}, {}
    for row in _json_objects(text, '{"step":'):
        if "answer" in row:
            checkpoints[int(row["step"])] = row
    for row in _json_objects(text, '{"oracle_patch_mean":'):
        patch = row
    return checkpoints, patch


def escape_step(checkpoints, condition, threshold):
    """First recorded update at which answer accuracy crosses the threshold."""
    for step in sorted(checkpoints):
        if checkpoints[step]["answer"].get(condition, 0.0) >= threshold:
            return step
    return None  # censored within the recorded horizon


def main() -> int:
    runs = {}
    for d in sorted(RUNS.iterdir()):
        log = d / "log.txt"
        if not log.exists():
            continue
        depth = int(d.name.split("_")[0][1:])
        seed = int(d.name.split("seed")[1])
        runs[(depth, seed)] = parse_run(log)

    depths = sorted({k[0] for k in runs})
    rows, summary = [], {"n_runs": len(runs), "thresholds": list(THRESHOLDS)}

    print("=== final-checkpoint answer accuracy (%), mean over seeds ===")
    print(f"{'D':>3} {'steps':>6} " + " ".join(f"{c:>16}" for c in CONDITIONS))
    table = {}
    for depth in depths:
        cells = []
        # compare every seed at the same checkpoint; a run censored before it is
        # excluded rather than averaged in at a different step
        common = max(max(ck) for (dd, _), (ck, _) in runs.items() if dd == depth and ck)
        for cond in CONDITIONS:
            vals, last = [], common
            for (dd, seed), (ck, _) in runs.items():
                if dd != depth or common not in ck:
                    continue
                vals.append(ck[common]["answer"][cond] * 100)
            m = st.mean(vals)
            s = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{m:7.2f} ± {s:<6.2f}")
            table[(depth, cond)] = (m, s, len(vals))
            rows.append({"depth": depth, "condition": cond, "metric": "final_answer",
                         "mean": m, "sd": s, "n": len(vals)})
        print(f"{depth:>3} {last:>6} " + " ".join(f"{c:>16}" for c in cells))
    summary["final_answer"] = {f"D{d}_{c}": table[(d, c)] for d in depths for c in CONDITIONS}

    for thr in THRESHOLDS:
        print(f"\n=== updates to first reach {thr:.0%} answer accuracy "
              f"(median over seeds; 'censored' = never within 6,000) ===")
        print(f"{'D':>3} " + " ".join(f"{c:>18}" for c in CONDITIONS))
        for depth in depths:
            cells = []
            for cond in CONDITIONS:
                steps, censored = [], 0
                for (dd, seed), (ck, _) in runs.items():
                    if dd != depth or not ck:
                        continue
                    s = escape_step(ck, cond, thr)
                    if s is None:
                        censored += 1
                    else:
                        steps.append(s)
                if steps:
                    cells.append(f"{st.median(steps):8.0f} ({censored} cens.)")
                else:
                    cells.append(f"{'censored':>8} ({censored}/{censored})")
                rows.append({"depth": depth, "condition": cond,
                             "metric": f"escape_{thr}",
                             "mean": st.median(steps) if steps else math.nan,
                             "sd": 0.0, "n": len(steps), "censored": censored})
            print(f"{depth:>3} " + " ".join(f"{c:>18}" for c in cells))

    patches = [p for _, p in runs.values() if p]
    if patches:
        oracle = st.mean(p["oracle_patch_mean"] for p in patches)
        rand = st.mean(p["random_patch_mean"] for p in patches)
        probe = st.mean(p.get("probe_subspace_oracle_mean", float("nan")) for p in patches)
        print(f"\n=== causal patch diagnostics, mean over {len(patches)} runs ===")
        print(f"  oracle patch {oracle:.4f}   random patch {rand:.4f}   "
              f"oracle probe subspace {probe:.4f}")
        summary["patch"] = {"oracle": oracle, "random": rand,
                            "probe_subspace": probe, "n": len(patches)}

    import csv
    with (OUT / "recovered.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["depth", "condition", "metric",
                                           "mean", "sd", "n", "censored"])
        w.writeheader()
        for r in rows:
            w.writerow({**{"censored": ""}, **r})
    (OUT / "recovered_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT/'recovered.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
