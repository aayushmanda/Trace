"""Gradient-polarity reliability threshold rho_grad from saved circuit-match runs.

body.tex's Theorem 3 is a population statement about an independently
parameterised local conditional law.  Proposition 3 refines it to a threshold
that depends on the *current* Transformer parameters: writing g+ and g- for the
clean- and corrupted-trace gradients and u for a useful-progress direction,

    A_pm  = <-g_pm, u>,          A_rho = rho*A+ + (1-rho)*A-,

the mixed gradient makes useful progress exactly when A_rho > 0.  Solving for
rho needs a case split on the sign of A+ - A-, because dividing by a negative
number reverses the inequality:

    A+ > A-   ->  useful iff rho > rho_grad := -A- / (A+ - A-)
    A+ < A-   ->  useful iff rho < rho_grad   (an upper, not a lower, threshold)
    A+ = A-   ->  useful iff A- > 0, independent of rho.

This script reads the alpha_{group}_{plus,minus} columns already written by
handcoded_circuit_match.py (there u = d_r^star, the direction toward the
hand-coded solution in parameter group r) and reports rho_grad per group, per
checkpoint, with across-seed spread.

Caveat this script is designed to expose: those runs train on *clean* process
supervision, so at convergence the model sits at a minimum of the clean loss
and A+ -> 0 by construction.  rho_grad is then not identified from the data;
what survives is the A+ = 0 limit, where A_rho = (1-rho)*A-, so the group is
helped by every rho < 1 when A- > 0 and by no rho at all when A- < 0.
Estimating rho_grad properly needs probes at parameters where the clean
gradient still carries signal: mixture-trained checkpoints, or early training.

Usage:
    uv run python -m src.experiments.circuit_match.gradient_polarity_threshold \
        --runs results/handcoded_circuit_match_symmetric_n20k \
               results/handcoded_circuit_match_coherent_n20k \
        --output results/paper/gradient_polarity_threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("QK", "OV", "MLP")
#: |A+| below this fraction of |A-| counts as "clean gradient exhausted".
IDENTIFIABILITY_RATIO = 1e-2


def rho_grad(a_plus: np.ndarray, a_minus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (threshold, sense) with sense = +1 lower bound, -1 upper bound, 0 degenerate."""
    denom = a_plus - a_minus
    sense = np.sign(denom).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        thr = np.where(denom != 0, -a_minus / denom, np.nan)
    return thr, sense


def summarise(frame: pd.DataFrame, run: str) -> list[dict]:
    rows: list[dict] = []
    for step in sorted(frame["step"].unique()):
        sub = frame[frame["step"] == step]
        for group in GROUPS:
            ap = sub[f"alpha_{group}_plus"].to_numpy(float)
            am = sub[f"alpha_{group}_minus"].to_numpy(float)
            thr, sense = rho_grad(ap, am)
            scale = np.abs(am).mean()
            identified = bool(scale > 0 and np.abs(ap).mean() / scale > IDENTIFIABILITY_RATIO)
            rows.append(
                {
                    "run": run,
                    "step": int(step),
                    "group": group,
                    "n_seeds": int(len(ap)),
                    "A_plus_mean": ap.mean(),
                    "A_plus_sd": ap.std(ddof=1) if len(ap) > 1 else np.nan,
                    "A_minus_mean": am.mean(),
                    "A_minus_sd": am.std(ddof=1) if len(am) > 1 else np.nan,
                    "abs_ratio": (np.abs(ap).mean() / scale) if scale > 0 else np.nan,
                    "identified": identified,
                    "rho_grad_mean": np.nanmean(thr),
                    "rho_grad_sd": np.nanstd(thr, ddof=1) if len(thr) > 1 else np.nan,
                    "n_lower_bound": int((sense > 0).sum()),
                    "n_upper_bound": int((sense < 0).sum()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="circuit-match run directories")
    parser.add_argument("--output", required=True, help="directory for the CSV and summary")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run_dir in args.runs:
        path = Path(run_dir) / "grad_match.csv"
        rows.extend(summarise(pd.read_csv(path), Path(run_dir).name))
    table = pd.DataFrame(rows)
    table.to_csv(out / "gradient_polarity_threshold.csv", index=False)

    final = table[table["step"] == table["step"].max()]
    summary = {
        "identifiability_ratio": IDENTIFIABILITY_RATIO,
        "checkpoints_total": int(len(table)),
        "checkpoints_identified": int(table["identified"].sum()),
        "final_step": int(table["step"].max()),
        "final": [
            {
                "run": r.run,
                "group": r.group,
                "A_plus_mean": r.A_plus_mean,
                "A_minus_mean": r.A_minus_mean,
                "A_minus_sd": r.A_minus_sd,
                "abs_ratio": r.abs_ratio,
                "identified": bool(r.identified),
                # A+ = 0 limit: A_rho = (1 - rho) A-, so the group is helped by
                # every rho < 1 iff A- > 0, and by no rho at all iff A- < 0.
                "limit_rho_grad": 0.0 if r.A_minus_mean > 0 else 1.0,
            }
            for r in final.itertuples()
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"checkpoints where rho_grad is identified: "
          f"{summary['checkpoints_identified']}/{summary['checkpoints_total']} "
          f"(|A+|/|A-| > {IDENTIFIABILITY_RATIO})")
    print(f"\nfinal checkpoint (step {summary['final_step']}):")
    for entry in summary["final"]:
        print(f"  {entry['run']:42s} {entry['group']:3s} "
              f"A+={entry['A_plus_mean']:+.2e}  A-={entry['A_minus_mean']:+8.3f}  "
              f"|A+|/|A-|={entry['abs_ratio']:.1e}  "
              f"identified={entry['identified']}  "
              f"rho_grad(A+=0 limit)={entry['limit_rho_grad']:.0f}")
    print(f"\nwrote {out/'gradient_polarity_threshold.csv'} and {out/'summary.json'}")


if __name__ == "__main__":
    main()
