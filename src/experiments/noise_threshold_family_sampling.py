"""Does the App. I recovery frontier depend on gate-string vs gate-family sampling?

App. I's tabular fit samples gates uniformly over the 52 legal strings; the
Transformer sweeps sample a gate family first, then a member of it. The two
frontiers being compared in body.tex (the tabular ~0.15 crossing vs. the
Transformer ~0.85 crossing) are therefore not sampled the same way. This
reruns exactly App. I's recovery-frontier sweep (same rhos, same seeds, same
N_CIRCUITS) with `sampling="family"` instead of the default `"uniform"`, to
check whether that confound explains any of the gap. Numpy only; no training;
does not touch noise_threshold.py's own default (uniform) or its output file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments import reject_extra_flags
from src.experiments.noise_threshold import K, M, D, N_CIRCUITS, SEEDS, corrupted_counts, decode, score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "noise_threshold"

RHOS = [0.02, 0.04, 1 / K, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.80, 1.00]


def frontier(sampling: str):
    rows = []
    for rho in RHOS:
        rule, ans = [], []
        for seed in SEEDS:
            counts = corrupted_counts(rho, N_CIRCUITS, seed=seed, sampling=sampling)
            r, a = score(decode(counts))
            rule.append(r)
            ans.append(a)
        rows.append({
            "rho": rho,
            "rule_mean": float(np.mean(rule)), "rule_sd": float(np.std(rule, ddof=1)),
            "answer_mean": float(np.mean(ans)), "answer_sd": float(np.std(ans, ddof=1)),
        })
        print(f"[{sampling:>7}] rho={rho:5.3f} rule={rows[-1]['rule_mean']*100:6.2f}% "
              f"answer={rows[-1]['answer_mean']*100:6.2f}%", flush=True)
    return rows


def crossing(rows, threshold=50.0):
    """First rho at which answer accuracy crosses `threshold`, by linear interp."""
    for a, b in zip(rows, rows[1:]):
        if a["answer_mean"] * 100 < threshold <= b["answer_mean"] * 100:
            span = b["answer_mean"] - a["answer_mean"]
            frac = (threshold / 100 - a["answer_mean"]) / span if span else 0.0
            return a["rho"] + frac * (b["rho"] - a["rho"])
    return None


def main(argv=None) -> int:
    reject_extra_flags(argv, __doc__)
    uniform = frontier("uniform")
    family = frontier("family")
    result = {
        "rhos": RHOS, "seeds": list(SEEDS), "n_circuits": N_CIRCUITS,
        "uniform": uniform, "family": family,
        "crossing_50pct_uniform": crossing(uniform),
        "crossing_50pct_family": crossing(family),
    }
    print(f"\n50% answer-accuracy crossing: uniform rho~={result['crossing_50pct_uniform']} "
          f"vs family rho~={result['crossing_50pct_family']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sampling_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT/'sampling_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
