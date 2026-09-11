"""How gradient descent under clean process supervision reaches T_g.

Corollary 5 (`cor:clean-convergence` in the paper) claims that, under clean
(rho=1) process supervision, the population process loss is convex and
separable in row-logit coordinates with minimizer P_g = T_g, so gradient
descent from any finite initialization drives every predicted row toward the
true rule's one-hot distribution -- never exactly reaching it, since T_g is a
permutation and a one-hot softmax needs unbounded logits.

This script tracks that trajectory directly, in the same shared kernel as
`noise_threshold.py`: K=16 states, M=52 gates, full-batch gradient descent
from random logits on clean (rho=1) counts, checkpointed over steps. It
reports, at each checkpoint:

  rule      fraction of (g, i) rows whose argmax is the true successor
  answer    greedy free-running answer accuracy of the decoded table
  mass      mean predicted probability mass on the true successor,
            averaged over all K*M rows -- the quantity Corollary 5 says
            tends to 1 but is not attained at finite steps
  max_logit largest logit magnitude reached, as evidence that convergence
            is via growing logits, not via reaching T_g at finite parameters

No model is trained; this is the closed-form tabular kernel only.

Writes results/clean_convergence/summary.json and
Paper/figures/clean_convergence.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import reject_extra_flags
from src.experiments.noise_threshold import K, M, D, N_CIRCUITS, SEEDS, PERM, corrupted_counts, score, softmax

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "clean_convergence"
FIG = ROOT / "Paper" / "figures"

CHECKPOINTS = (0, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000)
LR = 5.0
PROCESS = "#087eaa"


def trajectory(counts, steps=CHECKPOINTS, lr=LR, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(M, K, K))
    row_totals = counts.sum(axis=2, keepdims=True)
    n = counts.sum()
    rows = []
    done = 0
    for target in steps:
        while done < target:
            logits -= lr * (softmax(logits) * row_totals - counts) / n
            done += 1
        p = softmax(logits)
        table = p.argmax(axis=2)
        rule, ans = score(table)
        mass = float(p[np.arange(M)[:, None], np.arange(K)[None, :], PERM].mean())
        rows.append({
            "step": done, "rule": rule, "answer": ans,
            "mass": mass, "max_logit": float(np.abs(logits).max()),
        })
        print(f"step={done:5d}  rule={rule*100:6.2f}%  answer={ans*100:6.2f}%  "
              f"mass={mass:.6f}  max|logit|={rows[-1]['max_logit']:.2f}", flush=True)
    return rows


def draw(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, [r["rule"] * 100 for r in rows], "o-", color=PROCESS, label="rule cells (%)")
    axes[0].plot(steps, [r["answer"] * 100 for r in rows], "s--", color="#c56b08", label="answer accuracy (%)")
    axes[0].set_xscale("symlog")
    axes[0].set_xlabel("gradient steps")
    axes[0].set_ylabel("%")
    axes[0].set_title("(a) Recovery of $T_g$ over training")
    axes[0].legend(frameon=True, loc="lower right")

    axes[1].plot(steps, [1 - r["mass"] for r in rows], "o-", color=PROCESS)
    axes[1].set_xscale("symlog")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("gradient steps")
    axes[1].set_ylabel(r"$1-$ mean mass on $T_g$")
    axes[1].set_title("(b) Gap to the one-hot limit (never zero)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    reject_extra_flags(argv, __doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = corrupted_counts(1.0, N_CIRCUITS, SEEDS[0])
    rows = trajectory(counts)
    (OUT / "summary.json").write_text(json.dumps(
        {"K": K, "M": M, "D": D, "n_circuits": N_CIRCUITS, "lr": LR, "rows": rows},
        indent=2) + "\n")
    draw(rows, FIG / "clean_convergence")
    print(f"\nwrote {OUT / 'summary.json'} and {FIG / 'clean_convergence.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
