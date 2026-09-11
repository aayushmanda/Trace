"""Separating trace reliability (fraction) from corpus size (amount).

`noise_threshold.py`'s main sweep varies rho at one fixed N=20,000 and shows
recovery lags the population threshold; it does not, by itself, show that
this gap is a finite-sample effect that closes with more data at a *fixed*
reliability, rather than something that would improve if rho itself grew.
This script holds rho fixed at several values strictly above the population
threshold 1/K and varies N alone, using the same shared kernel (identical
corrupted-count corpora, decode, and score as `noise_threshold.py`). If the
gap is finite-sample noise, not a fraction effect, then at each fixed
rho > 1/K, rule-cell recovery and answer accuracy should climb toward the
same population ceiling as N grows, without rho ever changing.

Writes results/fraction_vs_amount/summary.json and
Paper/figures/fraction_vs_amount.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import reject_extra_flags
from src.experiments.noise_threshold import K, corrupted_counts, decode, score
from src.plot_style import apply_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "fraction_vs_amount"
FIG = ROOT / "Paper" / "figures"

RHOS = (0.08, 0.10, 0.15, 0.20, 0.30)
NS = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
SEEDS = (2001, 2002, 2003)
N_POPULATION = 400_000  # large-N anchor for the population ceiling at each rho

COLORS = {0.08: "#b42318", 0.10: "#c56b08", 0.15: "#8a7000",
          0.20: "#2f7d32", 0.30: "#087eaa"}


def run() -> dict:
    rows = []
    for rho in RHOS:
        for n in NS:
            rule_vals, ans_vals = [], []
            for seed in SEEDS:
                table = decode(corrupted_counts(rho, n, seed))
                r, a = score(table)
                rule_vals.append(r)
                ans_vals.append(a)
            rows.append({
                "rho": rho, "N": n,
                "rule_mean": float(np.mean(rule_vals)), "rule_sd": float(np.std(rule_vals, ddof=1)),
                "answer_mean": float(np.mean(ans_vals)), "answer_sd": float(np.std(ans_vals, ddof=1)),
            })
            print(f"rho={rho:.2f} N={n:6d}  rule={rows[-1]['rule_mean']*100:6.2f}% "
                  f"(+/-{rows[-1]['rule_sd']*100:4.2f})  "
                  f"answer={rows[-1]['answer_mean']*100:6.2f}% "
                  f"(+/-{rows[-1]['answer_sd']*100:4.2f})", flush=True)

    ceiling = {}
    for rho in RHOS:
        r, a = score(decode(corrupted_counts(rho, N_POPULATION, SEEDS[0])))
        ceiling[rho] = {"rule": r, "answer": a}
        print(f"ceiling rho={rho:.2f} N={N_POPULATION}  rule={r*100:6.2f}%  answer={a*100:6.2f}%")

    return {"K": K, "rhos": list(RHOS), "Ns": list(NS), "seeds": list(SEEDS),
            "n_population": N_POPULATION, "rows": rows, "ceiling": ceiling}


def draw(results: dict, path: Path) -> None:
    apply_style()
    rows = results["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for rho in results["rhos"]:
        sub = [r for r in rows if r["rho"] == rho]
        sub.sort(key=lambda r: r["N"])
        xs = [r["N"] for r in sub]
        color = COLORS.get(rho, "#333333")
        axes[0].errorbar(xs, [r["rule_mean"] * 100 for r in sub],
                          yerr=[r["rule_sd"] * 100 for r in sub],
                          marker="o", ms=4, lw=1.6, capsize=2, color=color,
                          label=rf"$\rho={rho:.2f}$")
        axes[1].errorbar(xs, [r["answer_mean"] * 100 for r in sub],
                          yerr=[r["answer_sd"] * 100 for r in sub],
                          marker="o", ms=4, lw=1.6, capsize=2, color=color,
                          label=rf"$\rho={rho:.2f}$")
        ceil = results["ceiling"][str(rho)] if isinstance(next(iter(results["ceiling"])), str) \
            else results["ceiling"][rho]
        axes[0].axhline(ceil["rule"] * 100, color=color, ls=":", lw=1.0)
        axes[1].axhline(ceil["answer"] * 100, color=color, ls=":", lw=1.0)

    for ax, title, ylabel in (
        (axes[0], "(a) Rule cells recovered", "rule cells (%)"),
        (axes[1], "(b) Answer accuracy", "answer accuracy (%)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("corpus size $N$ (fixed $\\rho$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-3, 105)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Fraction ($\\rho$) held fixed, amount ($N$) varied: dotted "
                  "lines are each $\\rho$'s population ceiling", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    reject_extra_flags(argv, __doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    results = run()
    # JSON keys must be strings; keep an str-keyed copy of `ceiling` on disk.
    results["ceiling"] = {str(k): v for k, v in results["ceiling"].items()}
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    draw(results, FIG / "fraction_vs_amount")
    print(f"\nwrote {OUT / 'summary.json'} and {FIG / 'fraction_vs_amount.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
