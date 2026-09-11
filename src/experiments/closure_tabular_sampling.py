"""Rerun the App. I tabular 1/K vs 1/2 fit under sweep gate-family sampling.

CPU. No training. Removes the known confound: App. I samples uniformly over
the 52 gate strings; Transformer sweeps sample a gate *family* first (x/c/s/t)
then an index. Same ρ grid, both corruption laws (`symmetric`, `coherent`).

Population thresholds stay 1/K and 1/2. This script only asks whether the
*tabular* frontiers move when sampling is matched to the sweep. It does not
predict the 0.15→0.85 Transformer offset (that remains unexplained; D.10).

Writes results/closure_handcoded/tabular_sampling/{summary.json,frontiers.csv,frontiers.pdf}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import noise_threshold as nt
from src.plot_style import apply_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "closure_handcoded" / "tabular_sampling"
RHOS = (0.05, 1 / nt.K, 0.10, 0.25, 0.40, 0.49, 0.51, 0.60, 0.80)
SAMPLINGS = ("uniform", "family")
LAWS = ("symmetric", "coherent")
PROCESS = "#087eaa"
COHERENT = "#b42318"


def coherent_wrong(seed=0):
    rs = np.random.default_rng(seed)
    sigma = rs.permutation(nt.K)
    while (sigma == np.arange(nt.K)).any():
        sigma = rs.permutation(nt.K)
    return np.array([[sigma[nt.PERM[g][i]] for i in range(nt.K)] for g in range(nt.M)])


def score(table, sampling, seed=99, n_eval=1000, depth=nt.D):
    """Answer accuracy of executing `table`; circuits drawn with `sampling`."""
    rule = float((table == nt.PERM).mean())
    rng = __import__("random").Random(seed)
    ok = 0
    for _ in range(n_eval):
        s = rng.randrange(nt.K)
        gates = [nt.sample_gate_index(rng, sampling) for _ in range(depth)]
        true, pred = s, s
        for g in gates:
            true, pred = nt.PERM[g][true], table[g][pred]
        ok += true == pred
    return rule, ok / n_eval


def rho50(rows, sampling, law):
    for r in sorted(rows, key=lambda x: x["rho"]):
        if r["sampling"] == sampling and r[f"{law}_answer"] >= 0.5:
            return float(r["rho"])
    return float("nan")


def draw(rows, path):
    apply_style({
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    styles = {
        ("uniform", "symmetric"): (PROCESS, "o", "-"),
        ("uniform", "coherent"): (COHERENT, "^", "-"),
        ("family", "symmetric"): (PROCESS, "o", "--"),
        ("family", "coherent"): (COHERENT, "^", "--"),
    }
    for ax, sampling in zip(axes, SAMPLINGS):
        xs = [r["rho"] for r in rows if r["sampling"] == sampling]
        for law in LAWS:
            color, marker, ls = styles[(sampling, law)]
            ys = [r[f"{law}_answer"] for r in rows if r["sampling"] == sampling]
            ax.plot(xs, ys, color=color, marker=marker, ls=ls, ms=4, lw=1.7, label=law)
        ax.axvline(1 / nt.K, color=PROCESS, ls=":", lw=0.9)
        ax.axvline(0.5, color=COHERENT, ls=":", lw=0.9)
        ax.axhline(1 / nt.K, color="#4a4a4a", ls=":", lw=0.8)
        ax.set_title(f"{sampling} gate sampling")
        ax.set_xlabel(r"$\rho$")
        ax.set_xlim(0, 0.85)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(frameon=True, loc="lower right")
    axes[0].set_ylabel("tabular greedy answer accuracy")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=nt.N_CIRCUITS)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="n=2000; plumbing, not the paper N")
    args = p.parse_args(argv)
    n = 2000 if args.quick else args.n
    OUT.mkdir(parents=True, exist_ok=True)
    wrong = coherent_wrong(0)
    rows = []
    for sampling in SAMPLINGS:
        for rho in RHOS:
            row = {"rho": float(rho), "sampling": sampling, "n": n}
            for law in LAWS:
                counts = nt.corrupted_counts(float(rho), n, nt.SEEDS[0], law, wrong, sampling=sampling)
                rule, ans = score(nt.decode(counts), sampling, n_eval=args.n_eval)
                row[f"{law}_rule"] = float(rule)
                row[f"{law}_answer"] = float(ans)
            rows.append(row)
            print(
                f"{sampling:8} ρ={rho:6.4f}  "
                f"sym={row['symmetric_answer']*100:6.2f}%  "
                f"coh={row['coherent_answer']*100:6.2f}%",
                flush=True,
            )

    frontiers = []
    for sampling in SAMPLINGS:
        rec = {
            "sampling": sampling,
            "symmetric_rho50": rho50(rows, sampling, "symmetric"),
            "coherent_rho50": rho50(rows, sampling, "coherent"),
        }
        rec["order_coh_minus_sym"] = rec["coherent_rho50"] - rec["symmetric_rho50"]
        at = next(r for r in rows if r["sampling"] == sampling and abs(r["rho"] - 0.25) < 1e-12)
        rec["symmetric_at_0.25"] = at["symmetric_answer"]
        rec["coherent_at_0.25"] = at["coherent_answer"]
        frontiers.append(rec)
        print(
            f"[frontier] {sampling:8}  ρ50_sym={rec['symmetric_rho50']:.4f}  "
            f"ρ50_coh={rec['coherent_rho50']:.4f}",
            flush=True,
        )

    keys = list(rows[0])
    with (OUT / "structure.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "frontiers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frontiers[0]))
        w.writeheader()
        w.writerows(frontiers)
    summary = {
        "note": (
            "uniform = App. I (52 strings equally). family = sweep sampler. "
            "Frontiers are tabular empirical-optimum ρ_50, not Transformer ρ_50. "
            "Matching sampling removes the confound; it does not explain 0.15→0.85."
        ),
        "n": n,
        "rhos": [float(x) for x in RHOS],
        "frontiers": frontiers,
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw(rows, OUT / "frontiers")
    print(f"wrote {OUT / 'frontiers.csv'} and {OUT / 'frontiers.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
