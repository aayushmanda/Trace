"""Four-panel reliability figure (register machine 16 + boolean circuit 8)."""
from pathlib import Path
import sys

from src.experiments import reject_extra_flags

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plot_style import apply_style
RM = ROOT / "results/reliability_sweeps/register_machine_16_seeds_2001-2003.csv"
BC = ROOT / "results/reliability_sweeps/boolean_circuit_8_seeds_1000-1800.csv"
OUT = ROOT / "docs/assets/reliability_rm16_bc8.png"
PROCESS, OUTCOME, EXACT = "#087eaa", "#c56b08", "#1b7f5a"


def split_frame(path):
    df = pd.read_csv(path)
    outcome = df[df.condition == "outcome"]
    process = df[df.condition != "outcome"].copy()
    process["rho"] = process["rho"].astype(float)
    return process, outcome


def stats(process, col):
    g = process.groupby("rho")[col].agg(["mean", "std", "count"])
    g["std"] = g["std"].fillna(0)
    return g.reset_index()


def plot_accuracy(ax, process, outcome, title, chance):
    for col, color, label, marker, ls in [
        ("answer_accuracy", OUTCOME, "Answer acc", "o", "-"),
        ("exact_trace_accuracy", EXACT, "Exact trace", "s", "--"),
        ("trace_step_accuracy", PROCESS, "Trace step", "^", ":"),
    ]:
        g = stats(process, col)
        yerr = np.where(g["count"] > 1, g["std"], 0)
        ax.errorbar(g.rho, g["mean"], yerr=yerr, color=color, marker=marker, ls=ls,
                    ms=6, lw=2, capsize=3, label=label)
        ax.scatter(process.rho, process[col], s=16, color=color, alpha=0.28, linewidths=0)
    ax.axhline(outcome.answer_accuracy.mean(), color=OUTCOME, ls=":", lw=2, label="Outcome acc")
    ax.axhline(chance, color="#4a4a4a", ls="--", lw=1.3, label=f"Chance ({100 * chance:.2f}%)")
    ax.set_title(title)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="lower right")


def plot_loss(ax, process, outcome, title, ymax):
    g = stats(process, "loss")
    yerr = np.where(g["count"] > 1, g["std"], 0)
    ax.errorbar(g.rho, g["mean"], yerr=yerr, color=PROCESS, marker="o", ms=6, lw=2,
                capsize=3, label="Trace loss")
    ax.axhline(outcome.loss.mean(), color=OUTCOME, ls=":", lw=2, label="Outcome loss")
    ax.set_title(title)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9, loc="upper right")


def main(argv=None):
    reject_extra_flags(argv, __doc__)
    apply_style(extra={"pdf.fonttype": 42, "savefig.dpi": 300})
    rm, rm_out = split_frame(RM)
    bc, bc_out = split_frame(BC)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    plot_accuracy(axes[0], rm, rm_out, "Register machine 16: accuracy", 0.0588)
    plot_loss(axes[1], rm, rm_out, "Register machine 16: loss", ymax=1.0)
    plot_accuracy(axes[2], bc, bc_out, "Boolean circuit 8: accuracy", 1 / 16)
    plot_loss(axes[3], bc, bc_out, "Boolean circuit 8: loss", ymax=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300)
    print(f"saved {OUT}")
    print("register_machine_16 outcome  acc={:.3f} loss={:.3f}".format(
        rm_out.answer_accuracy.mean(), rm_out.loss.mean()))
    print("boolean_circuit_8   outcome  acc={:.3f} loss={:.3f}".format(
        bc_out.answer_accuracy.mean(), bc_out.loss.mean()))
    print(stats(rm, "answer_accuracy").to_string(index=False))
    print(stats(bc, "answer_accuracy").to_string(index=False))


if __name__ == "__main__":
    main()
