"""Figure: outcome loss is flat to order D (left) and the margin orders register-machine recovery (right).

Run from the repository root:
  uv run --with pandas --with matplotlib python -m src.experiments.plotting.plot_margin_and_flatness
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#184f95"]  # sequential steps for D = 2, 4, 8
PROCESS, SCATTERED, COHERENT = "#eb6834", "#2a78d6", "#eb6834"
DATA = Path("results/paper/matched_corruption")
OUT = Path("Paper/figures/margin_and_flatness")


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def flatness_panel(ax, k=16):
    gamma = np.linspace(0, 1, 400)
    ax.plot(gamma, np.log(k) - np.log1p((k - 1) * gamma), color=PROCESS, lw=2, ls="--")
    ax.text(0.08, 0.55, "process", color=INK, fontsize=8.5)
    for depth, color in zip((2, 4, 8), BLUE_RAMP):
        loss = np.log(k) - np.log1p((k - 1) * gamma**depth)
        ax.plot(gamma, loss, color=color, lw=2)
        level = 1.9
        x = gamma[np.argmin(np.abs(loss - level))]  # label each curve where it crosses one height
        ax.text(x + 0.02, level, f"$D={depth}$", color=INK, fontsize=8.5, va="bottom")
    ax.text(0.97, 2.55, "outcome", color=INK, fontsize=8.5, ha="right")
    ax.set_xlabel(r"distance $\gamma$ along $\mathbf{U}+\gamma(\mathbf{T}_a-\mathbf{U})$", fontsize=9, color=INK)
    ax.set_ylabel("loss (nats)", fontsize=9, color=INK)
    ax.set_title(r"Outcome loss is flat to order $D$ ($K=16$)", fontsize=10, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.log(k) + 0.15)
    style(ax)


def register_panel(ax, k=289):
    rows = []
    for name, label in (("symmetric", "scattered errors"), ("coherent", "one wrong rule")):
        frame = pd.read_csv(DATA / f"register_machine_16_{name}.csv")
        frame = frame[frame.step == frame.step.max()]
        summary = frame.groupby("rho").answer_accuracy.agg(["mean", "std"]) * 100
        rows.append((label, summary))
    for (label, summary), color, marker in zip(rows, (SCATTERED, COHERENT), ("o", "^")):
        ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], color=color, marker=marker, ms=6,
                    lw=2, capsize=3, markeredgecolor="white", markeredgewidth=1)
        ax.text(summary.index[0] - 0.02, summary["mean"].iloc[0], label, color=INK, fontsize=8.5, ha="right",
                va="center")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.49, 97, r"one-rule threshold $1/2$", color=MUTED, fontsize=8, ha="right", va="top")
    ax.axhline(100 / 17, color=MUTED, lw=1, ls=":")
    ax.text(0.03, 100 / 17 + 2, "chance", color=MUTED, fontsize=8)
    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0, 100)
    ax.set_xticks([1 / k, 0.3, 0.5, 0.7], [r"$\frac{1}{K}$", ".3", ".5", ".7"])
    ax.set_xlabel(r"fraction of clean traces $\rho$", fontsize=9, color=INK)
    ax.set_ylabel("answer accuracy (%)", fontsize=9, color=INK)
    ax.set_title(r"Register machine ($K=289$), 3 seeds", fontsize=10, color=INK, loc="left")
    style(ax)


def main():
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 2.9), gridspec_kw={"wspace": 0.3})
    flatness_panel(left)
    register_panel(right)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
