"""Figure 1 of the paper: a corrupted trace, its two supervision targets, and the margin of one row.

Run from the repository root:
  python -m src.experiments.plotting.plot_task_schematic
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
CORRECT, WRONG = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
OUT = Path("Paper/figures/task_schematic")


def box(ax, x, y, text, edge=INK, face="white", color=INK, dashed=False):
    ax.add_patch(FancyBboxPatch((x - 0.32, y - 0.22), 0.64, 0.44, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor=edge, facecolor=face, linestyle="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color)


def arrow(ax, x0, x1, y, label):
    ax.add_patch(FancyArrowPatch((x0 + 0.34, y), (x1 - 0.34, y), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
    ax.text((x0 + x1) / 2, y + 0.3, label, ha="center", va="center", fontsize=9, color=MUTED)


def trace_panel(ax):
    xs = [0, 1.25, 2.5, 3.75, 5.0]
    states = ["$s_0$", "$s_1$", r"$\tilde s_2$", "$s_3$", "$s_4$"]
    for i, (x, s) in enumerate(zip(xs, states)):
        corrupted = i == 2
        box(ax, x, 1.6, s, edge=WRONG if corrupted else INK, color=WRONG if corrupted else INK)
        if i:
            arrow(ax, xs[i - 1], x, 1.6, f"$a_{i}$")
    ax.text(2.5, 2.25, "wrong step", ha="center", fontsize=8.5, color=WRONG)
    ax.text(-0.55, 0.75, "process", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(-0.55, 0.05, "outcome", ha="right", va="center", fontsize=9.5, color=INK)
    for i, x in enumerate(xs[1:], start=1):
        box(ax, x, 0.75, states[i], edge=WRONG if i == 2 else CORRECT, color=WRONG if i == 2 else CORRECT, dashed=True)
    box(ax, xs[-1], 0.05, states[-1], edge=CORRECT, color=CORRECT, dashed=True)
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-0.3, 2.45)
    ax.set_title("A trace and its supervision targets", fontsize=10.5, color=INK, loc="left")
    ax.axis("off")


def row_panel(ax, title, wrong):
    rho = 0.4
    values = [rho] + wrong
    labels = ["$y$"] + [f"$j_{i}$" for i in range(1, len(wrong) + 1)]
    colors = [CORRECT] + [WRONG] * len(wrong)
    bars = ax.bar(range(len(values)), values, width=0.62, color=colors, edgecolor="white", linewidth=2)
    strongest = 1 + max(range(len(wrong)), key=wrong.__getitem__)
    for index, (bar, value) in enumerate(zip(bars, values)):
        if index not in (0, strongest):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=INK)
    margin = rho - max(wrong)
    ax.set_title(f"{title}\n$m={margin:+.2f}$", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(values)), labels, fontsize=9, color=INK)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.25, 0.5], ["0", ".25", ".5"], fontsize=8, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def main():
    fig = plt.figure(figsize=(9.2, 2.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.35)
    trace_panel(fig.add_subplot(grid[0]))
    # Rows of Q_a at rho = 0.4, K = 5: wrong mass 0.6 spread over four states or put on one.
    ax_diffuse = fig.add_subplot(grid[1])
    row_panel(ax_diffuse, "Scattered errors", [0.15, 0.15, 0.15, 0.15])
    ax_diffuse.set_ylabel(r"$\bm{q}(\cdot)$ at $\rho=0.4$".replace(r"\bm", r"\mathbf"), fontsize=9, color=INK)
    row_panel(fig.add_subplot(grid[2]), "One wrong rule", [0.6, 0.0, 0.0, 0.0])
    for suffix in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
