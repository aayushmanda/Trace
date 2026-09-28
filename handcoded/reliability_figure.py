"""Reliability figure: counting mod 3 over 16 letters, scattered errors, 32k updates.

Data are the per-seed entries of the appendix reliability table (seeds 42-44).

    python handcoded/reliability_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("/home/hariguru/aayus/trace/Paper/figures")
BLUE, ORANGE, GREEN, INK, MUTED = "#0a7fad", "#c8680a", "#2e8b57", "#1a1a1a", "#6a6a6a"

RHO = np.array([0.3, 0.4, 0.5, 0.7, 1.0])
FULL_EXACT = [[17, 16, 17], [49, 92, 66], [98, 91, 93], [90, 94, 98], [100, 100, 100]]
FULL_ANSWER = [[46, 43, 43], [66, 95, 78], [98, 94, 96], [94, 96, 98], [100, 100, 100]]
CLEAN = [[100, 99.9, 99.9], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]]
OUTCOME = [50.0, 33.5, 30.1]          # correct answers only; exact trace = answer here
K = 3


def panel(ax, full, title):
    ax.set_facecolor("#f3f3f8")
    ax.grid(color="white", lw=1.15)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    full, clean = np.array(full, float), np.array(CLEAN, float)
    for data, color, mk, label in ((clean, GREEN, "^", "its clean traces only"),
                                   (full, BLUE, "o", "full corpus")):
        for k in range(3):
            ax.plot(RHO, data[:, k], mk, color=color, ms=3.5, alpha=0.35, zorder=2)
        ax.plot(RHO, data.mean(1), "-", color=color, lw=2.0, zorder=3, label=label)
    ax.axhspan(min(OUTCOME), max(OUTCOME), color=ORANGE, alpha=0.18, lw=0, zorder=1)
    ax.axhline(np.mean(OUTCOME), color=ORANGE, lw=2.0, zorder=3, label="correct answers only")
    ax.axhline(100 / K, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.axvline(1 / K, color="0.35", lw=1.3, ls=":", zorder=1)
    ax.text(1 / K + 0.01, 4, r"$\rho=1/K$", fontsize=7.5, color="0.35", ha="left")
    ax.text(0.985, 100 / K - 2, "chance", fontsize=7, color=MUTED, ha="right", va="top")
    ax.set_xlim(0.26, 1.02)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.3, 0.4, 0.5, 0.7, 1.0])
    ax.set_xlabel(r"fraction of clean traces  $\rho$")
    ax.set_ylabel(title)
    ax.tick_params(labelsize=7)


def draw():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stixsans", "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
    })
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.45))
    panel(a, FULL_EXACT, "exact-trace accuracy (%)")
    panel(b, FULL_ANSWER, "answer accuracy (%)")
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=10, color=INK,
                fontweight="bold", ha="left", va="bottom")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"reliability_n16.{ext}", dpi=300)
    print(f"wrote {FIG / 'reliability_n16.pdf'}")


if __name__ == "__main__":
    draw()
