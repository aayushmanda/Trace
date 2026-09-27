"""Main-text Figure: A_r vs W_r vs sqrt(A_r W_r) for counting at init.

    python handcoded/overlap_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/hariguru/aayus/trace")
FIG = ROOT / "Paper" / "figures"
REC = json.loads((ROOT / "results" / "paper" / "fig2" / "n8_s42.json").read_text())

# A_r: exact, from the closed form (stored in the init-grad artifact).
A = np.array(REC["A_r"], dtype=float)
# W_r: Table 12 / app:spectrum, n=8 seed 42. D(rho) is linear, so the
# letter-dependent energy is attributed to order 1.
W = np.zeros_like(A)
W[0] = 0.143
W[1] = 4.6e-4
overlap = np.sqrt(A * W)

BLUE, ORANGE, INK, MUTED = "#0a7fad", "#c8680a", "#1a1a1a", "#6a6a6a"
RS = np.arange(len(A))


def draw():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.38)

    specs = [
        (axes[0], A, False, ORANGE, r"$A_r$", r"(a)  task spectrum"),
        (axes[1], W, True, BLUE, r"$W_r$", r"(b)  model at init."),
        (axes[2], overlap, True, INK, r"$\sqrt{A_r W_r}$", r"(c)  overlap"),
    ]
    for ax, vals, logy, color, ylab, title in specs:
        ax.set_facecolor("#f3f3f8")
        ax.grid(color="white", lw=1.1, axis="y")
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.bar(RS, np.maximum(vals, 1e-18 if logy else 0.0),
               color=color, width=0.72, lw=0)
        ax.set_xticks(RS)
        ax.set_xlabel("order  $r$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
        ax.tick_params(labelsize=7)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(max(vals[vals > 0].min() / 8, 1e-8), max(vals.max() * 4, 1e-3))

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"overlap_spectra.{ext}", dpi=300)
    print(f"wrote {FIG / 'overlap_spectra.pdf'}")
    print("A_r shares", (A / A.sum()).round(4).tolist())
    print("W_r shares", (W / W.sum()).round(4).tolist())
    print("sqrt(A W)", [f"{v:.2e}" for v in overlap])


if __name__ == "__main__":
    draw()
