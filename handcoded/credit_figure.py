"""Figure 2: where the credit is. Left schematic; right, measured init signals.

    python handcoded/credit_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path("/home/hariguru/aayus/trace")
DATA = ROOT / "results" / "paper" / "fig2"
FIG = ROOT / "Paper" / "figures"

BLUE, ORANGE, INK, MUTED = "#0a7fad", "#c8680a", "#1a1a1a", "#6a6a6a"
NS = np.array([2, 4, 6, 8, 9, 10])


def load_signals():
    out, cot = [], []
    for n in NS:
        o, c = [], []
        for seed in (42, 43, 44):
            rec = json.loads((DATA / f"n{n}_s{seed}.json").read_text())
            o.append(rec["grad_outcome"])
            c.append(rec["grad_process"])
        out.append(o)
        cot.append(c)
    return np.array(out), np.array(cot)


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
    fig = plt.figure(figsize=(7.1, 2.7))
    ax = fig.add_axes([0.01, 0.08, 0.50, 0.84])
    bx = fig.add_axes([0.58, 0.20, 0.40, 0.70])

    # --- (a) schematic ---
    n = 6
    ax.set_xlim(-0.7, n + 0.7)
    ax.set_ylim(-2.25, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    for t in range(n + 1):
        ax.add_patch(Circle((t, 0), 0.27, fc="white", ec=INK, lw=1.2, zorder=3))
        ax.text(t, 0, f"$s_{{{t}}}$", ha="center", va="center", fontsize=9,
                color=INK, zorder=4)
    for t in range(n):
        ax.add_patch(FancyArrowPatch((t + 0.29, 0), (t + 0.71, 0),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=MUTED, lw=1.05))
        ax.text(t + 0.5, 0.20, f"$x_{{{t + 1}}}$", ha="center", va="bottom",
                fontsize=7, color=MUTED)
    for t in range(1, n + 1):
        ax.plot([t - 0.72, t - 0.72, t + 0.04, t + 0.04],
                [0.52, 0.72, 0.72, 0.52], color=BLUE, lw=1.7, solid_capstyle="butt")
    ax.text(n / 2 + 0.15, 1.28,
            r"$\mathrm{step\ target{:}\ one\ transition}$",
            ha="center", va="center", color=BLUE, fontsize=9)
    ax.plot([0.04, 0.04, n - 0.04, n - 0.04], [-0.52, -0.82, -0.82, -0.52],
            color=ORANGE, lw=1.7, solid_capstyle="butt")
    ax.text(n / 2, -1.22, r"$\mathrm{answer{:}\ all\ }n\mathrm{\ transitions}$",
            ha="center", color=ORANGE, fontsize=9)
    ax.text(n / 2, -1.68, r"$\mathrm{credit\ to\ order\ }r\ \leq\ \sqrt{A_r W_r}$",
            ha="center", color=INK, fontsize=8.5)
    ax.text(n / 2, -1.98, r"$\mathrm{reversible{:}\ \leq\,}\lambda^{n-r}$",
            ha="center", color=MUTED, fontsize=7.5)
    ax.text(-0.65, 1.78, "(a)", fontsize=10, color=INK, fontweight="bold")

    # --- (b) three-seed measurements ---
    out, cot = load_signals()
    bx.set_facecolor("#f3f3f8")
    bx.grid(color="white", lw=1.15)
    bx.set_axisbelow(True)
    for s in bx.spines.values():
        s.set_visible(False)
    for seed_i, alpha in enumerate((0.35, 0.35, 0.35)):
        bx.plot(NS, cot[:, seed_i], "o", color=BLUE, ms=3.5, alpha=alpha, zorder=2)
        bx.plot(NS, out[:, seed_i], "s", color=ORANGE, ms=3.5, alpha=alpha, zorder=2)
    bx.plot(NS, cot[:, 0], "-", color=BLUE, lw=2.0, zorder=3)
    bx.plot(NS, out[:, 0], "-", color=ORANGE, lw=2.0, zorder=3)
    grid_n = np.linspace(2, 10, 80)
    bx.plot(grid_n, out[0, 0] * (0.5 ** (grid_n - 2)), ":", color="0.35", lw=1.45, zorder=1)
    bx.set_yscale("log")
    bx.set_xticks(list(NS))
    bx.set_xlabel("steps  $n$")
    bx.set_ylabel("gradient norm at initialization")
    bx.tick_params(labelsize=7)
    bx.set_xlim(1.4, 12.3)
    bx.text(10.2, cot[-1, 0], "step", va="center", ha="left", fontsize=8, color=BLUE)
    bx.text(10.2, out[-1, 0], "answer", va="center", ha="left", fontsize=8, color=ORANGE)
    bx.text(5.0, out[0, 0] * (0.5 ** 2.85), r"$\lambda^{n}$", fontsize=8.5, color="0.35")
    bx.text(1.45, 1.15 * bx.get_ylim()[1], "(b)", fontsize=10, color=INK,
            fontweight="bold", ha="left", va="bottom")

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"credit_locality.{ext}", dpi=300)
    print(f"wrote {FIG / 'credit_locality.pdf'}")
    print("seed 42 answer", [f"{v:.2e}" for v in out[:, 0]])
    print("seed 42 step  ", [f"{v:.3e}" for v in cot[:, 0]])


if __name__ == "__main__":
    draw()
