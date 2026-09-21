"""Publication figures from existing CSVs only. No training, no invented numbers.

Writes the architecture schematic the paper cites, PDF+PNG (300 dpi), under
Paper/figures/architectures.pdf. The reliability/corruption/gradient/credit
figures live in plot_main_paper.py; this module also supplies the shared
style() and save() helpers those figures import.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import apply_style
from src.experiments import reject_extra_flags

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "Paper"
FIG = PAPER / "figures"

PROCESS = "#087eaa"
OUTCOME = "#c56b08"
MIXED = "#6b4c9a"
EXACT = "#1b7f5a"
STEP = "#b42318"
CHANCE = "#4a4a4a"
BASELINE = "#5c5c5c"

LABELS = {"process": "Process", "outcome": "Outcome", "both": "Mixed format"}
MODE_COLOR = {"process": PROCESS, "outcome": OUTCOME, "both": MIXED}

# apply_style() sets DejaVu / #EAEAF2. Extra sizes are NeurIPS-readable (not 21pt,
# which collides on two-column panels) while staying well above tiny 7–8pt captions.
PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "lines.linewidth": 2.0,
}


def style(**extra):
    params = dict(PAPER_RC)
    params.update(extra)
    apply_style(extra=params)


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / stem
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(".pdf"))
    return path.with_suffix(".pdf")


def plot_architectures():
    style()
    fig, ax = plt.subplots(figsize=(7.0, 3.05), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, text, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, 0.53, boxstyle="round,pad=0.06",
            edgecolor=color, facecolor="white", lw=1.6,
        ))
        ax.text(x + w / 2, y + 0.265, text, ha="center", va="center", fontsize=11)

    def arrow(x, y, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x, y),
                    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": ".3"})

    ax.text(0.15, 2.68, "Process architecture: one block reused across output positions",
            fontsize=13, weight="bold", color=PROCESS)
    box(0.2, 1.92, 1.3, "Prompt", PROCESS)
    box(2.15, 1.92, 2.1, "One causal block", PROCESS)
    box(5, 1.92, 3.9, r"Generated trace: $s_1,s_2,\ldots,s_D$", PROCESS)
    arrow(1.6, 2.18, 2.07, 2.18)
    arrow(4.32, 2.18, 4.92, 2.18)
    ax.plot([6.9, 6.9, 3.2], [1.84, 1.63, 1.63], lw=1.2, color=".35")
    arrow(3.2, 1.63, 3.2, 1.84)
    ax.text(5, 1.38, "Feed generated states back", ha="center", fontsize=11)

    ax.text(0.15, 1.02, "Outcome architecture: one internal block per transition",
            fontsize=13, weight="bold", color=OUTCOME)
    for x, w, t in [
        (0.2, 1.3, r"$h_0$"), (2.15, 1.4, r"$F_1\to h_1$"),
        (4.2, 1.4, r"$F_2\to h_2$"), (6.3, 1.4, r"$F_D\to h_D$"),
        (8.4, 1.3, r"Answer $s_D$"),
    ]:
        box(x, 0.2, w, t, OUTCOME)
    for x, x2 in [(1.6, 2.07), (3.62, 4.12), (5.67, 6.22), (7.77, 8.32)]:
        arrow(x, 0.465, x2, 0.465)
    ax.text(5.94, 0.64, r"$\cdots$", ha="center", fontsize=13)
    return save(fig, "architectures")


def main(argv=None):
    reject_extra_flags(argv, __doc__)
    plot_architectures()
