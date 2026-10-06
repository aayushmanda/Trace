"""Schematic of the second block of the two counting executors (main-text Section 3).
    uv run python handcoded/executor_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
STEP, OUTC, GREY = "#1f77b4", "#d95f02", "#555555"


def box(ax, x, y, w, h, text, col, fill=0.12):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=matplotlib.colors.to_rgba(col, fill), ec=col, lw=0.8))
    ax.text(x, y, text, ha="center", va="center")


def arrow(ax, a, b, col, text=None, rad=0.0, dx=0, dy=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=col, lw=0.8,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha="center", va="center", color=col, fontsize=6.5)


fig, ax = plt.subplots(1, 2, figsize=(6.6, 1.95))
for a in ax:
    a.set_xlim(0, 10); a.set_ylim(0, 4.2); a.axis("off")

a = ax[0]
a.set_title("(a) step executor: predict $s_t$", color=STEP)
box(a, 1.6, 0.6, 2.0, 0.55, "token $s_{t-1}$", GREY)
box(a, 4.4, 0.6, 2.2, 0.55, "letter $w_t$: $b_t$", GREY)
box(a, 1.6, 2.0, 2.0, 0.55, r"$\mathsf{prev}=\mathbf{e}_{s_{t-1}}$", STEP)
box(a, 4.4, 2.0, 2.0, 0.55, r"$\mathsf{mt}=b_t$", STEP)
box(a, 3.0, 3.45, 3.4, 0.6, "lookup, $2K$ units\n$s_t=s_{t-1}+b_t$", STEP)
arrow(a, (1.6, 0.9), (1.6, 1.7), STEP, "route", dx=-0.55)
arrow(a, (4.4, 0.9), (4.4, 1.7), STEP, "route", dx=0.55)
arrow(a, (1.8, 2.3), (2.6, 3.12), STEP)
arrow(a, (4.2, 2.3), (3.4, 3.12), STEP)
a.text(7.9, 2.0, "every head has\none source;\nno factor $n$", ha="center", va="center", fontsize=6.5, color=GREY)

a = ax[1]
a.set_title(r"(b) outcome executor: predict $s_n$", color=OUTC)
for i, x in enumerate((0.9, 2.0, 3.1)):
    box(a, x, 0.6, 0.9, 0.55, f"$b_{i + 1}$", GREY)
a.text(4.0, 0.6, r"$\cdots$", ha="center", va="center")
box(a, 4.9, 0.6, 0.9, 0.55, "$b_n$", GREY)
box(a, 7.6, 0.6, 1.6, 0.55, "token $s_0$", GREY)
box(a, 2.9, 2.0, 3.2, 0.55, r"$\mathsf{cnt}=n\cdot\frac{1}{n}\sum_tb_t=m$", OUTC)
box(a, 7.6, 2.0, 1.6, 0.55, r"$\mathsf{s0}=\mathbf{e}_{s_0}$", OUTC)
box(a, 5.2, 3.45, 4.2, 0.6, "lookup, $3K(n+1)$ units\n$s_n=s_0+m$", OUTC)
for x in (0.9, 2.0, 3.1, 4.9):
    arrow(a, (x, 0.9), (2.9 + (x - 2.9) * 0.35, 1.7), OUTC)
a.text(0.35, 1.35, "average\n$\\times n$", ha="center", va="center", color=OUTC, fontsize=6.5)
arrow(a, (7.6, 0.9), (7.6, 1.7), OUTC, "route", dx=0.55)
arrow(a, (3.2, 2.3), (4.6, 3.12), OUTC)
arrow(a, (7.4, 2.3), (6.0, 3.12), OUTC)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
fig.savefig("Paper/figures/executors.pdf")
print("wrote Paper/figures/executors.pdf")
