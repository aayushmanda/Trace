"""Figure: outcome held-out accuracy against lambda^n over the (n, lambda) grid.
Reads results/paper/lambda_grid/*.json (handcoded/lambda_grid.py).
    uv run python handcoded/lambda_grid_figure.py
"""
import collections
import glob
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42})
rows = [json.load(open(f)) for f in glob.glob("results/paper/lambda_grid/n*_A*_s*.json")]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["n"], r["alphabet"])].append(r)
lams = sorted({round(abs(1 - 2 / A), 3) for _, A in cells})
colors = dict(zip(lams, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]))
markers = {6: "o", 8: "s", 10: "^", 12: "D", 16: "v"}

fig, ax = plt.subplots(figsize=(3.3, 1.9))
for (n, A), rs in sorted(cells.items()):
    lam = round(abs(1 - 2 / A), 3)
    acc = [r["outcome"]["final_exact"] for r in rs]
    x, y = lam ** n, sum(acc) / len(acc)
    ax.errorbar(x, y, yerr=[[y - min(acc)], [max(acc) - y]], fmt=markers[n], color=colors[lam],
                ms=3.5, mec="black", mew=0.3, elinewidth=0.6, capsize=1.2)
ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
ax.text(1.2e-5, 0.515, "chance", color="0.4", fontsize=5.5)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda^n$")
ax.set_ylabel("outcome accuracy (held out)")
for lam in lams:
    ax.plot([], [], "s", color=colors[lam], mec="black", mew=0.3, ms=4, label=fr"$\lambda={lam:.2f}$")
for n, m in markers.items():
    ax.plot([], [], m, color="white", mec="black", mew=0.5, ms=3.5, label=f"$n={n}$")
ax.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.2, columnspacing=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("Paper/figures/lambda_grid.pdf")
print("wrote Paper/figures/lambda_grid.pdf")
