"""Appendix figure from results/landscape/{init,islands}.json.
    uv run python handcoded/landscape_figure.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "pdf.fonttype": 42})
init = {(r["n"], r["mode"]): r for r in json.load(open("results/landscape/init.json"))}
isl = {(r["n"], r["mode"]): r for r in json.load(open("results/landscape/islands.json"))}
fig, axs = plt.subplots(2, 2, figsize=(5.6, 4.0))
ax = axs.flatten()
vmax = 15
for a, mode, title in ((ax[0], "outcome", "(a) outcome, $n=8$"), (ax[1], "step", "(b) step, $n=8$")):
    r = init[(8, mode)]
    Z, x = np.array(r["Z"]), np.array(r["axis"])
    im = a.contourf(x, x, Z, levels=np.linspace(-vmax, vmax, 31), cmap="RdBu_r", extend="both")
    a.set_title(title)
    a.set_xlabel("random direction"); a.set_ylabel("own $-\\partial L/\\partial\\theta$")
    a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
fig.colorbar(im, cax=fig.add_axes([0.93, 0.6, 0.015, 0.28]), ticks=[-15, 0, 15])
for a, mode, title in ((ax[2], "process", "(c) step executor, $n=16$"), (ax[3], "outcome", "(d) outcome executor, $n=16$")):
    r = isl[(16, mode)]
    E, x = np.array(r["exact"]) == 1.0, np.array(r["axis"])
    a.contourf(x, x, E.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#eeeeee", "#1f77b4" if mode == "process" else "#d95f02"])
    a.plot([0], [0], "k+", ms=5)
    if mode == "outcome":
        a.text(0, -0.03, "exact only at the centre\n(narrower than one grid cell)", ha="center", fontsize=6)
    a.set_title(title + ", exact region"); a.set_xlabel("key tilt"); a.set_ylabel(r"Gaussian $\xi$")
    a.set_xticks([-0.05, 0, 0.05]); a.set_yticks([-0.05, 0, 0.05])
fig.subplots_adjust(hspace=0.5, wspace=0.45)
fig.savefig("Paper/figures/landscape.pdf", bbox_inches="tight")
print("wrote Paper/figures/landscape.pdf")
