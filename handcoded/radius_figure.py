"""Figure: geometry around the two programs. Reads results/radius/*.json.
    uv run python handcoded/radius_figure.py
"""
import glob
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42})
STEP, OUTC = "#1f77b4", "#d95f02"
fig, ax = plt.subplots(1, 3, figsize=(6.6, 1.75))

for rec in json.load(open("results/radius/axis.json")):
    if rec["n"] != 8:
        continue
    col = STEP if rec["mode"] == "process" else OUTC
    ax[0].plot(rec["alphas"], [max(v, 1e-4) for v in rec["loss"]], color=col, lw=1,
               label="step loss" if rec["mode"] == "process" else "outcome loss")
    good = [a for a, e in zip(rec["alphas"], rec["exact"]) if e == 1.0]
    ax[0].axvspan(min(good), max(good) + 0.004, color=col, alpha=0.18, lw=0)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"position $\alpha$ on the line $\theta_0\to\theta^\star$"); ax[0].set_ylabel("loss")
ax[0].set_title("(a) loss toward each program, $n=8$"); ax[0].legend(frameon=False, loc="lower left")

rows = json.load(open("results/radius/exact.json"))
for mode, col, lab in (("process", STEP, "step"), ("outcome", OUTC, "outcome")):
    rs = [r for r in rows if r["mode"] == mode]
    ns = [r["n"] for r in rs]; med = [statistics.median(r["radii"]) for r in rs]
    ax[1].fill_between(ns, [r["radii"][0] for r in rs], [r["radii"][-1] for r in rs], color=col, alpha=0.18, lw=0)
    ax[1].plot(ns, med, "o-", color=col, ms=2.5, lw=1, label=lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks([6, 8, 12, 16, 24, 32]); ax[1].set_xticklabels([6, 8, 12, 16, 24, 32])
ax[1].minorticks_off(); ax[1].set_xlabel("horizon $n$"); ax[1].set_ylabel("exactness radius")
ax[1].set_title("(b) random directions"); ax[1].legend(frameon=False)

runs = [r for f in glob.glob("results/radius/attract_n*.json") for r in json.load(open(f))]
for n, ls in ((8, "-"), (16, "--")):
    for mode, col, lab in (("process", STEP, "step"), ("outcome", OUTC, "outcome")):
        radii = sorted({r["r"] for r in runs if r["n"] == n and r["mode"] == mode})
        frac = [statistics.mean(r["final"] >= 0.99 for r in runs if r["n"] == n and r["mode"] == mode and r["r"] == x) for x in radii]
        ax[2].plot(radii, frac, "o" + ls, color=col, ms=2.5, lw=1, label=f"{lab}, $n={n}$")
ax[2].set_xscale("log"); ax[2].set_xlabel("start distance $r$"); ax[2].set_ylabel("fraction returning")
ax[2].set_title("(c) training from $\\theta^\\star+r\\xi$"); ax[2].legend(frameon=False, loc="lower left", fontsize=5.5)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("Paper/figures/program_geometry.pdf")
print("wrote Paper/figures/program_geometry.pdf")
