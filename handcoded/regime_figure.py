"""Figure: escape time of both losses inside the theorem's regime. Reads results/theorem_regime/both*.json.
    uv run python handcoded/regime_figure.py                      # original runs (nonzero readout)
    REGIME_FIG_ZERO=1 uv run python handcoded/regime_figure.py    # zero-readout runs of Theorem 3
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42})
STEP, OUTC = "#1f77b4", "#d95f02"
import glob
import os
ZERO = os.environ.get("REGIME_FIG_ZERO") == "1"   # zero-readout sweeps (Theorem 3); default: original runs
if ZERO:
    files = glob.glob("results/theorem_regime/both_zero_seed[0-9].json") + ["results/theorem_regime/both_zero_s02.json"]
else:
    files = [f for f in glob.glob("results/theorem_regime/both*.json") if "both_zero" not in f]
rows = [r for f in files for r in json.load(open(f))]
for r in rows:
    r.setdefault("seed", 0)
rows = list({(r["sigma"], r["n"], r["mode"], r["seed"]): r for r in rows}.values())
fig, ax = plt.subplots(1, 2, figsize=(5.4, 1.8))
for sigma, ls, mk in ((0.002, "-", "o"), (0.02, "--", "s")):
    for mode, col in (("step", STEP), ("outcome", OUTC)):
        for seed in sorted({r["seed"] for r in rows if r["sigma"] == sigma}):
            rs = sorted((r for r in rows if r["sigma"] == sigma and r["mode"] == mode and r["seed"] == seed and "T" in r),
                        key=lambda r: r["n"])
            if not rs:
                continue
            ns = [r["n"] for r in rs]
            lab = f"{mode}, $\\sigma={sigma}$" if seed == 0 else None
            ax[0].plot(ns, [r["T"] for r in rs], mk + ls, color=col, ms=2.2, lw=0.9, alpha=1 if seed == 0 else 0.55, label=lab)
            ax[1].plot(ns, [max(r["drop_over_sigma2"], 1e-12) / r["T"] for r in rs], mk + ls, color=col, ms=2.2, lw=0.9,
                       alpha=1 if seed == 0 else 0.55)
rows = [r for r in rows if r["seed"] == 0]
out = sorted((r for r in rows if r["sigma"] == 0.002 and r["mode"] == "outcome" and "T" in r), key=lambda r: r["n"])
if out:
    ns = [r["n"] for r in out]
    ax[0].plot(ns, [out[0]["T"] * 2.0 ** (n - ns[0]) for n in ns], ":", color="0.4", lw=0.8, label=r"$\lambda^{-n}$")
ax[0].set_yscale("log"); ax[0].set_xlabel("horizon $n$"); ax[0].set_ylabel(r"training time to move $0.1\,\sigma\|\tilde\theta\|$")
ax[0].set_title("(a) time to leave the initialization ball"); ax[1].legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax[1].set_yscale("log"); ax[1].set_xlabel("horizon $n$"); ax[1].set_ylabel(r"loss decrease $/\,(\sigma^2\cdot$ time$)$")
ax[1].set_title("(b) rate of loss decrease")
for a in ax:
    a.set_xticks([4, 6, 8, 10, 12, 14] if ZERO else [4, 6, 8, 10, 12, 14, 16]); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("Paper/figures/regime_both.pdf")
print("wrote Paper/figures/regime_both.pdf")
