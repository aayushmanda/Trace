"""LaTeX table for the fully random (family C) initialization runs under the anti-copy law.

Each cell: runs (of 10 seeds x the listed initializations) that reach 99% clean-rule accuracy within the budget.
Also prints the median time to leave the chance plateau, which does not depend on the law.

    python Paper2/checks/make_randinit_table.py
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = []
for f in glob.glob(str(ROOT / "checks" / "results" / "randinit" / "C_n*_*.json")):
    runs += json.load(open(f))["runs"]
learned = lambda r: any(c[2] >= .99 for c in r["curve"])
exit_ = lambda r: next((c[0] for c in r["curve"] if c[1] <= .68), None)
GROUPS = [("adam", None, r"Adam, all $\varepsilon$"), ("gd", .5, r"GD, $\varepsilon\ge0.5$")]
lines = [r"\begin{tabular}{@{}l" + "ccc" * len(GROUPS) + "@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{g[2]}}}" for g in GROUPS) + r"\\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(GROUPS))),
         r"$N$ & " + " & ".join([r"$\kappa=1$ & $\kappa=0$ & $\kappa=-1$"] * len(GROUPS)) + r"\\", r"\midrule"]
for n in (8, 16, 32):
    cells = []
    for opt, emin, _ in GROUPS:
        for kappa in (1., 0., -1.):
            rr = [r for r in runs if r["n"] == n and r["opt"] == opt and r["kappa"] == kappa and (emin is None or r["eps"] >= emin)]
            cells.append(f"{sum(map(learned, rr))}/{len(rr)}" if rr else "--")
    lines.append(f"{n} & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "figures" / "randinit_table.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for n in (8, 16, 32):
    for opt in ("adam", "gd"):
        ex = {k: np.median([e for r in runs if (r["n"], r["opt"], r["kappa"]) == (n, opt, k) and (e := exit_(r)) is not None] or [np.nan])
              for k in (1., 0., -1.)}
        stuck = [round(r["final_loss"], 3) for r in runs if (r["n"], r["opt"]) == (n, opt) and r["kappa"] < 1 and not learned(r)]
        print(n, opt, "median plateau exit by kappa", {k: round(v, 1) for k, v in ex.items()},
              "stuck losses", sorted(set(stuck))[:4])
