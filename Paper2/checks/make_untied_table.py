"""LaTeX table of the untied ladder under the anti-copy law: seeds reaching 99% clean accuracy, median flow time.

    python Paper2/checks/make_untied_table.py Paper2/checks/results/untied_ladder_anti_n16_v2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(sys.argv[1]))
runs = d["runs"]
EPS = [.003, .03, .1, .3, .5, 1.]
LEVELS = [("A", 0.), ("B", .1), ("B", .3), ("C", 0.)]
NAME = {("A", 0.): "A: directions", ("B", .1): r"B: $+\,0.1$ noise", ("B", .3): r"B: $+\,0.3$ noise",
        ("C", 0.): "C: random"}


def t99(r):
    return next((c[0] for c in r["curve"] if c[2] >= .99), None)


lines = [r"\begin{tabular}{@{}ll" + "c" * len(EPS) + "@{}}", r"\toprule",
         r"Initialization & $\kappa$ & " + " & ".join(f"$\\varepsilon={e:g}$" for e in EPS) + r"\\", r"\midrule"]
for li, key in enumerate(LEVELS):
    for ki, kappa in enumerate((1., 0., -1.)):
        cells = []
        for eps in EPS:
            rr = [r for r in runs if (r["level"], r["rho"], r["kappa"], r["eps"]) == (*key, kappa, eps)]
            ts = [t for t in map(t99, rr) if t is not None]
            cells.append(f"{len(ts)}/{len(rr)}" + (f" ({np.median(ts):,.0f})" if ts else ""))
        lines.append((NAME[key] if ki == 0 else "") + f" & ${kappa:+g}$ & " + " & ".join(cells) + r"\\")
    if li < len(LEVELS) - 1:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "figures" / "untied_table.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
