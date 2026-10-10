"""Build the LaTeX table of the reduced-ODE check (tab:J-ode).

    uv run python handcoded/make_ode_table.py   (from the repository root)
The first file holds K = 2 and the line gate W = w(I - R) for K >= 3 (the theorem); the second the full circulant gate readout.
"""
import json
import math
import statistics as st
import sys

import numpy as np

files = sys.argv[1:] or ["results/reduced_ode/runs.json", "results/reduced_ode/runs_fullgate.json"]
runs = [r for fn in files for r in json.load(open(fn))]
epss = sorted({r["eps"] for r in runs}, reverse=True)
lines = [r"\begin{table}[h]", r"\centering\small", r"\fitwidth{\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"$K$ & gate readout & solved & \multicolumn{3}{c}{median solve time} & slope of $\log T$ in $\log\frac{1}{\initscale}$ & checks\\",
         " & & & " + " & ".join(f"$\\initscale={e}$" for e in epss) + r" & (range) & passed\\", r"\midrule"]
fmt = lambda v: f"${v:.1e}".replace("e+0", r"\cdot10^{").replace("e+", r"\cdot10^{") + "}$"
groups = sorted({(r["K"], r.get("gate", "line")) for r in runs}, key=lambda kg: (kg[1] != "line", kg[0]))
for K, gate in groups:
    rs = [r for r in runs if r["K"] == K and r.get("gate", "line") == gate]
    solved = sum(r["solved_at"] is not None for r in rs)
    med = [st.median([r["solved_at"] for r in rs if r["eps"] == e and r["solved_at"]]) for e in epss]
    slopes = []
    for key in {(r["p"], r["seed"]) for r in rs}:
        pts = [(math.log(1 / r["eps"]), math.log(r["solved_at"])) for r in rs if (r["p"], r["seed"]) == key and r["solved_at"]]
        x, y = np.array(pts).T
        slopes.append(np.polyfit(x, y, 1)[0])
    checks = [r["G_keeps_sign"] and r["a_sign_changes"] <= 1 and r["invariant_after_alignment"] is True for r in rs]
    name = "--" if K == 2 else (r"$\R(\mathbf I-\mathbf R)$" if gate == "line" else "circulant")
    lines.append(f"{K} & {name} & {solved}/{len(rs)} & " + " & ".join(fmt(m) for m in med)
                 + f" & ${min(slopes):.2f}$--${max(slopes):.2f}$ & {sum(checks)}/{len(rs)}\\\\")
    if (K, gate) == (6, "line"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{\textbf{Reduced gate flow: every run solves, and solve times grow far more slowly than the proved bound.} "
          r"Rows with gate readout $\R(\mathbf I-\mathbf R)$ (and $K=2$) integrate \cref{eq:J-flow}, the setting of \cref{thm:joint}; rows marked circulant "
          r"train the full circulant gate readout of \cref{rem:J-line}. Each row: "
          r"$p\in\{0.1,0.25,0.75\}$, $\initscale\in\{0.3,0.2,0.1\}$, three seeds, $d_{\mathrm h}=8$; these $\initscale$ lie outside the range of "
          r"\cref{thm:joint} in the misaligned case. Checks: $g$ keeps its sign after prior fitting, $a$ crosses zero at most once, and from the "
          r"first time $|a|\ge\frac{a_{\mathrm{sc}}}{8}$ with the gate aligned, $ag\le0$ and $|a|$ is nondecreasing (\cref{lem:J-align}). "
          r"Implementation: \texttt{handcoded/reduced\_ode.py}.}",
          r"\label{tab:J-ode}", r"\end{table}"]
print("\n".join(lines))
