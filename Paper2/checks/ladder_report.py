"""Summarize untied_ladder.py output: per (level, rho, kappa, eps) solved/routed counts and median solve step.

    python Paper2/checks/ladder_report.py Paper2/checks/results/untied_ladder_n16.json
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
runs, tied = d["runs"], d["tied_ce_route_time"]
epss = sorted({r["eps"] for r in runs})
for level, rho in sorted({(r["level"], r["rho"]) for r in runs}):
    print(f"\n== level {level} rho={rho}: cell = solved/routed of seeds, median solve time (flow time); "
          f"[tied CE routing time]")
    print("kappa  " + "".join(f"{f'eps={e:g}':>22}" for e in epss))
    for kappa in sorted({r["kappa"] for r in runs}, reverse=True):
        cells = []
        for eps in epss:
            rr = [r for r in runs if (r["level"], r["rho"], r["kappa"], r["eps"]) == (level, rho, kappa, eps)]
            s = [r["solve_time"] for r in rr if r["solve_time"] is not None]
            nr = sum(r["route_time"] is not None for r in rr)
            t = tied.get(f"{kappa}|{eps}")
            med = f"{np.median(s):.0f}" if s else "-"
            cells.append(f"{len(s)}/{nr} {med} [{t:.0f}]" if t else f"{len(s)}/{nr} {med} [x]")
        print(f"{kappa:+.2f}  " + "".join(f"{c:>22}" for c in cells))

print("\n\n#### soft metric: final clean accuracy (mean over seeds) | median time to 99% clean accuracy (log grid)")
for level, rho in sorted({(r["level"], r["rho"]) for r in runs}):
    print(f"\n== level {level} rho={rho}")
    print("kappa  " + "".join(f"{f'eps={e:g}':>17}" for e in epss))
    for kappa in sorted({r["kappa"] for r in runs}, reverse=True):
        cells = []
        for eps in epss:
            rr = [r for r in runs if (r["level"], r["rho"], r["kappa"], r["eps"]) == (level, rho, kappa, eps)]
            acc = np.mean([r["final_clean_acc"] for r in rr])
            t99 = [next((c[0] for c in r["curve"] if c[2] >= .99), None) for r in rr]
            t99 = [t for t in t99 if t is not None]
            cells.append(f"{acc:.3f}|{np.median(t99):.0f}({len(t99)})" if t99 else f"{acc:.3f}|-")
        print(f"{kappa:+.2f}  " + "".join(f"{c:>17}" for c in cells))
