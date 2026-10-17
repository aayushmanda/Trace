"""Random-initialization (family C) sweep: learning and plateau exit by law, size, initialization and optimizer.

For each (N, optimizer, kappa, eps): seeds reaching 99% clean-rule accuracy within the budget, their median time, and
the median time to leave the chance plateau (loss below 0.68, against log 2 = 0.693). If plateau exit is the same for
every kappa while learning differs, the delay is in reading the rule; if plateau exit already differs, it is in escaping.

    python Paper2/checks/randinit_report.py
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def first(curve, test):
    return next((c[0] for c in curve if test(c)), None)


rows = []
for f in sorted(glob.glob(str(ROOT / "checks" / "results" / "randinit" / "C_n*_*.json"))):
    d = json.load(open(f))
    for r in d["runs"]:
        rows.append(dict(n=r["n"], opt=r["opt"], kappa=r["kappa"], eps=r["eps"], seed=r["seed"],
                         t99=first(r["curve"], lambda c: c[2] >= .99),
                         exit=first(r["curve"], lambda c: c[1] <= .68),
                         final_acc=r["final_clean_acc"]))
summary = {}
for n in sorted({r["n"] for r in rows}):
    for opt in sorted({r["opt"] for r in rows}):
        epss = sorted({r["eps"] for r in rows if (r["n"], r["opt"]) == (n, opt)})
        if not epss:
            continue
        print(f"\n== N={n} {opt}: cell = learned/10, median time to 99% | median plateau exit")
        print("kappa " + "".join(f"{f'eps={e:g}':>24}" for e in epss))
        for kappa in (1., 0., -1.):
            cells = []
            for eps in epss:
                rr = [r for r in rows if (r["n"], r["opt"], r["kappa"], r["eps"]) == (n, opt, kappa, eps)]
                t = [r["t99"] for r in rr if r["t99"] is not None]
                x = [r["exit"] for r in rr if r["exit"] is not None]
                cell = dict(learned=len(t), total=len(rr), t99=float(np.median(t)) if t else None,
                            exit=float(np.median(x)) if x else None, exited=len(x))
                summary[f"{n}|{opt}|{kappa}|{eps}"] = cell
                cells.append(f"{len(t)}/{len(rr)} {cell['t99'] or 0:,.0f} | {cell['exit'] or 0:,.0f}")
            print(f"{kappa:+.0f}    " + "".join(f"{c:>24}" for c in cells))
(ROOT / "checks" / "results" / "randinit_summary.json").write_text(json.dumps(summary, indent=1))
