"""Summarize the step-reachability sweep (handcoded/step_reach.py).

Per run: t_prior (loss within 0.005 of the state-only prior), t_solve (first eval margin > 0),
and the growth rate of the first saddle, fitted to log(gamma * v_ol) while both are small.
Then fits t_solve against n and log(1/sigma).

Usage: python handcoded/step_reach_report.py results/step_reach/sweep_K2_n*.json
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

runs = [r for f in sys.argv[1:] for r in json.load(open(f))]
rows = []
for r in runs:
    log, lr = r["log"], r["lr"]
    p = 1 / r["A"]
    prior = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    t_prior = next((x["step"] for x in log if x["loss"] <= prior + 5e-3), None)
    t_solve = next((x["step"] for x in log if x["margin"] > 0), None)
    # first-saddle growth: points after the prior with 3*init < gamma*v_ol < 0.05
    g0 = log[0]["gamma"] * log[0]["v_ol"]
    pts = [(x["step"], math.log(x["gamma"] * x["v_ol"])) for x in log
           if t_prior is not None and x["step"] >= t_prior and 3 * g0 < x["gamma"] * x["v_ol"] < 0.05]
    rate = np.polyfit(*zip(*pts), 1)[0] / lr if len(pts) >= 3 else float("nan")   # per unit time
    rows.append(dict(K=r["K"], n=r["n"], sigma=r["sigma"], seed=r["seed"], t_prior=t_prior,
                     t_solve=t_solve, rate=rate))

groups = defaultdict(list)
for x in rows:
    groups[(x["K"], x["n"], x["sigma"])].append(x)
print(f"{'K':>2} {'n':>3} {'sigma':>6}  solved  t_solve(median, range)   rate*n (median)")
for key in sorted(groups):
    g = groups[key]
    ts = [x["t_solve"] for x in g if x["t_solve"] is not None]
    rn = [x["rate"] * key[1] for x in g if not math.isnan(x["rate"])]
    rng = f"{min(ts)}-{max(ts)}" if ts else "-"
    med = f"{int(np.median(ts))}" if ts else "-"
    print(f"{key[0]:>2} {key[1]:>3} {key[2]:>6}  {len(ts)}/{len(g)}   {med:>6} ({rng:>11})   "
          f"{np.median(rn) if rn else float('nan'):.4f}")

for K in sorted({x["K"] for x in rows}):
    sel = [x for x in rows if x["K"] == K and x["t_solve"] is not None]
    if len(sel) < 4:
        continue
    X = np.array([[1, math.log(x["n"]), math.log(math.log(1 / x["sigma"]))] for x in sel])
    y = np.log([x["t_solve"] * runs[0]["lr"] for x in sel])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"K={K}: t_solve ~ n^{coef[1]:.2f} * log(1/sigma)^{coef[2]:.2f}  ({len(sel)} solved runs)")
