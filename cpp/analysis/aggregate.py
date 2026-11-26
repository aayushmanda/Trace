#!/usr/bin/env python3
"""Aggregate the 20-seed full-model sweep (causal_interventions) into solve rates, Wilson intervals,
Fisher exact tests and mechanism diagnostics. Writes summary.json and prints a table.

Solve rule (used everywhere in the paper): final free-running exact rollout >= 99.9% on the
1,000 held-out prompts.
"""
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
try:
    from scipy.stats import fisher_exact
except ImportError:
    def fisher_exact(table):
        a, b = table[0]
        c, d = table[1]
        r1, c1, n = a + b, a + c, a + b + c + d
        def pmf(x):
            return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
        p_obs = pmf(a)
        lo, hi = max(0, r1 + c1 - n), min(r1, c1)
        pv = sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1.0 + 1e-7))
        return ((a * d) / (b * c) if b * c > 0 else float("inf"), min(1.0, pv))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SWEEP_DIR = os.environ.get("SWEEP_DIR", os.path.join(ROOT, "results", "causal_intervention", "sweep"))
SOLVE = 99.9
FINAL = 2400

MODES = {0: "Natural", 1: "Suppressed", 2: "External offset", 3: "Restored", 4: "Weight-calibrated V(0)=c*",
         5: "Ext. offset + V(0)!=0", 6: "Ext. offset, removed at 300", 7: "Ext. offset + 10x lr(V)", 8: "Block-RMS SGD",
         9: "Outcome Block-RMS SGD", 10: "Outcome AdamW"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                break  # partially written line from a run still in progress
    return rows


def at(rows, step, key):
    for r in rows:
        if int(r["step"]) == step:
            return r[key]
    return float("nan")


def first_solve(rows):
    for r in rows:
        if r["rollout_acc"] >= SOLVE:
            return int(r["step"])
    return None


def main():
    groups = defaultdict(list)
    for sub in ("main", "rel600", "rel1000"):
        for path in glob.glob(os.path.join(SWEEP_DIR, sub, "*.csv")):
            m = re.search(r"n(\d+)_p([\d.]+)_m(\d+)_s(\d+)\.csv", path)
            _, p, mode, seed = int(m[1]), m[2], int(m[3]), int(m[4])
            rows = load(path)
            if not rows or int(rows[-1]["step"]) < FINAL:
                continue  # unfinished
            key = (p, mode if mode != 3 else f"3-{sub}")
            groups[key].append((seed, rows))

    out = {}
    print(f"{'p':>5} {'condition':32} {'solved':>7} {'Wilson95':>13} {'med solve':>9} "
          f"{'Edist@100':>9} {'chat@100':>8} {'lagn@400':>8} {'|V|@400':>7} {'probe@final':>11}")
    for key in sorted(groups, key=lambda k: (k[0], str(k[1]))):
        p, mode = key
        runs = sorted(groups[key])
        k = sum(r[-1]["rollout_acc"] >= SOLVE for _, r in runs)
        nrun = len(runs)
        lo, hi = wilson(k, nrun)
        fs = sorted(x for x in (first_solve(r) for _, r in runs) if x is not None)
        med = fs[len(fs) // 2] if fs else None
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        d = dict(p=float(p), mode=str(mode), solved=k, n=nrun, wilson=(lo, hi), median_first_solve=med,
                 first_solves=fs,
                 final_rollout=[r[-1]["rollout_acc"] for _, r in runs],
                 seeds=[s for s, _ in runs],
                 Edist0=mean([at(r, 0, "E_dist") for _, r in runs]),
                 Edist100=mean([at(r, 100, "E_dist") for _, r in runs]),
                 Edist400=mean([at(r, 400, "E_dist") for _, r in runs]),
                 chat100=mean([at(r, 100, "c_hat") for _, r in runs]),
                 lagn400=mean([at(r, 400, "lag_n_share") for _, r in runs]),
                 prefix_lagn400=mean([at(r, 400, "prefix_lag_n") for _, r in runs]),
                 prefix_lag1_400=mean([at(r, 400, "prefix_lag_1") for _, r in runs]),
                 normV100=mean([at(r, 100, "norm_V") for _, r in runs]),
                 normV400=mean([at(r, 400, "norm_V") for _, r in runs]),
                 qk400=mean([at(r, 400, "qk_score_rms") for _, r in runs]),
                 probe_final=mean([r[-1]["b1_probe_acc"] for _, r in runs]),
                 probe400=mean([at(r, 400, "b1_probe_acc") for _, r in runs]),
                 rollout600=mean([at(r, 600, "rollout_acc") for _, r in runs]),
                 rollout1200=mean([at(r, 1200, "rollout_acc") for _, r in runs]),
                 final_mean=mean([r[-1]["rollout_acc"] for _, r in runs]))
        out[f"p{p}_{mode}"] = d
        name = MODES.get(mode, f"Mode {mode}") if isinstance(mode, int) else f"Restored ({mode.split('-rel')[-1]})"
        print(f"{p:>5} {name:32} {k:>3}/{nrun:<3} [{lo:.2f},{hi:.2f}] {str(med):>9} "
              f"{d['Edist100']:9.4f} {d['chat100']:8.3f} {d['lagn400']:8.3f} {d['normV400']:7.3f} {d['probe_final']:11.2f}")

    tests = {}
    def test(a, b, label):
        if a in out and b in out:
            A, B = out[a], out[b]
            table = [[A["solved"], A["n"] - A["solved"]], [B["solved"], B["n"] - B["solved"]]]
            _, pv = fisher_exact(table)
            tests[label] = dict(table=table, p_two_sided=pv)
    for p in ("0.05", "0.10", "0.15", "0.25", "0.50", "0.75"):
        test(f"p{p}_0", f"p{p}_1", f"p={p} Natural vs Suppressed")
    for m in ("2", "4", "5", "6", "7", "8", "3-rel600", "3-rel1000"):
        test("p0.25_0", f"p0.25_{m}", f"p=0.25 Natural vs mode {m}")
    test("p0.25_2", "p0.25_5", "p=0.25 Ext offset vs Ext offset + V(0)!=0")
    test("p0.25_2", "p0.25_6", "p=0.25 Ext offset vs transient offset")
    test("p0.25_1", "p0.50_1", "Suppressed p=0.25 vs p=0.50")
    print("\nFisher exact (two-sided):")
    for k, v in tests.items():
        print(f"  {k:48} {v['table']}  p={v['p_two_sided']:.3g}")
    json.dump(dict(groups=out, tests=tests, solve_threshold=SOLVE), open(os.path.join(HERE, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
