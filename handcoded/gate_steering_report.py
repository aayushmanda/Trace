"""Summarize results/gate_steering/*.json (written by gate_steering.py) into the tables of the appendix.

    python3 handcoded/gate_steering_report.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1] / (sys.argv[1] if len(sys.argv) > 1 else "results/gate_steering_gf")


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        rows += json.loads(Path(f).read_text())
    return rows


def fmt(x):
    return "--" if x is None else f"{x:.1e}"


def main_table(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["start"], r["rho"], r["n"])].append(r)
    print("start rho n | in G | median T_D | leak_max (median) | att cur/own | align | path barrier-free | final margin")
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        rs = cells[key]
        inG = sum(r["in_G"] for r in rs)
        td = [r["T_enter_G"] for r in rs if r["T_enter_G"] is not None]
        paths = [r["path_barrier_free"] for r in rs if "path_barrier_free" in r]
        print(f"{key[0]:12s} {key[1]:<6g} {key[2]:3d} | {inG}/{len(rs)} | {fmt(np.median(td)) if td else '--'} | "
              f"{np.median([r['leak_max'] for r in rs]):.1e} | {np.median([r['att_current'] for r in rs]):.3f}/"
              f"{np.median([r['att_own'] for r in rs]):.3f} | {np.median([r['align'] for r in rs]):.3f} | "
              f"{(str(sum(paths)) + '/' + str(len(paths))) if paths else '--'} | "
              f"{min(r['final_margin'] for r in rs):+.3f}..{max(r['final_margin'] for r in rs):+.3f}")


def outcome_table(rows):
    print("\noutcome start rho n | final acc (min..max) | loss - log2 | logit spread | dist/sigma")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["start"], r.get("rho", 0.0), r["n"])].append(r["log"][-1])
    for key in sorted(cells):
        ls = cells[key]
        print(f"{key[0]:15s} {key[1]:<4g} {key[2]:3d} | {min(l['acc'] for l in ls):.3f}..{max(l['acc'] for l in ls):.3f} | "
              f"{min(l['loss_minus_log2'] for l in ls):.1e} | {max(l['logit_spread'] for l in ls):.1e} | "
              f"{max(l['dist_over_sigma'] for l in ls):.2f}")


if __name__ == "__main__":
    main_table(load("main_n*.json"))
    print("\n== fine ladder (B, no path)")
    main_table(load("fine_n*.json"))
    print("\n== per-block nuisance (B)")
    for f in sorted(glob.glob(str(ROOT / "block_*_n*.json"))):
        print("--", Path(f).stem)
        main_table(json.loads(Path(f).read_text()))
    outcome_table(load("outcome_n*.json"))
