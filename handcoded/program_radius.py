"""Geometry around the two handcoded counting programs (K = 2, four letters).

    uv run python handcoded/program_radius.py exact   [--n 6 8 12 16 24 32] [--dirs 20]
        exactness radius in random directions: smallest r such that theta* + r xi changes a
        greedy continuation, xi with independent standard Gaussian entries on every weight.
    uv run python handcoded/program_radius.py attract [--n 6 8 12 16] [--dirs 6] [--steps 3000] [--lr 1e-4]
        attraction: train each loss from theta* + r xi and record exact continuation at the
        start and at the end (AdamW, the data and batches of the letter-task protocol).
    uv run python handcoded/program_radius.py slice   [--n 8]
        both losses on the plane through theta_0, theta*_S and theta*_O.
    uv run python handcoded/program_radius.py axis    [--n 8 16]
        each loss, and exactness, along the line from theta_0 through its own program.
Results go to results/radius/*.json.
"""
import argparse
import json
import os
import random
import statistics
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(__file__))
import lettertrace as L

OUT = "results/radius"
RADII = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def direction(p, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=g).to(v.device) for k, v in p.items()}


def moved(p, xi, r):
    return {k: p[k] + r * xi[k] for k in p}


def prompts(n, count=128):
    rng = random.Random(0)
    return [L.make_example(rng) for _ in range(count)]


def exact(ns, dirs):
    os.makedirs(OUT, exist_ok=True)
    out = []
    for n in ns:
        L.configure(task="count", word_len=n, mod=2)
        sols, ex = L.handcoded_params(), prompts(n)
        for mode in ("process", "outcome"):
            radii = []
            for s in range(dirs):
                xi = direction(sols[mode], 1000 + s)
                lo, hi = 1e-7, 10.0                        # exact at lo, broken at hi
                for _ in range(16):
                    mid = (lo * hi) ** 0.5
                    if L.free_run(moved(sols[mode], xi, mid), ex, mode)[0] == 1.0:
                        lo = mid
                    else:
                        hi = mid
                radii.append(hi)
            radii.sort()
            out.append(dict(n=n, mode=mode, radii=radii))
            print(f"n={n:3d} {mode:8s} median {statistics.median(radii):.2e}  min {radii[0]:.2e}  max {radii[-1]:.2e}"
                  f"   median*n {statistics.median(radii) * n:.3f}  median*n^2 {statistics.median(radii) * n * n:.2f}", flush=True)
            json.dump(out, open(f"{OUT}/exact.json", "w"))


def attract(ns, dirs, steps, lr, tag):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/attract{tag}.json"
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {(r["n"], r["mode"], r["r"], r["dir"]) for r in done}
    for n in ns:
        L.configure(task="count", word_len=n, mod=2, seed=42, steps=steps, eval_every=steps, lr=lr)
        train, shown, test = L.make_corpus()
        sols = L.handcoded_params()
        for mode in ("process", "outcome"):
            for r in RADII:
                for s in range(dirs):
                    if (n, mode, r, s) in have:
                        continue
                    start = moved(sols[mode], direction(sols[mode], 1000 + s), r)
                    e0 = L.free_run(start, test, mode)[0]
                    p, hist = L.train_mode(mode, start, train, shown, test)
                    e1 = hist[-1]["exact"]
                    done.append(dict(n=n, mode=mode, r=r, dir=s, steps=steps, lr=lr, start=e0, final=e1))
                    print(f"n={n:2d} {mode:8s} r={r:.0e} dir={s} exact {e0:.3f} -> {e1:.3f}", flush=True)
                    json.dump(done, open(path, "w"))


@torch.no_grad()
def slice_(n, grid, rows):
    """Plane theta_0 + a (theta*_S - theta_0) + b (theta*_O - theta_0), a, b in [-0.25, 1.25]."""
    os.makedirs(OUT, exist_ok=True)
    L.configure(task="count", word_len=n, mod=2, seed=42)
    train, shown, _ = L.make_corpus()
    theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
    sols = L.handcoded_params()
    data = {m: L.encode([L.row(e, m, s) for e, s in zip(train[:rows], shown[:rows])], m) for m in ("process", "outcome")}
    axis = [-0.25 + 1.5 * i / (grid - 1) for i in range(grid)]
    loss = {m: [[0.0] * grid for _ in range(grid)] for m in data}
    for i, a in enumerate(axis):
        for j, b in enumerate(axis):
            p = {k: theta_0[k] + a * (sols["process"][k] - theta_0[k]) + b * (sols["outcome"][k] - theta_0[k]) for k in theta_0}
            for m, (inputs, targets) in data.items():
                loss[m][i][j] = float(F.cross_entropy(L.forward(p, inputs).flatten(0, 1), targets.flatten(), ignore_index=-100))
    json.dump(dict(n=n, axis=axis, loss=loss), open(f"{OUT}/slice_n{n}.json", "w"))
    print(f"wrote {OUT}/slice_n{n}.json")


@torch.no_grad()
def axis(ns, rows=1024, points=301):
    """Each loss along the line from theta_0 through its own program, theta_0 + a (theta* - theta_0)."""
    os.makedirs(OUT, exist_ok=True)
    out = []
    for n in ns:
        L.configure(task="count", word_len=n, mod=2, seed=42)
        train, shown, test = L.make_corpus()
        theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
        sols = L.handcoded_params()
        alphas = [1.5 * i / (points - 1) for i in range(points)]
        for m in ("process", "outcome"):
            inputs, targets = L.encode([L.row(e, m, s) for e, s in zip(train[:rows], shown[:rows])], m)
            loss, ok = [], []
            for a in alphas:
                p = {k: theta_0[k] + a * (sols[m][k] - theta_0[k]) for k in theta_0}
                loss.append(float(F.cross_entropy(L.forward(p, inputs).flatten(0, 1), targets.flatten(), ignore_index=-100)))
                ok.append(L.free_run(p, test[:200], m)[0])
            good = [a for a, e in zip(alphas, ok) if e == 1.0]
            out.append(dict(n=n, mode=m, alphas=alphas, loss=loss, exact=ok))
            print(f"n={n} {m:8s} exact for alpha in [{min(good):.3f}, {max(good):.3f}] ({len(good)} of {points} points)", flush=True)
    json.dump(out, open(f"{OUT}/axis.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exact", "attract", "slice", "axis"])
    ap.add_argument("--n", type=int, nargs="+")
    ap.add_argument("--dirs", type=int)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.what == "exact":
        exact(a.n or [6, 8, 12, 16, 24, 32], a.dirs or 20)
    elif a.what == "attract":
        attract(a.n or [6, 8, 12, 16], a.dirs or 6, a.steps, a.lr, a.tag)
    elif a.what == "axis":
        axis(a.n or [8, 16])
    else:
        slice_((a.n or [8])[0], a.grid, 1024)
