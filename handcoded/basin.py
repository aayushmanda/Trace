"""Basin of the outcome solution, measured three ways (paper, Section 5).

    uv run python handcoded/basin.py align   [--n 8 12 16 24] [--seeds 42 43 44]
        cosine between -grad L and (theta* - theta) along theta_a = theta_0 + a (theta* - theta_0),
        theta* the handcoded program of the same loss; also the loss along the path.
    uv run python handcoded/basin.py success [--n 6 8 10 12 14 16] [--seeds 20]
        fraction of random initializations whose free-running exact accuracy reaches 0.9
        within the budget, under both losses (same theta_0, data and batches per seed).
Results go to results/basin/*.json.
"""
import argparse
import json
import os
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(__file__))
import lettertrace as L

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
OUT = "results/basin"


def start_near(ns, seeds, alphas, steps, tag, lr=1e-3):
    """Basin along the path: train each loss from theta_a = theta_0 + a (theta* - theta_0)."""
    path = f"{OUT}/start{tag}.json"
    os.makedirs(OUT, exist_ok=True)
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {(r["n"], r["seed"], r["mode"], r["alpha"]) for r in done}
    for n in ns:
        for s in seeds:
            L.configure(task="count", word_len=n, mod=2, seed=s, steps=steps, eval_every=steps // 4, lr=lr)
            train, shown, test = L.make_corpus()
            theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
            hand = L.handcoded_params()
            for mode in ("process", "outcome"):
                for a in alphas:
                    if (n, s, mode, a) in have:
                        continue
                    start = {k: theta_0[k] + a * (hand[mode][k] - theta_0[k]) for k in theta_0}
                    exact0 = L.free_run(start, test, mode)[0]
                    p, hist = L.train_mode(mode, start, train, shown, test)
                    final = hist[-1]["exact"]
                    done.append(dict(n=n, seed=s, mode=mode, alpha=a, steps=steps, lr=lr,
                                     exact_at_start=exact0, final_exact=final, solved=final >= 0.9))
                    print(f"n={n:2d} seed={s} {mode:8s} alpha={a:.2f} start {exact0:.3f} final {final:.3f}", flush=True)
                    with open(path, "w") as f:
                        json.dump(done, f)


def alignment_along_path(mode, theta_0, train, shown, alphas=ALPHAS, rows=2048):
    hand = L.handcoded_params()[mode]
    inputs, targets = L.encode([L.row(e, mode, s) for e, s in zip(train[:rows], shown[:rows])], mode)
    cos, loss_path = [], []
    for a in alphas:
        p = {k: (theta_0[k] + a * (hand[k] - theta_0[k])).detach().requires_grad_() for k in hand}
        loss = F.cross_entropy(L.forward(p, inputs).flatten(0, 1), targets.flatten(), ignore_index=-100)
        grads = torch.autograd.grad(loss, list(p.values()))
        g = torch.cat([x.flatten() for x in grads])
        d = torch.cat([(hand[k] - p[k]).detach().flatten() for k in hand])
        cos.append(float(-(g @ d) / (g.norm() * d.norm() + 1e-30)))
        loss_path.append(float(loss))
    return cos, loss_path


def align(ns, seeds):
    out = []
    for n in ns:
        for s in seeds:
            L.configure(task="count", word_len=n, mod=2, seed=s)
            train, shown, _ = L.make_corpus()
            theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
            for mode in ("process", "outcome"):
                cos, lp = alignment_along_path(mode, theta_0, train, shown)
                out.append(dict(n=n, seed=s, mode=mode, alphas=ALPHAS, cos=cos, loss=lp))
                print(f"n={n:2d} seed={s} {mode:8s} cos " + " ".join(f"{c:+.3f}" for c in cos), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/align.json", "w") as f:
        json.dump(out, f)


def success(ns, n_seeds, steps, tag=""):
    path = f"{OUT}/success{tag}.json"
    os.makedirs(OUT, exist_ok=True)
    done = json.load(open(path)) if os.path.exists(path) else []
    if tag and os.path.exists(f"{OUT}/success.json"):           # reuse runs of the single-GPU sweep
        done += [r for r in json.load(open(f"{OUT}/success.json")) if r["n"] in ns and r not in done]
    have = {(r["n"], r["seed"], r["mode"]) for r in done}
    for n in ns:
        for s in range(42, 42 + n_seeds):
            if all((n, s, m) in have for m in ("process", "outcome")):
                continue
            L.configure(task="count", word_len=n, mod=2, seed=s, steps=steps, eval_every=steps // 4)
            train, shown, test = L.make_corpus()
            theta_0 = L.init_theta()
            for mode in ("process", "outcome"):
                p, hist = L.train_mode(mode, theta_0, train, shown, test)
                best = max(h["exact"] for h in hist if "exact" in h)
                final = hist[-1]["exact"]
                done.append(dict(n=n, seed=s, mode=mode, steps=steps, final_exact=final,
                                 best_exact=best, solved=final >= 0.9))
                print(f"n={n:2d} seed={s} {mode:8s} final exact {final:.3f}", flush=True)
            with open(path, "w") as f:
                json.dump(done, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["align", "success", "start"])
    ap.add_argument("--alphas", type=float, nargs="+")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    if a.what == "align":
        align(a.n or [8, 12, 16, 24], a.seeds or [42, 43, 44])
    elif a.what == "success":
        success(a.n or [6, 8, 10, 12, 14, 16], (a.seeds or [20])[0], a.steps, a.tag)
    else:
        start_near(a.n or [8, 16], a.seeds or [42, 43], a.alphas or [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                   a.steps, a.tag, a.lr)
