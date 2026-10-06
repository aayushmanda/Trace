"""Diagnostic: do AdamW updates point toward the constructed program of the trained loss?

Counting, K = 2, the letter-task protocol (AdamW, lr 1e-3, batch 256, sigma = 0.02). For each
loss and its own handcoded program theta*, over the first updates:
    |Delta_t|                    size of the update
    cos(Delta_t, theta* - theta_t)
    |theta_t - theta*|
and, at the same parameters, the cosine between theta* - theta_t and the credit gradients
(derivative of E<z, target part>): answer credit for outcome; prior and rule parts for step.
    uv run python handcoded/update_alignment.py [--n 8 16] [--seeds 42 43 44] [--steps 400]
"""
import argparse
import json
import math
import os
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(__file__))
import lettertrace as L

AT = [1, 2, 5, 10, 20, 50, 100, 200, 400]


def flat(p):
    return torch.cat([p[k].flatten() for k in sorted(p)])


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def credit_cosines(p, mode, inputs, targets, d):
    n, A, V = L.n, L.A, L.V
    z = L.forward(p, inputs)
    eye = torch.eye(L.VOCAB, device=z.device)
    out = {}
    if mode == "outcome":
        pos = n + 3                                           # ':' predicts the answer
        tgt = eye[targets[:, pos]].clone()
        tgt[:, A:A + V] -= 1 / V
        parts = {"answer": (z[:, pos] * tgt).sum(1).mean()}
    else:
        pos = torch.arange(n + 3, 2 * n + 2, device=z.device)  # hold s_1..s_{n-1}, predict s_2..s_n
        prev, nxt = inputs[:, pos] - A, targets[:, pos] - A
        prior = torch.zeros(*prev.shape, L.VOCAB, device=z.device)
        prior.scatter_(2, (A + prev)[..., None], 1 - 1 / A)
        prior.scatter_add_(2, (A + (prev + 1) % V)[..., None], torch.full((*prev.shape, 1), 1 / A, device=z.device))
        full = eye[A + nxt]
        unif = torch.zeros(L.VOCAB, device=z.device); unif[A:A + V] = 1 / V
        parts = {"prior": (z[:, pos] * (prior - unif)).sum(2).mean(), "rule": (z[:, pos] * (full - prior)).sum(2).mean()}
    keys = sorted(p)
    for name, phi in parts.items():
        g = torch.autograd.grad(phi, [p[k] for k in keys], retain_graph=True)
        out[name] = cos(torch.cat([x.flatten() for x in g]), d)
    return out


def run(n, seed, steps):
    L.configure(task="count", word_len=n, mod=2, seed=seed)
    train, shown, _ = L.make_corpus()
    theta_0, hand = L.init_theta(), L.handcoded_params()
    rows = []
    for mode in ("process", "outcome"):
        p = {k: t.clone().to(L.device).requires_grad_() for k, t in theta_0.items()}
        star = flat(hand[mode])
        opt = torch.optim.AdamW(p.values(), lr=L.lr, weight_decay=0.0)
        inputs, targets = L.encode([L.row(e, mode, s) for e, s in zip(train, shown)], mode)
        gen = torch.Generator().manual_seed(seed)
        for it in range(1, steps + 1):
            idx = torch.randint(len(train), (L.batch_size,), generator=gen).to(L.device)
            before = flat(p).detach().clone()
            if it in AT:
                credits = credit_cosines(p, mode, inputs[idx], targets[idx], star - before)
            loss = F.cross_entropy(L.forward(p, inputs[idx]).flatten(0, 1), targets[idx].flatten(), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = flat({k: v.grad for k, v in p.items()}).clone()
            torch.nn.utils.clip_grad_norm_(p.values(), 1.0)
            opt.step()
            if it in AT:
                delta, d = flat(p).detach() - before, star - before
                rows.append(dict(n=n, seed=seed, mode=mode, step=it, loss=float(loss), update_norm=float(delta.norm()),
                                 cos_update=cos(delta, d), cos_neg_grad=cos(-grad, d), dist=float(d.norm()), credit_cos=credits))
                c = "  ".join(f"{k} {v:+.4f}" for k, v in credits.items())
                print(f"n={n} seed={seed} {mode:8s} step {it:3d}  |Delta|={float(delta.norm()):.3f}  cos(Delta,d)={cos(delta, d):+.4f}"
                      f"  cos(-grad,d)={cos(-grad, d):+.4f}  |theta-theta*|={float(d.norm()):.1f}  credit cos: {c}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    out = []
    os.makedirs("results/radius", exist_ok=True)
    for n in a.n:
        for s in a.seeds:
            out += run(n, s, a.steps)
            json.dump(out, open("results/radius/update_alignment.json", "w"))
