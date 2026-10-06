"""Landscape illustrations (appendix).

  init     log K - L on a plane through theta_0 inside the ball (regime_both setting, sigma = 0.02):
           the plane is spanned by the loss's own negative gradient and a random direction scaled
           per tensor to theta_0, both of length 0.1 sigma |theta~|; exact loss over all words.
  islands  exact decoding on a plane through each counting program (lettertrace, K = 2), spanned
           by a Gaussian direction xi and the key tilt of Theorem 1(iii) (entries +-1).
    uv run python landscape.py init|islands
"""
import json
import math
import os
import random
import sys

import torch

import regime_both as RB

OUT = "../results/landscape"
G = 21


def unit(d):
    nrm = math.sqrt(sum(float((v ** 2).sum()) for v in d.values()))
    return {k: v / nrm for k, v in d.items()}


@torch.no_grad()
def lossval(p, d, mode, n):
    return float(RB.loss(p, d, mode, n))


def init_slice():
    os.makedirs(OUT, exist_ok=True)
    tilde, sigma = RB.theta_tilde(0), 0.02
    rad = 0.1 * sigma * math.sqrt(sum(float((v ** 2).sum()) for v in tilde.values()))
    g = torch.Generator().manual_seed(7)
    res = []
    for n in (4, 8, 12):
        d = RB.data(n)
        for mode in ("outcome", "step"):
            p0 = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
            grads = torch.autograd.grad(RB.loss(p0, d, mode, n), list(p0.values()))
            d1 = unit({k: -gg for k, gg in zip(p0, grads)})
            d2 = {k: torch.randn(v.shape, generator=g).to(v) for k, v in p0.items()}
            d2 = unit({k: v * p0[k].detach().norm() / (v.norm() + 1e-30) for k, v in d2.items()})
            axis = [-1 + 2 * i / (G - 1) for i in range(G)]
            Z = [[(math.log(RB.K) - lossval({k: p0[k].detach() + rad * (a * d1[k] + b * d2[k]) for k in p0}, d, mode, n)) / sigma ** 2
                  for b in axis] for a in axis]
            res.append(dict(n=n, mode=mode, axis=axis, Z=Z))
            flat = [v for r in Z for v in r]
            print(f"init n={n:2d} {mode:8s} (log K - L)/sigma^2 on the slice: max {max(flat):.3e}  min {min(flat):.3e}", flush=True)
            json.dump(res, open(f"{OUT}/init.json", "w"))


def islands():
    import lettertrace as L
    os.makedirs(OUT, exist_ok=True)
    res = []
    for n in (8, 16):
        L.configure(task="count", word_len=n, mod=2)
        rng = random.Random(0)
        ex = [L.make_example(rng) for _ in range(96)]
        for q in range(L.A):
            word = (q,) * (n // 2) + ((q + 1) % L.A,) * (n - n // 2)
            for s0 in range(L.V):
                st, s = [], s0
                for t, c in enumerate(word, 1):
                    s = L.step(s, t, c, q); st.append(s)
                ex.append((s0, q, word, st))
        sols = L.handcoded_params()
        g = torch.Generator().manual_seed(1000)
        xi = {k: torch.randn(v.shape, generator=g).to(v) for k, v in sols["process"].items()}
        tilt = {k: torch.zeros_like(v) for k, v in sols["process"].items()}
        sign = torch.full((L.P,), -1.0); sign[2:2 + n // 2] = 1.0
        tilt["1.k"][0][:, L.POS] = sign[None, :].to(tilt["1.k"])
        r = 0.05
        axis = [-r + 2 * r * i / 40 for i in range(41)]
        for mode in ("process", "outcome"):
            p = sols[mode]
            E = [[L.free_run({k: p[k] + a * xi[k] + b * tilt[k] for k in p}, ex, mode)[0] for b in axis] for a in axis]
            res.append(dict(n=n, mode=mode, axis=axis, exact=E))
            frac = sum(e == 1.0 for row in E for e in row) / len(axis) ** 2
            print(f"islands n={n} {mode:8s} exact fraction of the plane [-{r},{r}]^2: {frac:.3f}", flush=True)
            json.dump(res, open(f"{OUT}/islands.json", "w"))


if __name__ == "__main__":
    init_slice() if sys.argv[1] == "init" else islands()
