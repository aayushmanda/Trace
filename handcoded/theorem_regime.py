"""Outcome training inside the regime of Proposition 3 / Theorem 2.

Two blocks at fixed widths (the same for every n), K value logits read at the ':' position,
theta_0 = sigma * theta_tilde with every MLP bias |b~| >= 1, fixed start state and query,
counting mod K over 4 letters, full-batch gradient descent on the exact outcome loss over all
4^n words (small steps: each moves the parameters by at most 1e-3 of sigma * ||theta~||).

Reports, per n:
  T_exit(eps)  first gradient-flow time with ||theta_t - theta_0|| = eps * sigma * ||theta~||
  the product T_exit * (sigma^2 + sqrt(A_<=1)), which Theorem 2 bounds below by a constant
  log K - L_O at exit, in units of sigma^2
  whether any MLP unit's sign depends on the word, at start and at exit

    uv run python handcoded/theorem_regime.py [K] [sigma]
"""
import itertools
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, Wd, Hh, Dh, F = 4, 16, 3, 8, 16


def setup(n, K, seed=0):
    vocab, SEP, COLON = A + K + 3, A + K, A + K + 1
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    Tmax = 16                                              # same tensors for every n <= 12
    th = {"wte": rn(vocab, Wd), "wpe": rn(Tmax, Wd), "R": rn(K, Wd)}
    for b in range(2):
        th[f"{b}.q"], th[f"{b}.k"], th[f"{b}.v"] = rn(Hh, Dh, Wd), rn(Hh, Dh, Wd), rn(Hh, Wd, Wd)
        th[f"{b}.mi"], th[f"{b}.mo"] = rn(F, Wd), rn(Wd, F)
        mb = rn(F)
        th[f"{b}.mb"] = torch.sign(mb) * (1 + mb.abs())
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    m = len(words)
    ids = torch.cat([torch.full((m, 1), A + 0), torch.zeros(m, 1, dtype=torch.long), words,
                     torch.full((m, 1), SEP), torch.full((m, 1), COLON)], 1)
    tgt = (words == 0).sum(1) % K                          # start state 0, query letter 0
    return {k: v.to(dev) for k, v in th.items()}, ids.to(dev), tgt.to(dev)


def forward(p, ids, want_signs=False):
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    switching = 0
    for b in range(2):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(Dh)).masked_fill(future, -math.inf)
        x = x + (att.softmax(-1) @ v).sum(1)
        pre = x @ p[f"{b}.mi"].T + p[f"{b}.mb"]
        if want_signs:
            on = pre > 0
            switching += int((on.any(0) & ~on.all(0)).any(0).sum())
        x = x + torch.relu(pre) @ p[f"{b}.mo"].T
    z = x[:, -1] @ p["R"].T
    return (z, switching) if want_signs else z


def loss(p, ids, tgt, chunk=1 << 16):
    tot = 0.0
    for i in range(0, len(ids), chunk):
        tot = tot + torch.nn.functional.cross_entropy(forward(p, ids[i:i + chunk]), tgt[i:i + chunk], reduction="sum")
    return tot / len(ids)


def switching_units(p, ids, chunk=1 << 16):
    with torch.no_grad():
        return max(forward(p, ids[i:i + chunk], True)[1] for i in range(0, len(ids), chunk))


def low_degree_mass(n, K, pm=0.25):
    lam = [abs(1 - pm + pm * complex(math.cos(2 * math.pi * j / K), math.sin(2 * math.pi * j / K))) for j in range(1, K)]
    return sum(l ** (2 * n) + n * l ** (2 * (n - 1)) * (1 - l * l) for l in lam) / K


def run(n, K, sigma, eps_list=(0.05, 0.1, 0.2), max_steps=400_000):
    tilde, ids, tgt = setup(n, K)
    norm0 = math.sqrt(sum(float((v ** 2).sum()) for v in tilde.values()))
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    theta0 = {k: v.detach().clone() for k, v in p.items()}
    A1 = low_degree_mass(n, K)
    rate = sigma ** 2 + math.sqrt(A1)
    out = dict(n=n, K=K, sigma=sigma, sqrtA=math.sqrt(A1), switch_start=switching_units(p, ids), exits=[])
    t, pending = 0.0, list(eps_list)
    for step in range(max_steps):
        L = loss(p, ids, tgt)
        grads = torch.autograd.grad(L, list(p.values()))
        gn = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
        dt = 1e-3 * sigma * norm0 / gn                     # each step moves 1e-3 of the ball scale
        with torch.no_grad():
            for v, g_ in zip(p.values(), grads):
                v -= dt * g_
        t += dt
        dist = math.sqrt(sum(float(((p[k] - theta0[k]) ** 2).sum()) for k in p)) / (sigma * norm0)
        if step == 0:
            out["grad0_over_sigma"] = gn / sigma
        while pending and dist >= pending[0]:
            e = pending.pop(0)
            gap = math.log(K) - float(loss(p, ids, tgt))
            out["exits"].append(dict(eps=e, T=t, T_times_rate=t * rate, gap_over_sigma2=gap / sigma ** 2,
                                     switching=switching_units(p, ids), steps=step + 1))
            print(f"  n={n} eps={e}: T_exit={t:.2f}  T*(sigma^2+sqrtA)={t * rate:.3f}  "
                  f"(logK-L)/sigma^2={gap / sigma ** 2:+.3f}  switching units={out['exits'][-1]['switching']}", flush=True)
        if not pending:
            break
    return out


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    res = []
    for n in (4, 6, 8, 10):
        print(f"n={n} K={K} sigma={sigma}  sqrt(A_<=1)={math.sqrt(low_degree_mass(n, K)):.3e}", flush=True)
        res.append(run(n, K, sigma))
        print(f"  switching units at start: {res[-1]['switch_start']}  ||grad L_O(theta_0)||/sigma={res[-1]['grad0_over_sigma']:.3e}", flush=True)
        os.makedirs("results/theorem_regime", exist_ok=True)
        json.dump(res, open(f"results/theorem_regime/K{K}_sigma{sigma}.json", "w"))
