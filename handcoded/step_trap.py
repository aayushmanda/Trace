"""Sublevel-set trapping around the step executor (Theorem 2(iv)), checked on all prompts.

L_S of (A4): cross-entropy over the K value logits of s_2..s_n under teacher forcing, averaged
over all N = K |Sigma|^(n+1) prompts (start, query, word) and the T = n - 1 positions. If
L_S(theta) < log 2 / (N T), every position decodes correctly, and under gradient flow this stays
true forever. For the executor with readout scale a (value rows 20 a) and random-sign
perturbations of entrywise size x / n, report L_S * N T / log 2 (below 1: inside the trap) and
whether small gradient steps (300; 60 at n = 8; step size 1e-2 / a^2) keep every position correct and the loss decreasing.
    uv run python step_trap.py
"""
import itertools
import math

import torch
from torch.nn import functional as F

import lettertrace as L

torch.set_default_dtype(torch.float64)


def all_prompts(n):
    A, K = L.A, L.V
    rows, tgts = [], []
    for s0 in range(K):
        for q in range(A):
            for word in itertools.product(range(A), repeat=n):
                s, states = s0, []
                for c in word:
                    s = (s + (c == q)) % K
                    states.append(s)
                rows.append([A + s0, q, *word, L.SEP] + [A + v for v in states[:-1]])
                tgts.append(states[1:])
    return torch.tensor(rows, device=L.device), torch.tensor(tgts, device=L.device)


def step_loss(p, ids, tgt, n, chunk=1 << 15):
    tot, ok = 0.0, True
    for i in range(0, len(ids), chunk):
        z = L.forward(p, ids[i:i + chunk])[:, n + 3:2 * n + 2, L.A:L.A + L.V]
        tot = tot + F.cross_entropy(z.flatten(0, 1), tgt[i:i + chunk].flatten(), reduction="sum")
        ok = ok and bool((z.argmax(-1) == tgt[i:i + chunk]).all())
    return tot / tgt.numel(), ok


def signs(p, eps, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: v + eps * (2 * torch.randint(0, 2, v.shape, generator=g) - 1).to(v) for k, v in p.items()}


if __name__ == "__main__":
    for n in (4, 6):                     # n = 8 (524k prompts) takes hours
        L.configure(task="count", word_len=n, mod=2)
        star = {k: v.double() for k, v in L.handcoded_params()["process"].items()}
        ids, tgt = all_prompts(n)
        NT = len(ids) * (n - 1)
        for a in (1, 2, 3):
            base = dict(star); base["readout"] = star["readout"].clone()
            base["readout"][L.A:L.A + L.V] *= a
            for x in (0.0, 0.01, 0.02, 0.05):
                for draw in range(3 if x else 1):
                    p = {k: v.clone().requires_grad_() for k, v in signs(base, x / n, draw).items()}
                    l0, ok0 = step_loss(p, ids, tgt, n)
                    ratio = float(l0.detach()) * NT / math.log(2)
                    mono, ok_all, prev = True, bool(ok0), float(l0)
                    for _ in range(300 if n < 8 else 60):
                        l, ok = step_loss(p, ids, tgt, n)
                        g = torch.autograd.grad(l, list(p.values()))
                        with torch.no_grad():
                            for v, gg in zip(p.values(), g):
                                v -= 1e-2 / a ** 2 * gg
                        mono &= float(l) <= prev + 1e-15; prev = float(l); ok_all &= bool(ok)
                    print(f"n={n} a={a} x={x} draw={draw}: L_S*NT/log2 = {ratio:.3e}  correct at start {bool(ok0)}"
                          f"  300 steps: always correct {ok_all}, loss monotone {mono}, final L_S {prev:.2e}", flush=True)
