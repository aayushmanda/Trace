"""Worst-case weight perturbations of the two handcoded counting programs.

Tests whether the exactness radius (largest eps such that every theta with
|theta - theta*|_inf <= eps keeps the greedy continuation) is n-independent for the step
program and of order 1/n for the outcome program. Three explicit perturbations, each of
entrywise size eps; the smallest eps that breaks a program is found by bisection.
  match   : block-1 MLP output weights onto `match` lowered by eps (one entry per letter)
  fallback: every position embedding raised by eps on all `pos` coordinates
  keys    : block-2 key matrix of head 1 shifted by +-eps so the first half of the word
            gets more attention than the second half
    uv run python handcoded/exactness_radius.py
"""
import random
import sys

import torch

sys.path.insert(0, "handcoded")
import lettertrace as L

torch.set_num_threads(1)


def examples(n):
    rng = random.Random(0)
    ex = [L.make_example(rng) for _ in range(16)]
    for q in range(L.A):                                    # all-match and half-match words
        for word in ((q,) * n, (q,) * (n // 2) + ((q + 1) % L.A,) * (n - n // 2)):
            for s0 in range(L.mod):
                states, s = [], s0
                for t, c in enumerate(word, 1):
                    s = L.step(s, t, c, q); states.append(s)
                ex.append((s0, q, word, states))
    return ex


def perturbed(p, kind, eps):
    p = {k: v.clone() for k, v in p.items()}
    n = L.n
    if kind == "match":
        p["0.mlp_out"][L.at(L.MATCH), : L.A] -= eps
    elif kind == "fallback":
        p["wpe"][:, L.POS] += eps
    elif kind == "keys":
        sign = torch.full((L.P,), -1.0); sign[2 : 2 + n // 2] = 1.0
        p["1.k"][0][:, L.POS] += eps * sign[None, :].to(p["1.k"].device)
    return p


def radius(p, mode, kind, ex):
    lo, hi = 0.0, 1.0
    if L.free_run(perturbed(p, kind, hi), ex, mode)[0] == 1.0:
        return float("inf")
    for _ in range(14):
        mid = (lo + hi) / 2
        if L.free_run(perturbed(p, kind, mid), ex, mode)[0] == 1.0:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__" and len(sys.argv) == 1:
    print("smallest breaking eps, and eps*n")
    for n in (8, 16, 32):
        L.configure(task="count", word_len=n, mod=2, device="cpu")
        sols, ex = L.handcoded_params(), examples(n)
        for mode, name in (("process", "step   "), ("outcome", "outcome")):
            assert L.free_run(sols[mode], ex, mode)[0] == 1.0
            out = []
            for kind in ("match", "fallback", "keys"):
                r = radius(sols[mode], mode, kind, ex)
                out.append(f"{kind} {r:.2e} (x n = {r * n:.3f}, x n^2 = {r * n * n:.2f})")
            print(f"n={n:3d} {name}: " + "   ".join(out), flush=True)


def random_signs(p, eps, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: v + eps * (2 * torch.randint(0, 2, v.shape, generator=g) - 1).to(v) for k, v in p.items()}


def sign_test(draws=30):
    """Step program under dense random-sign perturbations of entrywise size x / n."""
    print("step program, random signs on every entry: fraction of draws that stay exact")
    for n in (8, 16, 32):
        L.configure(task="count", word_len=n, mod=2, device="cpu")
        p, ex = L.handcoded_params()["process"], examples(n)
        row = []
        for x in (0.05, 0.1, 0.2, 0.4, 0.8):
            ok = sum(L.free_run(random_signs(p, x / n, s), ex, "process")[0] == 1.0 for s in range(draws))
            row.append(f"{x}/n: {ok}/{draws}")
        print(f"n={n:3d}  " + "   ".join(row), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "signs":
    sign_test()
