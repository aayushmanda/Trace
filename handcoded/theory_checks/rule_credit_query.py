"""Rule credit with a fixed query against a uniformly random query.

Setting of regime_both.py ((A1)-(A4), width 16, MLP biases bounded away from zero), but with
|Sigma| = 4 uniform letters, so the match probability is p = 1/4 and lambda = 1/2 for K = 2. Exact
over all 4^n words, both start states and, for the random query, all four query letters. The
rule credit is || d/dtheta E_t E <z_t, e_{s_t} - Tbar^T e_{s_{t-1}}> || as in rule_credit.py.
With a fixed query the leading coefficient is of order 1/n; averaging over a uniform query
cancels it, leaving order 1/n^2.
    uv run python handcoded/rule_credit_query.py
"""
import itertools
import json
import math
import os

import torch

import regime_both as RB

A, K = 4, RB.K
RB.A = A                                             # token layout of theta_tilde: letters, values, SEP, ':', spare
SEP = A + K


def data(n, queries):
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    rows, prob = [], []
    for q in queries:
        for s0 in range(K):
            match = (words == q).long()
            states = (s0 + match.cumsum(1)) % K
            head = torch.cat([torch.full((len(words), 1), A + s0), torch.full((len(words), 1), q), words,
                              torch.full((len(words), 1), SEP)], 1)
            rows.append(torch.cat([head, A + states[:, : n - 1]], 1))
            prob.append(torch.full((len(words),), 1.0 / (len(words) * K * len(queries))))
            rows[-1] = (rows[-1], states[:, 1:])
    ids = torch.cat([r[0] for r in rows]).to(RB.dev)
    tgt = torch.cat([r[1] for r in rows]).to(RB.dev)
    return ids, tgt, torch.cat(prob).to(RB.dev)


def rule_credit(n, sigma, tilde, queries):
    ids, tgt, prob = data(n, queries)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    pm = 1 / A
    eye = torch.eye(K, device=RB.dev)
    Tbar = (1 - pm) * eye + pm * eye.roll(1, 1)
    prev = ids[:, n + 3:] - A
    rule = eye[tgt] - Tbar[prev]
    phi = 0.0
    for i in range(0, len(prob), 1 << 16):
        z = RB.residual(p, ids[i:i + (1 << 16)])[:, n + 3:] @ p["R"].T
        phi = phi + (prob[i:i + (1 << 16), None, None] * z * rule[i:i + (1 << 16)]).sum() / z.shape[1]
    g = torch.autograd.grad(phi, list(p.values()))
    return math.sqrt(sum(float((x ** 2).sum()) for x in g))


if __name__ == "__main__":
    res = []
    for seed in (0, 1):
        tilde = RB.theta_tilde(seed)
        for n in (4, 5, 6, 7, 8):
            row = {}
            for name, qs in (("fixed", [0]), ("random", list(range(A)))):
                r = {s: rule_credit(n, s, tilde, qs) for s in (0.01, 0.02)}
                row[name] = dict(R=r[0.02], expo=math.log(r[0.02] / r[0.01]) / math.log(2))
            res.append(dict(seed=seed, n=n, **row))
            f, rr = row["fixed"], row["random"]
            print(f"draw {seed} n={n}: fixed q  R={f['R']:.3e}  nR/s^6={n * f['R'] / 0.02 ** 6:.1f}  exp {f['expo']:.2f} | "
                  f"random q  R={rr['R']:.3e}  nR/s^6={n * rr['R'] / 0.02 ** 6:.2f}  n^2R/s^6={n * n * rr['R'] / 0.02 ** 6:.1f}"
                  f"  exp {rr['expo']:.2f}", flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/rule_credit_query.json", "w"))
