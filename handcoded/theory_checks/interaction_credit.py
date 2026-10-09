"""Outcome credit by degree against step rule credit, at initialization.

Setting of regime_both.py ((A1)-(A4), exact over all 2^n words and both start states). For each
start state the answer target y - u is split exactly into Efron-Stein components over the n
letters (independent, query letter with probability PM), by a degree-tracking pass over the
letter axes. Reports, at theta_0 = sigma * theta~,
    outcome credit at degree r   || d/dtheta E <z, (y - u)_r> ||   (r = 0, 1, 2, 3)
    outcome interaction credit   || d/dtheta E <z, (y - u)_{>=2}> ||
    step rule credit             as in rule_credit.py
and their exponents in sigma. The matched comparison of the outcome interaction credit with the
rule credit is the one the two-site argument predicts: both need one attention score.
    uv run python handcoded/theory_checks/interaction_credit.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # handcoded/
import json
import math
import os

import torch

import regime_both as RB
import rule_credit as RC

K, PM, A = RB.K, RB.PM, RB.A


def degree_parts(f, n):
    """f: (K * 2^n, K) target rows in data() order. Returns list over r of the degree-r part."""
    w = torch.tensor([PM, 1 - PM], device=f.device)                  # letter 0 is the query letter
    g = f.view(K, *([2] * n), K)                                       # start state, letters, logit
    parts = [g]                                                        # parts[r]: degree exactly r so far
    for t in range(n):
        ax = 1 + t
        new = [None] * (len(parts) + 1)
        for r, h in enumerate(parts):
            mean = (h * w.view(*([1] * ax), 2, *([1] * (n - t)))).sum(ax, keepdim=True).expand_as(h)
            new[r] = mean if new[r] is None else new[r] + mean
            new[r + 1] = h - mean
        parts = new
    return [p.reshape(-1, K) for p in parts]


def credits(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye = torch.eye(K, device=RB.dev)
    parts = degree_parts(eye[d["out_tgt"]] - 1 / K, n)
    zo = RB.residual(p, d["out_ids"])[:, -1] @ p["R"].T
    w = d["prob"][:, None]
    targets = {f"r{r}": parts[r] for r in range(min(4, n + 1))}
    targets["ge2"] = sum(parts[2:])
    targets["all"] = sum(parts)
    out = {}
    for name, tgt in targets.items():
        g = torch.autograd.grad((w * zo * tgt).sum(), list(p.values()), retain_graph=True)
        out[name] = math.sqrt(sum(float((x ** 2).sum()) for x in g))
    out["rule"] = RC.credits(n, sigma, tilde)["rule"]
    return out


if __name__ == "__main__":
    lam, res = 1 - 2 * PM, []
    for seed in (0, 1, 2):
        tilde = RB.theta_tilde(seed)
        for n in (4, 6, 8, 10, 12, 14):
            row = {s: credits(n, s, tilde) for s in (0.005, 0.02)}
            expo = {k: math.log(row[0.02][k] / row[0.005][k]) / math.log(4) for k in row[0.02]}
            res.append(dict(seed=seed, n=n, credits=row, exponent=expo))
            c = row[0.02]
            print(f"draw {seed} n={n:2d} sigma=0.02: outcome all {c['all']:.2e} r0 {c['r0']:.2e} r1 {c['r1']:.2e}"
                  f" r2 {c['r2']:.2e} r3 {c['r3']:.2e} >=2 {c['ge2']:.2e} | rule {c['rule']:.2e}"
                  f" | rule/(>=2) {c['rule'] / c['ge2']:.3g}"
                  f" | (>=2)/(sigma^6 lam^(n-2)) {c['ge2'] / (0.02 ** 6 * lam ** (n - 2)):.3g}"
                  f" | exps r2 {expo['r2']:.2f} r3 {expo['r3']:.2f} >=2 {expo['ge2']:.2f} rule {expo['rule']:.2f}",
                  flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/interaction_credit.json", "w"))
