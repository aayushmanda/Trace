"""Credit at initialization for the rule part of the step target, against the outcome credit.

Setting of regime_both.py ((A1)-(A4), exact over all words and both start states). The step
target splits as  e_{s_t} - u = [Tbar^T e_{s_{t-1}} - u]  +  [e_{s_t} - Tbar^T e_{s_{t-1}}]:
the prior of s_t given the displayed state, and the rule part, a pure interaction between the
displayed state and the current move. Reports the norms of
    prior credit   || d/dtheta E <z_i, Tbar^T e_{s_{t-1}} - u> ||
    rule credit    || d/dtheta E <z_i, e_{s_t} - Tbar^T e_{s_{t-1}}> ||
    outcome credit || d/dtheta E <z, e_y - u> ||
and their exponents in sigma.
    uv run python handcoded/rule_credit.py
"""
import json
import math
import os

import torch

import regime_both as RB

K, PM = RB.K, RB.PM


def credits(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye = torch.eye(K, device=RB.dev)
    Tbar = (1 - PM) * eye + PM * eye.roll(1, 1)                      # row s -> distribution of s + b
    prev = d["step_ids"][:, n + 3:] - RB.A                            # displayed s_{t-1}, t = 2..n
    z = RB.residual(p, d["step_ids"])[:, n + 3:] @ p["R"].T
    prior_t, full_t = Tbar[prev] - 1 / K, eye[d["step_tgt"]] - 1 / K
    w = d["prob"][:, None, None]
    out = {}
    for name, tgt in (("prior", prior_t), ("rule", full_t - prior_t)):
        phi = (w * z * tgt).sum() / z.shape[1]
        g = torch.autograd.grad(phi, list(p.values()), retain_graph=True)
        out[name] = math.sqrt(sum(float((x ** 2).sum()) for x in g))
    zo = RB.residual(p, d["out_ids"])[:, -1] @ p["R"].T
    phi = (d["prob"][:, None] * zo * (eye[d["out_tgt"]] - 1 / K)).sum()
    g = torch.autograd.grad(phi, list(p.values()))
    out["outcome"] = math.sqrt(sum(float((x ** 2).sum()) for x in g))
    return out


if __name__ == "__main__":
    res = []
    for seed in (0, 1, 2):
        tilde = RB.theta_tilde(seed)
        for n in (4, 6, 8, 10, 12, 14):
            row = {s: credits(n, s, tilde) for s in (0.005, 0.01, 0.02)}
            expo = {k: math.log(row[0.02][k] / row[0.005][k]) / math.log(4) for k in ("prior", "rule", "outcome")}
            res.append(dict(seed=seed, n=n, credits=row, exponent=expo))
            c = row[0.02]
            print(f"draw {seed} n={n:2d} sigma=0.02: prior {c['prior']:.3e}  rule {c['rule']:.3e}  outcome {c['outcome']:.3e}"
                  f"   rule*n/sigma^6 {c['rule'] * n / 0.02 ** 6:.3g}"
                  f"   sigma-exponents prior {expo['prior']:.2f} rule {expo['rule']:.2f} outcome {expo['outcome']:.2f}", flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/rule_credit.json", "w"))
