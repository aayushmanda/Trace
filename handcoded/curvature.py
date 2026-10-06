"""Curvature that carries answer information, at initialization (setting of regime_both.py).

Phi(theta) = E <z_theta, e_y - u> is the answer-dependent part of -L_O. Reports the largest and
smallest eigenvalues of d^2 Phi / d theta^2 at theta_0 (Lanczos on Hessian-vector products), and
the largest eigenvalue of d^2 L_S / d theta^2 for comparison.
    uv run python curvature.py
"""
import json
import math
import os

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

import regime_both as RB

K = RB.K


def phi_out(p, d):
    eye = torch.eye(K, device=RB.dev)
    z = RB.residual(p, d["out_ids"])[:, -1] @ p["R"].T
    return (d["prob"][:, None] * z * (eye[d["out_tgt"]] - 1 / K)).sum()


def extremes(f, p):
    keys = sorted(p)
    shapes = [p[k].shape for k in keys]
    sizes = [p[k].numel() for k in keys]
    flat0 = torch.cat([p[k].detach().flatten() for k in keys]).requires_grad_()

    def unflat(v):
        out, i = {}, 0
        for k, s, m in zip(keys, shapes, sizes):
            out[k] = v[i:i + m].view(s); i += m
        return out

    g = torch.autograd.grad(f(unflat(flat0)), flat0, create_graph=True)[0]

    def hvp(v):
        v = torch.as_tensor(np.asarray(v).ravel(), device=RB.dev)
        return torch.autograd.grad(g, flat0, v, retain_graph=True)[0].detach().cpu().numpy()

    op = LinearOperator((flat0.numel(), flat0.numel()), matvec=hvp, dtype=np.float64)
    top = eigsh(op, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    bot = eigsh(op, k=1, which="SA", tol=1e-6, return_eigenvectors=False)[0]
    return float(top), float(bot)


if __name__ == "__main__":
    res = []
    for seed in (0, 1):
        tilde = RB.theta_tilde(seed)
        for sigma in (0.02, 0.005):
            for n in (4, 6, 8, 10, 12, 14):
                d = RB.data(n)
                p = {k: (sigma * v) for k, v in tilde.items()}
                pt, pb = extremes(lambda q: phi_out(q, d), p)
                st = extremes(lambda q: RB.loss(q, d, "step", n), p)[0] if n <= 8 else float("nan")
                delta = 0.5 ** n + sigma * 0.5 ** (n - 1)
                res.append(dict(seed=seed, sigma=sigma, n=n, phi_top=pt, phi_bot=pb, step_top=st))
                print(f"draw {seed} sigma={sigma} n={n:2d}  Hess Phi: top {pt:+.3e} bottom {pb:+.3e}  "
                      f"max|eig|/delta_n {max(abs(pt), abs(pb)) / delta:.3f}   Hess L_S top {st:.4f}", flush=True)
                os.makedirs("../results/theorem_regime", exist_ok=True)
                json.dump(res, open("../results/theorem_regime/curvature.json", "w"))
