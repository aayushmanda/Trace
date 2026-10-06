"""Does the step-loss gradient move the rule part? Alignment of -dL_S with dPsi_R at theta_0.

Setting of rule_credit.py (regime_both.py: (A1)-(A4), fixed query, two letters, p = 1/4, exact).
Psi_R = E_t E <z_t, e_{s_t} - Tbar^T e_{s_{t-1}}> (rule), Psi_P = E_t E <z_t, Tbar^T e_{s_{t-1}} - u> (prior).
Reports, over all parameters and per block group,
    ||grad Psi_R||^2,  <grad Psi_R, grad Psi_P>,  <grad Psi_R, -grad L_S>
and the change of Psi_R along a few full-batch GD steps on L_S, against eta j sigma^12 / n^2.
    uv run python handcoded/rule_alignment.py
"""
import math

import torch

import regime_both as RB

K, PM = RB.K, RB.PM
GROUPS = {"readout": ("R",), "embed": ("wte", "wpe"),
          "score": tuple(f"{b}.{x}" for b in range(2) for x in "qk"), "value": ("0.v", "1.v"),
          "mlp": tuple(f"{b}.{x}" for b in range(2) for x in ("mi", "mo", "mb"))}


def parts(p, d, n):
    eye = torch.eye(K, device=RB.dev)
    Tbar = (1 - PM) * eye + PM * eye.roll(1, 1)
    prev = d["step_ids"][:, n + 3:] - RB.A
    z = RB.residual(p, d["step_ids"])[:, n + 3:] @ p["R"].T
    w = d["prob"][:, None, None]
    full = eye[d["step_tgt"]] - 1 / K
    prior = Tbar[prev] - 1 / K
    psi_r = (w * z * (full - prior)).sum() / z.shape[1]
    psi_p = (w * z * prior).sum() / z.shape[1]
    return psi_r, psi_p


def grads(f, p):
    return dict(zip(p, torch.autograd.grad(f, list(p.values()), retain_graph=True)))


def dot(a, b, keys=None):
    keys = keys or a.keys()
    return sum(float((a[k] * b[k]).sum()) for k in keys)


if __name__ == "__main__":
    tilde = RB.theta_tilde(0)
    for sigma in (0.02, 0.01):
        for n in (4, 8, 12):
            d = RB.data(n)
            p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
            psi_r, psi_p = parts(p, d, n)
            gR, gP = grads(psi_r, p), grads(psi_p, p)
            gL = grads(RB.loss(p, d, "step", n), p)
            mL = {k: -v for k, v in gL.items()}
            line = [f"sigma={sigma} n={n:2d}  ||gR||^2={dot(gR, gR):.2e}"]
            for name, keys in GROUPS.items():
                keys = [k for k in keys if k in p]
                line.append(f"{name}: |gR|^2={dot(gR, gR, keys):.1e} <gR,gP>={dot(gR, gP, keys):+.1e} <gR,-gL>={dot(gR, mL, keys):+.1e}")
            print("  ".join(line[:1]), f"<gR,gP>={dot(gR, gP):+.2e}  <gR,-gL>={dot(gR, mL):+.2e}", flush=True)
            for l in line[1:]:
                print("     ", l, flush=True)
            # a few GD steps on L_S
            eta, q = 0.5, {k: v.detach().clone().requires_grad_() for k, v in p.items()}
            r0 = float(psi_r)
            for j in range(1, 6):
                g = grads(RB.loss(q, d, "step", n), q)
                with torch.no_grad():
                    for k in q:
                        q[k] -= eta * g[k]
                r, _ = parts(q, d, n)
                print(f"      GD step {j}: dPsi_R={float(r) - r0:+.3e}   eta*j*||gR||^2={eta * j * dot(gR, gR):.3e}", flush=True)
