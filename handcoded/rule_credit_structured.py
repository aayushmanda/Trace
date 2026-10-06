"""Step 4 of the proof of the rule-credit lower bound: one explicit theta_tilde where the
sigma^6 coefficient of the rule credit is nonzero, checked against its closed form.

theta_tilde has one block-2 head and nothing else: the query reads chi(s) = (-1)^s from the
displayed state, the key and the value read phi = 1[letter = query letter]. Then at the
position predicting s_t (m = n + 2 + t positions in view)
    d/dR_{k,l} E <z, rule target>  =  sigma^6 / sqrt(d_head) * w_k * p (1 - p) / m * (1 - (3 + 2 (n - 1) p) / m) + O(sigma^10),
with w_k = (chi(k - 1) - chi(k)) / K, averaged over t = 2..n.
    uv run python rule_credit_structured.py
"""
import math

import torch

import regime_both as RB

K, PM = RB.K, RB.PM


def structured():
    th = {k: torch.zeros_like(v) for k, v in RB.theta_tilde(0).items()}
    for s in range(K):
        th["wte"][RB.A + s, 0] = (-1.0) ** s          # chi(s) on coordinate 0
    th["wte"][0, 1] = 1.0                              # phi on coordinate 1 (query letter is letter 0)
    th["1.q"][0, 0, 0] = 1.0
    th["1.k"][0, 0, 1] = 1.0
    th["1.v"][0, 2, 1] = 1.0                           # value writes phi to coordinate 2
    for b in range(2):
        th[f"{b}.mb"][:] = -1.0                        # all MLP units off: the MLPs are zero maps
    return th


def measured(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye = torch.eye(K, device=RB.dev)
    Tbar = (1 - PM) * eye + PM * eye.roll(1, 1)
    prev = d["step_ids"][:, n + 3:] - RB.A
    z = RB.residual(p, d["step_ids"])[:, n + 3:] @ p["R"].T
    rule = eye[d["step_tgt"]] - Tbar[prev]
    phi = (d["prob"][:, None, None] * z * rule).sum() / z.shape[1]
    return torch.autograd.grad(phi, p["R"])[0][:, 2]   # derivative in R_{k,2}


def closed_form(n, sigma):
    tot = 0.0
    for t in range(2, n + 1):
        m = n + 2 + t
        tot += PM * (1 - PM) / m * (1 - (3 + 2 * (n - 1) * PM) / m)   # the query token also has phi = 1
    w = [((-1.0) ** (k - 1) - (-1.0) ** k) / K for k in range(K)]
    return [sigma ** 6 / math.sqrt(RB.Dh) * wk * tot / (n - 1) for wk in w]


if __name__ == "__main__":
    tilde = structured()
    for n in (2, 4, 6, 8, 10, 12):
        for sigma in (0.05, 0.02):
            got, want = measured(n, sigma, tilde).tolist(), closed_form(n, sigma)
            print(f"n={n:2d} sigma={sigma}: measured {got[0]:+.6e} {got[1]:+.6e}   closed form {want[0]:+.6e} {want[1]:+.6e}"
                  f"   ratio {got[0] / want[0]:.6f}", flush=True)
