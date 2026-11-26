"""Gate flow on the full two-block network with BOTH readouts unconstrained 2 x 2 matrices (Appendix E.5).

research/gate_network/run.py trains U as a full 2 x 2 matrix but keeps the gate readout on the line W = w (I - R). Here W is
a free 2 x 2 matrix too, started at 0 and trained with the Frobenius metric. By the count-flip symmetry W stays circulant,
W = w0 I + w1 R, and its I + R component adds the same constant to both logits, so it receives no gradient: W should stay
on the line I - R. For the 27 runs of Figure 3b this script records the solve time, its relative difference from the
reduced three-scalar ODE (reduced_ode.py), and the largest distance of U from the circulant family and of W from the line
R(I - R) along the trajectory.

    python handcoded/checks/full_readout_check.py      # -> handcoded/checks/full_readout_check.json
"""
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

HANDCODED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDCODED))
sys.path.insert(0, str(HANDCODED / "research" / "gate_network"))
import reduced_ode as ro                                   # noqa: E402
import run as gn                                           # noqa: E402

torch.set_default_dtype(torch.float64)
K = gn.K
R = torch.roll(torch.eye(K), 1, 0)
I_MINUS_R = (torch.eye(K) - R) / math.sqrt(2 * K)          # unit vector of the line R(I - R)


def off_line(W):
    """Frobenius distance of W from the line R(I - R)."""
    return float(torch.linalg.norm(W - (W * I_MINUS_R).sum() * I_MINUS_R))


def off_circulant(U):
    """Frobenius distance of U from the circulant family u0 I + u1 R."""
    return float(torch.linalg.norm(U - R @ U @ R.T) / 2)


def run(job):
    torch.set_num_threads(1)
    p, eps, name, xq, xk = job
    rows, probs, tgts = gn.build_data(gn.N_LET, p)
    X = gn.embed(rows, eps)
    full, gate = gn.masks(gn.N_LET, X.shape[1])
    v, thr = eps, 0.5 * min(p, 1 - p) * math.log(2)
    drift = dict(W=0.0, U=0.0, W_rel=0.0, U_rel=0.0)

    def forward(U, W, q, k):
        # gn.forward with the gate readout block replaced by a free 2 x 2 matrix W
        PiQ = torch.zeros(gn.DH, gn.D); PiQ[0, gn.CNT:gn.CNT + K] = 1.0
        PiK = torch.zeros(gn.DH, gn.D); PiK[0, gn.EL] = 1.0
        PiV = torch.zeros(gn.D, gn.D)
        for s in range(K):
            PiV[gn.GAT + s, gn.CNT + s] = 1.0
        W_in, b_in, W_out = torch.zeros(16, gn.D), eps * torch.tensor([(-1.0) ** m for m in range(16)]), torch.zeros(gn.D, 16)
        h = X + torch.relu(X @ W_in.T + b_in) @ W_out.T
        S = ((h @ (q * PiQ).T) @ (h @ (k * PiK).T).transpose(1, 2)) / math.sqrt(gn.DH)
        h = h + S.masked_fill(~gate, -math.inf).softmax(-1) @ (h @ (v * PiV).T)
        h = h + torch.relu(h @ W_in.T + b_in) @ W_out.T
        readout = torch.cat([torch.zeros(K, gn.CNT), U, W], 1)
        return h[:, torch.arange(gn.N_LET + 3, 2 * gn.N_LET + 2)] @ readout.T

    def field(y):
        U = torch.tensor(y[:4]).view(K, K).requires_grad_()
        W = torch.tensor(y[4:8]).view(K, K).requires_grad_()
        q, k = torch.tensor(y[8]).requires_grad_(), torch.tensor(y[9]).requires_grad_()
        z = forward(U, W, q, k)
        ce = torch.nn.functional.cross_entropy(z.reshape(-1, K), tgts.reshape(-1), reduction="none").view(len(probs), -1)
        L = (probs * ce.mean(1)).sum()
        marg = float((z.gather(2, tgts[..., None]) - z.gather(2, (1 - tgts)[..., None])).min())
        dW, dU = off_line(W.detach()), off_circulant(U.detach())
        drift["W"], drift["U"] = max(drift["W"], dW), max(drift["U"], dU)
        drift["W_rel"] = max(drift["W_rel"], dW / max(float(torch.linalg.norm(W)), 1e-300))
        drift["U_rel"] = max(drift["U_rel"], dU / max(float(torch.linalg.norm(U)), 1e-300))
        g = torch.autograd.grad(L, [U, W, q, k])
        return np.concatenate([-g[0].flatten().numpy(), -g[1].flatten().numpy(), [-float(g[2]), -float(g[3])]]), float(L), marg

    def ev(tau, y):
        _, L, m = field(y)
        return 1.0 if m <= 0 else L - thr
    ev.terminal = True
    y0 = np.zeros(10)
    y0[8], y0[9] = eps * xq, eps * xk
    sol = solve_ivp(lambda tau, y: math.exp(tau) * field(y)[0], (0, 75.0), y0, method="LSODA", events=ev,
                    rtol=1e-10, atol=1e-18)
    t_full = float(math.expm1(sol.t_events[0][0]))
    t_red = ro.run((2, p, eps, gn.DH, xq, xk, 1.0, 75.0, 601, "line"))["solved_at"]
    return dict(p=p, eps=eps, seed=name, solved_full_W=t_full, solved_reduced=t_red,
                rel_diff=abs(t_full - t_red) / t_red, max_W_off_line=drift["W"], max_U_off_circulant=drift["U"],
                max_W_off_line_rel=drift["W_rel"], max_U_off_circulant_rel=drift["U_rel"])


if __name__ == "__main__":
    seeds = [("aligned", 1.3, -0.8), ("misaligned", 1.3, 0.8), ("misaligned near-balanced", 1.0, 0.98)]
    jobs = [(p, e, n, xq, (1 if p < 0.5 else -1) * xk) for p in (0.1, 0.25, 0.75) for e in (0.3, 0.2, 0.1)
            for n, xq, xk in seeds]
    with ProcessPoolExecutor(len(jobs)) as ex:
        out = list(ex.map(run, jobs))
    for r in out:
        print(f"p={r['p']:<5} eps={r['eps']:<4} {r['seed']:25s} rel diff {r['rel_diff']:.1e}  "
              f"W off line {r['max_W_off_line']:.1e}  U off circulant {r['max_U_off_circulant']:.1e}")
    summary = dict(runs=len(out), max_rel_diff=max(r["rel_diff"] for r in out),
                   max_W_off_line=max(r["max_W_off_line"] for r in out),
                   max_U_off_circulant=max(r["max_U_off_circulant"] for r in out),
                   max_W_off_line_rel=max(r["max_W_off_line_rel"] for r in out),
                   max_U_off_circulant_rel=max(r["max_U_off_circulant_rel"] for r in out))
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, runs=out), open(Path(__file__).with_name("full_readout_check.json"), "w"), indent=1)
