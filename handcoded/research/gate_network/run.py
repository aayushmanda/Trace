"""Gradient flow on the actual two-block gate network of Theorem 2 (Appendix D.1), not on the reduced four-scalar system.

The network is built from explicit matrices: token embeddings, zero position and format embeddings, block 1 with zero heads
and a zero-output MLP with biases +-eps, block 2 with three heads of which only the gate head is nonzero (W_Q = q Pi_Q,
W_K = k Pi_K, W_V = v Pi_V, mask {i_t, j_t} at the positions predicting s_t, full causal mask elsewhere), and a readout whose
count block is U and whose gate block is W = w (I - R). The loss is L_step (t = 2..n, teacher forced), exact over all words
and both starts. The flow uses the theorem's metric: Frobenius on U, Frobenius on the line R(I - R) for W, Euclidean on q, k;
gradients come from autograd through the whole network. Same seeds and solve criterion as handcoded/reduced_ode.py (K = 2).

    python handcoded/research/gate_network/run.py
"""
import itertools
import json
import math
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
K, DH, N_LET, ALPHA = 2, 8, 4, 2                      # counting modulo 2, head dim 8, n = 4 letters, alphabet {query, other}
D = 2 + 2 * K                                         # residual: e0, e_ell, count register, gate register
E0, EL, CNT, GAT = 0, 1, 2, 2 + K


def basis(i, d=D):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def build_data(n, p):
    """Token sequences up to the last step prediction, targets and probabilities."""
    rows, probs, tgts = [], [], []
    for s0 in range(K):
        for word in itertools.product(range(ALPHA), repeat=n):          # letter 0 is the query
            b = [1 if w == 0 else 0 for w in word]
            s = [s0]
            for bt in b:
                s.append((s[-1] + bt) % K)
            # positions: s0@0, query@1, w_t@1+t, SEP@n+2, s_t@n+2+t (t=1..n-1); position n+1+t predicts s_t
            toks = [("cnt", s0), ("let", 0)] + [("let", w) for w in word] + [("fmt", 0)] + [("cnt", s[t]) for t in range(1, n)]
            rows.append(toks)
            probs.append(np.prod([p if bt else 1 - p for bt in b]) / K)
            tgts.append(s[2:])                                            # s_2..s_n
    return rows, torch.tensor(probs), torch.tensor(tgts)


def embed(rows, eps):
    X = torch.zeros(len(rows), len(rows[0]), D)
    for i, toks in enumerate(rows):
        for j, (kind, val) in enumerate(toks):
            if kind == "cnt":
                X[i, j] = eps * basis(CNT + val)
            elif kind == "let":
                X[i, j] = eps * (basis(E0) - basis(EL) if val == 0 else basis(E0) + basis(EL))
    return X                                                               # format token and positions: zero


def masks(n, T):
    full = torch.ones(T, T).tril().bool()
    gate = full.clone()
    for t in range(2, n + 1):                                              # position n+1+t holds s_{t-1}, predicts s_t
        i, j = n + 1 + t, 1 + t
        gate[i] = False
        gate[i, i] = gate[i, j] = True
    return full, gate


def forward(theta, X, n, eps, v, full, gate):
    U, w, q, k = theta["U"], theta["w"], theta["q"], theta["k"]
    PiQ = torch.zeros(DH, D)
    PiQ[0, CNT:CNT + K] = 1.0
    PiK = torch.zeros(DH, D)
    PiK[0, EL] = 1.0
    PiV = torch.zeros(D, D)
    for s in range(K):
        PiV[GAT + s, CNT + s] = 1.0
    h = X
    # block 1: zero heads; MLP with biases +-eps and zero output map
    W_in, b_in, W_out = torch.zeros(16, D), eps * torch.tensor([(-1.0) ** m for m in range(16)]), torch.zeros(D, 16)
    h = h + torch.relu(h @ W_in.T + b_in) @ W_out.T
    # block 2: gate head (head 1) with its mask; heads 2, 3 are zero
    Q, Kk, V = h @ (q * PiQ).T, h @ (k * PiK).T, h @ (v * PiV).T
    S = (Q @ Kk.transpose(1, 2)) / math.sqrt(DH)
    S = S.masked_fill(~gate, -math.inf)
    h = h + S.softmax(-1) @ V
    h = h + torch.relu(h @ W_in.T + b_in) @ W_out.T
    R = torch.roll(torch.eye(K), 1, 0)                                     # R e_s = e_{s+1}
    Wg = w * (torch.eye(K) - R)
    readout = torch.zeros(K, D)
    readout = readout + torch.cat([torch.zeros(K, CNT), U, Wg], 1)
    pos = torch.arange(n + 3, 2 * n + 2)                                   # positions n+3..2n+1 predict s_2..s_n
    return h[:, pos] @ readout.T


def loss_fn(theta, data, n, eps, v, full, gate, p):
    X, probs, tgts = data
    z = forward(theta, X, n, eps, v, full, gate)
    ce = torch.nn.functional.cross_entropy(z.reshape(-1, K), tgts.reshape(-1), reduction="none").view(len(probs), -1)
    marg = (z.gather(2, tgts[..., None]).squeeze(2) - z.gather(2, (1 - tgts)[..., None]).squeeze(2)).min()
    return (probs * ce.mean(1)).sum(), float(marg.detach()) if torch.is_tensor(marg) else marg


def run(p, eps, xq, xk, xv=1.0, tau_max=75.0):
    rows, probs, tgts = build_data(N_LET, p)
    X = embed(rows, eps)
    full, gate = masks(N_LET, X.shape[1])
    data, v = (X, probs, tgts), eps * xv
    thr = 0.5 * min(p, 1 - p) * math.log(2)

    def unpack(y):
        return dict(U=torch.tensor(y[:4]).view(K, K).requires_grad_(), w=torch.tensor(y[4]).requires_grad_(),
                    q=torch.tensor(y[5]).requires_grad_(), k=torch.tensor(y[6]).requires_grad_())

    def field(y):
        th = unpack(y)
        L, m = loss_fn(th, data, N_LET, eps, v, full, gate, p)
        gU, gw, gq, gk = torch.autograd.grad(L, [th["U"], th["w"], th["q"], th["k"]])
        nIR = 2 * K                                                        # ||I - R||_F^2: Frobenius metric on the line
        return np.concatenate([-gU.flatten().numpy(), [-float(gw) / nIR, -float(gq), -float(gk)]]), float(L), m

    def ev(tau, y):
        _, L, m = field(y)
        return 1.0 if m <= 0 else L - thr
    ev.terminal = True
    y0 = np.array([0, 0, 0, 0, 0, eps * xq, eps * xk], dtype=float)
    sol = solve_ivp(lambda tau, y: math.exp(tau) * field(y)[0], (0, tau_max), y0, method="LSODA", events=ev,
                    rtol=1e-10, atol=1e-18)
    return float(math.expm1(sol.t_events[0][0])) if len(sol.t_events[0]) else None


def job(args):
    torch.set_num_threads(1)
    p, eps, name, xq, xk = args
    return dict(K=K, p=p, eps=eps, seed=name, xi_Q=xq, xi_K=xk, solved_at=run(p, eps, xq, xk))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    seeds = [("aligned", 1.3, -0.8), ("misaligned", 1.3, 0.8), ("misaligned near-balanced", 1.0, 0.98)]
    jobs = [(p, eps, name, xq, (1 if p < 0.5 else -1) * xk) for p in (0.1, 0.25, 0.75) for eps in (0.3, 0.2, 0.1)
            for name, xq, xk in seeds]
    with ProcessPoolExecutor(len(jobs)) as ex:
        out = list(ex.map(job, jobs))
    for r in out:
        print(f"p={r['p']} eps={r['eps']} {r['seed']:25s} full-network solve time {r['solved_at']}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.json"), "w"), indent=1)
