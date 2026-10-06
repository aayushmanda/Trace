"""Numerical check of the two-sided tabular theorem (T1).

Tabular model: one learned row-stochastic matrix per move, P_x = row-softmax(Theta_x).
Counting mod K over a 4-letter alphabet: moves are the match bit, pi = (3/4, 1/4).
  outcome  q(y | s0, x) = e_{s0}^T P_{x_1} ... P_{x_n} e_y,   L_O = E[-log q(s_n)]
  step     L_S = sum_x pi(x)/K sum_s -log P_x[s, F(s, x)]      (clean traces, uniform start)
Checks
  (a) inside ||Theta||_max <= r = 1/(8K): log K - L_O <= 6K(2r)^n, by adversarial
      projected gradient ascent on the gap (several restarts), exact over all move strings;
  (b) gradient descent on L_S with any step size eta: L_S(theta_j) <= |X| K (1 + S0) / (j eta),
      S0 = (K-1) e^{2 r0}.
"""
import itertools
import math

import torch

torch.set_default_dtype(torch.float64)
PI = torch.tensor([0.75, 0.25])


def outcome_loss(theta, K, n):
    P = torch.softmax(theta, -1)                              # (2, K, K)
    words = torch.tensor(list(itertools.product((0, 1), repeat=n)))
    w = PI[words].prod(1)                                     # P(x)
    M = torch.eye(K).expand(len(words), K, K)
    for t in range(n):
        M = M @ P[words[:, t]]
    shift = words.sum(1)                                      # s_n = s0 + #matches mod K
    s0 = torch.arange(K)
    sn = (s0[None, :] + shift[:, None]) % K                   # (words, K)
    q = M[torch.arange(len(words))[:, None], s0[None, :], sn]
    return -(w[:, None] * torch.log(q)).sum(0).mean()         # uniform s0


def step_loss(theta, K):
    logp = torch.log_softmax(theta, -1)
    s = torch.arange(K)
    return sum(PI[b] / K * -logp[b, s, (s + b) % K].sum() for b in (0, 1))


def check_outcome(K, n, restarts=6, iters=300):
    r = 1 / (8 * K)
    best = -1.0
    for seed in range(restarts):
        g = torch.Generator().manual_seed(seed)
        th = ((torch.rand(2, K, K, generator=g) * 2 - 1) * r).requires_grad_()
        opt = torch.optim.Adam([th], lr=r / 20)
        for _ in range(iters):
            gap = math.log(K) - outcome_loss(th, K, n)
            opt.zero_grad(); (-gap).backward(); opt.step()
            with torch.no_grad():
                th.clamp_(-r, r)
        best = max(best, float(math.log(K) - outcome_loss(th.detach(), K, n)))
    return best, 6 * K * (2 * r) ** n


def check_step(K, r0=0.01, dt=0.05, T=400.0):
    g = torch.Generator().manual_seed(0)
    th = ((torch.rand(2, K, K, generator=g) * 2 - 1) * r0).requires_grad_()
    S0 = (K - 1) * math.exp(2 * r0)
    worst, t, rows = 0.0, 0.0, []
    while t < T:
        L = step_loss(th, K)
        (gr,) = torch.autograd.grad(L, th)
        with torch.no_grad():
            th -= dt * gr
        t += dt
        bound = 2 * K * (1 + S0) / t
        Lt = float(step_loss(th, K))
        worst = max(worst, Lt / bound)
        if any(abs(t - c) < dt / 2 for c in (10, 50, 100, 200, 400)):
            rows.append((round(t), Lt, bound))
    return worst, rows


if __name__ == "__main__":
    for K in (2, 3):
        for n in (4, 6, 8, 10, 12):
            gap, bound = check_outcome(K, n)
            print(f"outcome K={K} n={n:2d}: max gap {gap:.3e}  bound 6K(2r)^n {bound:.3e}  "
                  f"{'OK' if gap <= bound else 'VIOLATED'}")
    for K in (2, 3):
        worst, rows = check_step(K)
        print(f"step K={K}: max L_S/bound {worst:.3f} ({'OK' if worst <= 1 else 'VIOLATED'})  "
              + "  ".join(f"t={t}: L={L:.3f} bound={b:.3f}" for t, L, b in rows))
