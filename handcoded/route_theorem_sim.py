"""Numerical checks for the Theorem 3 proof plan: learned routing from Theorem 4's structured start.

Exact feature formula (Step 0 of the plan) on the paper's layout  s0 q w_1..w_n SEP s_1..s_{n-1}.
At the position i_t = n+1+t predicting s_t (t >= 2) the gate head attends over all j <= i_t with
    score_j = a * x_t * m_j   (letter-type tokens: q has m = -1, w_j has m_j)   + beta[i_t - j]
    value_j = g               (letter-type tokens), 0 otherwise (s0, SEP, states),
and the logit difference is f = sigma*v_c + sigma*v_s*x_t + g*v_r*M with M the attention mass on letter-type tokens.
x_t = (-1)^{s_{t-1}}, m_t = (-1)^{[w_t = q]}, y_t = x_t m_t. Planted constants: g = sigma^2 A_V,
a = sigma^4 A_Q A_K / sqrt(d_h), d_h = 2n+6, random signs A. Trained: (v_c, v_s, v_r) and beta, from zero,
by full-batch gradient descent (small step ~ gradient flow) on the mean logistic loss of y f over t >= 2.

Logged (the plan's order parameters): sigma*v_s, v_r, Delta = beta_n - beta_0, the attention on the current letter,
on the own position, on other letters (harmful leakage eps, mean and max over t), on q, and the sign check of
Step 3: the number of offsets delta not in {0, n} with sign(g a v_r) * dL/dbeta_delta < 0, and the most negative one.
Solved = every held-out (prompt, t >= 2) margin positive.

Usage: python handcoded/route_theorem_sim.py --n 6 8 --sigma 1.0 0.8 --seeds 0 1 --out results/route/x.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, nargs="+", default=[6])
P.add_argument("--sigma", type=float, nargs="+", default=[1.0])
P.add_argument("--p", type=float, default=0.25)
P.add_argument("--seeds", type=int, nargs="+", default=[0])
P.add_argument("--lr", type=float, default=1.0)
P.add_argument("--steps", type=int, default=2_000_000)
P.add_argument("--batch", type=int, default=4096)
P.add_argument("--every", type=int, default=2000)
P.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
P.add_argument("--out", default=None)
args = P.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)


def run(n, sigma, seed):
    torch.manual_seed(seed)
    A = (torch.randint(2, (3,)) * 2 - 1).double()                  # random signs A_Q, A_K, A_V
    dh = 2 * n + 6
    g = sigma ** 2 * A[2].item()
    a = sigma ** 4 * (A[0] * A[1]).item() / math.sqrt(dh)
    L = 2 * n + 2                                                    # s0 q w_1..w_n SEP s_1..s_{n-1}
    T = n - 1                                                        # prediction positions t = 2..n
    it = torch.arange(2, n + 1, device=dev) + n + 1                  # i_t = n+1+t
    j = torch.arange(L, device=dev)
    rel = (it[:, None] - j[None, :])                                 # (T, L) offsets
    vis = rel >= 0
    relc = rel.clamp(min=0)
    letter_pos = (j >= 1) & (j <= n + 1)                             # q at 1, w_t at 1+t
    cur = (j[None, :] == (it - n)[:, None])                          # current letter w_t at 1+t = i_t - n
    own = (j[None, :] == it[:, None])
    other_letter = letter_pos[None, :] & ~cur & (j[None, :] >= 2)    # letters w_u, u != t (q excluded)
    qpos = (j == 1)

    def sample(B):
        mw = 1 - 2 * (torch.rand(B, n, device=dev) < args.p).double()  # m for w_1..w_n
        x0 = 1 - 2 * torch.randint(2, (B, 1), device=dev).double()
        xs = torch.cat([x0, x0 * mw[:, :-1].cumprod(1)], 1)          # x_t = (-1)^{s_{t-1}}, t = 1..n
        mkey = torch.zeros(B, L, device=dev)
        mkey[:, 1] = -1.0                                            # q
        mkey[:, 2:n + 2] = mw
        x = xs[:, 1:]                                                # t = 2..n
        y = x * mw[:, 1:]
        return mkey, x, y

    tr, ev = sample(args.batch), sample(4 * args.batch)
    beta = torch.zeros(L, device=dev, requires_grad=True)
    v = torch.zeros(3, device=dev, requires_grad=True)               # v_c, v_s, v_r

    def forward(mkey, x, y):
        s = a * x[:, :, None] * mkey[:, None, :] + beta[relc][None]
        al = s.masked_fill(~vis[None], float("-inf")).softmax(-1)    # (B, T, L)
        M = (al * letter_pos).sum(-1)
        f = sigma * v[0] + sigma * v[1] * x + g * v[2] * M
        return y * f, al

    log, solved = [], None
    for step in range(args.steps + 1):
        mg, _ = forward(*tr)
        loss = F.softplus(-mg).mean()
        beta.grad = v.grad = None
        loss.backward()
        if step % args.every == 0:
            with torch.no_grad():
                me, al = forward(*ev)
                x0, y0 = ev[1], ev[2]                                    # local case (x, y) of each prediction
                sgn = math.copysign(1.0, g * a * v[2].item()) if v[2].item() != 0 else 1.0
                gb = sgn * beta.grad
                cand = torch.tensor([d for d in range(L) if d not in (0, n)], device=dev)
                gneg = gb[cand]
                leak = (al * other_letter).sum(-1)
                log.append(dict(step=step, loss=loss.item(), min_margin=me.min().item(),
                                correct=(me > 0).double().mean().item(), sv_s=sigma * v[1].item(), v_r=v[2].item(),
                                Delta=(beta[n] - beta[0]).item(), cur=(al * cur).sum(-1).mean().item(),
                                own=(al * own).sum(-1).mean().item(), eps_mean=leak.mean().item(),
                                eps_max=leak.max().item(), q=(al * qpos).sum(-1).mean().item(),
                                n_neg=int((gneg < 0).sum().item()), min_signed_grad=gneg.min().item(),
                                neg_deltas=cand[gneg < 0].tolist(),
                                case_correct=[((me > 0) & (x0 == xs) & (y0 == ys)).sum().item()
                                              / max(((x0 == xs) & (y0 == ys)).sum().item(), 1)
                                              for xs in (-1, 1) for ys in (-1, 1)]))
                if me.min() > 0:
                    solved = step
                    break
        with torch.no_grad():
            beta -= args.lr * beta.grad
            v -= args.lr * v.grad
    return dict(g=g, a=a, A=A.tolist()), log, solved


results = []
for n in args.n:
    for sigma in args.sigma:
        for seed in args.seeds:
            t0 = time.time()
            consts, log, solved = run(n, sigma, seed)
            last = log[-1]
            neg = sum(1 for r in log if r["n_neg"] > 0)
            print(f"n={n:3d} sigma={sigma:<5} seed={seed} g={consts['g']:+.3f} a={consts['a']:+.4f} solved_at={solved} "
                  f"correct {last['correct']:.4f} eps_max {last['eps_max']:.4f} Delta {last['Delta']:+.2f} "
                  f"logs_with_wrong_sign {neg}/{len(log)} ({time.time() - t0:.0f}s)", flush=True)
            results.append(dict(n=n, sigma=sigma, seed=seed, p=args.p, lr=args.lr, solved=solved, **consts, log=log))
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results))
