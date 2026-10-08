"""Planted model with learned routing: does gradient descent learn the relative bias (routing) and the
readout jointly, from zero?

Counting mod 2, binary alphabet {q, r}, P(q) = p. Layout [w_1..w_n, s_0..s_{n-1}]; the position holding s_{t-1}
(index n+t-1) predicts s_t; the current letter sits exactly n positions back. x = (-1)^{s_{t-1}},
m = (-1)^{[w=q]}, target y = x*m.

One attention head at state positions (causal over all earlier positions):
  score(i, j) = a * x_i * m_j * [j is a letter] + b[i - j]     (a: planted gate strength, fixed)
  value(j)    = g * [j is a letter]                             (letter-independent)
  feature     = [1, x, sum_{letters} alpha_ij * g, onehot(position)]
Readout v and relative bias b are trained from zero by full-batch gradient descent on the logistic loss of the
margin y * v.feature. Logs the attention on the current letter, on the own position and on other letters
(leakage), and the first update at which every held-out context has positive margin.

Usage: python handcoded/routing_sim.py --n 6 8 --a 0.3 1 --seeds 0 1 2 --lr 0.1 --steps 400000 --out x.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, nargs="+", default=[8])
P.add_argument("--p", type=float, default=0.25)
P.add_argument("--a", type=float, nargs="+", default=[0.3])
P.add_argument("--g", type=float, default=1.0)
P.add_argument("--lr", type=float, default=0.1)
P.add_argument("--steps", type=int, default=400000)
P.add_argument("--batch", type=int, default=8192)
P.add_argument("--every", type=int, default=1000)
P.add_argument("--seeds", type=int, nargs="+", default=[0])
P.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
P.add_argument("--out", default=None)
args = P.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)


def run(n, a, seed):
    torch.manual_seed(seed)
    L = 2 * n

    def sample(B):
        m = 1 - 2 * (torch.rand(B, n) < args.p).double()           # m_t = (-1)^{b_t}
        x0 = 1 - 2 * torch.randint(2, (B, 1)).double()             # (-1)^{s_0}
        x = torch.cat([x0, x0 * m[:, :-1].cumprod(1)], 1)          # (-1)^{s_{t-1}}
        return m.to(dev), x.to(dev), (x * m).to(dev)               # y_t = x_t m_t

    tr, ev = sample(args.batch), sample(4 * args.batch)
    rel = torch.arange(L, device=dev)[:, None] - torch.arange(L, device=dev)[None, :]
    causal = rel >= 0
    rel = rel.clamp(min=0)
    qi = torch.arange(n, L, device=dev)                             # state positions
    is_letter = torch.arange(L, device=dev) < n
    ar = torch.arange(n, device=dev)
    b = torch.zeros(L, device=dev, requires_grad=True)
    v = torch.zeros(3 + n, device=dev, requires_grad=True)

    def margins(m, x, y):
        B = m.shape[0]
        mfull = torch.cat([m, torch.zeros(B, n, device=dev)], 1)   # letters carry m
        s = a * x[:, :, None] * mfull[:, None, :] * is_letter + b[rel[qi]]
        al = s.masked_fill(~causal[qi], float("-inf")).softmax(-1)
        hr = args.g * (al * is_letter).sum(-1)                     # (B, n)
        pos = torch.eye(n, device=dev).expand(B, n, n)
        feat = torch.cat([torch.ones(B, n, 1, device=dev), x[..., None], hr[..., None], pos], -1)
        cur = al[:, ar, ar]                                        # weight on the current letter
        slf = al[:, ar, n + ar]                                    # weight on the own position
        oth = (al * is_letter).sum(-1) - cur                       # weight on other letters (leakage)
        return y * (feat @ v), (cur, slf, oth)

    log, solved = [], None
    for step in range(args.steps + 1):
        mg, _ = margins(*tr)
        loss = torch.nn.functional.softplus(-mg).mean()
        if step % args.every == 0:
            with torch.no_grad():
                me, (cur, slf, oth) = margins(*ev)
            log.append(dict(step=step, loss=loss.item(), min_margin=me.min().item(),
                            correct=(me > 0).double().mean().item(), cur=cur.mean().item(),
                            self=slf.mean().item(), leak=oth.mean().item(), v_r=v[2].item()))
            if me.min() > 0:
                solved = step
                break
        b.grad = v.grad = None
        loss.backward()
        with torch.no_grad():
            b -= args.lr * b.grad
            v -= args.lr * v.grad
    return log, solved


results = []
for n in args.n:
    for a in args.a:
        for seed in args.seeds:
            t0 = time.time()
            log, solved = run(n, a, seed)
            last = log[-1]
            print(f"n={n:3d} a={a:<5} seed={seed} solved_at={solved} correct {last['correct']:.4f} "
                  f"leak {last['leak']:.4f} cur {last['cur']:.3f} self {last['self']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            results.append(dict(n=n, a=a, seed=seed, p=args.p, lr=args.lr, solved=solved, log=log))
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results))
