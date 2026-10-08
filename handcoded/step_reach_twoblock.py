"""Conjecture 1 in the two-block model: restricted step training from small initialization.

The architecture of lettertrace.py (two blocks, three heads of dimension P, ReLU MLPs, no LayerNorm) plus a
relative position bias rb[b][h, i - j] per block and head, initialized at zero (Assumption 1). Counting mod K with
a fixed query letter (letter 0), block layout:  s0 q w_1..w_n SEP s_1..s_n.  The position holding s_{t-1}
(SEP for t = 1) predicts s_t; the loss is cross-entropy over the K value logits at those n positions.

theta_0 = sigma * theta_tilde with the fan-in-scaled theta_tilde of Remark B.9: Gaussian entries of variance 1/fan-in
(embeddings 1/d_res) and MLP biases |b~_m| in [1, 2] (Assumption 2).
Trained by full-batch gradient descent: Q, K, the relative bias and the readout. Frozen at theta_0: token and
position embeddings, value maps, MLPs.

Logged: loss, the smallest held-out state margin (> 0 means every held-out state prediction is correct), and the
norms of the relative bias and of the readout's value rows.

With --format outcome the prompt ends at SEP and only that position is supervised, with the final count s_n; this
is the outcome baseline under the same restricted protocol (same initialization, trained subset and step size).

Usage: python handcoded/step_reach_twoblock.py --n 6 8 --sigma 0.3 --seeds 0 1 --lr 2 --steps 150000 --out x.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, nargs="+", default=[6])
p.add_argument("--K", type=int, default=2)
p.add_argument("--A", type=int, default=4)        # alphabet; letter 0 is the query
p.add_argument("--sigma", type=float, nargs="+", default=[0.3])
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--lr", type=float, default=2.0)
p.add_argument("--steps", type=int, default=150000)
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--every", type=int, default=500)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", default=None)
p.add_argument("--format", choices=("step", "outcome"), default="step")
p.add_argument("--armijo", type=int, default=0,
               help="after solving, up to this many full-batch Armijo updates; record when 4L/pi_min < 1")
args = p.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)
H = 3
TRAINED = ("q", "k", "rb", "readout")


def run(n, K, A, sigma, seed):
    torch.manual_seed(seed)
    SEP, VOCAB, P = A + K, A + K + 1, 2 * n + 6
    W = 2 * A + 4 * K + P + 3                          # residual width of lettertrace.py
    FW = max(A, 3 * K * (n + 1), 2 * n + 1)            # MLP width of lettertrace.py

    def sample(m):
        w = torch.randint(A, (m, n))
        b = (w == 0).long()
        s0 = torch.randint(K, (m,))
        s = (s0[:, None] + b.cumsum(1)) % K                                   # s_1..s_n
        ids = torch.cat([A + s0[:, None], torch.zeros(m, 1, dtype=torch.long), w,
                         torch.full((m, 1), SEP), A + s[:, :-1]], 1)          # s0 q w SEP s_1..s_{n-1}
        if args.format == "outcome":                                          # s0 q w SEP -> s_n
            return ids[:, :n + 3].to(dev), s[:, -1:].to(dev)
        return ids.to(dev), s.to(dev)

    x_tr, y_tr = sample(args.batch)
    x_ev, y_ev = sample(4 * args.batch)
    T = x_tr.shape[1]                                                         # 2n + 2
    pred = torch.arange(n + 2, 2 * n + 2, device=dev)                         # positions predicting s_1..s_n
    if args.format == "outcome":
        pred = torch.tensor([n + 2], device=dev)                              # SEP predicts s_n

    g = lambda fan, *s: sigma * torch.randn(*s, device=dev) / math.sqrt(fan)     # variance sigma^2 / fan-in
    th = {"wte": g(W, VOCAB, W), "wpe": g(W, P, W), "readout": g(W, K, W)}
    for b in range(2):
        th[f"{b}.q"], th[f"{b}.k"], th[f"{b}.v"] = g(W, H, P, W), g(W, H, P, W), g(W, H, W, W)
        th[f"{b}.rb"] = torch.zeros(H, P, device=dev)
        th[f"{b}.mlp_in"], th[f"{b}.mlp_out"] = g(W, FW, W), g(FW, W, FW)
        sign = torch.randint(2, (FW,), device=dev) * 2 - 1
        th[f"{b}.mlp_b"] = sigma * sign * (1 + torch.rand(FW, device=dev))   # |b~_m| in [1, 2]
    params = [t.requires_grad_() for k, t in th.items() if k.split(".")[-1] in TRAINED]

    rel = (torch.arange(T, device=dev)[:, None] - torch.arange(T, device=dev)[None, :]).clamp(min=0)
    future = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)

    def forward(ids):
        x = th["wte"][ids] + th["wpe"][:T]
        for b in range(2):
            q = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, th[f"{b}.v"])
            att = q @ k.transpose(-1, -2) / math.sqrt(P) + th[f"{b}.rb"][:, rel]
            x = x + (att.masked_fill(future, -math.inf).softmax(-1) @ v).sum(1)
            x = x + F.relu(x @ th[f"{b}.mlp_in"].T + th[f"{b}.mlp_b"]) @ th[f"{b}.mlp_out"].T
        return x[:, pred] @ th["readout"].T                                   # K value logits

    def margin(ids, y):
        with torch.no_grad():
            z = forward(ids)
            gold = z.gather(-1, y[..., None]).squeeze(-1)
            other = z.scatter(-1, y[..., None], float("-inf")).max(-1).values
            return (gold - other).min().item(), (z.argmax(-1) == y).all(1).double().mean().item()

    log = []
    for step in range(args.steps + 1):
        loss = F.cross_entropy(forward(x_tr).reshape(-1, K), y_tr.reshape(-1))
        if step % args.every == 0:
            m, exact = margin(x_ev, y_ev)
            log.append(dict(step=step, loss=loss.item(), margin=m, exact=exact,
                            rb=max(th[f"{b}.rb"].abs().max().item() for b in range(2)),
                            readout=th["readout"].norm().item()))
            solved = next((r["step"] for r in log if r["margin"] > 0), None)
            if solved is not None and (args.armijo or step >= solved + 5000):   # hand over to Armijo, or stop
                break
        for q in params:
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in params:
                q -= args.lr * q.grad
    basin = []
    if args.armijo and any(r["margin"] > 0 for r in log):
        # Theorem 4 on the training set: items are (prompt, position) with weight 1/(N n), so pi_min = 1/(N n).
        pi_min = 1.0 / (x_tr.shape[0] * n)
        lossf = lambda: F.cross_entropy(forward(x_tr).reshape(-1, K), y_tr.reshape(-1))
        eta_bar, entered = 100.0, None
        for j in range(args.armijo):
            loss = lossf()
            for q in params:
                q.grad = None
            loss.backward()
            g2 = sum((q.grad ** 2).sum() for q in params).item()
            ratio = 4 * loss.item() / pi_min
            if entered is None and ratio < 1:
                entered = j
            m, _ = margin(x_ev, y_ev)
            basin.append(dict(update=j, loss=loss.item(), ratio=ratio, heldout_margin=m))
            if entered is not None and j >= entered + 100:
                break
            eta = eta_bar
            with torch.no_grad():
                old = [q.detach().clone() for q in params]
                while True:                                   # Armijo: L(theta - eta g) <= L - eta/2 |g|^2
                    for q, o in zip(params, old):
                        q.copy_(o - eta * q.grad)
                    if lossf().item() <= loss.item() - eta / 2 * g2 or eta < 1e-8:
                        break
                    eta /= 2
        basin = dict(entered=entered, trace=basin)
    return log, basin


results = []
for n in args.n:
    for sigma in args.sigma:
        for seed in args.seeds:
            t0 = time.time()
            log, basin = run(n, args.K, args.A, sigma, seed)
            solved = next((r["step"] for r in log if r["margin"] > 0), None)
            last = log[-1]
            print(f"n={n:3d} sigma={sigma:<6} seed={seed} solved_at={solved} final loss {last['loss']:.4f} "
                  f"margin {last['margin']:+.3f} exact {last['exact']:.3f} max|rb| {last['rb']:.2f} "
                  f"basin_entered_after={basin.get('entered') if basin else None} Armijo updates "
                  f"({time.time() - t0:.0f}s)", flush=True)
            results.append(dict(n=n, K=args.K, A=args.A, sigma=sigma, seed=seed, lr=args.lr, log=log, basin=basin))
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results))
