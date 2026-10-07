"""Proof model for step-training reachability: counting mod K, fixed query.

Sequence: [w_1..w_n, s_0, s_1, .., s_{n-1}]; the position holding s_{t-1}
predicts s_t (teacher forcing). The letter w_t sits exactly n positions back.

Model (one trained attention layer, frozen one-hot embeddings and values):
  h_i     = onehot(token_i)                          (dim D = A + K)
  psi_ij  = h_i^T W h_j + b[i - j]   (causal)        token-bilinear + relative bias
  o_i     = sum_j softmax_j(psi_ij) h_j              value = identity
  z_i     = V [h_i ; o_i]                            readout, K logits
Trained: W, b, V by full-batch gradient descent (small step ~ gradient flow).

Logged order parameters:
  beta : W[state, a] - W[state, other letters]  (score bonus for a matching letter), mean over states
  gamma: same bonus, centred over states (its norm): the state-dependent part
  self : W[state, state] - W[state, other states], mean over states (bonus for attending to one's own state)
  b0, bn : relative bias on self and on the current letter (offset n)
  v_hs : readout on the direct state token h (block V[:, A:D])
  v_ol, v_os : readout on the attention output, letter and state blocks
  margin : smallest gold-logit margin over the eval set (> 0 means all correct)

Usage: python handcoded/step_reach.py --n 6 8 --sigma 0.02 --seeds 0 1 --out results/step_reach/x.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, nargs="+", default=[8])
p.add_argument("--K", type=int, default=2)
p.add_argument("--A", type=int, default=4)        # alphabet size; letter 0 is the query
p.add_argument("--sigma", type=float, nargs="+", default=[0.02])
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--lr", type=float, default=0.5)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=4096)  # fixed full batch
p.add_argument("--every", type=int, default=50)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", default=None)
p.add_argument("--scaled", action="store_true",
               help="frozen embeddings sigma*onehot and value map sigma*I, as at small initialization")
args = p.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)


def run(n, K, A, sigma, seed):
    torch.manual_seed(seed)
    D, L = A + K, 2 * n

    def sample(m):
        w = torch.randint(A, (m, n))
        b = (w == 0).long()                              # match bit, query letter 0
        s0 = torch.randint(K, (m,))
        s = (s0[:, None] + b.cumsum(1)) % K              # s_1..s_n
        prev = torch.cat([s0[:, None], s[:, :-1]], 1)    # s_0..s_{n-1}
        x = torch.cat([w, A + prev], 1)                  # tokens, length 2n
        return x.to(dev), s.to(dev)                      # targets s_1..s_n

    x_tr, y_tr = sample(args.batch)
    x_ev, y_ev = sample(4 * args.batch)
    emb = sigma if args.scaled else 1.0                  # frozen embedding scale
    val = sigma if args.scaled else 1.0                  # frozen value-map scale
    H_tr, H_ev = emb * F.one_hot(x_tr, D).double(), emb * F.one_hot(x_ev, D).double()

    rel = torch.arange(L, device=dev)[:, None] - torch.arange(L, device=dev)[None, :]
    causal = rel >= 0
    rel = rel.clamp(min=0)

    W = (sigma * torch.randn(D, D, device=dev)).requires_grad_()
    b = (sigma * torch.randn(L, device=dev)).requires_grad_()
    V = (sigma * torch.randn(K, 2 * D, device=dev)).requires_grad_()
    params = [W, b, V]

    def forward(H):
        psi = H @ W @ H.transpose(1, 2) + b[rel]
        psi = psi.masked_fill(~causal, float("-inf"))
        o = val * (psi.softmax(-1) @ H)
        z = torch.cat([H, o], -1) @ V.T
        return z[:, n:]                                  # logits at state positions

    def stats(H, y):
        with torch.no_grad():
            z = forward(H)
            gold = z.gather(-1, y[..., None]).squeeze(-1)
            other = z.scatter(-1, y[..., None], float("-inf")).max(-1).values
            st = slice(A, A + K)
            bonus = W[st, 0] - W[st, 1:A].mean(1)
            Ws = W[st, st]
            selfb = (Ws.diag().mean() - (Ws.sum() - Ws.diag().sum()) / max(K * (K - 1), 1)).item()
            return dict(gamma=(bonus - bonus.mean()).norm().item(), self=selfb,
                margin=(gold - other).min().item(),
                exact=(z.argmax(-1) == y).all(1).double().mean().item(),
                beta=bonus.mean().item(),
                b0=b[0].item(), bn=b[n].item(),
                v_hs=V[:, A:D].norm().item(),
                v_ol=V[:, D:D + A].norm().item(),
                v_os=V[:, D + A:].norm().item(),
            )

    log = []
    for step in range(args.steps + 1):
        loss = F.cross_entropy(forward(H_tr).reshape(-1, K), y_tr.reshape(-1))
        if step % args.every == 0:
            log.append(dict(step=step, loss=loss.item(), **stats(H_ev, y_ev)))
        for q in params:
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in params:
                q -= args.lr * q.grad
    return log


results = []
for n in args.n:
    for sigma in args.sigma:
        for seed in args.seeds:
            t0 = time.time()
            log = run(n, args.K, args.A, sigma, seed)
            solved = next((r["step"] for r in log if r["margin"] > 0), None)
            last = log[-1]
            print(f"n={n:3d} sigma={sigma:<6} seed={seed} solved_at={solved} final loss {last['loss']:.4f} "
                  f"margin {last['margin']:+.3f} beta {last['beta']:+.3f} b0 {last['b0']:+.3f} "
                  f"bn {last['bn']:+.3f} v_os {last['v_os']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            results.append(dict(n=n, K=args.K, A=args.A, sigma=sigma, seed=seed, lr=args.lr,
                                scaled=args.scaled, log=log))

if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results))
