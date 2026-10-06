"""Numerical check of the progress separation in the two-block architecture (Theorem 3).

Two blocks as in lettertrace.py, theta = sigma * theta_tilde with MLP biases bounded away
from zero, K value logits, float64, counting mod K over 4 letters, query fixed, uniform start,
exact over all 4^n words.
  step loss     state tokens s_t, t = 2..n, in the block layout (position n+1+t holds s_{t-1})
  outcome loss  the answer s_n at the ':' position
Checks
  (a) ||grad_R L_S|| / sigma  vs  s1 * sigma_min(E_val) and the leading term ||C E_val||_F,
      with C[y,s] = (1/K)(P(s_t=y | s_{t-1}=s) - 1/K), s1 = lambda/K;      (n-independent)
  (b) log K - L_O <= C sigma^2 (sqrt(A_<=1) + sigma^5): the ratio Phi / (sigma^2 sqrt(A_<=1));
  (c) Euler gradient flow on each loss: loss decrease after time t.
"""
import itertools
import math
import sys

import torch

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, Wd, Hh, Dh, F = 4, 16, 3, 8, 16


def setup(n, K, seed=0):
    vocab, SEP, COLON = A + K + 3, A + K, A + K + 1
    T = 2 * n + 4
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    th = {"wte": rn(vocab, Wd), "wpe": rn(T, Wd), "R": rn(K, Wd)}
    for b in range(2):
        th[f"{b}.q"], th[f"{b}.k"], th[f"{b}.v"] = rn(Hh, Dh, Wd), rn(Hh, Dh, Wd), rn(Hh, Wd, Wd)
        th[f"{b}.mi"], th[f"{b}.mo"] = rn(F, Wd), rn(Wd, F)
        mb = rn(F)
        th[f"{b}.mb"] = torch.sign(mb) * (1 + mb.abs())
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    m = len(words)
    rows_s, tgt_s, rows_o, tgt_o = [], [], [], []
    for s0 in range(K):
        states = (s0 + (words == 0).cumsum(1)) % K                       # query letter 0
        head = torch.cat([torch.full((m, 1), A + s0), torch.zeros(m, 1, dtype=torch.long), words,
                          torch.full((m, 1), SEP)], 1)
        rows_s.append(torch.cat([head, A + states[:, :n - 1]], 1))       # ... SEP s_1 .. s_{n-1}
        tgt_s.append(states[:, 1:])                                      # s_2 .. s_n
        rows_o.append(torch.cat([head, torch.full((m, 1), COLON)], 1))
        tgt_o.append(states[:, -1])
    data = dict(step=(torch.cat(rows_s).to(dev), torch.cat(tgt_s).to(dev)),
                outcome=(torch.cat(rows_o).to(dev), torch.cat(tgt_o).to(dev)))
    return {k: v.to(dev) for k, v in th.items()}, data


def hidden(p, ids):
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    for b in range(2):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(Dh)).masked_fill(future, -math.inf)
        x = x + (att.softmax(-1) @ v).sum(1)
        x = x + torch.relu(x @ p[f"{b}.mi"].T + p[f"{b}.mb"]) @ p[f"{b}.mo"].T
    return x


def loss(p, data, mode, n, chunk=1 << 15):
    ids, tgt = data[mode]
    tot = 0.0
    for i in range(0, len(ids), chunk):
        h = hidden(p, ids[i:i + chunk])
        z = (h[:, n + 3:] if mode == "step" else h[:, -1]) @ p["R"].T    # positions n+3.. hold s_1..
        tot = tot + torch.nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), tgt[i:i + chunk].reshape(-1),
                                                      reduction="sum")
    return tot / tgt.numel()


def low_degree_mass(n, K, p=0.25):
    lam = [abs(1 - p + p * complex(math.cos(2 * math.pi * j / K), math.sin(2 * math.pi * j / K))) for j in range(1, K)]
    return sum(l ** (2 * n) + n * l ** (2 * (n - 1)) * (1 - l * l) for l in lam) / K, max(lam)


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for n in (4, 6, 8):
        tilde, data = setup(n, K)
        A1, lam = low_degree_mass(n, K)
        s1 = lam / K
        E_val = tilde["wte"][A:A + K]
        P = torch.zeros(K, K, device=dev)                                 # P[s, y]
        for s in range(K):
            P[s, s] += 0.75; P[s, (s + 1) % K] += 0.25
        C = (P.T - 1 / K) / K
        lead, floor = torch.linalg.norm(C @ E_val).item(), s1 * torch.linalg.svdvals(E_val).min().item()
        print(f"K={K} n={n}  s1={s1:.4f}  sqrt(A_<=1)={math.sqrt(A1):.3e}  ||C E_val||_F={lead:.4f}  s1*smin={floor:.4f}")
        for sigma in (1e-2, 3e-3):
            p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
            LS, LO = loss(p, data, "step", n), loss(p, data, "outcome", n)
            gS = torch.autograd.grad(LS, list(p.values()))
            gR = gS[list(p).index("R")]
            gO = torch.autograd.grad(LO, list(p.values()))
            nS, nO = math.sqrt(sum(float((g ** 2).sum()) for g in gS)), math.sqrt(sum(float((g ** 2).sum()) for g in gO))
            gap = math.log(K) - float(LO)
            print(f"  sigma={sigma:.0e}  ||grad_R L_S||/sigma={gR.norm().item() / sigma:.4f}  "
                  f"||grad L_S||/sigma={nS / sigma:.3f}  ||grad L_O||/sigma={nO / sigma:.3e}  "
                  f"(logK-L_O)/sigma^2={gap / sigma ** 2:+.3e}  /(sigma^2 sqrtA)={gap / sigma ** 2 / math.sqrt(A1):+.3f}")
            # Euler gradient flow for time 1 on each loss
            for mode in ("step", "outcome"):
                q = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
                L0, dt = float(loss(q, data, mode, n)), 0.05
                out = []
                for it in range(1, 21):
                    L = loss(q, data, mode, n)
                    gr = torch.autograd.grad(L, list(q.values()))
                    with torch.no_grad():
                        for v, g_ in zip(q.values(), gr):
                            v -= dt * g_
                    if it in (10, 20):
                        out.append((it * dt, L0 - float(loss(q, data, mode, n))))
                print(f"     flow {mode:7s} " + "  ".join(f"t={t:.1f}: decrease/sigma^2={d / sigma ** 2:.4e}" for t, d in out), flush=True)


if __name__ == "__main__":
    main()
