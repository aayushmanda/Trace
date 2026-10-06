"""Set-sums T_r = sum_{|S|=r} ||J_S||_{L2} of the answer-logit Jacobian, by degree.

One layer (the model of one_layer_degrees.py) and two blocks (the model of
two_block_spectrum.py, MLP biases bounded away from zero), theta = sigma * theta_tilde,
exact over all 4^n words, float64. J_S is the Efron-Stein component of the Jacobian on the
letter set S; ||J_S||^2 is estimated with random parameter directions.
The degree-resolved credit bound is  credit_r <= lambda^{n-r} T_r,  and the paper proves
T_r <= C sigma^2 kappa^{r-1} with kappa = C1 sigma^2 for two blocks (exponent 2r in sigma).
Measured exponents for two blocks: 2, 6, 7, 11, 12 at r = 1..5 (sigma in 0.03..0.05).
    uv run python handcoded/degree_setsums.py 6 0.03 0.04 0.05
    uv run python handcoded/degree_setsums.py [n]
"""
import itertools
import math
import sys

import torch
from torch.func import jvp

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, K, Wd, Hh, Dh, F, DIRS = 4, 2, 16, 3, 8, 16, 8


def models(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    m = len(words)
    # one layer: query token, n letters, answer token
    ids1 = torch.cat([torch.full((m, 1), A), words, torch.full((m, 1), A + 1)], 1).to(dev)
    t1 = dict(E=rn(A + 2, Wd), P=rn(n + 2, Wd), Wq=rn(Wd, Wd), Wk=rn(Wd, Wd), Wv=rn(Wd, Wd), R=rn(K, Wd))

    def f1(p):
        x = p["E"][ids1] + p["P"][None]
        q = x[:, -1] @ p["Wq"].T; k = x @ p["Wk"].T; v = x @ p["Wv"].T
        a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(Wd), -1)
        return (x[:, -1] + (a.unsqueeze(-1) * v).sum(1)) @ p["R"].T

    # two blocks: s0 q w1..wn SEP ':'
    vocab, T = A + K + 3, n + 4
    ids2 = torch.cat([torch.full((m, 1), A), torch.zeros(m, 1, dtype=torch.long), words,
                      torch.full((m, 1), A + K), torch.full((m, 1), A + K + 1)], 1).to(dev)
    t2 = {"wte": rn(vocab, Wd), "wpe": rn(T, Wd), "R": rn(K, Wd)}
    for b in range(2):
        t2[f"{b}.q"], t2[f"{b}.k"], t2[f"{b}.v"] = rn(Hh, Dh, Wd), rn(Hh, Dh, Wd), rn(Hh, Wd, Wd)
        t2[f"{b}.mi"], t2[f"{b}.mo"] = rn(F, Wd), rn(Wd, F)
        mb = rn(F); t2[f"{b}.mb"] = torch.sign(mb) * (1 + mb.abs())
    future = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)

    def f2(p):
        x = p["wte"][ids2] + p["wpe"]
        for b in range(2):
            q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(Dh)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            x = x + torch.relu(x @ p[f"{b}.mi"].T + p[f"{b}.mb"]) @ p[f"{b}.mo"].T
        return x[:, -1] @ p["R"].T

    return ({k: v.to(dev) for k, v in t1.items()}, f1), ({k: v.to(dev) for k, v in t2.items()}, f2)


def setsums(tilde, f, n, sigma, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=torch.Generator().manual_seed(1))], 1))
    Q = (Q.T * torch.sign(Q.T[:, :1])).to(dev)            # rows orthonormal, row 0 constant
    idx = torch.tensor(list(itertools.product(range(A), repeat=n)), device=dev)
    support = ((idx != 0).long() * (2 ** torch.arange(n, device=dev))).sum(1)   # set S as a bitmask
    th = {k: sigma * v for k, v in tilde.items()}
    e_S = torch.zeros(2 ** n, device=dev)
    for _ in range(DIRS):
        d = {k: torch.randn(v.shape, generator=g, device=dev) for k, v in th.items()}
        c = jvp(f, (th,), (d,))[1].reshape((A,) * n + (-1,))
        for ax in range(n):
            c = torch.tensordot(Q, c, dims=([1], [ax])).movedim(0, ax)
        e = (c.reshape(A ** n, -1) ** 2).sum(1) / A ** n / DIRS
        e_S.index_add_(0, support, e)
    size = torch.tensor([bin(s).count("1") for s in range(2 ** n)], device=dev)
    return [float(e_S[size == r].sqrt().sum()) for r in range(n + 1)]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    (t1, f1), (t2, f2) = models(n)
    sig = [float(x) for x in sys.argv[2:5]] if len(sys.argv) > 4 else [0.1, 0.2, 0.3]
    for name, tilde, f in (("one layer ", t1, f1), ("two blocks", t2, f2)):
        T = {s: setsums(tilde, f, n, s) for s in sig}
        for r in range(1, 6):
            e1 = math.log(T[sig[1]][r] / T[sig[0]][r]) / math.log(sig[1] / sig[0])
            e2 = math.log(T[sig[2]][r] / T[sig[1]][r]) / math.log(sig[2] / sig[1])
            print(f"{name} n={n} r={r}:  T_r(sigma={sig[0]})={T[sig[0]][r]:.3e}  sigma-exponent {e1:.2f}, {e2:.2f}", flush=True)
