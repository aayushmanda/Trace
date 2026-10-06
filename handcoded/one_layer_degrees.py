"""Per-degree Jacobian spectrum W_r of the one-layer attention model at initialization.

Same model as one_layer_spectrum.py (no MLP). theta = sigma * theta_tilde with one fixed
theta_tilde. Tests the all-degree envelope
    W_{r+1} / W_r  =  c (n - r)(r + 1) sigma^8 / N^2      (r >= 1),
i.e. each extra interacting letter costs sigma^4 in amplitude and one factor 1/N.
    uv run python handcoded/one_layer_degrees.py
"""
import math
import numpy as np
import torch
from torch.func import jvp

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, d, dirs = 4, 16, 8


def spectrum(n, sigma, seed=0):
    N, V_TOK = n + 2, A + 2
    words = torch.tensor((np.arange(A ** n)[:, None] // A ** np.arange(n)[::-1]) % A, device=dev)
    m = len(words)
    ids = torch.cat([torch.full((m, 1), A, device=dev), words, torch.full((m, 1), A + 1, device=dev)], 1)
    Qb = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
    Qb = (Qb * math.sqrt(A) * torch.sign(Qb[0, 0])).to(dev)
    order = (words != 0).sum(1)

    def energy(f):
        c = f.reshape((A,) * n + (-1,))
        for t in range(n):
            c = torch.tensordot(c, Qb, dims=([t], [0])) / A
            c = c.movedim(-1, t)
        return torch.zeros(n + 1, device=dev).index_add_(0, order, (c ** 2).sum(-1).reshape(-1))

    def forward(p):
        x = p["E"][ids] + p["P"][None]
        q = x[:, -1] @ p["Wq"].T; k = x @ p["Wk"].T; v = x @ p["Wv"].T
        a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(d), -1)
        return (x[:, -1] + (a.unsqueeze(-1) * v).sum(1)) @ p["R"].T

    g0 = torch.Generator(device=dev).manual_seed(seed)
    shapes = dict(E=(V_TOK, d), P=(N, d), Wq=(d, d), Wk=(d, d), Wv=(d, d), R=(2, d))
    tilde = {k: torch.randn(s, generator=g0, device=dev) for k, s in shapes.items()}
    th = {k: sigma * t for k, t in tilde.items()}
    W = torch.zeros(n + 1, device=dev)
    for _ in range(dirs):
        g = {k: torch.randn(t.shape, generator=g0, device=dev) for k, t in th.items()}
        W += energy(jvp(forward, (th,), (g,))[1]) / dirs
    return W.cpu().numpy(), N


if __name__ == "__main__":
    for n in (4, 6, 8):
        for sigma in (0.1, 0.2, 0.3):
            W, N = spectrum(n, sigma)
            line = f"n={n} sigma={sigma}:  W0={W[0]:.2e} W1={W[1]:.2e}"
            for r in range(1, min(n, 5)):
                ratio = W[r + 1] / W[r]
                c = ratio / ((n - r) * (r + 1) * sigma ** 8 / N ** 2)
                line += f" | W{r+1}/W{r}={ratio:.2e} c={c:.3g}"
            print(line, flush=True)
