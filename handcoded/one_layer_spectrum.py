"""H/W at initialization for a one-layer softmax-attention model with a linear readout.

Positions: query (fixed token), n letters (4-letter alphabet), answer position (fixed token).
x_i = E[tok_i] + P[i];  h = x_ans + sum_j a_j V x_j,  a = softmax_j(q.k_j / sqrt d) over all
positions j <= ans;  z = R h (2 logits). Every tensor ~ N(0, sigma^2).
Exact Efron-Stein split of J g over the 4^n letter grid, averaged over random directions g.
    python handcoded/one_layer_spectrum.py n=6 d=16 dirs=16 mlp=0
mlp=1 inserts h <- h + W2 relu(W1 h) ; mlp=2 uses tanh instead of relu.
"""
import ast, math, sys
import numpy as np, torch
from torch.func import jvp
n, d, dirs, seed, mlp = 6, 16, 16, 0, 0
sigmas = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
for a_ in sys.argv[1:]:
    k, v = a_.split("="); globals()[k] = ast.literal_eval(v)
torch.set_default_dtype(torch.float64); dev = "cuda"
A = 4; V_TOK = A + 2; N = n + 2           # tokens: 4 letters, query token, answer token
words = torch.tensor((np.arange(A ** n)[:, None] // A ** np.arange(n)[::-1]) % A, device=dev)
ids = torch.cat([torch.full((len(words), 1), A, device=dev), words, torch.full((len(words), 1), A + 1, device=dev)], 1)
Qb = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
Qb = (Qb * math.sqrt(A) * torch.sign(Qb[0, 0])).to(dev)
ORDER = (words != 0).sum(1)

def energy(f):
    c = f.reshape((A,) * n + (-1,))
    for t in range(n):
        c = torch.tensordot(c, Qb, dims=([t], [0])) / A
        c = c.movedim(-1, t)
    e = (c ** 2).sum(-1).reshape(-1)
    return torch.zeros(n + 1, device=dev).index_add_(0, ORDER, e)

def forward(p, ids):
    x = p["E"][ids] + p["P"][None]
    q = x[:, -1] @ p["Wq"].T; k = x @ p["Wk"].T; v = x @ p["Wv"].T
    a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(d), -1)
    h = x[:, -1] + (a.unsqueeze(-1) * v).sum(1)
    if mlp == 1: h = h + torch.relu(h @ p["W1"].T) @ p["W2"].T
    if mlp == 2: h = h + torch.tanh(h @ p["W1"].T) @ p["W2"].T
    return h @ p["R"].T

for sg in sigmas:
    g0 = torch.Generator(device=dev).manual_seed(seed)
    shapes = dict(E=(V_TOK, d), P=(N, d), Wq=(d, d), Wk=(d, d), Wv=(d, d), R=(2, d))
    if mlp: shapes |= dict(W1=(4 * d, d), W2=(d, 4 * d))
    th = {k: torch.randn(s, generator=g0, device=dev) * sg for k, s in shapes.items()}
    Wr = torch.zeros(n + 1, device=dev)
    for _ in range(dirs):
        g = {k: torch.randn(t.shape, generator=g0, device=dev) for k, t in th.items()}
        _, jg = jvp(lambda p: forward(p, ids), (th,), (g,))
        Wr += energy(jg) / dirs
    Wr = Wr.cpu().numpy(); tot = Wr.sum()
    print(f"sigma={sg:g}  H/W={Wr[2:].sum()/tot:.3e}  deg1/W={Wr[1]/tot:.3e}  W={tot:.3e}")
