"""H/W at initialization for the two-block architecture of lettertrace.py (Proposition B.x).

theta = sigma * theta_tilde, float64, answer logits (K = 2) at the ':' position, start state
and query fixed, all 4^n words. H/W is the share of answer-Jacobian energy above letter
degree 1, from the exact Efron--Stein spectrum over the words.

    uv run python handcoded/two_block_spectrum.py [n] [width] [bias] [seed]
        bias: bigbias  MLP biases of theta_tilde bounded away from zero (slope 10)
              bias     Gaussian MLP biases (a small one switches with the letters)
              nobias   zero MLP biases (no fixed activation pattern)
    uv run python handcoded/two_block_spectrum.py switch
        MLP units whose sign depends on the letters at the experiments' initialization
        (count, K=2, n=8, sigma=0.02, seeds 42-44).
"""
import itertools
import math
import sys

import torch

torch.set_default_dtype(torch.float64)


def spectrum(n=4, width=16, bias_mode="bigbias", seed=0, alphabet=4, K=2, heads=3, head_dim=8):
    A, F = alphabet, width
    vocab, SEP, COLON = A + K + 3, A + K, A + K + 1
    T = n + 4                                       # s0 q w1..wn SEP COLON
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    tilde = {"wte": rn(vocab, width), "wpe": rn(T, width), "R": rn(K, width)}
    for b in range(2):
        tilde[f"{b}.q"], tilde[f"{b}.k"] = rn(heads, head_dim, width), rn(heads, head_dim, width)
        tilde[f"{b}.v"] = rn(heads, width, width)
        tilde[f"{b}.mi"], tilde[f"{b}.mb"], tilde[f"{b}.mo"] = rn(F, width), rn(F), rn(width, F)
        if bias_mode == "bigbias":
            tilde[f"{b}.mb"] = torch.sign(tilde[f"{b}.mb"]) * (1 + tilde[f"{b}.mb"].abs())
        elif bias_mode == "nobias":
            tilde[f"{b}.mb"] = torch.zeros(F)
    names = list(tilde)
    shapes, sizes = [tilde[k].shape for k in names], [tilde[k].numel() for k in names]
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    m = len(words)
    ids = torch.cat([torch.full((m, 1), A), torch.zeros(m, 1, dtype=torch.long), words,
                     torch.full((m, 1), SEP), torch.full((m, 1), COLON)], 1)
    future = torch.ones(T, T, dtype=torch.bool).triu(1)

    def logits(flat):
        p = dict(zip(names, (t.view(s) for t, s in zip(flat.split(sizes), shapes))))
        x = p["wte"][ids] + p["wpe"]
        for b in range(2):
            q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(head_dim)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            x = x + torch.relu(x @ p[f"{b}.mi"].T + p[f"{b}.mb"]) @ p[f"{b}.mo"].T
        return x[:, -1] @ p["R"].T

    # orthonormal letter basis whose first row is constant
    Q, _ = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=g)], 1))
    Q = Q.T * torch.sign(Q.T[:, :1])
    degree = torch.tensor([sum(i != 0 for i in idx) for idx in itertools.product(range(A), repeat=n)])
    flat = torch.cat([tilde[k].flatten() for k in names])
    sigmas = [2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2] if bias_mode == "bigbias" else \
        [3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    print(f"n={n} width={width} {bias_mode} seed={seed}")
    prev = None
    for sigma in sigmas:
        f = torch.func.jacrev(logits)(sigma * flat).reshape((A,) * n + (-1,))
        for ax in range(n):
            f = torch.tensordot(Q, f, dims=([1], [ax])).movedim(0, ax)
        energy = (f.reshape(A ** n, -1) ** 2).sum(1)
        ratio = (energy[degree >= 2].sum() / energy.sum()).item()
        slope = "" if prev is None else f"  slope {math.log(ratio / prev[1]) / math.log(sigma / prev[0]):.2f}"
        print(f"  sigma={sigma:.1e}  H/W={ratio:.3e}  W/sigma^2={energy.sum().item() / m / sigma ** 2:.2f}{slope}")
        prev = (sigma, ratio)


def switching_units():
    sys.path.insert(0, "handcoded")
    import lettertrace as L
    for seed in (42, 43, 44):
        L.configure(task="count", word_len=8, mod=2, seed=seed, device="cpu")
        th = {k: t.double() for k, t in L.init_theta().items()}
        ids = torch.tensor([[L.A, 0, *w, L.SEP, L.COLON]
                            for w in itertools.product(range(L.A), repeat=L.n)])
        T = ids.shape[1]
        x = th["wte"][ids] + th["wpe"][:T]
        future = torch.ones(T, T, dtype=torch.bool).triu(1)
        counts = []
        for b in range(L.n_blocks):
            q = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, th[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(L.P)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            pre = x @ th[f"{b}.mlp_in"].T + th[f"{b}.mlp_b"]
            on = pre > 0
            counts.append(f"{(on.any(0) & ~on.all(0)).any(0).sum().item()}/{pre.shape[-1]}")
            x = x + torch.relu(pre) @ th[f"{b}.mlp_out"].T
        print(f"seed {seed}: MLP units switching with the letters, per block: {counts}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["switch"]:
        switching_units()
    else:
        a = sys.argv[1:]
        spectrum(n=int(a[0]) if a else 4, width=int(a[1]) if len(a) > 1 else 16,
                 bias_mode=a[2] if len(a) > 2 else "bigbias", seed=int(a[3]) if len(a) > 3 else 0)
