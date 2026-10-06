"""Exact letter-order spectra W_r of logit Jacobians (no fit), for the plateau bound.

For fixed (s0, q) the logits are a V-vector function on the 4^n letter grid. For a
Gaussian parameter direction g, E_g ||(J g)_r||^2 = ||J_r||_F^2, so W_r is estimated
without bias from forward-mode JVPs, and the Efron-Stein split of each J g is exact
(orthonormal transform along every letter axis under the uniform letter law).

    python handcoded/exact_spectrum.py site=answer dirs=16 ckpt=ckpt/n8_block_s42/s42_outcome_step1000.pt
    python handcoded/exact_spectrum.py site=every_k dirs=4 ckpt=ckpt/n8_block_k2/s42_process_step1000.pt

site=answer: outcome ':' position; prints W_r / W, A_r, H_1 / W, the exact answer-dependent
  gradient ||grad Phi|| (per (s0, q) and averaged), and the two terms of the plateau bound.
site=every_k: the position predicting s_{jk} with displayed states as free inputs; prints the
  energy share on subsets of the supervised window {(j-1)k+1..jk} by order, and outside it.
"""

import ast, math, sys
import numpy as np
import torch
from torch.func import jvp
from torch.nn import functional as F

ckpt = ""
site = "answer"          # answer | every_k
every_k = 2
shown_draws = 2
dirs = 16
seed = 0
init_std = 0.02          # init scale when no ckpt is given
init_seed = 42           # seed of theta_0 when no ckpt is given
n, A, mod, n_blocks = 8, 4, 2, 2
device = "cuda"
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    globals()[k] = v if k in ("ckpt", "device", "site") else ast.literal_eval(v)
torch.set_default_dtype(torch.float64)

V = mod
SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
VOCAB, P, H = A + V + 3, 2 * n + 6, 3
F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
W = 2 * A + 4 * V + P + 3
QPOS = n + 3


def forward(p, ids):  # copied from handcoded/spectrum.py
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    for b in range(n_blocks):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -math.inf)
        x = x + (att.softmax(-1) @ v).sum(1)
        x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
    return x @ p["readout"].T


if ckpt:
    saved = torch.load(ckpt, map_location=device)
    cfg = saved["config"]
    assert (cfg["task"], cfg["word_len"], cfg["alphabet"], cfg["mod"], cfg["n_blocks"]) == ("count", n, A, mod, n_blocks), cfg
    theta = {k: t.to(device).double() for k, t in saved["params"].items()}
    label = f"{saved['mode']} step {saved['step']}"
else:  # theta_0 of seed 42, as in spectrum.py
    torch.manual_seed(init_seed)
    shapes = {"wte": (VOCAB, W), "wpe": (P, W), "readout": (VOCAB, W)}
    for b in range(n_blocks):
        shapes |= {f"{b}.q": (H, P, W), f"{b}.k": (H, P, W), f"{b}.v": (H, W, W),
                   f"{b}.mlp_in": (F_WIDTH, W), f"{b}.mlp_b": (F_WIDTH,), f"{b}.mlp_out": (W, F_WIDTH)}
    theta = {k: (torch.randn(s, dtype=torch.float32) * init_std).double().to(device) for k, s in shapes.items()}
    label = f"init std={init_std} seed={init_seed}"

words = torch.tensor((np.arange(A ** n)[:, None] // A ** np.arange(n)[::-1]) % A, device=device)


def ids_for(s0, q):
    m = len(words)
    return torch.cat([torch.full((m, 1), A + s0, device=device), torch.full((m, 1), q, device=device),
                      words, torch.full((m, 1), SEP, device=device), torch.full((m, 1), COLON, device=device)], 1)


def logits(p, ids):
    return forward(p, ids)[:, QPOS, A:A + V]


# orthonormal basis for L2(uniform on A letters): column 0 constant
Qb = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
Qb = (Qb * math.sqrt(A) * torch.sign(Qb[0, 0])).to(device)   # E_x[phi_i phi_j] = delta_ij
ORDER = (words != 0).sum(1)   # word index doubles as the multi-index of coefficients


def order_energy(f):
    """f: (A^n, V) values on the grid -> energy per order under the uniform law."""
    c = f.reshape((A,) * n + (V,))
    for t in range(n):
        c = torch.tensordot(c, Qb, dims=([t], [0])) / A      # coefficient <f, phi_i>
        c = c.movedim(-1, t)
    e = (c ** 2).sum(-1).reshape(-1)
    return torch.zeros(n + 1, device=device).index_add_(0, ORDER, e)


def answer_onehot(s0, q):
    y = (s0 + (words == q).sum(1)) % mod
    return torch.eye(V, device=device)[y]


if site == "answer":
    gen = torch.Generator(device=device).manual_seed(seed)
    Wr = torch.zeros(n + 1, device=device)
    Ar = torch.zeros(n + 1, device=device)
    grad_phi = {k: torch.zeros_like(t) for k, t in theta.items()}
    combos = [(s0, q) for s0 in range(mod) for q in range(A)]
    for s0, q in combos:
        ids = ids_for(s0, q)
        Ar += order_energy(answer_onehot(s0, q) - 1 / V) / len(combos)
        for _ in range(dirs):
            g = {k: torch.randn(t.shape, generator=gen, device=device) for k, t in theta.items()}
            _, jg = jvp(lambda p: logits(p, ids), (theta,), (g,))
            Wr += order_energy(jg) / (len(combos) * dirs)
        p = {k: t.clone().requires_grad_() for k, t in theta.items()}
        phi = ((answer_onehot(s0, q) - 1 / V) * logits(p, ids)).sum(1).mean() / len(combos)
        gs = torch.autograd.grad(phi, list(p.values()))
        print(f"s0={s0} q={q} per-combo ||grad Phi|| {len(combos)*math.sqrt(sum((t**2).sum().item() for t in gs)):.4e}")
        for k, gk in zip(p, gs):
            grad_phi[k] += gk

    Wr, Ar = Wr.cpu().numpy(), Ar.cpu().numpy()
    gphi = math.sqrt(sum((t ** 2).sum().item() for t in grad_phi.values()))
    Wt = Wr.sum()
    print(f"== {label}")
    print("W_r / W :", " ".join(f"{x:.3e}" for x in Wr / Wt), f"  (W = {Wt:.4e})")
    print("A_r     :", " ".join(f"{x:.3e}" for x in Ar), f"  (sum = {Ar.sum():.4f})")
    print(f"order-0 share {Wr[0] / Wt:.6f}   H_0/W {Wr[1:].sum() / Wt:.3e}   H_1/W {Wr[2:].sum() / Wt:.3e}")
    print(f"||grad Phi|| exact {gphi:.4e}   sum_r sqrt(A_r W_r) {np.sqrt(Ar * Wr).sum():.4e}")
    for qq in range(3):
        lo = math.sqrt(Ar[:qq + 1].sum() * Wr[:qq + 1].sum()); hi = math.sqrt(Ar[qq + 1:].sum() * Wr[qq + 1:].sum())
        print(f"  q={qq}: sqrt(A<=q W<=q) {lo:.3e}   sqrt(A>q H_q) {hi:.3e}   bound {lo + hi:.3e}")
    raise SystemExit(0)

# every-k site: position n+1+j predicts v_j = s_{jk} from the displayed v_{j-1} (free input).
assert saved["config"].get("every_k", 1) == every_k if ckpt else True
k = every_k; NV = n // k
rng = np.random.default_rng(seed)
gen = torch.Generator(device=device).manual_seed(seed)


def state_logits(p, ids, pos):
    return forward(p, ids)[:, pos, A:A + V]


def ids_state(s0, q, shown):
    m = len(words)
    cols = [torch.full((m, 1), A + s0, device=device), torch.full((m, 1), q, device=device), words,
            torch.full((m, 1), SEP, device=device)]
    cols += [torch.full((m, 1), A + int(v), device=device) for v in shown]
    return torch.cat(cols, 1)


def subset_energy(f):
    """energy per coefficient multi-index (A^n,) summed over V"""
    c = f.reshape((A,) * n + (V,))
    for t in range(n):
        c = torch.tensordot(c, Qb, dims=([t], [0])) / A
        c = c.movedim(-1, t)
    return (c ** 2).sum(-1).reshape(-1)


rows = []
for j in range(1, NV + 1):
    win = list(range((j - 1) * k, j * k))                 # letter axes the target depends on
    nz = (words != 0)
    in_win_only = nz[:, [t for t in range(n) if t not in win]].sum(1) == 0
    win_order = nz[:, win].sum(1)
    masks = {"win_order%d" % r: (in_win_only & (win_order == r)) for r in range(k + 1)}
    masks["outside"] = ~in_win_only
    acc = {m: 0.0 for m in masks}; tot = 0.0; a_acc = {m: 0.0 for m in masks}
    count = 0
    for s0 in range(mod):
        for q in range(A):
            for _ in range(shown_draws if j > 1 else 1):
                shown = rng.integers(mod, size=j - 1)
                prev = shown[-1] if j > 1 else s0
                ids = ids_state(s0, q, shown)
                pos = n + 2 + (j - 1)
                y = (prev + (words[:, win] == q).sum(1)) % mod
                ea = subset_energy(torch.eye(V, device=device)[y] - 1 / V)
                for m, mk in masks.items():
                    a_acc[m] += ea[mk].sum().item()
                for _ in range(dirs):
                    g = {kk: torch.randn(t.shape, generator=gen, device=device) for kk, t in theta.items()}
                    _, jg = jvp(lambda p: state_logits(p, ids, pos), (theta,), (g,))
                    e = subset_energy(jg)
                    tot += e.sum().item() / dirs
                    for m, mk in masks.items():
                        acc[m] += e[mk].sum().item() / dirs
                count += 1
    rows.append((j, {m: v / tot for m, v in acc.items()}, {m: v / count for m, v in a_acc.items()}, tot / count))
print(f"== {label}  every_k={k}")
for j, wsh, a, wt in rows:
    print(f"v{j}: W share " + " ".join(f"{m}={v:.3e}" for m, v in wsh.items()) + f"  | A " + " ".join(f"{m}={v:.3e}" for m, v in a.items() if v > 1e-12) + f"  | W={wt:.3e}")
