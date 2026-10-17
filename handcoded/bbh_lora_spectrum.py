"""Degree split of the LoRA Jacobian of Qwen2.5-0.5B on BIG-Bench Hard object tracking (3, 5 and 7 objects).

    python handcoded/bbh_lora_spectrum.py K=3 mode=exact contexts=10 seed=2001 out=results/paper/bbh_lora_spectrum/s2001.json
    python handcoded/bbh_lora_spectrum.py K=3 mode=validate contexts=4 N=64 reps=20 seed=2001 out=results/paper/bbh_lora_spectrum/validate_K3.json
    python handcoded/bbh_lora_spectrum.py K=5 mode=mc contexts=8 N=64 seed=2001 out=results/paper/bbh_lora_spectrum/mc_K5.json

A context fixes the scene, the item list, the start assignment and the queried person (drawn as in training,
src/shuffled.py --format bbh). Its n = K moves each take one of V = 2 * C(K, 2) values (pair x name order, uniform,
as in training): V = 6, 20, 42 for K = 3, 5, 7. For each prompt we take the logits z of the K option letters at the
answer position (the token after "Answer: (") and their gradients f = dz/dB with respect to every LoRA up-projection
B. At the standard LoRA initialization B = 0, so the gradient with respect to A is exactly zero and the Jacobian is
that with respect to B. The base model is run in float32; LoRA is r = 8, alpha = 16, all-linear, as in training.

Two quantities, each split by Efron--Stein degree over the n moves at a fixed context:
  W_r   energy of the Jacobian, sum_{|S|=r} E||f_S||^2 (W_0 = ||E f||^2).
  g_r   degree-r outcome credit E[f_r^T a_r] (a vector over B), a = y - 1/K the centred one-hot target; the
        outcome gradient at init is g = sum_r g_r. We report ||g_r||.
The target split (A_r, and the conditional means a_t(v) = E[a | m_t = v]) is exact: E[y | m_t = v] =
M^{t-1} P_v M^{n-t} e_q, with P_v the transposition of move v and M the mean transposition.

mode=exact (K = 3 only): the full V^n grid and an orthonormal transform along the move axes, as in the paper.
mode=mc: unbiased Monte Carlo at a fixed context, accumulating running sums (J is never stored). Per base sample i
  draw independent move vectors A_i, B_i and, for each t, C_it = A_i with coordinate t taken from B_i; evaluate
  f at A_i, B_i and every C_it (n + 2 prompts).
    W_0       (||S||^2 - Q) / (M(M-1)) over the M = 2N iid prompts A, B (S = sum f, Q = sum ||f||^2).
    Var       (Q - ||S||^2/M) / (M-1);  also 0.5 ||f(A_i) - f(B_i)||^2 per i for a standard error.
    W_1       sum_t <f(B_i) - f(B_{i-1}), f(C_it) - f(A_i)> (Saltelli pick-freeze; C_it shares only m_t with B_i;
              the independent centre f(B_{i-1}) removes the mean term without bias).
    W_{>=2}   Var - W_1.
    g_total   mean of f^T a;  g_1 = mean of f^T sum_t (a_t(m_t) - E a) (exact, linear in f);  g_0 = (mean f)^T E a;
              g_{>=2} = g_total - g_0 - g_1. The degree >= 1 credits use f - c with c the mean of f over `pilot`
              independent prompts (a control variate: exact in expectation because a - E a and a_t(m_t) - E a have
              mean zero; it removes the degree-0 part of f, >99% of its energy, from the noise). Squared norms come from G independent groups of base samples,
              (||sum_g g_g||^2 - sum_g ||g_g||^2) / (G(G-1)), which is unbiased; the reported norm is its signed sqrt.
mode=validate (K = 3): the exact grid per context, then the same MC estimator run `reps` times by looking up rows of
  the exact grid, and once through the model, so estimator bias and spread are measured against the exact split.
"""
import ast
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shuffled import parse_bbh, sample, prompt, answer, pairs  # noqa: E402

model_name = "Qwen/Qwen2.5-0.5B"
K = 3
mode = "exact"
contexts = 10
seed = 2001
N = 64          # MC base samples per context (n + 2 prompts each)
G = 8           # independent groups for the credit norms
pilot = 8       # pilot prompts for the control variate c
nb = 64         # credit batches per context
bs = 32         # prompts per credit batch
reps = 20       # validate: MC repetitions on the exact grid
out = ""
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    globals()[k] = v if k in ("model_name", "out", "mode") else ast.literal_eval(v)

n, V = K, 2 * len(pairs(K))           # BBH uses K swaps for K objects
assert mode in ("exact", "mc", "validate", "checkbatch", "validcredit") and (mode in ("mc", "checkbatch") or K == 3)
dev = "cuda"
torch.manual_seed(seed)               # LoRA A initialization
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(dev)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                                         bias="none", target_modules="all-linear"))
model.eval()
params = [p for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name]
assert all(float(p.detach().abs().max()) == 0.0 for p in params), "LoRA B must start at zero"
P = sum(p.numel() for p in params)
letters = [tok("\nAnswer: (" + chr(65 + j) + ")", add_special_tokens=False)["input_ids"][4] for j in range(K)]
assert [tok.decode([t]) for t in letters] == [chr(65 + j) for j in range(K)]
suffix = tok("\nAnswer: (", add_special_tokens=False)["input_ids"]


def example(base, vals):
    return dict(base, moves=[int(m // 2) for m in vals], flips=[bool(m % 2) for m in vals])


def evaluate(base, vals):
    """(K, P) Jacobian of the option logits w.r.t. lora_B, and the logits, for one move sequence."""
    ex = example(base, vals)
    ids = tok(prompt(ex, K), add_special_tokens=True)["input_ids"] + suffix
    z = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1, letters]
    f = torch.empty(K, P, device=dev)
    for k in range(K):
        g = torch.autograd.grad(z[k], params, retain_graph=k < K - 1)
        f[k] = torch.cat([x.reshape(-1) for x in g])
    return f, z.detach()


# ---------------------------------------------------------------- exact target split

Pm = []
for (i, j) in pairs(K):
    T = torch.eye(K, dtype=torch.float64)
    T[[i, j]] = T[[j, i]]
    Pm.append(T)
Mbar = sum(Pm) / len(Pm)


def target_parts(q):
    """Exact E a (K,), conditional means h[t, v] = E[a | m_t = v] - E a (n, V, K), and A_0, A_1, A_total."""
    e = torch.zeros(K, dtype=torch.float64)
    e[q] = 1.0
    mp = [torch.linalg.matrix_power(Mbar, s) for s in range(n + 1)]
    abar = mp[n] @ e - 1.0 / K
    h = torch.zeros(n, V, K, dtype=torch.float64)
    for t in range(n):                 # move t+1 of n
        for v in range(V):
            h[t, v] = mp[t] @ Pm[v // 2] @ mp[n - 1 - t] @ e - 1.0 / K - abar
    A0 = float(abar @ abar)
    A1 = float((h ** 2).sum(-1).mean(1).sum())
    return abar, h, dict(A0=A0, A1=A1, A_ge2=(1 - 1 / K) - A0 - A1, total=1 - 1 / K)


def target(ex):
    y = torch.zeros(K, device=dev)
    y[ord(answer(ex, K)[1]) - 65] = 1.0
    return y


# ---------------------------------------------------------------- Monte Carlo estimator

def mc_estimate(fz_of, base, rng, Nb):
    """Unbiased degree-0/1/>=2 energies of f and degree-split credit norms at one context (see docstring)."""
    abar, h, _ = target_parts(base["q"])
    abar_d, h_d = abar.to(dev), h.to(dev)
    D = torch.float64
    Sg = torch.zeros(G, K, P, device=dev, dtype=D)       # per-group sum of f over A and B prompts
    Ge1 = torch.zeros(G, P, device=dev, dtype=D)         # (f - c)^T (a - E a): degrees >= 1
    G1 = torch.zeros(G, P, device=dev, dtype=D)          # (f - c)^T sum_t h_t(m_t): degree 1
    cnt = torch.zeros(G, device=dev, dtype=D)
    Q, half_sq, w1_i, w1_t = 0.0, [], [], np.zeros(n)
    # control variate: c is the mean of f over independent pilot prompts; since a - E a and h_t have mean zero,
    # subtracting a constant c from f leaves the degree >= 1 credits unbiased and removes the large degree-0 noise
    c = torch.stack([fz_of(rng.integers(V, size=n))[0] for _ in range(pilot)]).mean(0).double()

    def credit(vals, f, g):
        a = (target(example(base, vals)) - 1.0 / K).double()
        w1 = sum(h_d[t, vals[t]] for t in range(n))
        fd = f.double()
        fc = fd - c
        Ge1[g] += (a - abar_d) @ fc
        G1[g] += w1 @ fc
        Sg[g] += fd
        cnt[g] += 1

    prev = None
    while prev is None:               # independent centre for i = 0
        d = rng.integers(V, size=n)
        prev, _ = fz_of(d)
    for i in range(Nb):
        g = i % G
        A = rng.integers(V, size=n)
        B = rng.integers(V, size=n)
        fA, zA = fz_of(A)
        fB, zB = fz_of(B)
        for vals, f in ((A, fA), (B, fB)):
            credit(vals, f, g)
            Q += float((f.double() ** 2).sum())
        half_sq.append(0.5 * float(((fA - fB).double() ** 2).sum()))
        cen = (fB - prev).double()
        tot = 0.0
        for t in range(n):
            C = A.copy()
            C[t] = B[t]
            fC, _ = fz_of(C)
            x = float((cen * (fC - fA).double()).sum())
            w1_t[t] += x
            tot += x
        w1_i.append(tot)
        prev = fB
    M = 2 * Nb
    S = Sg.sum(0)
    S2 = float((S ** 2).sum())
    W0 = (S2 - Q) / (M * (M - 1))
    var = (Q - S2 / M) / (M - 1)
    W1 = float(np.mean(w1_i))
    hs, wi = np.array(half_sq), np.array(w1_i)
    se = lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))  # noqa: E731

    def sqnorm(groups):               # unbiased ||E||^2 from G independent group means
        m = groups / cnt[:, None]
        tot_ = m.sum(0)
        pair = [float(m[a] @ m[b]) for a, b in itertools.combinations(range(G), 2)]
        est = (float(tot_ @ tot_) - float((m ** 2).sum())) / (G * (G - 1))
        return est, float(np.std(pair, ddof=1) / math.sqrt(G))   # rough SE

    g0 = torch.einsum("k,gkp->gp", abar_d, Sg)
    cred = {}
    for name, arr in (("total", g0 + Ge1), ("deg0", g0), ("deg1", G1), ("deg_ge2", Ge1 - G1), ("deg_ge1", Ge1)):
        est, s = sqnorm(arr)
        cred[name] = dict(sq=est, sq_se=s, norm=math.copysign(math.sqrt(abs(est)), est))
    return dict(W=[W0, W1, var - W1], W_total=W0 + var, Var=var, Var_pair=float(hs.mean()), Var_pair_se=se(hs),
                W1_se=se(wi), W_ge2_se=se(hs - wi), W1_per_move=(w1_t / Nb).tolist(), credit_small=cred,
                _mean_f=(S / M).float())


def ustat(m):
    """Unbiased ||E x||^2 from nb iid rows of m (nb, P), and its jackknife standard error."""
    nb = m.shape[0]
    Gm = (m.double() @ m.double().T).cpu().numpy()
    off = Gm.sum() - np.trace(Gm)
    est = off / (nb * (nb - 1))
    loo = (off - 2 * (Gm.sum(1) - np.diag(Gm))) / ((nb - 1) * (nb - 2))
    return float(est), float(math.sqrt((nb - 1) / nb * ((loo - loo.mean()) ** 2).sum()))


def credit_batched(base, rng, nb, c):
    """Degree-split outcome credit from nb batches of bs iid prompts (one backward per degree per batch).

    The batch gradient of sum_i w_i . z_i is sum_i f_i^T w_i, so per-prompt Jacobians are never formed. Rows use
    left padding with explicit position ids; c (the mean of f from an independent pass) is the control variate.
    Returns the per-degree squared norms (unbiased U-statistic over batches) and the context-mean vectors."""
    abar, h, _ = target_parts(base["q"])
    abar_d, h_d = abar.float().to(dev), h.float().to(dev)
    cd = c.to(dev)
    vec = {k: torch.zeros(nb, P, device=dev) for k in ("deg0", "deg1", "deg_ge1")}
    for b in range(nb):
        vals = rng.integers(V, size=(bs, n))
        exs = [example(base, v) for v in vals]
        rows = [tok(prompt(e, K), add_special_tokens=True)["input_ids"] + suffix for e in exs]
        w = max(map(len, rows))
        ids = torch.full((bs, w), tok.pad_token_id)
        mask = torch.zeros(bs, w, dtype=torch.long)
        for r, x in enumerate(rows):
            ids[r, w - len(x):] = torch.tensor(x)
            mask[r, w - len(x):] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        z = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                  logits_to_keep=1).logits[:, -1, letters]
        if b == 0:                    # padding check against the single-prompt path
            z0 = evaluate(base, vals[0])[1]
            assert torch.allclose(z[0].detach(), z0, atol=2e-3), (z[0], z0)
        a = torch.stack([target(e) for e in exs]) - 1.0 / K
        w1 = h_d[torch.arange(n)[None, :], torch.tensor(vals)].sum(1)          # (bs, K)
        weights = dict(deg0=abar_d.expand(bs, K), deg1=w1, deg_ge1=a - abar_d)
        for j, (k, wt) in enumerate(weights.items()):
            g = torch.autograd.grad((z * wt).sum(), params, retain_graph=j < 2)
            v = torch.cat([x.reshape(-1) for x in g])
            if k != "deg0":
                v = v - wt.sum(0) @ cd
            vec[k][b] = v / bs
    vec["total"] = vec["deg0"] + vec["deg_ge1"]
    vec["deg_ge2"] = vec["deg_ge1"] - vec["deg1"]
    out_ = {}
    for k, m in vec.items():
        est, se_ = ustat(m)
        out_[k] = dict(sq=est, sq_se=se_, norm=math.copysign(math.sqrt(abs(est)), est))
    return out_, {k: m.mean(0).cpu() for k, m in vec.items()}


# ---------------------------------------------------------------- exact grid (K = 3)

def exact_grid(base):
    grid = np.array(list(itertools.product(range(V), repeat=n)))
    J = torch.zeros(V ** n, K, P, device=dev)
    Z = torch.zeros(V ** n, K, device=dev)
    Y = torch.zeros(V ** n, K, device=dev)
    for i, vals in enumerate(grid):
        J[i], Z[i] = evaluate(base, vals)
        Y[i] = target(example(base, vals))
    return grid, J, Z, Y


Qb = torch.linalg.qr(torch.cat([torch.ones(V, 1), torch.randn(V, V - 1, generator=torch.Generator().manual_seed(1))], 1).double())[0]
Qb = (Qb * math.sqrt(V) * torch.sign(Qb[0, 0])).to(dev)


def coefficients(f):
    """f: (V^n, ...) on the grid -> orthonormal coefficients (V^n, ...) and the degree of each."""
    shape = f.shape[1:]
    c = f.reshape((V,) * n + (-1,))
    for t in range(n):
        c = torch.tensordot(c, Qb.to(c.dtype), dims=([t], [0])) / V
        c = c.movedim(-1, t)
    grid = np.array(list(itertools.product(range(V), repeat=n)))
    return c.reshape(V ** n, *shape), torch.tensor((grid != 0).sum(1), device=dev)


def exact_split(J, Y, base):
    a = Y - 1.0 / K
    ca, order = coefficients(a.double())
    W = torch.zeros(n + 1, device=dev, dtype=torch.float64)
    Ar = torch.zeros(n + 1, device=dev, dtype=torch.float64).index_add_(0, order, (ca ** 2).sum(1))
    g = torch.zeros(n + 1, P, device=dev, dtype=torch.float64)
    for s in range(0, P, 1 << 17):                      # float64, in chunks over parameters
        cJ, _ = coefficients(J[:, :, s:s + (1 << 17)].double())
        W.index_add_(0, order, (cJ ** 2).sum((1, 2)))
        for r in range(n + 1):
            sel = order == r
            g[r, s:s + (1 << 17)] = torch.einsum("ikp,ik->p", cJ[sel], ca[sel])
        del cJ
    gt = torch.cat([torch.einsum("ikp,ik->p", J[:, :, s:s + (1 << 17)].double(), a.double())
                    for s in range(0, P, 1 << 17)]) / V ** n
    assert torch.allclose(g.sum(0), gt, rtol=1e-6, atol=1e-9 * float(gt.abs().max()))
    W = W.cpu().numpy()
    gn = g.norm(dim=1).cpu().numpy()
    return dict(W_r=W.tolist(), W=[W[0], W[1], W[2:].sum()], W_total=W.sum(), A_r=Ar.cpu().numpy().tolist(),
                credit=dict(total=float(gt.norm()), deg_r=gn.tolist(), deg0=gn[0], deg1=gn[1],
                            deg_ge2=float(g[2:].sum(0).norm()))), \
        dict(total=gt.float().cpu(), deg0=g[0].float().cpu(), deg1=g[1].float().cpu(), deg_ge2=g[2:].sum(0).float().cpu())


# ---------------------------------------------------------------- main

scenes, _ = parse_bbh(K)
rng = np.random.default_rng(seed)
if mode == "checkbatch":      # batched (left-padded) credit gradient vs the sum of single-prompt Jacobians
    base = sample(K, n, rng, scenes)
    vals = rng.integers(V, size=(bs, n))
    wts = torch.randn(bs, K, device=dev)
    single = sum(wts[i] @ evaluate(base, vals[i])[0] for i in range(bs))
    exs = [example(base, v) for v in vals]
    rows = [tok(prompt(e, K), add_special_tokens=True)["input_ids"] + suffix for e in exs]
    w = max(map(len, rows))
    ids = torch.full((bs, w), tok.pad_token_id)
    mask = torch.zeros(bs, w, dtype=torch.long)
    for r, x in enumerate(rows):
        ids[r, w - len(x):] = torch.tensor(x)
        mask[r, w - len(x):] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    z = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
              logits_to_keep=1).logits[:, -1, letters]
    batched = torch.cat([x.reshape(-1) for x in torch.autograd.grad((z * wts).sum(), params)])
    print("lengths", sorted(map(len, rows)), "rel. diff", float((batched - single).norm() / single.norm()))
    sys.exit()
results, ctx_vecs = [], []
t0 = time.time()
for c in range(contexts):
    base = sample(K, n, rng, scenes)
    _, _, tgt = target_parts(base["q"])
    ex0 = example(base, rng.integers(V, size=n))
    y0 = target(ex0).cpu().double()
    # check the Markov-chain target against src.shuffled.answer on one sequence
    e = torch.zeros(K, dtype=torch.float64)
    e[base["q"]] = 1.0
    for m in reversed(ex0["moves"]):
        e = Pm[m] @ e
    assert torch.equal(e, y0)
    row = dict(context=c, q=base["q"], target=tgt)
    crng = np.random.default_rng([seed, c, 7])
    if mode == "mc":
        r = mc_estimate(lambda v: evaluate(base, v), base, np.random.default_rng([seed, c]), N)
        row["credit"], vecs = credit_batched(base, crng, nb, r.pop("_mean_f"))
        ctx_vecs.append(vecs)
        row.update(r)
    else:
        grid, J, Z, Y = exact_grid(base)
        row["exact"], gvec = exact_split(J, Y, base)
        if mode == "validate":
            lookup = lambda v: (J[int(np.ravel_multi_index(tuple(v), (V,) * n))],                # noqa: E731
                                Z[int(np.ravel_multi_index(tuple(v), (V,) * n))])
            row["mc_grid"] = [mc_estimate(lookup, base, np.random.default_rng([seed, c, r_]), N) for r_ in range(reps)]
            for m in row["mc_grid"]:
                m.pop("_mean_f")
            mm = mc_estimate(lambda v: evaluate(base, v), base, np.random.default_rng([seed, c, 999]), N)
            del J
            row["credit"], vecs = credit_batched(base, crng, nb, mm.pop("_mean_f"))
            row["mc_model"] = mm
            row["credit_vec_rel_err"] = {k: float((vecs[k] - gvec[k]).norm() / gvec["total"].norm())
                                         for k in gvec}
            ctx_vecs.append(vecs)
        elif mode == "validcredit":   # the batched credit estimator, rows looked up in the exact grid, `reps` times
            a = Y - 1.0 / K
            abar = target_parts(base["q"])[0].float().to(dev)
            reps_ = []
            for r_ in range(reps):
                rg = np.random.default_rng([seed, c, 100 + r_])
                cc = J[torch.tensor(rg.integers(V ** n, size=2 * N), device=dev)].mean(0)    # control variate
                m = torch.stack([((a[ix] - abar)[:, :, None] * (J[ix] - cc)).sum((0, 1)) / bs
                                 for ix in [torch.tensor(rg.integers(V ** n, size=bs), device=dev) for _ in range(nb)]])
                reps_.append(ustat(m)[0])
            row["credit_lookup_sq"] = dict(mean=float(np.mean(reps_)), sd=float(np.std(reps_, ddof=1)),
                                           exact=row["exact"]["credit"]["deg_ge2"] ** 2, reps=reps_)
            print("   credit lookup sq:", {k: v for k, v in row["credit_lookup_sq"].items() if k != "reps"}, flush=True)
            del J
        else:
            del J
    results.append(row)
    W = np.array(row["exact"]["W"] if mode != "mc" else row["W"])
    cr = row["exact"]["credit"] if mode in ("exact", "validcredit") else {k: v["norm"] for k, v in row["credit"].items()}
    print(f"[{time.time() - t0:.0f}s] context {c}: W share (0,1,>=2) = {np.round(W / W.sum(), 6).tolist()} "
          f"W={W.sum():.4e}  credit total={cr['total']:.3e} deg0={cr['deg0']:.3e} deg1={cr['deg1']:.3e} "
          f"deg>=2={cr['deg_ge2']:.3e}", flush=True)
    if mode == "validate":
        mcW = np.array([m["W"] for m in row["mc_grid"]])
        print(f"   exact W = {np.round(W, 4).tolist()}  mc-grid W mean = {np.round(mcW.mean(0), 4).tolist()} "
              f"sd = {np.round(mcW.std(0), 4).tolist()}  mc-model W = {np.round(row['mc_model']['W'], 4).tolist()}\n"
              f"   exact credit = {row['exact']['credit']}\n   vector rel. err = {row['credit_vec_rel_err']}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


summary = dict(model=model_name, K=K, n=n, V=V, mode=mode, seed=seed, contexts=contexts, N=N, G=G, reps=reps,
               nb=nb, bs=bs, pilot=pilot,
               lora_B_params=P, per_context=results)
if mode not in ("exact", "validcredit"):
    Ws = np.array([r["W"] if mode == "mc" else r["mc_model"]["W"] for r in results])
    sh = Ws / Ws.sum(1, keepdims=True)
    summary["W_share_mean"], summary["W_share_sd"] = sh.mean(0).tolist(), sh.std(0, ddof=1).tolist() if contexts > 1 else None
    summary["W_share_min"], summary["W_share_max"] = sh.min(0).tolist(), sh.max(0).tolist()
    summary["W_var_share_ge2_mean"] = float((Ws[:, 2] / Ws[:, 1:].sum(1)).mean())
    summary["W_mean"] = Ws.mean(0).tolist()
    cred = {}
    for k in ("total", "deg0", "deg1", "deg_ge2"):
        sq = np.array([r["credit"][k]["sq"] for r in results])
        # pooled credit (mean over contexts, as in training): unbiased via per-context U-stats + cross terms
        cross = sum(float(ctx_vecs[a][k].double() @ ctx_vecs[b][k].double())
                    for a in range(contexts) for b in range(contexts) if a != b)
        pooled = (sq.sum() + cross) / contexts ** 2
        cred[k] = dict(per_context_norm_mean=float(np.mean(np.sign(sq) * np.sqrt(np.abs(sq)))),
                       per_context_norm_sd=float(np.std(np.sign(sq) * np.sqrt(np.abs(sq)), ddof=1)) if contexts > 1 else None,
                       per_context_sq_mean=float(sq.mean()), pooled_sq=pooled,
                       pooled_norm=math.copysign(math.sqrt(abs(pooled)), pooled))
    summary["credit"] = cred
    print("W share mean", np.round(sh.mean(0), 6).tolist(), "sd", np.round(sh.std(0), 6).tolist())
    print("credit", json.dumps(cred, indent=1))
if mode == "exact":           # keep the original summary fields
    Wm = np.mean([r["exact"]["W_r"] for r in results], 0)
    summary.update(W_r_mean=Wm.tolist(), share=(Wm / Wm.sum()).tolist())
if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(jsonable(summary), indent=1))
