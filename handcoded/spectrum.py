"""
Interaction spectrum of the real Transformer at initialization, one file, only torch.

Theorem 1 says the outcome gradient at uniform prediction is
    grad L_O = - sum_S E[J_S^T a_S],
where J = sum_S J_S is the Efron-Stein split of the answer-logit Jacobian over the
letters and a_S the same split of the centered answer. This script measures the
model side and computes the task side exactly, for the same architecture and the
same theta_0 that lettertrace.py trains (same seed => same init).

  model side   C_J(rho) = E <J(x), J(x_rho)>_F = sum_r rho^r W_r      (measured)
               x_rho: each letter kept w.p. rho, else resampled; s0, q kept
  task side    A_r = sum_{|S|=r} E ||a_S||^2                            (exact)
  bound        ||grad L_O|| <= sum_r sqrt(W_r A_r)
                            <= sqrt(C_J(rho)) * sqrt(sum_r rho^-r A_r)  for every rho
The last line needs no inversion of C_J, so it is the number to trust. W_r itself
is also fitted (nonnegative least squares) for the figure, as a description only.

    python handcoded/spectrum.py task=count word_len=8 mod=2
    python handcoded/spectrum.py task=first word_len=8
    python handcoded/spectrum.py ckpt=ckpt/<tag>_outcome_step2000.pt          # any checkpoint
    python handcoded/spectrum.py site=state ckpt=ckpt/<tag>_process_step2000.pt

site=answer: the answer position of an outcome prompt (what outcome training must learn).
site=state:  the position predicting s_n in a process row whose displayed states are free
             inputs (uniform, independent of the letters). This is the per-step function
             chain-of-thought training must learn; "externalized composition" predicts its
             letter spectrum stays at orders 0-1 while accuracy rises.
"""
import ast
import json
import math
import sys

import numpy as np
import torch
from torch.func import jacrev, vmap
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# config (key=value on the command line)

task = "count"
alphabet = 4
word_len = 8
mod = 2
n_blocks = 2
init_std = 0.02
seed = 42                       # same seed as lettertrace.py => same theta_0
rhos = [i / 10 for i in range(11)]
pairs = 512                     # (x, x_rho) pairs per rho
chunk = 32                      # examples per vmap'd Jacobian
grad_prompts = 4096             # prompts for the measured gradients
exact = True                    # enumerate every prompt for the gradients when feasible
max_exact = 2_000_000           # largest prompt count enumerated exactly (n=10 needs 8.4M)
trace_rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # wrong-trace gradient test (count, mod>=3)
out = ""                        # optional json path
ckpt = ""                       # load parameters from a lettertrace.py checkpoint
site = "answer"                 # "answer" or "state" (see docstring)
layout = "block"                # process rows: "block" (same prompt as outcome) or "stream"
zero_readout = False            # init mode only: readout = 0, so predictions are exactly uniform
grads_only = False              # skip C_J / W_r; only the two signals in fig:credit
device = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    assert key in globals(), f"unknown setting {key}"
    globals()[key] = value if key in ("task", "device", "out", "ckpt", "site", "layout") else ast.literal_eval(value)
assert task in ("count", "first") and site in ("answer", "state") and layout in ("block", "stream")
assert site == "answer" or task == "count", "site=state is defined for the count task"

n, A = word_len, alphabet
p_match = 1 / A
V = mod if task == "count" else n + 1
SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
VOCAB, P, H = A + V + 3, 2 * n + 6, 3
F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
W = 2 * A + 4 * V + P + 3                   # slot widths of lettertrace.py

# -----------------------------------------------------------------------------
# the architecture of lettertrace.py, and the same theta_0


def zero_params():
    p = {"wte": torch.zeros(VOCAB, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(VOCAB, W)}
    for b in range(n_blocks):
        p[f"{b}.q"], p[f"{b}.k"] = torch.zeros(H, P, W), torch.zeros(H, P, W)
        p[f"{b}.v"] = torch.zeros(H, W, W)
        p[f"{b}.mlp_in"], p[f"{b}.mlp_b"] = torch.zeros(F_WIDTH, W), torch.zeros(F_WIDTH)
        p[f"{b}.mlp_out"] = torch.zeros(W, F_WIDTH)
    return p


def forward(p, ids):
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


torch.manual_seed(seed)
theta = {k: (torch.randn_like(t) * init_std).to(device) for k, t in zero_params().items()}
step_label = "init"
if ckpt:
    saved = torch.load(ckpt, map_location=device)
    cfg = saved["config"]
    for k_, v_ in dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks).items():
        assert cfg[k_] == v_, f"checkpoint has {k_}={cfg[k_]}, this run has {v_}"
    assert cfg.get("layout", layout) == layout, f"checkpoint has layout={cfg['layout']}, this run has {layout}"
    assert set(saved["params"]) == set(theta), "checkpoint tensors do not match the architecture"
    theta = {k: saved["params"][k].to(device).float() for k in theta}
    step_label = f"{saved['mode']} step {saved['step']}"
elif zero_readout:
    theta["readout"].zero_()
n_params = sum(t.numel() for t in theta.values())

# -----------------------------------------------------------------------------
# prompts: letters i.i.d. uniform (no conditioning on a match, as in the theory)

rng = np.random.default_rng(seed + 1)


def answer(s0, q, word):
    b = word == q[:, None]
    if task == "count":
        return (s0 + b.sum(1)) % mod
    return np.where(b.any(1), b.argmax(1) + 1, 0)


def sample(m):
    q = rng.integers(A, size=m)
    word = rng.integers(A, size=(m, n))
    s0 = rng.integers(mod, size=m) if task == "count" else np.zeros(m, dtype=int)
    return s0, q, word


def outcome_ids(s0, q, word):
    """Prompt up to ':'; the ':' position n+3 predicts the answer."""
    m = len(q)
    cols = [A + s0[:, None], q[:, None], word, np.full((m, 1), SEP), np.full((m, 1), COLON)]
    return torch.tensor(np.concatenate(cols, 1), device=device)


# Positions that predict s_1..s_n in a process row.
#   block:  s0 q w_1 .. w_n SEP s_1 .. s_n     position n+1+t predicts s_t (SEP for t = 1)
#   stream: s0 q w_1 s_1 .. w_n s_n            position 2t (the letter w_t) predicts s_t
PRED = np.arange(1, n + 1) + (n + 1 if layout == "block" else np.arange(1, n + 1))


def trace_ids(s0, q, word, shown):
    """Process row with displayed states `shown` (m x k, k <= n), cut after the last state."""
    m, k = shown.shape
    if layout == "block":
        cols = [A + s0[:, None], q[:, None], word, np.full((m, 1), SEP), A + shown]
    else:
        cols = [A + s0[:, None], q[:, None]]
        for t in range(n):
            cols.append(word[:, t : t + 1])
            if t < k:
                cols.append(A + shown[:, t : t + 1])
    return torch.tensor(np.concatenate(cols, 1), device=device)


def process_rows(s0, q, word):
    """Clean process rows up to s_n; position PRED[t-1] predicts s_t."""
    b = word == q[:, None]
    states, s = [], s0.copy()
    for t in range(n):
        s = (s + b[:, t]) % mod if task == "count" else np.where((s == 0) & b[:, t], t + 1, s)
        states.append(s.copy())
    states = np.stack(states, 1)
    return trace_ids(s0, q, word, states), states

def sample_state(m):
    """Prompts for site=state: displayed states are free inputs, uniform and independent."""
    s0, q, word = sample(m)
    shown = rng.integers(mod, size=(m, n - 1))
    return s0, q, word, shown


def state_ids(s0, q, word, shown):
    """Process row with n-1 free displayed states; position PRED[-1] predicts
    s_n = s_{n-1} + [w_n = q]."""
    return trace_ids(s0, q, word, shown)


def state_target(s0, q, word, shown):
    prev = shown[:, -1] if n > 1 else s0
    return (prev + (word[:, -1] == q)) % mod


QPOS = n + 3 if site == "answer" else int(PRED[-1])
draw = sample if site == "answer" else sample_state
make_ids = (lambda d, w: outcome_ids(d[0], d[1], w)) if site == "answer" else \
           (lambda d, w: state_ids(d[0], d[1], w, d[3]))

# -----------------------------------------------------------------------------
# model side: C_J(rho) from per-example answer-logit Jacobians


def value_logits(params, ids):
    return forward(params, ids[None])[0, QPOS, A : A + V]


jac = vmap(jacrev(value_logits), in_dims=(None, 0))


def flat_jacobian(ids):
    j = jac(theta, ids)
    return torch.cat([t.reshape(ids.shape[0], V, -1) for t in j.values()], 2)


def stability(rho):
    """C_J(rho) and D(rho) = C_J(rho) - C_J(0) = sum_{r>=1} rho^r W_r. D uses one fresh
    word for both x_rho and x_0 (coupled), so the large order-0 energy cancels per pair."""
    c_vals, d_vals = [], []
    for start in range(0, pairs, chunk):
        m = min(chunk, pairs - start)
        d = draw(m)
        word = d[2]
        fresh = rng.integers(A, size=(m, n))
        word_rho = np.where(rng.random((m, n)) < rho, word, fresh)
        j = flat_jacobian(make_ids(d, word)).detach()
        j_rho = flat_jacobian(make_ids(d, word_rho)).detach()
        j_0 = flat_jacobian(make_ids(d, fresh)).detach()
        c_vals.append((j * j_rho).sum((1, 2)).double().cpu())
        d_vals.append((j * (j_rho - j_0)).sum((1, 2)).double().cpu())
    c, d = torch.cat(c_vals), torch.cat(d_vals)
    se = lambda v: v.std().item() / math.sqrt(len(v))
    return c.mean().item(), se(c), d.mean().item(), se(d)


# -----------------------------------------------------------------------------
# task side: exact answer spectrum A_r over the match bits


def state_spectrum():
    """Target s_{n-1} + b_n with s_{n-1} fixed: only orders 0 and 1 (the last letter)."""
    p = p_match
    e0, e1 = np.eye(V)[0], np.eye(V)[1 % V]
    a0 = (1 - p) * e0 + p * e1 - 1 / V
    A_st = np.zeros(n + 1)
    A_st[0] = (a0 ** 2).sum()
    A_st[1] = p * (1 - p) * ((e1 - e0) ** 2).sum()
    return A_st


def answer_spectrum():
    """Biased Fourier transform over b in {0,1}^n, b_t ~ Bern(p). The answer depends on
    the letters only through b, so its Efron-Stein energies over letters equal these."""
    assert n <= 18, "exact answer spectrum needs n <= 18"
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[::-1]) & 1
    y = answer(np.zeros(2 ** n, dtype=int), np.zeros(2 ** n, dtype=int), (1 - bits))  # q=0, letter 0 iff b=1
    f = np.eye(V)[y] - 1 / V
    f = f.reshape((2,) * n + (V,))
    c = math.sqrt(p_match * (1 - p_match))
    for t in range(n):
        f0, f1 = np.take(f, 0, axis=t), np.take(f, 1, axis=t)
        f = np.stack([(1 - p_match) * f0 + p_match * f1, c * (f1 - f0)], axis=t)
    energy = (f ** 2).sum(-1).reshape(-1)
    order = bits.sum(1)
    return np.array([energy[order == r].sum() for r in range(n + 1)])

# -----------------------------------------------------------------------------
# measured gradients at theta_0


def all_prompts():
    """Every (s0, q, word) with its probability, or None if there are too many."""
    n_s0 = mod if task == "count" else 1
    total = n_s0 * A * A ** n
    if not exact or total > max_exact:
        return None
    idx = np.arange(total)
    word = (idx[:, None] // A ** np.arange(n)[::-1]) % A
    q = (idx // A ** n) % A
    s0 = idx // (A ** (n + 1)) if task == "count" else np.zeros(total, dtype=int)
    return s0, q, word


PROMPTS = all_prompts()


def grad_vec(make_scalar, m_total, batch=256):
    """Population gradient as one flat vector (exact when PROMPTS is set, else Monte Carlo)."""
    grads = []
    source = PROMPTS if PROMPTS is not None else None
    total = len(PROMPTS[0]) if PROMPTS is not None else m_total
    step = 4096 if PROMPTS is not None else batch
    for start in range(0, total, step):
        if PROMPTS is not None:
            args = tuple(a[start:start + step] for a in PROMPTS)
        else:
            args = sample(min(batch, m_total - start))
        params = {k: t.clone().requires_grad_() for k, t in theta.items()}
        s = make_scalar(params, *args) * (len(args[0]) / total)
        grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
    return torch.stack(grads).sum(0)


def grad_norm(make_scalar, m_total, batch=256):
    """Norm of the population gradient. Exact (every prompt) when feasible; otherwise
    a Monte Carlo mean over batches, with a standard error of the norm."""
    grads = []
    if PROMPTS is not None:
        total = len(PROMPTS[0])
        for start in range(0, total, 4096):
            sl = slice(start, start + 4096)
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            s = make_scalar(params, *(a[sl] for a in PROMPTS)) * (len(PROMPTS[0][sl]) / total)
            grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
        return torch.stack(grads).sum(0).norm().item(), 0.0
    for start in range(0, m_total, batch):
        params = {k: t.clone().requires_grad_() for k, t in theta.items()}
        s = make_scalar(params, *sample(min(batch, m_total - start)))
        grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
    g = torch.stack(grads)
    mean = g.mean(0)
    se = math.sqrt(((g - mean) ** 2).sum().item() / (len(g) * (len(g) - 1)))
    return mean.norm().item(), se


def outcome_theory(params, s0, q, word):             # E[J^T (e_y - u)], u uniform over values
    z = forward(params, outcome_ids(s0, q, word))[:, n + 3, A : A + V]
    e = torch.eye(V, device=device)[torch.tensor(answer(s0, q, word), device=device)]
    return ((e - 1 / V) * z).sum(1).mean()


def outcome_residual(params, s0, q, word):           # E[J^T (p_v - u)]: non-uniformity at init
    logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
    pv = logits.softmax(-1)[:, A : A + V].detach()
    return ((pv - 1 / V) * logits[:, A : A + V]).sum(1).mean()


def outcome_full(params, s0, q, word):               # the actual answer-token loss
    logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
    return F.cross_entropy(logits, torch.tensor(A + answer(s0, q, word), device=device))


def outcome_nonvalue(params, s0, q, word):          # E[J_nv^T p_nv]: mass on non-answer tokens
    logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
    p = logits.softmax(-1).detach()
    mask = torch.ones(VOCAB, device=device)
    mask[A : A + V] = 0
    return (p * mask * logits).sum(1).mean()


def state_theory(params, s0, q, word, shown):        # E[J^T (e_target - u)] at site=state
    z = forward(params, state_ids(s0, q, word, shown))[:, QPOS, A : A + V]
    e = torch.eye(V, device=device)[torch.tensor(state_target(s0, q, word, shown), device=device)]
    return ((e - 1 / V) * z).sum(1).mean()


def process_theory(params, s0, q, word):             # E_t E[J_t^T (e_{s_t} - u)], clean process rows
    ids, states = process_rows(s0, q, word)
    z = forward(params, ids)[:, PRED, A : A + V]
    e = torch.eye(V, device=device)[torch.tensor(states, device=device)]
    return ((e - 1 / V) * z).sum(-1).mean()

def corrupted_rows(s0, q, word, corrupt):
    """Process rows of a trace that is wrong at every step, and the true successor F_t of each
    displayed previous state (the rule applied to what the model actually sees)."""
    b = word == q[:, None]
    shown, succ, s = [], [], s0.copy()
    for t in range(n):
        f = (s + b[:, t]) % mod
        eps = np.ones_like(f) if corrupt == "shift" else rng.integers(1, mod, size=len(f))
        s = (f + eps) % mod
        succ.append(f)
        shown.append(s.copy())
    shown, succ = np.stack(shown, 1), np.stack(succ, 1)
    return trace_ids(s0, q, word, shown), succ


def make_corrupt(corrupt):
    """E[J^T (R_F - u)] on corrupted contexts, with the step target replaced by its exact law
    given the context (Rao-Blackwell): scatter R_F = (K u - e_F)/(K-1), shift R_F = e_{F+1}."""
    def scalar(params, s0, q, word):
        ids, succ = corrupted_rows(s0, q, word, corrupt)
        z = forward(params, ids)[:, PRED, A : A + V]
        e = torch.eye(V, device=device)
        f = torch.tensor(succ, device=device)
        R = (1 - e[f]) / (V - 1) if corrupt == "scatter" else e[(f + 1) % V]
        return ((R - 1 / V) * z).sum(-1).mean()
    return scalar


# -----------------------------------------------------------------------------
# run

print(f"task={task} A={A} n={n}" + (f" K={mod}" if task == "count" else "") +
      f"  L={n_blocks} width={W} params={n_params:,}  site={site}  layout={layout}  weights: {step_label}"
      + (" (zero readout)" if zero_readout and not ckpt else ""))

A_r = answer_spectrum() if site == "answer" else state_spectrum()
if task == "count" and site == "answer":
    om = np.exp(2j * np.pi / mod)
    g2 = [abs(1 - p_match + p_match * om ** j) ** 2 for j in range(1, mod)]
    closed = [math.comb(n, r) * sum(x ** (n - r) * (1 - x) ** r for x in g2) / mod for r in range(n + 1)]
    assert np.allclose(A_r, closed), "answer spectrum disagrees with the closed form"
print("target spectrum A_r (exact):", np.round(A_r, 5).tolist(), f" total {A_r.sum():.4f}")

C = None
if grads_only:
    print("grads_only: skipping C_J / W_r")
else:
    C = [stability(r) for r in rhos]
    print("\n  rho    C_J(rho)     s.e.      D(rho)=C_J(rho)-C_J(0)   s.e.")
    for r, (c_, cse, d_, dse) in zip(rhos, C):
        print(f"  {r:4.2f}   {c_:.4e}  {cse:.1e}   {d_:+.3e}              {dse:.1e}")

W_r = a_ = b_ = hi_share = hi_share_se = best = best_upper = rho_best = None
if C is not None:
    X = np.array([[r ** k for k in range(1, n + 1)] for r in rhos if r > 0])
    y = np.array([c[2] for r, c in zip(rhos, C) if r > 0])
    try:
        from scipy.optimize import nnls
        W_hi = nnls(X, y)[0]
    except ImportError:
        W_hi = np.clip(np.linalg.lstsq(X, y, rcond=None)[0], 0, None)
    W0 = C[0][0]
    W_r = np.concatenate([[W0], W_hi])

    rr = np.array([r for r in rhos if r > 0])
    dd = np.array([c[2] for r, c in zip(rhos, C) if r > 0])
    ww = 1 / np.maximum(np.array([c[3] for r, c in zip(rhos, C) if r > 0]), 1e-30)
    Xq = np.c_[rr, rr ** 2] * ww[:, None]
    coef = np.linalg.lstsq(Xq, dd * ww, rcond=None)[0]
    cov = np.linalg.inv(Xq.T @ Xq)
    a_, b_ = coef
    grad_f = np.array([-b_, a_]) / (a_ + b_) ** 2
    hi_share, hi_share_se = b_ / (a_ + b_), float(np.sqrt(grad_f @ cov @ grad_f))
    print("\nmodel spectrum W_r (W_0 measured, W_r>=1 fitted, descriptive):", [f"{w:.2e}" for w in W_r])
    print(f"share of Jacobian energy at order 0: {W0 / C[-1][0]:.4f}")
    print(f"letter-dependent energy: order 1 ~ {a_:.3e}, orders >= 2 ~ {b_:.3e}; "
          f"order >= 2 share {hi_share:+.2f} +- {hi_share_se:.2f} (weighted fit)")

    tail = lambda r: sum(A_r[k] * r ** -k for k in range(1, n + 1))
    cands = [(math.sqrt(W0 * A_r[0]) + math.sqrt(max(c[2], 0.0) * tail(r)),
              math.sqrt(W0 * A_r[0]) + math.sqrt(max(c[2] + 2 * c[3], 0.0) * tail(r)), r)
             for r, c in zip(rhos, C) if r > 0]
    best, best_upper, rho_best = min(cands)
    print(f"\nbound sqrt(W_0 A_0) + sqrt(D(rho) sum_r>=1 rho^-r A_r): {best:.3e}  "
          f"(with D+2se: {best_upper:.3e}; best rho = {rho_best:.2f})")

mode = "exact over all prompts" if PROMPTS is not None else f"Monte Carlo, {grad_prompts} prompts"
grads = {}
if site == "answer":
    tgt = grad_vec(outcome_theory, grad_prompts)
    proc = grad_vec(process_theory, grad_prompts)
    grads = dict(grad_outcome=tgt.norm().item(), grad_process=proc.norm().item())
    print(f"gradients: {mode}")
    print(f"  answer-dependent  ||E[J_v^T(e_y-u)]|| (Theorem 1): {grads['grad_outcome']:.3e}")
    print(f"  process ||E_t E[J_t^T(e_st-u)]||:                  {grads['grad_process']:.3e}"
          f"   ratio process/outcome {grads['grad_process'] / max(grads['grad_outcome'], 1e-30):.1f}")
    if not grads_only:
        res = grad_vec(outcome_residual, grad_prompts)
        nv = grad_vec(outcome_nonvalue, grad_prompts)
        full = grad_vec(outcome_full, grad_prompts)
        err = (full - (res - tgt + nv)).norm().item() / max(full.norm().item(), 1e-30)
        grads.update(grad_residual=res.norm().item(), grad_nonvalue=nv.norm().item(),
                     grad_full=full.norm().item(), split_error=err)
        print(f"  answer-value residual ||E[J_v^T(p_v-u)]||:         {grads['grad_residual']:.3e}")
        print(f"  non-answer tokens ||E[J_nv^T p_nv]||:              {grads['grad_nonvalue']:.3e}")
        print(f"  full answer-token gradient:                        {grads['grad_full']:.3e}"
              f"   (full = residual - dependent + non-answer; relative error {err:.1e})")
        print(f"  answer-dependent share of the full gradient:       {grads['grad_outcome'] / grads['grad_full']:.2e}")
else:
    g_state, g_state_se = 0.0, 0.0
    vecs = []
    for start in range(0, grad_prompts, 256):
        params = {k: t.clone().requires_grad_() for k, t in theta.items()}
        sc = state_theory(params, *sample_state(min(256, grad_prompts - start)))
        vecs.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(sc, list(params.values()))]).double())
    vecs = torch.stack(vecs)
    mean = vecs.mean(0)
    g_state = mean.norm().item()
    g_state_se = math.sqrt(((vecs - mean) ** 2).sum().item() / (len(vecs) * (len(vecs) - 1)))
    grads = dict(grad_state=g_state, grad_state_se=g_state_se)
    print(f"state-position ||E[J^T(e_target-u)]||: {g_state:.3e}  (s.e. {g_state_se:.1e}, Monte Carlo)")

if out:
    payload = dict(task=task, A=A, n=n, K=mod, L=n_blocks, seed=seed, init_std=init_std,
                   layout=layout, A_r=A_r.tolist(), gradients=mode, site=site,
                   weights=step_label, ckpt=ckpt, zero_readout=zero_readout,
                   grads_only=grads_only, **grads)
    if C is not None:
        payload.update(rhos=rhos, C_J=[c[0] for c in C], C_J_se=[c[1] for c in C],
                       D=[c[2] for c in C], D_se=[c[3] for c in C], W_r=W_r.tolist(),
                       bound=best, bound_upper=best_upper, rho_best=rho_best,
                       W1_fit=a_, Whi_fit=b_, hi_share=hi_share, hi_share_se=hi_share_se)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)

# -----------------------------------------------------------------------------
# wrong traces at initialization: is the net process gradient mu(rho) times the clean one?

if task == "count" and mod >= 3 and site == "answer":
    g_clean, g_out_vec = proc, tgt
    cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()
    print("\nwrong traces at theta_0 (clean process gradient g_1; corrupted part g_R)")
    report = {}
    for corrupt in ("scatter", "shift"):
        g_R = grad_vec(make_corrupt(corrupt), grad_prompts)
        c = 1 / (mod - 1) if corrupt == "scatter" else 1.0
        print(f"  {corrupt}: cos(g_R, g_1) = {cos(g_R, g_clean):+.4f}  "
              f"||g_R||/||g_1|| = {g_R.norm().item() / g_clean.norm().item():.4f}"
              + (f"  (blind prediction: cos = -1, ratio = {1 / (mod - 1):.4f})" if corrupt == "scatter" else ""))
        rows = []
        for r in trace_rhos:
            g = r * g_clean + (1 - r) * g_R
            mu = r * (1 + c) - c
            rows.append(dict(rho=r, mu=mu, norm_ratio=g.norm().item() / g_clean.norm().item(),
                             cos_clean=cos(g, g_clean), vs_clean_outcome=g.norm().item() / g_out_vec.norm().item()))
            print(f"    rho={r:.2f}  mu={mu:+.3f}  ||g_rho||/||g_1||={rows[-1]['norm_ratio']:.4f}  "
                  f"cos(g_rho,g_1)={rows[-1]['cos_clean']:+.4f}  vs clean outcome x{rows[-1]['vs_clean_outcome']:.1f}")
        report[corrupt] = dict(cos_R=cos(g_R, g_clean), ratio_R=g_R.norm().item() / g_clean.norm().item(), rows=rows)
    ratio = g_out_vec.norm().item() / g_clean.norm().item()
    print(f"  noisy CoT beats clean outcome (scatter, blind) once rho > (1+(K-1)r)/K = {(1 + (mod - 1) * ratio) / mod:.4f},"
          f" r = ||g_O||/||g_1|| = {ratio:.2e}")
    if out:
        with open(out) as f:
            record = json.load(f)
        record["wrong_traces"] = report
        with open(out, "w") as f:
            json.dump(record, f, indent=1)