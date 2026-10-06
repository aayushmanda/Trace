"""
Numerical checks of the theorems against handcoded/lettertrace.py, count task only.

    python handcoded/theory_checks.py spectrum n=8 mod=2
    python handcoded/theory_checks.py credit   n=4,6,8 sigma=0.005,0.01,0.02
    python handcoded/theory_checks.py jacobian n=8 sigma=0.02
    python handcoded/theory_checks.py rule     n=4,6,8 sigma=0.02
    python handcoded/theory_checks.py plateau  n=8 sigma=0.005,0.01,0.02 lr=50 steps=20000

What each one tests
    spectrum  Eq. (4): A_r of the answer vs C(n,r) (1-1/K) lam^(2(n-r)); degree 0 is attained.
    jacobian  Fig. 3b: Jacobian mass W_r at init, via random JVPs, split by degree.
    credit    Thm 1(i): |grad| of the degree>=2 part of the answer at theta_0, vs sigma and n.
              Expect slope ~6 in log sigma and ~log(lam) per letter (or the sigma^10 floor).
    rule      Thm 2(ii): rule credit R(theta_0) with a fixed query, expect ~sigma^6 / n, no lam.
    plateau   Thm 1(ii): full-batch GD on the ANSWER token only, no clipping, large MLP biases.
              Logs time t = steps*lr, answer loss, |theta - theta_0|, max attention score,
              fraction of ReLUs that switched, E|yhat - u|. Exit time vs sigma tells you whether
              the sigma^-2 cap is real.

Uses the lettertrace architecture, whose widths grow with n, so the constants differ from Table 2
(handcoded/theorem_regime.py, fixed width 16); the exponents and n-trends are what to compare.

Conventions that match the theory, not the training script
    * float64 throughout: sigma^10 at sigma = 0.02 is 1e-17.
    * logits restricted to the K value tokens, so u = 1/K as in Section 4.1.
    * degree is Efron-Stein over the n letters, with start state and query fixed (Section 2).
    * mlp_bias sets every MLP bias to mlp_bias * sigma, as (A2) asks (theta_0 = sigma * theta~
      with |b~| >= 1), so mlp_bias=2 means b = 2 sigma.
"""
import itertools
import math
import sys

import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
import lettertrace as lt  # noqa: E402  (must import after the dtype is set)

# -----------------------------------------------------------------------------
# setup


def setup(n, mod, alphabet=4):
    lt.configure(task="count", word_len=n, mod=mod, alphabet=alphabet, device="cpu")


def lam():
    """Contraction coefficient: largest |eigenvalue| of the mean move off the uniform vector."""
    p = 1 / lt.A
    T = (1 - p) * torch.eye(lt.mod) + p * torch.roll(torch.eye(lt.mod), 1, dims=1)
    return sorted(torch.linalg.eigvals(T).abs().tolist())[-2] if lt.mod > 1 else 0.0


def init(sigma, mlp_bias=None):
    theta = {k: torch.randn_like(t) * sigma for k, t in lt.zero_params().items()}
    if mlp_bias is not None:
        for b in range(lt.n_blocks):
            theta[f"{b}.mlp_b"].fill_(mlp_bias * sigma)
    return theta


def words():
    return list(itertools.product(range(lt.A), repeat=lt.n))


def example(s0, q, w):
    states, s = [], s0
    for t, c in enumerate(w, 1):
        s = lt.step(s, t, c, q)
        states.append(s)
    return s0, q, w, states


def ids(examples, mode):
    return torch.tensor([lt.row(e, mode) for e in examples])[:, :-1]


def vlogits(p, x, pos):
    """K-way logits over the value tokens at row position pos (int or tensor of positions)."""
    return lt.forward(p, x)[:, pos, lt.A:lt.A + lt.V]


def flatgrad(p, loss_sum, rows, chunk=8192):
    """Gradient of (1/rows) sum_i loss_sum(slice) over chunks, as one flat vector."""
    total = None
    for i in range(0, rows, chunk):
        g = torch.autograd.grad(loss_sum(slice(i, i + chunk)), list(p.values()), allow_unused=True)
        g = torch.cat([(gi if gi is not None else torch.zeros_like(t)).flatten() for gi, t in zip(g, p.values())])
        total = g if total is None else total + g
    return total / rows


# -----------------------------------------------------------------------------
# Efron-Stein split over the letters


def degree_parts(f, n):
    """f has shape [A]*n + rest. Returns [f_0, ..., f_n], f_r the degree-r part."""
    parts = [f]
    for i in range(n):
        new = [torch.zeros_like(f) for _ in range(len(parts) + 1)]
        for d, g in enumerate(parts):
            m = g.mean(i, keepdim=True).expand_as(g)
            new[d] += m
            new[d + 1] += g - m
        parts = new
    return parts


def centred_answer(s0, q):
    """(y - u) for every word, shape [A]*n + [K]."""
    y = torch.tensor([example(s0, q, w)[3][-1] for w in words()])
    y = F.one_hot(y, lt.V).double() - 1 / lt.V
    return y.reshape(*[lt.A] * lt.n, lt.V)

# -----------------------------------------------------------------------------
# checks


def spectrum():
    K, l, n = lt.V, lam(), lt.n
    parts = degree_parts(centred_answer(0, 0), n)
    A = [x.pow(2).sum(-1).mean().item() for x in parts]
    bound = [math.comb(n, r) * (1 - 1 / K) * l ** (2 * (n - r)) for r in range(n + 1)]
    print(f"n={n} K={K} lam={l:.4f}   share at degrees 0,1: {sum(A[:2]) / sum(A):.2e}")
    for r in range(n + 1):
        print(f"  r={r}  A_r {A[r]:.3e}   bound {bound[r]:.3e}")


def credit(sigma, mlp_bias=None, seed=0):
    """Norms at theta_0 of the answer gradient split as degrees <=1 and >=2, start fixed and uniform."""
    torch.manual_seed(seed)
    p = {k: t.requires_grad_() for k, t in init(sigma, mlp_bias).items()}
    W, q, out = words(), 0, {}
    g = {"le1": 0, "ge2": 0}
    for s0 in range(lt.V):
        parts = degree_parts(centred_answer(s0, q), lt.n)
        targ = {"le1": parts[0] + parts[1], "ge2": sum(parts[2:])}
        x = ids([example(s0, q, w) for w in W], "outcome")
        for k, t in targ.items():
            t = t.reshape(-1, lt.V)
            # sum t = 0 per word, so the gradient of -<t, log yhat> is exactly -E[J^T t]
            f = lambda sl: -(t[sl] * vlogits(p, x[sl], lt.n + 3).log_softmax(-1)).sum()
            gk = flatgrad(p, f, len(W))
            g[k] = g[k] + gk / lt.V
            if s0 == 0:
                out[f"{k}_fixed"] = gk.norm().item()
    out.update(le1_uniform=g["le1"].norm().item(), ge2_uniform=g["ge2"].norm().item())
    return out


def jacobian(sigma, draws=4, mlp_bias=None, seed=0):
    """W_r = E |(dz/dtheta)_r|_F^2, estimated with random tangent vectors."""
    torch.manual_seed(seed)
    p = init(sigma, mlp_bias)
    x = ids([example(0, 0, w) for w in words()], "outcome")
    W = torch.zeros(lt.n + 1)
    for _ in range(draws):
        v = {k: torch.randn_like(t) for k, t in p.items()}
        _, jv = torch.func.jvp(lambda q: vlogits(q, x, lt.n + 3), (p,), (v,))
        parts = degree_parts(jv.reshape(*[lt.A] * lt.n, lt.V), lt.n)
        W += torch.tensor([y.pow(2).sum(-1).mean().item() for y in parts]) / draws
    return (W / W.sum()).tolist()


def rule(sigma, mlp_bias=None, seed=0):
    """R(theta_0): gradient of the rule part e_{s_t} - T^T e_{s_{t-1}}, t = 2..n, fixed query."""
    torch.manual_seed(seed)
    p = {k: t.requires_grad_() for k, t in init(sigma, mlp_bias).items()}
    P = 1 / lt.A
    T = (1 - P) * torch.eye(lt.V) + P * torch.roll(torch.eye(lt.V), 1, dims=1)
    ex = [example(s0, 0, w) for s0 in range(lt.V) for w in words()]
    x = ids(ex, "process")
    pos = torch.tensor([lt.n + 1 + t for t in range(2, lt.n + 1)])     # position predicting s_t
    s = torch.tensor([e[3] for e in ex])                                # s_1..s_n
    eye = torch.eye(lt.V)
    targ = eye[s[:, 1:]] - eye[s[:, :-1]] @ T                           # e_{s_t} - T^T e_{s_{t-1}}
    f = lambda sl: -(targ[sl] * vlogits(p, x[sl], pos).log_softmax(-1)).sum() / (lt.n - 1)
    return flatgrad(p, f, len(ex), chunk=4096).norm().item()


@torch.no_grad()
def probe(p, x):
    """Max |pre-softmax attention score| and the ReLU on/off pattern, over all blocks."""
    h, worst, pats = p["wte"][x] + p["wpe"][:x.shape[1]], 0.0, []
    fut = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
    for b in range(lt.n_blocks):
        q = torch.einsum("btw,hdw->bhtd", h, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", h, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", h, p[f"{b}.v"])
        a = q @ k.transpose(-1, -2) / math.sqrt(lt.P)
        worst = max(worst, a.masked_fill(fut, 0).abs().max().item())
        h = h + (a.masked_fill(fut, -math.inf).softmax(-1) @ v).sum(1)
        pre = h @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]
        pats.append((pre > 0).flatten())
        h = h + F.relu(pre) @ p[f"{b}.mlp_out"].T
    return worst, torch.cat(pats)


def plateau(sigma, lr, steps, mlp_bias=None, seed=0, eps=0.01, every=500, size=4096):
    """Full-batch GD on the answer token only. Returns the time the loss first drops eps below log K."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    W = words()
    pick = torch.randperm(len(W), generator=rng)[:size].tolist()
    ex = [example(s0, q, W[i]) for i in pick for s0 in range(lt.V) for q in [0]]
    x = ids(ex, "outcome")
    y = torch.tensor([e[3][-1] for e in ex])
    theta0 = init(sigma, mlp_bias)
    p = {k: t.clone().requires_grad_() for k, t in theta0.items()}
    pat0, logK = probe(p, x)[1], math.log(lt.V)
    for it in range(steps + 1):
        z = vlogits(p, x, lt.n + 3)
        loss = F.cross_entropy(z, y)
        if it % every == 0 or loss.item() < logK - eps:
            with torch.no_grad():
                dist = math.sqrt(sum((p[k] - theta0[k]).pow(2).sum().item() for k in p))
                gap = (z.softmax(-1) - 1 / lt.V).norm(dim=-1).mean().item()
                worst, pat = probe(p, x)
                flip = (pat != pat0).double().mean().item()
            print(f"  t={it * lr:9.1f}  loss {loss.item():.5f}  |th-th0| {dist:.3e}  "
                  f"max score {worst:.3e}  relu flips {flip:.3f}  E|yhat-u| {gap:.2e}")
            if loss.item() < logK - eps:
                return it * lr
        g = torch.autograd.grad(loss, list(p.values()), allow_unused=True)
        with torch.no_grad():
            for t, gi in zip(p.values(), g):
                if gi is not None:
                    t -= lr * gi
    return math.inf

# -----------------------------------------------------------------------------


def main():
    what, args = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    ns = [int(v) for v in args.get("n", "8").split(",")]
    sigmas = [float(v) for v in args.get("sigma", "0.02").split(",")]
    mod = int(args.get("mod", 2))
    bias = float(args["mlp_bias"]) if "mlp_bias" in args else None
    for n in ns:
        setup(n, mod)
        print(f"== {what}  n={n}  K={mod}  lam={lam():.4f}")
        for s in sigmas if what != "spectrum" else [None]:
            if what == "spectrum":
                spectrum()
            elif what == "credit":
                c = credit(s, bias)
                print(f"  sigma={s:g}  " + "  ".join(f"{k} {v:.3e}" for k, v in c.items())
                      + f"   sigma^6 lam^(n-3) {s ** 6 * lam() ** (n - 3):.3e}  sigma^10 {s ** 10:.3e}")
            elif what == "jacobian":
                print(f"  sigma={s:g}  W_r share: " + " ".join(f"{w:.2e}" for w in jacobian(s, mlp_bias=bias)))
            elif what == "rule":
                print(f"  sigma={s:g}  R {rule(s, bias):.3e}   sigma^6/n {s ** 6 / n:.3e}")
            elif what == "plateau":
                lr, steps = float(args.get("lr", 50)), int(args.get("steps", 20000))
                t = plateau(s, lr, steps, bias)
                print(f"  sigma={s:g}  exit time {t:.1f}   1/sigma^2 {s ** -2:.0f}   lam^-n {lam() ** -n:.0f}")


if __name__ == "__main__":
    main()
