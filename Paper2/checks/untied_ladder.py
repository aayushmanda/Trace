"""Untied one-block training from three initialization families (the Paper 1 ladder).

Every matrix entry of the one-block network (query, key, value, output, both MLP layers with bias, readout) is an
independent trainable coordinate. Exact population (16 rows of D_kappa), binary cross-entropy, full-batch GD.
Runs are batched along a leading run axis, so one GD step updates every run at once.

    A  scaffold: the directions of Section 2.1 with magnitudes U[1/2,2]; rule factors eps*U[1/2,2] with random signs,
       independent per MLP unit; scores start at zero (uniform attention).
    B  scaffold + rho * fan-in Gaussian noise on every matrix (rho in --rho).
    C  fan-in Gaussian initialization of every matrix scaled by eps, 16-unit MLP, nothing hand-coded.
    Cp fan-in Gaussian, nothing hand-coded, but only routing and the MLP small: W_Q, W_V, W_O ~ Z/sqrt(fan-in),
       W_K, W_1, W_2 ~ eps Z/sqrt(fan-in), 16-unit MLP. Variants (looped like rho for B):
       --cp_bias gate: b_1 = -B 1 (B = --bound); rand: b_1 ~ U(-1/sqrt(7), 1/sqrt(7)) (the nn.Linear default).
       --cp_readout one: u ~ Z/sqrt(7); eps: u ~ eps Z/sqrt(7).

Tested prediction (Theorem 1(iii)): kappa = 1 routes; kappa = 0 is delayed by 1/eps; for kappa < 0 routing fails below
the barrier eps^2 ~ -F(q0) and returns above it. The tied cross-entropy reduced flow gives the reference boundary.

    python Paper2/checks/untied_ladder.py --n 16 --out Paper2/checks/results/untied_ladder_n16.json
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--eta", type=float, default=0.1)
ap.add_argument("--law", default="shared", choices=["shared", "anti"],
                help="shared: D_kappa of Definition 2 (one decoy value w). anti: m decoys equal -x_N, the rest iid")
ap.add_argument("--kappa", type=float, nargs="+", default=[1., .5, 0., -.02, -.05, -.1, -.2, -.4])
ap.add_argument("--m", type=int, nargs="+", default=[0, 1, 2], help="anti law: number of anti-copies (kappa = 1 - m)")
ap.add_argument("--eps", type=float, nargs="+", default=[.003, .01, .03, .1, .2, .3, .5, 1.])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--levels", nargs="+", default=["A", "B", "C"])
ap.add_argument("--rho", type=float, nargs="+", default=[.1, .3])
ap.add_argument("--cp_bias", nargs="+", default=["gate", "rand"], choices=["gate", "rand"])
ap.add_argument("--cp_readout", nargs="+", default=["one", "eps"], choices=["one", "eps"])
ap.add_argument("--check_grad", action="store_true",
                help="print the population gradients of each matrix at the first initialization of each spec "
                     "with W_K set to zero (correlation loss and BCE, BCE minus label-marginal fitting), then exit")
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=400000, help="cap on GD iterations")
ap.add_argument("--t_max", type=float, default=30000., help="flow-time budget per run")
ap.add_argument("--bound", type=float, default=4.)
ap.add_argument("--hidden_c", type=int, default=16)
ap.add_argument("--opt", default="gd", choices=["gd", "adam"],
                help="gd: backtracking GD on the flow clock; adam: Adam with step size --lr, clock counts steps")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)
D = 7


def population(n, kappa, eta):
    rho = (1 - kappa) / (n - 1)
    rows = [(x, w, b, b * x * e, (1 - rho * x * w) / 8 * (eta if e == -1 else 1 - eta))
            for x, w, b, e in itertools.product((-1., 1.), repeat=4)]
    return torch.tensor(rows).T


def embed(x, w, b, n):
    e = torch.zeros(x.numel(), n + 1, D)
    e[:, :n - 1, 0], e[:, n - 1, 0], e[:, :, 2] = w[:, None], x, 1
    e[:, :n - 1, 4], e[:, n - 1, 3], e[:, n, 1] = 1, 1, b
    return e


def law_rows(law, par, n, eta):
    """Rows of one law: embeddings, label, clean target, uniform-read reverse rule, routing mask, probability.

    shared: par = kappa, the 16 rows of D_kappa. anti: par = m; decoys are m copies of -x_N and n-1-m iid uniform
    signs (k of them +1), so kappa = 1 - m, and for m = 1 the source is independent of the attended sum."""
    if law == "shared":
        x, w, b, y, p = population(n, par, eta)
        e = embed(x, w, b, n)
        return e, y, b * x, -b * w, x != w, p
    m = int(par)
    E, Y, C, Rv, P = [], [], [], [], []
    for x, k, b, err in itertools.product((-1., 1.), range(n - m), (-1., 1.), (-1., 1.)):
        vals = [-x] * m + [1.] * k + [-1.] * (n - 1 - m - k)
        e = torch.zeros(n + 1, D)
        e[:n - 1, 0], e[n - 1, 0], e[:, 2] = torch.tensor(vals), x, 1
        e[:n - 1, 4], e[n - 1, 3], e[n, 1] = 1, 1, b
        E.append(e); Y.append(b * x * err); C.append(b * x); Rv.append(-b * float(np.sign(x + sum(vals))))
        P.append(math.comb(n - 1 - m, k) / 2 ** (n - 1 - m) / 4 * (eta if err == -1 else 1 - eta))
    T = lambda v: torch.tensor(v)
    return torch.stack(E), T(Y), T(C), T(Rv), torch.ones(len(Y), dtype=torch.bool), T(P)


def init_run(level, eps, rho, seed, H, bias="gate", readout="one"):
    g = torch.Generator().manual_seed(seed)
    U = lambda *s: .5 + 1.5 * torch.rand(*s, generator=g)
    sign = lambda: float(torch.randint(2, (1,), generator=g) * 2 - 1)
    gauss = lambda *s: torch.randn(*s, generator=g)
    th = dict(Wq=torch.zeros(D, D), Wk=torch.zeros(D, D), Wv=torch.zeros(D, D), Wo=torch.zeros(D, D),
              W1=torch.zeros(H, D), b1=torch.zeros(H), W2=torch.zeros(D, H), u=torch.zeros(D))
    if level == "C":
        for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            th[k] = eps * gauss(*th[k].shape) / math.sqrt(th[k].shape[1])
        th["u"] = eps * gauss(D) / math.sqrt(D)
        return th
    if level == "Cp":
        for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            th[k] = (eps if k in ("Wk", "W1", "W2") else 1.) * gauss(*th[k].shape) / math.sqrt(th[k].shape[1])
        th["b1"] = -args.bound * torch.ones(H) if bias == "gate" else (2 * torch.rand(H, generator=g) - 1) / math.sqrt(D)
        th["u"] = (eps if readout == "eps" else 1.) * gauss(D) / math.sqrt(D)
        return th
    B = args.bound
    th["Wq"][0, 2], th["Wv"][0, 0], th["Wo"][5, 0], th["u"][6] = U(1), U(1), U(1), U(1)
    s1, s2 = sign(), sign()
    th["W1"][:4, 5] = s1 * eps * U(4) * torch.tensor([1., -1., 1., -1.])
    th["W1"][:4, 1] = B * torch.tensor([1., 1., -1., -1.])
    th["b1"][:4] = -B
    th["W2"][6, :4] = s2 * eps * U(4) * torch.tensor([1., -1., -1., 1.])
    if level == "B":
        for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            th[k] = th[k] + rho * gauss(*th[k].shape) / math.sqrt(th[k].shape[1])
        th["u"] = th["u"] + rho * gauss(D) / math.sqrt(D)
    return th


def forward(th, e, n):
    eq, ek = e[:, n], e[:, :n]
    q = torch.einsum("bd,red->rbe", eq, th["Wq"])
    k = torch.einsum("bnd,red->rbne", ek, th["Wk"])
    a = torch.softmax(torch.einsum("rbe,rbne->rbn", q, k), -1)
    v = torch.einsum("bnd,red->rbne", ek, th["Wv"])
    h = eq + torch.einsum("rbe,rde->rbd", torch.einsum("rbn,rbne->rbe", a, v), th["Wo"])
    pre = torch.einsum("rbd,rhd->rbh", h, th["W1"]) + th["b1"][:, None]
    h = h + torch.einsum("rbh,rdh->rbd", torch.relu(pre), th["W2"])
    return torch.einsum("rbd,rd->rb", h, th["u"]), a[..., n - 1], (pre > 0).to(pre.dtype).mean(-1)


def tied_ce_routes(n, kappa, eta, eps, t_max):
    """Tied cross-entropy reduced flow (Appendix A.6): routing time, or None if not routed by t_max."""
    d, c = (1 - kappa) / (n - 1), 1 - 2 * eta
    def rhs(t, s):
        m, q = s
        a = 1 / (1 + (n - 1) * math.exp(-q)); f = 2 * a - 1
        A = c * ((1 + d) * a - d) - (1 - d) / 2 * math.tanh(m / 2) - (1 + d) / 2 * f * math.tanh(m * f / 2)
        return [m * A, m * (1 + d) * a * (1 - a) * (c - math.tanh(m * f / 2))]
    ev = lambda t, s: s[1] - math.log(n - 1)
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0, t_max), (eps * eps, 0.), events=ev, rtol=1e-9, atol=1e-15, method="LSODA")
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else None


def main():
    n = args.n
    pars = args.kappa if args.law == "shared" else args.m
    segs, rows = {}, []
    for par in pars:
        r = law_rows(args.law, par, n, args.eta)
        segs[par] = (sum(len(q[1]) for q in rows), len(r[1]))
        rows.append(r)
    e, y, clean, decoy, differ, _ = (torch.cat([r[i] for r in rows]).to(dev) for i in range(6))

    def weights(par):
        w = torch.zeros(len(y))
        o, L = segs[par]
        w[o:o + L] = rows[pars.index(par)][5]
        return w
    specs = []
    for level in args.levels:
        for rho in (args.rho if level == "B" else [0.]):
            for bias, readout in (itertools.product(args.cp_bias, args.cp_readout) if level == "Cp" else [("-", "-")]):
                for par, eps, seed in itertools.product(pars, args.eps, args.seeds):
                    kappa = par if args.law == "shared" else 1. - par
                    specs.append(dict(level=level, rho=rho, bias=bias, readout=readout, law=args.law, par=par,
                                      kappa=kappa, eps=eps, seed=seed))
    Hs = {s["level"]: (args.hidden_c if s["level"] in ("C", "Cp") else 4) for s in specs}
    results = []
    for H in sorted(set(Hs.values())):
        group = [s for s in specs if Hs[s["level"]] == H]
        inits = [init_run(s["level"], s["eps"], s["rho"], 1000 * s["seed"] + 17, H, s["bias"], s["readout"])
                 for s in group]
        th = {k: torch.stack([i[k] for i in inits]).to(dev).requires_grad_() for k in inits[0]}
        P = torch.stack([weights(s["par"]) for s in group]).to(dev)
        if args.check_grad:
            check_grad(th, P, e, y, n, group)
            return
        R = len(group)
        # Backtracking GD on the flow clock: a step that raises a run's loss (or gives NaN) is undone and that run's
        # step size halved; accepted steps grow it by 5% up to --lr. Flow time is the sum of accepted step sizes.
        lr = torch.full((R,), args.lr, device=dev)
        clock = torch.zeros(R, device=dev)
        taken = torch.zeros(R, device=dev)
        prev = torch.full((R,), math.inf, device=dev)
        backup = {k: v.detach().clone() for k, v in th.items()}
        mom = {k: torch.zeros_like(v) for k, v in th.items()}
        sec = {k: torch.zeros_like(v) for k, v in th.items()}
        hit = torch.full((R,), math.nan, device=dev)
        solve = torch.full((R,), math.nan, device=dev)
        grid = np.concatenate([[0.], np.geomspace(1, args.t_max, 80)])
        nxt = torch.zeros(R, dtype=torch.long, device=dev)
        curves = [[] for _ in range(R)]
        t0 = time.time()
        for step in range(args.steps):
            z, a, act = forward(th, e, n)
            loss = (P * torch.nn.functional.softplus(-y * z)).sum(1)
            grads = torch.autograd.grad(loss.sum(), list(th.values()))
            with torch.no_grad():
                bad = ~torch.isfinite(loss)
                if args.opt == "gd":
                    bad = bad | (loss > prev + 1e-13)
                good = ~bad
                for k, v in th.items():
                    v[bad] = backup[k][bad]
                    backup[k][good] = v[good]
                clock[bad] -= taken[bad]
                lr[bad] /= 2
                prev[good] = loss[good]
                acc = (P * (z * clean > 0)).sum(1)
                src = torch.where((P > 0) & differ, a, torch.ones_like(a)).min(1).values
                solve[good & (acc >= 1 - 1e-12) & solve.isnan()] = clock[good & (acc >= 1 - 1e-12) & solve.isnan()]
                ok = good & (src >= .5) & (acc >= 1 - 1e-12) & hit.isnan()
                hit[ok] = clock[ok]
                gt = torch.tensor(grid, device=dev)[nxt.clamp(max=len(grid) - 1)]
                rec = good & (clock >= gt) & (nxt < len(grid))
                if rec.any():
                    dec = (P * (z * decoy > 0)).sum(1)
                    gate = (P * a * e[:, n, 1]).sum(1)          # E[b alpha_src]: instruction-gated routing
                    live = (P * act).sum(1)                     # mean fraction of active MLP units
                    for r in rec.nonzero().flatten().tolist():
                        curves[r].append((float(clock[r]), float(loss[r]), float(acc[r]), float(src[r]),
                                          float((P[r] * a[r]).sum()), float(dec[r]), float(gate[r]), float(live[r])))
                    nxt[rec] = torch.searchsorted(torch.tensor(grid, device=dev), clock[rec], right=True)
                active = good & (clock < args.t_max)
                if args.opt == "gd":
                    for p_, g_ in zip(th.values(), grads):
                        p_[active] -= lr[active].view(-1, *[1] * (p_.dim() - 1)) * g_[active]
                    taken = torch.where(active, lr, torch.zeros_like(lr))
                    lr[active] = torch.clamp(lr[active] * 1.05, max=args.lr)
                else:                                   # Adam, elementwise, so runs stay independent
                    for k, g_ in zip(th, grads):
                        mom[k][active] = .9 * mom[k][active] + .1 * g_[active]
                        sec[k][active] = .999 * sec[k][active] + .001 * g_[active] ** 2
                        mh, vh = mom[k][active] / (1 - .9 ** (step + 1)), sec[k][active] / (1 - .999 ** (step + 1))
                        th[k][active] -= lr[active].view(-1, *[1] * (mh.dim() - 1)) * mh / (vh.sqrt() + 1e-8)
                    taken = active.to(clock.dtype)
                clock += taken
                if not active.any() and not bad.any():
                    break
            if step % 20000 == 0:
                print(f"H={H} step {step} clock min {clock.min().item():.0f} solved {(~solve.isnan()).sum().item()} "
                      f"routed {(~hit.isnan()).sum().item()}/{R} ({time.time() - t0:.0f}s)", flush=True)
        for r, s in enumerate(group):
            last = curves[r][-1]
            s = dict(s, n=n, eta=args.eta, lr=args.lr, opt=args.opt, t_max=args.t_max, H=H, final_time=float(clock[r]),
                     route_time=None if math.isnan(hit[r]) else float(hit[r]),
                     solve_time=None if math.isnan(solve[r]) else float(solve[r]),
                     final_loss=last[1], final_clean_acc=last[2], final_min_src_differ=last[3], final_mean_src=last[4],
                     final_decoy_acc=last[5], curve=curves[r])
            results.append(s)
    t_max = args.t_max
    tied = {}
    for kappa, eps in itertools.product(sorted({s["kappa"] for s in specs}), args.eps):
        tied[f"{kappa}|{eps}"] = tied_ce_routes(n, kappa, args.eta, eps, t_max)
    out = dict(args=vars(args), runs=results, tied_ce_route_time=tied)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))
    summary(results, tied)


def check_grad(th, P, e, y, n, group):
    """Gradients at the given initialization with W_K = 0 (uniform attention).

    corr: |grad of -E[y z]/2| per matrix; bce: |grad of E[softplus(-y z)]|; marg: |grad of the label-marginal loss
    E[(softplus(z) + softplus(-z))/2]|. For y = +-1, softplus(-y z) = (softplus(z) + softplus(-z))/2 - y z/2 exactly, so
    bce = marg + corr. At W_K = 0, z depends only on (b, sum of displayed values); under one anti-copy (kappa = 0)
    E[y | b, displayed multiset] = 0, so corr vanishes for every non-key matrix and BCE is pure label-marginal fitting."""
    with torch.no_grad():
        th["Wk"].zero_()
    sp = torch.nn.functional.softplus
    losses = dict(corr=lambda z: -.5 * (P * y * z).sum(1), bce=lambda z: (P * sp(-y * z)).sum(1),
                  marg=lambda z: (P * (.5 * sp(z) + .5 * sp(-z))).sum(1))
    seen = set()
    for r, s in enumerate(group):
        key = (s.get("bias"), s.get("readout"), s["kappa"], s["eps"])
        if key in seen:
            continue
        seen.add(key)
        out = []
        for name, f in losses.items():
            z = forward(th, e, n)[0]
            g = torch.autograd.grad(f(z).sum(), list(th.values()))
            out.append(name + " " + " ".join(f"{k}={gk[r].norm():.1e}" for k, gk in zip(th, g)))
        print(s["level"], *key, " | ".join(out), flush=True)


def summary(results, tied):
    var = lambda r: (r["level"], r["rho"], r.get("bias", "-"), r.get("readout", "-"))
    keys = sorted({var(r) for r in results})
    epss = sorted({r["eps"] for r in results})
    for key in keys:
        print(f"\nlevel/rho/bias/readout {key}: routed/total, T/. = tied CE flow routes or not, (solved)   eps ->",
              " ".join(f"{e:g}" for e in epss))
        for kappa in sorted({r["kappa"] for r in results}, reverse=True):
            cells = []
            for eps in epss:
                rr = [r for r in results if var(r) == key and (r["kappa"], r["eps"]) == (kappa, eps)]
                t = tied.get(f"{kappa}|{eps}")
                cells.append(f"{sum(r['route_time'] is not None for r in rr)}/{len(rr)}{'T' if t else '.'}")
                cells[-1] += f"({sum(r['solve_time'] is not None for r in rr)})"
            print(f"  kappa={kappa:+.2f}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
