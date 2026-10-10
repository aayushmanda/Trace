"""Direct numerical integration of the reduced gate flow of Theorem 4 (fixed mask, trained U, W, q, k; value factor fixed).

K = 2 integrates the exact three-variable system (c, G, a). For K >= 3 the gate readout is trained on the line W = w(I - R), as
in the theorem; by cyclic symmetry and the symmetry among the N = K - 2 background offsets the step logits are
    offset 0: A0 + (G/2) r_m,   offset 1: A1 - (G/2) r_m,   background offsets: 0,
with A_i = eps (u_i - u_bg), G = 2 g w, r_+ = sigma(-a) (non-query letter, target 0, probability 1-p), r_- = sigma(a) (query
letter, target 1, probability p), and matrix-space gradient flow
    dA/dt = -(eps^2/K)(I + 11^T/N) grad_A F,   dG/dt = -(2 g^2/K) F_G,   da/dt = -(eps^4 Q/d_h) F_a.
--gate full integrates the unconstrained circulant gate readout instead (B0, B1 in place of G/2, -G/2).
Initialization as in the theorem: U = W = 0, q0 = eps A_Q xi_Q, k0 = eps A_K xi_K, gamma = eps xi_V (so g = eps^2 xi_V).

For each run it records the time at which every prediction has positive margin and the loss is below min(p,1-p) log 2 / 2,
and checks the invariants of the proof along the trajectory: G keeps its sign after prior fitting, a crosses zero at most
once, and from the first time the gate is aligned with |a| >= a_sc/8 on, a G <= 0 and |a| is nondecreasing.

    python handcoded/reduced_ode.py                     # K = 2 (the theorem) and K = 3, 4, 6
    python handcoded/reduced_ode.py --K 2 --eps 0.3 0.2 0.1 0.05
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp


def sig(x):
    return 0.5 * (1 + math.tanh(x / 2))


def rhs_K2(p, eps, g, dh):
    """State (c, G, q, k); returns (dstate/dt, loss, diagnostics)."""
    def f(y):
        c, G, q, k = y
        a = eps ** 2 * q * k / math.sqrt(dh)
        rp, rm = sig(-a), sig(a)
        dp, dm = c + G * rp, c + G * rm
        ep, em = sig(-dp), sig(dm)
        F = (1 - p) * np.logaddexp(0, -dp) + p * np.logaddexp(0, dm)
        Fc = -(1 - p) * ep + p * em
        FG = -(1 - p) * ep * rp + p * em * rm
        Fa = G * rp * rm * ((1 - p) * ep + p * em)
        d = np.array([-eps ** 2 * Fc, -g ** 2 * FG, -Fa * eps ** 2 * k / math.sqrt(dh), -Fa * eps ** 2 * q / math.sqrt(dh)])
        return d, F, dict(a=a, G=G, margin=min(dp, -dm))
    return f


def rhs_K(K, p, eps, g, dh, gate="line"):
    """State (A0, A1, G, q, k) for K >= 3 with W = w(I - R); (A0, A1, B0, B1, q, k) with gate="full"."""
    N = K - 2
    M = np.eye(2) + np.ones((2, 2)) / N
    def f(y):
        if gate == "line":
            A0, A1, G, q, k = y
            B0, B1 = G / 2, -G / 2
        else:
            A0, A1, B0, B1, q, k = y
        a = eps ** 2 * q * k / math.sqrt(dh)
        rp, rm = sig(-a), sig(a)
        rho = rp * rm
        F, Fa, gA, gB, margin = 0.0, 0.0, np.zeros(2), np.zeros(2), np.inf
        for pi, r, dr, tgt in ((1 - p, rp, -rho, 0), (p, rm, rho, 1)):
            l = np.array([A0 + B0 * r, A1 + B1 * r])
            mx = max(l.max(), 0.0)
            e = np.exp(l - mx)
            Z = e.sum() + N * math.exp(-mx)
            v = e / Z
            F += pi * (math.log(Z) + mx - l[tgt])
            margin = min(margin, l[tgt] - max(l[1 - tgt], 0.0))
            v[tgt] -= 1
            gA += pi * v
            gB += pi * r * v
            Fa += pi * dr * (B0 * v[0] + B1 * v[1])
        da = [-Fa * eps ** 2 * k / math.sqrt(dh), -Fa * eps ** 2 * q / math.sqrt(dh)]
        if gate == "line":                                  # F_G = (F_B0 - F_B1)/2 on the line B = (G/2, -G/2)
            d = np.concatenate([-(eps ** 2 / K) * M @ gA, [-(2 * g ** 2 / K) * 0.5 * (gB[0] - gB[1])], da])
        else:
            d = np.concatenate([-(eps ** 2 / K) * M @ gA, -(g ** 2 / K) * M @ gB, da])
        return d, F, dict(a=a, G=B0 - B1, margin=margin)
    return f


def run(job):
    K, p, eps, dh, xq, xk, xv, tau_max, npts, gate = job
    g = eps ** 2 * xv
    f = rhs_K2(p, eps, g, dh) if K == 2 else rhs_K(K, p, eps, g, dh, gate)
    y0 = np.zeros(4 if K == 2 else (5 if gate == "line" else 6))
    y0[-2:] = eps * xq, eps * xk
    thr = 0.5 * min(p, 1 - p) * math.log(2)
    def ev(tau, y):
        _, F, info = f(y)
        return 1.0 if info["margin"] <= 0 else F - thr
    ev.terminal = True
    taus = np.linspace(0, tau_max, npts)                    # integrate in log time, t = e^tau - 1
    sol = solve_ivp(lambda tau, y: math.exp(tau) * f(y)[0], (0, tau_max), y0, method="LSODA", t_eval=taus,
                    events=ev, rtol=1e-10, atol=1e-18)
    rows = [f(sol.y[:, i]) for i in range(sol.y.shape[1])]
    F = np.array([r[1] for r in rows]); a = np.array([r[2]["a"] for r in rows]); G = np.array([r[2]["G"] for r in rows])
    H = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    s = 1 if p < 0.5 else -1                                # after prior fitting sign(G) = s; aligned means a G < 0
    fitted = np.searchsorted(-F, -(H + 0.05 * (math.log(K) - H)))
    below = F < H
    solved = float(math.expm1(sol.t_events[0][0])) if len(sol.t_events[0]) else None
    asc = eps ** 4 / math.sqrt(dh)
    al = np.nonzero(s * a <= -asc / 8)[0]                   # first sample with the gate aligned and |a| >= a_sc/8
    post = slice(al[0], None) if len(al) else slice(0, 0)
    return dict(K=K, p=p, eps=eps, dh=dh, xi_Q=xq, xi_K=xk, xi_V=xv, gate=gate, solved_at=solved,
                aligned_from=float(math.expm1(sol.t[al[0]])) if len(al) else None,
                invariant_after_alignment=bool(np.all((a * G)[post] <= 0) and np.all(np.diff(np.abs(a[post])) >= -1e-15))
                if len(al) else None,
                G_keeps_sign=bool(np.all(s * G[fitted:] > 0)),
                a_sign_changes=int(np.sum(np.diff(np.sign(a)) != 0)),
                aligned_below_prior=bool(np.all((a * G)[below] < 0)) if below.any() else None,
                abs_a_monotone_below_prior=bool(np.all(np.diff(np.abs(a[below])) >= 0)) if below.sum() > 1 else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--p", type=float, nargs="+", default=[0.1, 0.25, 0.75])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.3, 0.2, 0.1])
    ap.add_argument("--dh", type=int, default=8)
    ap.add_argument("--tau_max", type=float, default=75.0)
    ap.add_argument("--points", type=int, default=601)
    ap.add_argument("--gate", choices=["line", "full"], default="line")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/reduced_ode/runs.json")
    args = ap.parse_args()
    seeds = [("aligned", 1.3, -0.8), ("misaligned", 1.3, 0.8), ("misaligned near-balanced", 1.0, 0.98)]
    jobs, names = [], []
    for K in args.K:
        for p in args.p:
            s = 1 if p < 0.5 else -1
            for eps in args.eps:
                for name, xq, xk in seeds:
                    jobs.append((K, p, eps, args.dh, xq, s * xk, 1.0, args.tau_max, args.points, args.gate))
                    names.append(name)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs))
    for r, name in zip(res, names):
        r["seed"] = name
        t = "never" if r["solved_at"] is None else f"{r['solved_at']:.3g}"
        print(f"K={r['K']} p={r['p']:<4} eps={r['eps']:<4} {name:25s} solved t={t:>9}  G keeps sign={r['G_keeps_sign']}  "
              f"a sign changes={r['a_sign_changes']}  invariant after alignment={r['invariant_after_alignment']}  "
              f"|a| monotone below H={r['abs_a_monotone_below_prior']}", flush=True)
    for K in args.K:                                        # slope of log T against log(1/eps), per p and seed
        for p in args.p:
            for name in dict.fromkeys(names):
                pts = [(math.log(1 / r["eps"]), math.log(r["solved_at"])) for r in res
                       if r["K"] == K and r["p"] == p and r["seed"] == name and r["solved_at"]]
                if len(pts) >= 2:
                    x, y = np.array(pts).T
                    print(f"K={K} p={p} {name:25s} T ~ eps^-{np.polyfit(x, y, 1)[0]:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
