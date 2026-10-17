"""Routing time across the cancellation coefficient: clean-fraction crossover and over-cancellation barrier.

For kappa in [0, 1], D_kappa is the mixture kappa D_1 + (1 - kappa) D_0, so kappa is the fraction of traces without an
anti-copy. Checks the corollary T = Theta((N/gamma) min{1/eps, (1 + log(1 + kappa/eps))/kappa}) with explicit constants,
the primitive bounds behind it, and draws T against kappa, including the barrier for kappa < 0.

    python Paper2/checks/kappa_sweep.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GAMMA = 0.8                                          # eta = 0.1, as in Table 1


def fns(n, kappa):
    rho = (1 - kappa) / (n - 1)
    al = lambda q: 1 / (1 + (n - 1) * math.exp(-q))
    phi = lambda q: (1 + rho) * al(q) - rho
    dphi = lambda q: (1 + rho) * al(q) * (1 - al(q))
    return al, phi, dphi


def F(n, kappa, q):
    _, phi, dphi = fns(n, kappa)
    return quad(lambda v: phi(v) / dphi(v), 0, q, limit=200)[0]


def routing_time(n, kappa, eps):
    """Integrate F' = phi/phi' and T' = 1/(gamma (eps^2 + F) phi') together in q, from 0 to log(N - 1)."""
    _, phi, dphi = fns(n, kappa)
    q1 = math.log(n - 1)
    if kappa < 0 and eps ** 2 <= -F(n, kappa, math.log(1 - kappa)):
        return math.inf
    rhs = lambda q, y: [phi(q) / dphi(q), 1 / (GAMMA * (eps ** 2 + y[0]) * dphi(q))]
    sol = solve_ivp(rhs, (0, q1), [0., 0.], method="DOP853", rtol=1e-10, atol=1e-14)
    return float(sol.y[1, -1])


def I(eps, kappa):
    """The reference integral int_0^1 dq / (eps^2 + kappa q + q^2)."""
    return quad(lambda q: 1 / (eps ** 2 + kappa * q + q * q), 0, 1, limit=200, points=[eps, kappa] if kappa > 0 else [eps])[0]


def closed(eps, kappa):
    return 1 / eps if kappa <= eps else (1 + math.log(1 + kappa / eps)) / kappa


def main():
    out = {}
    # Primitive bounds: (3/32)(kappa q + q^2/2) <= F(q) <= 2(kappa q + q^2/2) on [0, 1].
    worst = [math.inf, 0.]
    for n in (8, 16, 64, 256):
        for kappa in (0., 1e-4, 1e-3, .01, .05, .1, .3, .6, 1.):
            for q in np.linspace(.01, 1, 25):
                r = F(n, kappa, q) / (kappa * q + q * q / 2)
                worst = [min(worst[0], r), max(worst[1], r)]
    assert 3 / 32 <= worst[0] and worst[1] <= 2, worst
    out["primitive_ratio_range"] = worst
    # Constants of the corollary: (1/(2e)) I <= gamma T/(N-1) <= (64/3)(4 I + 4/e) and I within [1/3, 3] x closed form.
    rows, lo, hi, ilo, ihi = [], math.inf, 0., math.inf, 0.
    for n in (16, 64):
        for eps in (1e-2, 1e-3, 1e-4):
            for kappa in (0., 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, .01, .03, .1, .3, 1.):
                T = routing_time(n, kappa, eps)
                Iv = I(eps, kappa)
                lo, hi = min(lo, GAMMA * T / ((n - 1) * Iv)), max(hi, GAMMA * T / ((n - 1) * Iv))
                ilo, ihi = min(ilo, Iv / closed(eps, kappa)), max(ihi, Iv / closed(eps, kappa))
                rows.append(dict(n=n, eps=eps, kappa=kappa, T=T, I=Iv, closed=closed(eps, kappa)))
    out.update(T_over_I=[lo, hi], I_over_closed=[ilo, ihi], rows=rows)
    assert 1 / (2 * math.e) <= lo and hi <= 64 / 3 * (4 + 4 / math.e), (lo, hi)
    # Barrier location for kappa < 0 at N = 16.
    barrier = {}
    for eps in (1e-2, 1e-3):
        g = lambda k: eps ** 2 + F(16, k, math.log(1 - k))
        barrier[eps] = brentq(g, -0.9, -1e-9)
    out["barrier_kappa_N16"] = barrier
    (ROOT / "checks" / "results" / "kappa_sweep.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

    # Figure: (a) clean fraction kappa > 0, log-log; (b) through zero to the barrier.
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
    cols = {1e-2: "#174f94", 1e-3: "#7b4fb0", 1e-4: "#b22e38"}
    ks = np.geomspace(1e-5, 1, 40)
    for eps, c in cols.items():
        T = [routing_time(16, k, eps) for k in ks]
        axs[0].loglog(ks, T, color=c, lw=2, label=fr"$\varepsilon={eps:g}$")
        axs[0].axvline(eps, color=c, ls=":", lw=1)
    axs[0].set_xlabel(r"clean fraction $\lambda=\kappa$")
    axs[0].set_ylabel("routing time $T$")
    axs[0].set_title(r"(a) a fraction $\lambda\gtrsim\varepsilon$ removes the bottleneck", fontsize=9.5)
    axs[0].legend(fontsize=8, frameon=False)
    for eps, c in list(cols.items())[:2]:
        kb = barrier[eps]
        kk = np.concatenate([np.linspace(kb * 0.999, kb * 0.5, 25), np.linspace(kb * 0.5, 0.05, 60)])
        T = [routing_time(16, k, eps) for k in kk]
        axs[1].semilogy(kk, T, color=c, lw=2, label=fr"$\varepsilon={eps:g}$")
        axs[1].axvline(kb, color=c, ls="--", lw=1)
    axs[1].axvline(0, color="0.5", lw=0.8)
    axs[1].set_xlabel(r"cancellation coefficient $\kappa$")
    axs[1].set_title(r"(b) over-cancellation: $T\to\infty$ at the barrier", fontsize=9.5)
    axs[1].legend(fontsize=8, frameon=False)
    for ax in axs:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "kappa_sweep.pdf")


if __name__ == "__main__":
    main()
