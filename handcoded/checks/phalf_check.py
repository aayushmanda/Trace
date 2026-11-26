"""Unbiased parity (Proposition E.12, Appendix F.8.2): integrate the reduced gate flow at p = 1/2 in 24 configurations
and check the proof's claims: D <= 0, g < 0 and a nondecreasing for a(0) > 0, |D| <= 28.6 eps^2, and T_star (first time
with A = m_0 + m_1 = 4) below the bound 259 eps^-5 sqrt(d_h log(4 sqrt(d_h)/eps^4)) + 2040 eps^-4.

    python handcoded/checks/phalf_check.py      # -> results/reduced_ode/phalf_check.json

The constants are those of the proof: 28.6 = 4/(2 * 0.070) rounded up, 259 = sqrt(2 * 4 / (0.0043 * 0.0279)) rounded up.
"""
import json
import math
import os

import numpy as np
from scipy.integrate import solve_ivp


def sig(x):
    return 1 / (1 + math.exp(-x))


def run(eps, dh, q0, k0, v):
    """Reduced flow of (u, g, q, k) at p = 1/2 until A = 4; returns T_star, the bound and the trajectory diagnostics."""
    def f(t, y):
        u, g, q, k = y
        a = eps ** 2 * q * k / math.sqrt(dh)
        a0, a1 = sig(-a), sig(a)
        m0, m1 = u + g * a0, -u - g * a1
        r0, r1 = .5 * sig(-m0), .5 * sig(-m1)
        dLda = g * a1 * a0 * (r0 + r1)
        return [eps ** 2 * (r0 - r1), eps ** 2 * v ** 2 * (a0 * r0 - a1 * r1),
                -(eps ** 2 * k / math.sqrt(dh)) * dLda, -(eps ** 2 * q / math.sqrt(dh)) * dLda]

    def reach(t, y):
        a = eps ** 2 * y[2] * y[3] / math.sqrt(dh)
        return (y[0] + y[1] * sig(-a)) + (-y[0] - y[1] * sig(a)) - 4
    reach.terminal = True
    sol = solve_ivp(f, (0, 1e12), [0, 0, q0, k0], method='LSODA', rtol=1e-10, atol=1e-16, events=reach,
                    dense_output=True)
    T = sol.t_events[0][0]
    Y = sol.sol(np.linspace(0, T, 4000))
    s = 1 if q0 * k0 > 0 else -1                        # reflect so that a(0) > 0
    a, g, D = s * eps ** 2 * Y[2] * Y[3] / math.sqrt(dh), s * Y[1], s * (2 * Y[0] + Y[1])
    bound = 259 * math.sqrt(dh * math.log(4 * math.sqrt(dh) / eps ** 4)) / eps ** 5 + 2040 / eps ** 4
    return dict(eps=eps, dh=dh, q0=q0, k0=k0, v=v, T_star=T, bound=bound, bound_over_T=bound / T,
                D_max=float(D.max()), D_abs_max=float(np.abs(D).max()), D_bound=28.6 * eps ** 2,
                g_max_after_0=float(g[1:].max()), a_min_increment=float(np.diff(a).min()))


if __name__ == "__main__":
    rows = []
    for eps in (0.13, 0.1, 0.05):
        for dh in (8, 64):
            for q0, k0, v in ((eps * 1.3, eps * 0.8, eps), (eps * 0.5, eps * 0.5, eps * 2),
                              (eps * 2, -eps * 0.6, eps * 0.5), (eps, eps, eps)):
                rows.append(run(eps, dh, q0, k0, v))
    checks = dict(
        configurations=len(rows),
        T_below_bound=all(r["T_star"] <= r["bound"] for r in rows),
        D_nonpositive=all(r["D_max"] <= 1e-12 for r in rows),
        g_negative=all(r["g_max_after_0"] < 0 for r in rows),
        a_nondecreasing=all(r["a_min_increment"] >= -1e-12 for r in rows),
        min_D_margin=min(r["D_bound"] / r["D_abs_max"] for r in rows),
        bound_over_T_range=[min(r["bound_over_T"] for r in rows), max(r["bound_over_T"] for r in rows)])
    for r in rows:
        print(f"eps={r['eps']:<5} dh={r['dh']:<3} T*={r['T_star']:.3e} bound/T*={r['bound_over_T']:6.1f} "
              f"|D|max={r['D_abs_max']:.2e} (28.6 eps^2 = {r['D_bound']:.2e})")
    print(json.dumps(checks, indent=1))
    os.makedirs("results/reduced_ode", exist_ok=True)
    json.dump(dict(checks=checks, runs=rows), open("results/reduced_ode/phalf_check.json", "w"), indent=1)
