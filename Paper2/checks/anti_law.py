"""The anti-copy law D_kappa and its tied reduced flows.

Decoys: M copies of -x_N at uniformly random decoy positions, the other N-1-M decoys iid uniform.
M = 1 - kappa for kappa in {1, 0, -1}; for fractional kappa, M mixes the two nearest integers with mean 1 - kappa.
The tied read is v = alpha x_N + (1-alpha) S/(N-1), S the decoy sum; only u = x_N S/(N-1) enters the flows.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp


def anti_mix(kappa):
    """(M, probability) pairs with E[M] = 1 - kappa."""
    lo = math.floor(1 - kappa + 1e-12)
    fr = (1 - kappa) - lo
    return [(lo, 1 - fr)] + ([(lo + 1, fr)] if fr > 1e-12 else [])


def u_law(n, kappa):
    """Support and probabilities of u = x_N S_dec/(N-1)."""
    out = {}
    for M, pm in anti_mix(kappa):
        k = n - 1 - M
        for j in range(k + 1):                       # j iid decoys agree with x_N
            u = (-M + j - (k - j)) / (n - 1)
            out[u] = out.get(u, 0.) + pm * math.comb(k, j) / 2 ** k
    us = np.array(sorted(out)); return us, np.array([out[u] for u in us])


def ce_field(q, m, n, us, ps, gamma):
    a = 1 / (1 + (n - 1) * math.exp(-q)); ap = a * (1 - a)
    s = a + (1 - a) * us
    th = np.tanh(m * s / 2)
    f = gamma * np.dot(ps, s) - np.dot(ps, s * th)
    j = ap * np.dot(ps, (1 - us) * (gamma - th))
    return f, j


def ce_route(n, kappa, eta, eps, t_max=1e7, target=None):
    us, ps = u_law(n, kappa); g = 1 - 2 * eta
    tgt = math.log(n - 1) if target is None else target
    def rhs(t, y):
        f, j = ce_field(y[1], y[0], n, us, ps, g); return [y[0] * f, y[0] * j]
    ev = lambda t, y: y[1] - tgt; ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0, t_max), (eps * eps, 0.), events=ev, rtol=2e-10, atol=1e-15, method="DOP853",
                    dense_output=False)
    jmin = min(ce_field(q, m, n, us, ps, g)[1] for m, q in sol.y.T)
    return (float(sol.t_events[0][0]) if len(sol.t_events[0]) else None), jmin, float(sol.y[0, -1])


if __name__ == "__main__":
    for n in (16, 64):
        for kappa in (1., 0., -.1, -1.):
            for eps in (.03, .01, .001):
                T, jmin, m = ce_route(n, kappa, .1, eps)
                print(f"N={n} kappa={kappa:+.1f} eps={eps}: T_ce={T} min j={jmin:.2e} m_end={m:.3g}")
