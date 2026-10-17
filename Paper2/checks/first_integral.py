"""Check the exact first integral and time formula of the correlation-loss flow.

Flow (r1 = r2 = r, s = xi_s - xi_d):  r' = c r h(s) / 2,  s' = c r^2 h'(s),
  alpha(s) = e^s / (e^s + N - 1),  h(s) = (1 + rho) alpha - rho,  rho = (1 - kappa)/(N - 1).
Claims: r^2 - H(s) = eps0^2 with H(s) = int_0^s h/h';
        T(s1) = int_0^{s1} ds / (c (eps0^2 + H(s)) h'(s)).
Also: bounds of Theorem 2 and the over-cancellation dichotomy.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp, quad

def funcs(N, kappa):
    rho = (1 - kappa) / (N - 1)
    al = lambda s: 1 / (1 + (N - 1) * math.exp(-s))
    h = lambda s: (1 + rho) * al(s) - rho
    hp = lambda s: (1 + rho) * al(s) * (1 - al(s))
    return al, h, hp

def H(N, kappa, s):
    al, h, hp = funcs(N, kappa)
    return quad(lambda x: h(x) / hp(x), 0, s, limit=200)[0]

def T_formula(N, kappa, eps0, c=1.0):
    al, h, hp = funcs(N, kappa)
    s1 = math.log(N - 1)                     # alpha = 1/2
    return quad(lambda x: 1 / (c * (eps0**2 + H(N, kappa, x)) * hp(x)), 0, s1, limit=200)[0]

def T_ode(N, kappa, eps0, c=1.0):
    al, h, hp = funcs(N, kappa)
    s1 = math.log(N - 1)
    ev = lambda t, y: y[1] - s1; ev.terminal = True
    sol = solve_ivp(lambda t, y: [c * y[0] * h(y[1]) / 2, c * y[0]**2 * hp(y[1])],
                    (0, 1e9), [eps0, 0.0], events=ev, rtol=1e-10, atol=1e-14, method="LSODA")
    inv = sol.y[0]**2 - np.array([H(N, kappa, s) for s in sol.y[1][::max(1, len(sol.t)//20)]]).mean() if False else None
    drift = max(abs(sol.y[0][i]**2 - H(N, kappa, sol.y[1][i]) - eps0**2) for i in range(0, len(sol.t), max(1, len(sol.t)//25)))
    return (sol.t_events[0][0] if sol.t_events[0].size else math.inf), drift

print("N kappa eps0 | T_ode T_formula invariant_drift | bound")
for N in (16, 64, 256):
    for eps0 in (1e-2, 1e-3):
        for kappa in (1.0, 0.0):
            To, d = T_ode(N, kappa, eps0); Tf = T_formula(N, kappa, eps0)
            if kappa == 1.0:
                lb = (N-1)/(2*math.e) * math.log(1+2/eps0**2)
                ub = 4*(N-1)*(math.log(1+eps0**-2)+1/math.e)
            else:
                lb = (N-1)/(5*eps0)
                ub = 4*(N-1)*(math.sqrt(2)*math.pi/eps0+8/math.e)
            assert lb <= To <= ub, (N, kappa, eps0, To, lb, ub)
            assert abs(To-Tf)/Tf < 1e-7
            assert d < 1e-8
            b = f"{lb:.4g} <= T <= {ub:.4g}: True"
            print(f"{N} {kappa} {eps0} | {To:.5g} {Tf:.5g} {d:.1e} | {b}")
# over-cancellation dichotomy
for N in (64,):
    for eps0 in (1e-2,):
        for kappa in (-0.5 * math.sqrt(2) * eps0, -2 * math.sqrt(2) * eps0):
            al, h, hp = funcs(N, kappa)
            rho = (1-kappa)/(N-1)
            theta = rho/(1+rho)
            s_min = math.log((N-1)*theta/(1-theta))
            Hmin = H(N,kappa,s_min)
            To, _ = T_ode(N, kappa, eps0)
            assert math.isfinite(To) == (eps0**2 > -Hmin)
            print(f"over-cancel N={N} kappa={kappa:.4f} eps0^2={eps0**2:.1e} -minH={-Hmin:.2e} "
                  f"approx kappa^2/2={kappa**2/2:.2e} -> T={To}")
