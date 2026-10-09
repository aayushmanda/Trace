"""Checks of the staged routing proof (Theorem 4, thm:routing).

    python3 handcoded/theory_checks/routing_checks.py constants   # rigorous check of the proof's decimal constants
    python3 handcoded/theory_checks/routing_checks.py flow        # Stages A-D integrated on the reduced gate model
    python3 handcoded/theory_checks/routing_checks.py             # both

constants: every exponential is enclosed in a rational interval. For 0 <= x <= 1 the alternating series of e^{-x} has
  decreasing terms, so consecutive partial sums bracket it, and e^x = (e^{x/m})^m for larger x. Logarithms are avoided by
  exponentiating both sides. Each check is a strict comparison of rational bounds, so a pass is a proof of the inequality,
  not a floating-point test. Bounds that must hold for every n >= 2 are reduced to n = 2 by a monotonicity check.

flow: reduced model of Lemma R-features, K = 2, exact population over all letter patterns and both start counts:
  f_t = sigma v_c + sigma v_s c_t + g v_r M_t, with M_t the gate head's attention mass on the letters.
  A flow on (v_c, v_s) to tolerance alpha_B; B flow on v_r until alpha = sign(a) g v_r hits alpha_B = 0.01 p(1-p)|a|/n^2;
  C flow on beta until the largest attention outside {own position, current letter} is at most zeta = 0.01|a|/(sqrt(K) B);
  D flow on (v_c, v_s, v_r). Along Stage C it checks the corridor: b = (beta_n + beta_0)/2 nondecreasing,
  |beta_n - beta_0| <= 1, beta_delta <= 0 for every other offset; it compares the hitting times with the proved bounds
  T_B, T_C and bar T_D (eq:R-Dconst). Stage D is run to positive margins.
"""
import itertools
import math
import sys
from fractions import Fraction as F

import numpy as np
from scipy.integrate import solve_ivp

# --- constants: rational interval arithmetic --------------------------------------------------------------

TERMS = 40


def exp_neg_unit(x):
    """Enclosure (lo, hi) of e^{-x} for rational 0 <= x <= 1."""
    assert 0 <= x <= 1
    term, s, sums = F(1), F(0), []
    for k in range(TERMS):
        s += term
        sums.append(s)
        term *= -x / (k + 1)
    a, b = sums[-1], sums[-2]
    return min(a, b), max(a, b)


def exp_iv(x):
    """Enclosure (lo, hi) of e^x for rational x."""
    x = F(x)
    m = max(1, int(abs(x)) + 1)
    lo, hi = exp_neg_unit(abs(x) / m)
    lo, hi = lo ** m, hi ** m
    return (lo, hi) if x <= 0 else (1 / hi, 1 / lo)


def lo(x):
    return exp_iv(x)[0]


def hi(x):
    return exp_iv(x)[1]


def constants_checks():
    checks = []

    def check(name, ok, where):
        checks.append((name, ok, where))

    d = F
    # lem:R-field, mass of the useful sites: Z/e^{beta_0} and Z/e^{u+beta_n} <= 3.12 n for all n >= 2.
    # (A + (n-1)B)/n = B + (A-B)/n decreases in n when A > B, so n = 2 is the worst case.
    for B_exp, label in ((d("0.7"), "Z/e^beta0"), (d("0.8"), "Z/e^(u+beta_n)")):
        A_hi = 1 + hi(d("1.1"))
        check(f"{label}: monotone in n (1+e^1.1 > e^{float(B_exp)})", 1 + lo(d("1.1")) > hi(B_exp), "lem:R-field")
        check(f"{label} <= 3.12 n at n=2", (A_hi + hi(B_exp)) / 2 <= d("3.12"), "lem:R-field")
    check("3.1 n fails at n=2 for Z/e^(u+beta_n) (the corrected slip)",
          (1 + lo(d("1.1")) + lo(d("0.8"))) / 2 > d("3.1"), "lem:R-field")
    # Signs certified in lem:R-field.
    check("common strength: 2e^-0.4/3.12 - 0.0075 >= 0.42", 2 * lo(d("-0.4")) / d("3.12") - d("0.0075") >= d("0.42"),
          "lem:R-field(a)")
    check("common strength: drive/error >= 57", 2 * lo(d("-0.4")) / d("3.12") / d("0.0075") >= 57, "lem:R-field(a)")
    check("irrelevant offsets: -4e^-0.4/3.12 + 0.015 < 0", -4 * lo(d("-0.4")) / d("3.12") + d("0.015") < 0,
          "lem:R-field(a)")
    check("imbalance: 1 - e^-0.9 >= 0.59", 1 - hi(d("-0.9")) >= d("0.59"), "lem:R-field(b)")
    check("imbalance: 3.12^2 <= 9.74", d("3.12") ** 2 <= d("9.74"), "lem:R-field(b)")
    check("imbalance: -2.36 e^-0.2/9.74 + 0.06 < 0", -d("2.36") * lo(d("-0.2")) / d("9.74") + d("0.06") < 0,
          "lem:R-field(b)")
    check("imbalance: drive/error >= 3.3", d("2.36") * lo(d("-0.2")) / d("9.74") / d("0.06") >= d("3.3"),
          "lem:R-field(b)")
    check("leakage: e^-0.4/3.12 >= 0.21", lo(d("-0.4")) / d("3.12") >= d("0.21"), "lem:R-field(c)")
    check("leakage: 0.42 * 0.21 = 0.0882", d("0.42") * d("0.21") == d("0.0882"), "lem:R-field(c)")
    # lem:R-C.
    check("leak: e^0.6 <= 1.83", hi(d("0.6")) <= d("1.83"), "lem:R-C")
    check("rate: 0.0882 * 0.01 = 0.000882", d("0.0882") * d("0.01") == d("0.000882"), "lem:R-C")
    check("T_C: 1/0.000882 <= 1134", 1 / d("0.000882") <= 1134, "lem:R-C")
    # lem:R-coeff: Z <= 1 + n e^0.1 <= 1.61 n for n >= 2 (linear in n, holds at n = 2 and slope 1.61 > e^0.1).
    check("Z <= 1.61 n at n=2", 1 + 2 * hi(d("0.1")) <= d("1.61") * 2, "lem:R-coeff")
    check("Z <= 1.61 n: slope 1.61 > e^0.1", d("1.61") > hi(d("0.1")), "lem:R-coeff")
    check("d_u M >= e^-0.1/1.61^2 >= 0.349", lo(d("-0.1")) / d("1.61") ** 2 >= d("0.349"), "lem:R-coeff")
    check("kappa: 2 * 0.349 >= 0.69", 2 * d("0.349") >= d("0.69"), "lem:R-coeff")
    check("kappa >= 69 alpha_B: 0.69 / 0.01 = 69", d("0.69") / d("0.01") == 69, "lem:R-B")
    # lem:R-B.
    check("bar r <= 1.5 alpha_B <= kappa/3: 1.5 <= 69/3", d("1.5") <= d(69, 3), "lem:R-B")
    check("rate: (4/3) * 0.69 >= 0.92", d(4, 3) * d("0.69") >= d("0.92"), "lem:R-B")
    check("T_B: 0.01/0.92 <= 0.011", d("0.01") / d("0.92") <= d("0.011"), "lem:R-B")
    # lem:R-D separator: |D| >= 2|a| sigma'(1.1) >= 0.37|a|, sigma'(x) = e^-x/(1+e^-x)^2 decreasing for x > 0.
    sp = lo(d("-1.1")) / (1 + hi(d("-1.1"))) ** 2
    check("sigma'(1.1) >= 0.185", sp >= d("0.185"), "lem:R-D")
    check("N bar: 1 + 2 sqrt2/(0.37 |a|) <= 7.84/|a| for |a| <= 0.1",
          d("0.1") + 2 * d(14143, 10000) / d("0.37") <= d("7.84"), "lem:R-D")
    # lem:R-D margins, with zeta*B*Nbar = 0.01 * 7.84 and zeta*B = 0.01|a|/sqrt K <= 0.001/sqrt 2.
    check("6 zeta B Nbar = 0.4704 <= 0.48", 6 * d("0.01") * d("7.84") <= d("0.48"), "lem:R-D")
    check("early: log(1+e^-0.96) >= 0.32  <=>  1+e^-0.96 >= e^0.32", 1 + lo(d("-0.96")) >= hi(d("0.32")), "lem:R-D")
    check("early: 0.2 < 0.32", d("0.2") < d("0.32"), "lem:R-D")
    check("4 zeta B <= 0.004/sqrt2 <= 0.003", d("0.004") / d(14142, 10000) <= d("0.003"), "lem:R-D")
    check("8 zeta Nbar B = 0.627 <= 0.63", 8 * d("0.01") * d("7.84") <= d("0.63"), "lem:R-D")
    check("B >= 2 log 40 >= 2.52  <=>  40 >= e^1.26", 40 >= hi(d("1.26")), "lem:R-D")
    check("C_D >= 2 log 40 at K=2: 10(K-1)/pi_0 >= 20K(K-1) = 40", 20 * 2 * 1 >= 40, "lem:R-D")
    check("late: log2 * e^-0.003 >= 0.69  <=>  2 >= e^{0.69 e^0.003}", 2 >= hi(d("0.69") * hi(d("0.003"))), "lem:R-D")
    check("late, first term: 4.3 >= 1/0.33", d("4.3") >= 1 / d("0.33"), "lem:R-D")
    check("late, second term: (0.33 * 4.3)^2 >= 2 >= (1+t)/t for t >= 1", (d("0.33") * d("4.3")) ** 2 >= 2, "lem:R-D")
    check("late: 0.33 + 0.33 < 0.69", d("0.33") + d("0.33") < d("0.69"), "lem:R-D")

    w = max(len(c[0]) for c in checks)
    for name, ok, where in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  [{where}]")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks pass")
    return not failed


# --- flow: Stages A-D on the reduced gate model ----------------------------------------------------------------

def setup(n, p):
    pats = np.array(list(itertools.product([1., -1.], repeat=n)))          # m_j (-1 = match)
    m = np.tile(pats, (2, 1))
    c0 = np.repeat([1., -1.], len(pats))
    w = np.prod(np.where(m < 0, p, 1 - p), axis=1) / 2
    c = np.column_stack([c0, c0[:, None] * np.cumprod(m[:, :-1], axis=1)])  # c_t, t = 1..n
    return m, c, w


def features(beta, a, m, c, n):
    """M[:, t], attention own/current/others and dM/dbeta for t = 2..n (index 1..n-1)."""
    M, own, cur, oth, dM = [], [], [], [], []
    for t in range(1, n):
        offs = n + t - np.arange(n)                      # offset of letter j from count position t
        sc = a * c[:, t, None] * m + beta[offs][None, :]
        top = np.maximum(sc.max(1), beta[0])
        e_l = np.exp(sc - top[:, None]); e_0 = np.exp(beta[0] - top)
        Z = e_0 + e_l.sum(1)
        pl = e_l / Z[:, None]; p0 = e_0 / Z
        M.append(pl.sum(1)); own.append(p0); cur.append(pl[:, t]); oth.append(pl.sum(1) - pl[:, t])
        g = np.zeros((len(Z), len(beta)))               # dM/dbeta_delta = p0 p_j; dM/dbeta_0 = -p0 M
        for j in range(n):
            g[:, offs[j]] += p0 * pl[:, j]
        g[:, 0] -= p0 * pl.sum(1)
        dM.append(g)
    return np.array(M).T, np.array(own).T, np.array(cur).T, np.array(oth).T, np.stack(dM, 1)


def proof_constants(n, p, a, sig, g, K=2):
    """alpha_B, zeta and the stage-time bounds of the routing appendix (eq:R-Dconst)."""
    a = abs(a)
    pi0 = min(p, 1 - p) / K
    alpha_B = .01 * p * (1 - p) * a / n**2
    ell_p, L_A = math.log(1 / min(p, 1 - p)), math.log(4 * K / alpha_B**2)
    N_A = 3 * math.sqrt(K) * math.sqrt(2 * ell_p**2 + K * L_A**2)
    Nbar, Nbar0 = 7.84 * math.sqrt(K) / a, 1 + sig * N_A + K * alpha_B
    C_D = 2 * math.log(10 * (K - 1) / pi0)
    TD_bar = 5 * (Nbar0 + C_D * Nbar)**2 / (pi0 * g * g)
    R_D = 4.3 * math.sqrt(K - 1) * (Nbar0**2 + Nbar**2) / (pi0 * g * g)
    log_Tp = 2 * math.log(100 * R_D * math.log(100 * (K - 1) * R_D)**2)
    B = max(C_D, 2 * (math.log(K - 1) + log_Tp + math.log1p(math.exp(-log_Tp))), Nbar0 + 1.5)
    zeta = .01 * a / (math.sqrt(K) * B)
    TB_bound = alpha_B * K * n**2 / (.92 * g * g * p * (1 - p) * a)
    TC_bound = 1134 * n**5 * math.log(1.83 * n / zeta) / (p**2 * (1 - p)**2 * a * a * zeta)
    return alpha_B, zeta, TB_bound, TC_bound, TD_bar


def run(n, p, a, sig, g):
    m, c, w = setup(n, p)
    T = n - 1
    y = c[:, 1:] * m[:, 1:]
    ustar = math.log((1 - p) / p)
    alpha_B, eps, TB_bound, TC_bound, TD_bar = proof_constants(n, p, a, sig, g)
    delta_A = alpha_B
    sa = math.copysign(1, a)

    def loss_grad(v, beta, need_beta=False):
        M, p0, pc, po, dM = features(beta, a, m, c, n)
        f = sig * v[0] + sig * v[1] * c[:, 1:] + g * v[2] * M
        s = 1 / (1 + np.exp(y * f))                      # sigmoid(-y f)
        W = w[:, None] / T
        L = np.sum(W * np.logaddexp(0, -y * f))
        r = -y * s * W                                   # dL/df
        gv = np.array([sig * r.sum(), sig * (r * c[:, 1:]).sum(), g * (r * M).sum()])
        gb = g * v[2] * np.einsum('nt,ntk->k', r, dM) if need_beta else None
        return L, gv, gb, (M, p0, pc, po)

    beta = np.zeros(2 * n + 1)
    # Stage A: the (v_c, v_s) flow keeps v_c = 0 and sends sigma v_s monotonically to u_*; stop within delta_A.
    v = np.array([0., (ustar - math.copysign(delta_A / 2, ustar)) / sig if ustar else 0., 0.])
    TA = math.log(1 + abs(ustar) / (delta_A / 2)) / (sig**2 * p * (1 - p)) if ustar else 0.

    # Stage B: flow on v_r until alpha = sign(a) g v_r = alpha_B.
    def rhsB(t, z):
        return [-loss_grad(np.array([v[0], v[1], z[0]]), beta)[1][2]]
    evB = lambda t, z: sa * g * z[0] - alpha_B
    evB.terminal = True
    solB = solve_ivp(rhsB, (0, 1e30), [0.], events=evB, rtol=1e-10, atol=1e-30, method='LSODA')
    v[2] = solB.y[0, -1]; TB = solB.t[-1]
    assert solB.status == 1 and TB <= TB_bound * (1 + 1e-9), (TB, TB_bound)

    # Stage C: flow on beta until the leakage is at most zeta; track the corridor.
    def leak(bb):
        return loss_grad(v, bb)[3][3].max()

    def rhsC(t, bb):
        return -loss_grad(v, bb, need_beta=True)[2]
    evC = lambda t, bb: leak(bb) - eps
    evC.terminal = True
    solC = solve_ivp(rhsC, (0, 1e40), beta, events=evC, rtol=1e-9, atol=1e-12, method='LSODA', dense_output=True)
    assert solC.status == 1, solC.message
    TC = solC.t[-1]
    bs = solC.sol(np.linspace(0, TC, 2000))
    bmid = (bs[n] + bs[0]) / 2; Delta = bs[n] - bs[0]
    others = np.delete(bs, [0, n], axis=0)
    ok_b = np.all(np.diff(bmid) >= -1e-9)
    ok_D = np.all(np.abs(Delta) <= 1)
    ok_o = np.all(others <= 1e-9)
    beta = solC.y[:, -1]

    # Stage D: flow on (v_c, v_s, v_r) until every margin is positive.
    def margins(vv):
        M, *_ = features(beta, a, m, c, n)
        return np.min(y * (sig * vv[0] + sig * vv[1] * c[:, 1:] + g * vv[2] * M))

    def rhsD(tau, vv):                                    # log time tau = log(1 + t)
        return -math.exp(tau) * loss_grad(vv, beta)[1]
    evD = lambda tau, vv: margins(vv)
    evD.terminal = True
    solD = solve_ivp(rhsD, (0, 200), v, events=evD, rtol=1e-10, atol=1e-12, method='LSODA')
    TD = math.expm1(solD.t[-1]) if solD.status == 1 else None
    return dict(n=n, p=p, a=a, zeta=eps, alpha_B=alpha_B, TA=TA, TB=TB, TB_bound=TB_bound, TC=TC, TC_bound=TC_bound,
                b_end=float(bmid[-1]), Delta_max=float(np.abs(Delta).max()), others_max=float(others.max()),
                b_monotone=bool(ok_b), corridor=bool(ok_D and ok_o), TD=TD, TD_bar=TD_bar)


def flow_checks():
    for n, p, a in itertools.product([2, 4, 6], [.1, .25, .5, .9], [.05, -.05, .01]):
        r = run(n, p, a, sig=.1, g=.01)
        assert r['b_monotone'] and r['corridor'] and r['TC'] <= r['TC_bound'], r
        assert r['TD'] is not None and r['TD'] <= r['TD_bar'], r
        print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print('all cases: Stage B and C hitting times within their bounds, corridor invariant, margins positive in Stage D')
    return True


if __name__ == '__main__':
    which = sys.argv[1:] or ["constants", "flow"]
    ok = True
    if "constants" in which:
        ok &= constants_checks()
    if "flow" in which:
        ok &= flow_checks()
    raise SystemExit(0 if ok else 1)
