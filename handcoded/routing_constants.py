"""Rigorous check of the decimal constants in the staged routing proof (Appendix F).

Every exponential is enclosed in a rational interval: for 0 <= x <= 1 the alternating series of
e^{-x} has decreasing terms, so consecutive partial sums bracket it, and e^x = (e^{x/m})^m for
larger x. Logarithms are avoided by exponentiating both sides. Each check is a strict comparison
of rational bounds, so a pass is a proof of the inequality, not a floating-point test. Bounds that
must hold for every n >= 2 are reduced to n = 2 by a monotonicity check printed alongside.
    python3 handcoded/routing_constants.py
"""
from fractions import Fraction as F

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
check("3.1 n fails at n=2 for Z/e^(u+beta_n) (the corrected slip)", (1 + lo(d("1.1")) + lo(d("0.8"))) / 2 > d("3.1"), "lem:R-field")
# Signs certified in lem:R-field.
check("common strength: 2e^-0.4/3.12 - 0.0075 >= 0.42", 2 * lo(d("-0.4")) / d("3.12") - d("0.0075") >= d("0.42"), "lem:R-field(a)")
check("common strength: drive/error >= 57", 2 * lo(d("-0.4")) / d("3.12") / d("0.0075") >= 57, "lem:R-field(a)")
check("irrelevant offsets: -4e^-0.4/3.12 + 0.015 < 0", -4 * lo(d("-0.4")) / d("3.12") + d("0.015") < 0, "lem:R-field(a)")
check("imbalance: 1 - e^-0.9 >= 0.59", 1 - hi(d("-0.9")) >= d("0.59"), "lem:R-field(b)")
check("imbalance: 3.12^2 <= 9.74", d("3.12") ** 2 <= d("9.74"), "lem:R-field(b)")
check("imbalance: -2.36 e^-0.2/9.74 + 0.06 < 0", -d("2.36") * lo(d("-0.2")) / d("9.74") + d("0.06") < 0, "lem:R-field(b)")
check("imbalance: drive/error >= 3.3", d("2.36") * lo(d("-0.2")) / d("9.74") / d("0.06") >= d("3.3"), "lem:R-field(b)")
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
raise SystemExit(1 if failed else 0)
