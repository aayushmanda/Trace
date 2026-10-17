"""Mechanical audit of the proof of Theorem 4 for every K >= 2 (Appendix D).

    uv run python handcoded/theory_checks/theorem4_checks.py

A. The threshold eps* implies eq. (J-eps) and the scale relations of "Conditions and scales" (T_1 <= T_max/2, T_cross < T_max/2,
   a_sc <= mu/16), on a grid of (p, d_h, gap, K), in 60-digit arithmetic (eps* can be tiny, so eps^12 underflows doubles).
B. The derivative identities of Lemma "Derivatives", the properties of phi, the drift bounds of Lemma "Drift identities"(e),
   the gradient bound and the loss levels, against finite differences and on random points, for K = 2, 3, 4, 6, 50.
C. The reduced flow of Lemma "Reduced loss and flow" against autograd gradients of the softmax model with a full K x K count
   readout and the gate readout W = w(I - R), for K = 2, 3, 4, 6; the count-readout gradient stays circulant and
   background-symmetric.
D. Elementary constants and sigmoid facts used in the proof.
Every check prints PASS or FAIL; the exit code is the number of failures.
"""
import math
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 60
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail and not ok else ''}")


# ----------------------------------------------------------------------------------------------- A. threshold
def threshold_inequalities(p, dh, gap, K):
    p, dh, gap, K = mp.mpf(p), mp.mpf(dh), mp.mpf(gap), mp.mpf(K)
    hs = mp.log((1 - p) / p)
    mu = p * (1 - 2 * p) / 14
    rho = 2 / K
    eps = mp.sqrt(gap * p) * hs / (712 * (1 + mp.log(K)))
    asc = eps ** 4 / mp.sqrt(dh)
    T1 = hs / (2 * rho * eps ** 2 * mu)
    Tcross = 15840 * mp.sqrt(dh) / (gap * p * hs * eps ** 4)
    Tmax = hs * mp.sqrt(dh) / (16 * rho * eps ** 6)
    return {
        "eps <= 1/2": (eps, mp.mpf(1) / 2),
        "eps^4 <= mu/16": (eps ** 4, mu / 16),
        "eps^6 <= p/(16(1+log K))": (eps ** 6, p / (16 * (1 + mp.log(K)))),
        "eps^2 <= gap p h*^2/506944": (eps ** 2, gap * p * hs ** 2 / 506944),
        "a_sc <= mu/16": (asc, mu / 16),
        "T_1 <= T_max/2": (T1, Tmax / 2),
        "T_cross < T_max/2 (with margin)": (Tcross * (1 + mp.mpf("1e-6")), Tmax / 2),
        "(iv) a_sc < mu/4": (asc, mu / 4 * (1 - mp.mpf("1e-6"))),
        "(v) 8 eps^6 (1+log K)/p <= 1/2": (8 * eps ** 6 * (1 + mp.log(K)) / p, mp.mpf(1) / 2),
        "4 a_sc <= 1/4": (4 * asc, mp.mpf(1) / 4),
        "a_sc/8 <= 2 (tanh bound applies)": (asc / 8, 2),
    }


bad, n_cases = [], 0
for p in [1e-8, 1e-5, 1e-3, 0.01, 0.1, 0.25, 0.4, 0.49, 0.499, 0.4999999]:
    for dh in [1, 8, 64, 1e3, 1e6]:
        for gap in (1, 0.1, 1e-3, 1e-6):
            for K in (2, 3, 6, 100, 10 ** 6):
                n_cases += 1
                for name, (lhs, rhs) in threshold_inequalities(p, dh, gap, K).items():
                    if not lhs <= rhs * (1 + mp.mpf("1e-40")):
                        bad.append((name, p, dh, gap, K, mp.nstr(lhs, 5), mp.nstr(rhs, 5)))
check(f"A. eps* implies all {len(threshold_inequalities(0.25, 8, 0.1, 3))} inequalities on {n_cases} (p, d_h, gap, K) cases",
      not bad, str(bad[:3]))

# ----------------------------------------------------------------------------------------------- D. constants and sigmoid facts
sig = lambda x: 1 / (1 + math.exp(-x))
dsig = lambda x: sig(x) * (1 - sig(x))
pp = np.linspace(1e-12, 0.5 - 1e-9, 2_000_001)
hh = np.log((1 - pp) / pp)
check("D. max p h*^4/(1-2p) <= 10", float((pp * hh ** 4 / (1 - 2 * pp)).max()) <= 10)
check("D. max p^2 h*^6 <= (3/e)^6 and max sqrt(p) h* <= 2/e",
      float((pp ** 2 * hh ** 6).max()) <= (3 / math.e) ** 6 and float((np.sqrt(pp) * hh).max()) <= 2 / math.e)
check("D. h* <= (1-2p)/p and h* <= 1/p", bool(np.all(hh <= (1 - 2 * pp) / pp + 1e-12)) and bool(np.all(hh <= 1 / pp)))
check("D. sigma'(1) >= 1/6, sigma(-1/4)/2 >= 1/5, e^(-1/2)/4 >= 1/7",
      dsig(1) >= 1 / 6 and sig(-0.25) / 2 >= 1 / 5 and math.exp(-0.5) / 4 >= 1 / 7)
xs = np.linspace(0, 2, 20001)
check("D. tanh(x/2) >= x/4 on [0,2]; sigma(-z) >= e^(-z)/2 for z >= 0",
      bool(np.all(np.tanh(xs / 2) >= xs / 4 - 1e-15)) and all(sig(-z) >= math.exp(-z) / 2 for z in np.linspace(0, 40, 4001)))
check("D. (1-p) - sqrt(p(1-p)) >= (1-2p)/2 and e^(-1/2)/4 (p/(1-p))^(1/4) >= p/7",
      bool(np.all((1 - pp) - np.sqrt(pp * (1 - pp)) >= (1 - 2 * pp) / 2 - 1e-15))
      and bool(np.all(math.exp(-0.5) / 4 * (pp / (1 - pp)) ** 0.25 >= pp / 7)))
check("D. (log 2)^2/128 >= 1/267", math.log(2) ** 2 / 128 >= 1 / 267)
check("D. 6144*267 = 1640448 <= 1.7e6, 6144*32 = 196608 <= 2e5", 6144 * 267 == 1640448 <= 1.7e6 and 6144 * 32 == 196608 <= 2e5)
check("D. 33/8*3840 = 15840, 32*15840 = 506880 < 712^2 = 506944", 33 / 8 * 3840 == 15840 and 32 * 15840 == 506880 < 712 ** 2 == 506944)
check("D. 3*4*1024/2 = 6144, 128*6*5 = 3840", 3 * 4 * 1024 // 2 == 6144 and 128 * 6 * 5 == 3840)
check("D. 7/2 <= 6 log 2 (T_1 absorbed for K >= 2)", 3.5 <= 6 * math.log(2))
check("D. growth lemma: 80*6*17/2 = 4080, 3*4*5*267/2 = 8010, tanh^2(1/2) >= 1/5, 1/sqrt(267) >= 1/17",
      80 * 6 * 17 // 2 == 4080 and 3 * 4 * 5 * 267 // 2 == 8010 and math.tanh(0.5) ** 2 >= 0.2 and 1 / math.sqrt(267) >= 1 / 17)
check("D. floor lemma: (5/4)/20 + 1/16 = 1/8 and v^2 h*/20 >= eps^2 min(1,h*)/80",
      abs(1.25 / 20 + 1 / 16 - 1 / 8) < 1e-15)

# ----------------------------------------------------------------------------------------------- B. identities for general K
rng = np.random.default_rng(0)


def F(m, s, N):
    terms = [0.0, -m] + ([math.log(N) - s - m / 2] if N else [])
    return float(np.logaddexp.reduce(terms))


def phi(m, s, N):
    x = math.exp(-m)
    y = N * math.exp(-s - m / 2) if N else 0.0
    return (x + y / 2) / (1 + x + y)


def P_target(m, s, N):
    return 1 / (1 + math.exp(-m) + (N * math.exp(-s - m / 2) if N else 0.0))


def L_red(c, g, s, a, p, N):
    m0, m1 = c + g * sig(-a), -(c + g * sig(a))
    return (1 - p) * F(m0, s, N) + p * F(m1, s, N)


worst = dict(fd=0.0, Lg=0.0, phi=0.0, mono=0.0, lip=0.0, lowF=0.0, drift0=0.0, driftmu=0.0, gradlb=0.0, levb=0.0, levc=0.0)
for K in (2, 3, 4, 6, 50):
    N = K - 2
    for _ in range(1500):
        p = rng.uniform(0.01, 0.49); c, g, a = rng.normal(0, 3, 3); s = rng.normal(0, 3) if N else 0.0
        hs = math.log((1 - p) / p); mu = p * (1 - 2 * p) / 14
        r0, r1 = sig(-a), sig(a); tau = math.tanh(a / 2)
        m0, m1 = c + g * r0, -(c + g * r1)
        e0, e1 = (1 - p) * phi(m0, s, N), p * phi(m1, s, N); err = e0 + e1
        Lc, Lg, La = e1 - e0, r1 * e1 - r0 * e0, g * dsig(a) * err
        d = 1e-6
        fd = [(L_red(c + d, g, s, a, p, N) - L_red(c - d, g, s, a, p, N)) / (2 * d),
              (L_red(c, g + d, s, a, p, N) - L_red(c, g - d, s, a, p, N)) / (2 * d),
              (L_red(c, g, s, a + d, p, N) - L_red(c, g, s, a - d, p, N)) / (2 * d)]
        worst["fd"] = max(worst["fd"], abs(fd[0] - Lc), abs(fd[1] - Lg), abs(fd[2] - La))
        worst["Lg"] = max(worst["Lg"], abs(Lg - 0.5 * (Lc + tau * err)))
        m = rng.normal(0, 4)
        fdphi = -(F(m + d, s, N) - F(m - d, s, N)) / (2 * d)
        Pt = P_target(m, s, N)
        worst["phi"] = max(worst["phi"], abs(fdphi - phi(m, s, N)), 0.5 * sig(-m) - phi(m, s, N),
                           0.5 * (1 - Pt) - phi(m, s, N), phi(m, s, N) - 1)
        x = abs(rng.normal(0, 2))
        worst["mono"] = max(worst["mono"], phi(m + x, s, N) - phi(m, s, N))
        worst["lip"] = max(worst["lip"], phi(m - 2 * x, s, N) - math.exp(2 * x) * phi(m, s, N))
        worst["lowF"] = max(worst["lowF"], math.log1p(math.exp(-m)) - F(m, s, N), -m - math.log1p(math.exp(-m)))
        h = c + g / 2
        if h <= hs / 2:
            worst["drift0"] = max(worst["drift0"], Lc)
        h2 = hs / 4 - abs(rng.normal(0, 1)); D2 = rng.uniform(-1, 1)          # (e), second claim: h <= h*/4, |Delta| <= 1
        Lc2 = p * phi(-h2 + D2 / 2, s, N) - (1 - p) * phi(h2 + D2 / 2, s, N)
        worst["driftmu"] = max(worst["driftmu"], Lc2 + mu)
        worst["gradlb"] = max(worst["gradlb"], tau ** 2 / 3 * (e0 ** 2 + e1 ** 2) - (Lc ** 2 + Lg ** 2))
        L = L_red(c, g, s, a, p, N)
        if P_target(m0, s, N) >= 0.5 and P_target(m1, s, N) >= 0.5:
            worst["levb"] = max(worst["levb"], L ** 2 / 32 - (e0 ** 2 + e1 ** 2))
        if L > 0.5 * p * math.log(2):
            worst["levc"] = max(worst["levc"], p ** 2 / 267 - (e0 ** 2 + e1 ** 2))
check("B. dL/dc, dL/dg, dL/da match finite differences (K = 2,3,4,6,50; 7500 points)", worst["fd"] < 1e-7, f"{worst['fd']:.2e}")
check("B. dL/dg = (dL/dc + tau err)/2", worst["Lg"] < 1e-14)
check("B. phi = -dF/dm, 0 < phi < 1, phi >= sigma(-m)/2, phi >= (1-P)/2", worst["phi"] < 1e-7, f"{worst['phi']:.2e}")
check("B. phi nonincreasing and phi(m-2x) <= e^{2x} phi(m)", worst["mono"] <= 1e-15 and worst["lip"] <= 1e-15)
check("B. F(m;s) >= l(m) >= -m", worst["lowF"] <= 1e-12)
check("B. Drift (e): dL/dc <= 0 if h <= h*/2;  dL/dc <= -mu if h <= h*/4, |Delta| <= 1",
      worst["drift0"] <= 1e-15 and worst["driftmu"] <= 1e-15, f"{worst['drift0']:.2e} {worst['driftmu']:.2e}")
check("B. (dL/dc)^2 + (dL/dg)^2 >= tau^2(err0^2+err1^2)/3", worst["gradlb"] <= 1e-15)
check("B. Levels (b), (c)", worst["levb"] <= 1e-15 and worst["levc"] <= 1e-15, f"{worst['levb']:.2e} {worst['levc']:.2e}")

# ----------------------------------------------------------------------------------------------- C. reduction vs full model
try:
    import torch
    torch.set_default_dtype(torch.float64)

    def full_loss(U, w, q, k, eps, v, p, dh, K):
        a = eps ** 2 * q * k / math.sqrt(dh)
        R = torch.roll(torch.eye(K), 1, 0)                   # R e_s = e_{s+1}
        W = w * (torch.eye(K) - R)
        loss = 0
        for s0 in range(K):
            x = torch.zeros(K); x[s0] = 1
            for b, pb in ((0, 1 - p), (1, p)):
                r = torch.sigmoid(-((-1) ** b) * a)
                z = eps * U @ x + eps * v * r * (W @ x)
                loss = loss + pb / K * torch.nn.functional.cross_entropy(z[None], torch.tensor([(s0 + b) % K]), reduction="sum")
        return loss

    worst_flow, worst_sym = 0.0, 0.0
    for K in (2, 3, 4, 6):
        N = K - 2
        R = torch.roll(torch.eye(K), 1, 0)
        for _ in range(60):
            eps, v, p, dh = rng.uniform(0.05, 0.5), rng.uniform(0.02, 1.0), rng.uniform(0.05, 0.95), int(rng.integers(1, 64))
            u0, u1, ubg, w0 = rng.normal(0, 2, 4)
            coef = [u0, u1] + [ubg] * N
            U = sum(cf * torch.linalg.matrix_power(R, j) for j, cf in enumerate(coef)).requires_grad_(True)
            w = torch.tensor(w0, requires_grad=True)
            q = torch.tensor(rng.normal(0, 1), requires_grad=True); k = torch.tensor(rng.normal(0, 1), requires_grad=True)
            Lfull = full_loss(U, w, q, k, eps, v, p, dh, K)
            gU, gw, gq, gk = torch.autograd.grad(Lfull, (U, w, q, k))
            worst_sym = max(worst_sym, float((gU - R.T @ gU @ R).abs().max()))
            du = [-float(gU[j, 0]) for j in range(K)]            # U[j, 0] is the coefficient u_j of R^j; dU/dt = -grad
            if N:
                worst_sym = max(worst_sym, max(abs(x - du[2]) for x in du[2:]))
            c, g = eps * (u0 - u1), 2 * eps * v * w0
            s = eps * ((u0 + u1) / 2 - ubg) if N else 0.0
            a = eps ** 2 * q.item() * k.item() / math.sqrt(dh)
            dc_full = eps * (du[0] - du[1])
            ds_full = eps * ((du[0] + du[1]) / 2 - du[2]) if N else 0.0
            dg_full = 2 * eps * v * (-gw.item() / (2 * K))      # dw/dt = -(dL/dw)/||I - R||_F^2
            da_full = eps ** 2 / math.sqrt(dh) * (-gq.item() * k.item() - q.item() * gk.item())
            m0, m1 = c + g * sig(-a), -(c + g * sig(a))
            e0, e1 = (1 - p) * phi(m0, s, N), p * phi(m1, s, N); err = e0 + e1
            Lc, Lg, La = e1 - e0, sig(a) * e1 - sig(-a) * e0, g * dsig(a) * err
            dd = 1e-6
            Ls = (L_red(c, g, s + dd, a, p, N) - L_red(c, g, s - dd, a, p, N)) / (2 * dd) if N else 0.0
            S = q.item() ** 2 + k.item() ** 2
            rho = 2 / K
            red = (-rho * eps ** 2 * Lc, -rho * (eps * v) ** 2 * Lg, -(eps ** 2 / (2 * N)) * Ls if N else 0.0, -eps ** 4 * S / dh * La)
            worst_flow = max(worst_flow, abs(dc_full - red[0]), abs(dg_full - red[1]), abs(ds_full - red[2]),
                             abs(da_full - red[3]), abs(Lfull.item() - L_red(c, g, s, a, p, N)))
    check("C. count-readout gradient stays circulant and background-symmetric (K = 2,3,4,6)", worst_sym < 1e-12, f"{worst_sym:.2e}")
    check("C. full-model loss and flow equal the reduced (c, g, s, a) system (K = 2,3,4,6; 240 points)", worst_flow < 1e-8,
          f"{worst_flow:.2e}")
except ImportError:
    print("SKIP  C. torch not available")

print(f"\n{fails} failure(s)")
sys.exit(fails)
