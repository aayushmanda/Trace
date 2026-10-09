"""Numerical check of the gradient-flow version of the staged routing theorem (Appendix G).

    python3 handcoded/theory_checks/routing_flow_check.py

Reduced model of Lemma R-features, exact population over all letter patterns and both start counts:
f_t = sigma v_c + sigma v_s c_t + g v_r M_t, with M_t the gate head's attention mass on the letters.
Stages: A flow on (v_c, v_s) to tolerance delta_A; B flow on v_r until alpha = sign(a) g v_r hits alpha_B;
C flow on beta until the largest attention outside {own position, current letter} is at most eps;
D flow on (v_c, v_s, v_r). Along Stage C it checks the corridor claims of the routing lemma:
b = (beta_n + beta_0)/2 nondecreasing, |beta_n - beta_0| <= 1, beta_delta <= 0 for every other offset,
and compares the hitting times with the proved bounds. Stage D is checked up to positive margins.
"""
import itertools
import math

import numpy as np
from scipy.integrate import solve_ivp


def setup(n, p):
    pats = np.array(list(itertools.product([1., -1.], repeat=n)))          # m_j (-1 = match)
    m = np.tile(pats, (2, 1))
    c0 = np.repeat([1., -1.], len(pats))
    w = np.prod(np.where(m < 0, p, 1 - p), axis=1) / 2
    c = np.column_stack([c0, c0[:, None] * np.cumprod(m[:, :-1], axis=1)])  # c_t, t = 1..n
    return m, c, w


def features(beta, a, m, c, n):
    """M[:, t], attention own/current/others for t = 2..n (index 1..n-1)."""
    M, own, cur, oth, dM = [], [], [], [], []
    for t in range(1, n):
        offs = n + t - np.arange(n)                      # offset of letter j from count position t
        sc = a * c[:, t, None] * m + beta[offs][None, :]
        top = np.maximum(sc.max(1), beta[0])
        e_l = np.exp(sc - top[:, None]); e_0 = np.exp(beta[0] - top)
        Z = e_0 + e_l.sum(1)
        pl = e_l / Z[:, None]; p0 = e_0 / Z
        M.append(pl.sum(1)); own.append(p0); cur.append(pl[:, t]); oth.append(pl.sum(1) - pl[:, t])
        # dM/dbeta_delta: + p0 * p_j for each letter at offset delta; dM/dbeta_0 = -p0 * M
        g = np.zeros((len(Z), len(beta)))
        np.add.at(g, (slice(None), offs), 0)            # placeholder for shape
        for j in range(n):
            g[:, offs[j]] += p0 * pl[:, j]
        g[:, 0] -= p0 * pl.sum(1)
        dM.append(g)
    return np.array(M).T, np.array(own).T, np.array(cur).T, np.array(oth).T, np.stack(dM, 1)


def run(n, p, a, sig, g):
    m, c, w = setup(n, p)
    T = n - 1
    y = c[:, 1:] * m[:, 1:]
    ustar = math.log((1 - p) / p)
    pi0 = min(p, 1 - p) / 2
    C0 = 2 * math.log(10 / pi0)
    yv = 625 / (pi0 * g * g * a * a)
    log_Tp = 2 * (math.log(64 * yv) + 2 * math.log(math.log(64 * yv)))
    B = max(C0, 2 * log_Tp)
    eps = .015 * abs(a) / B
    alpha_B = .05 * p * (1 - p) * abs(a) / n**2
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
    TB_bound = alpha_B * n**2 / (.65 * g**2 * p * (1 - p) * abs(a))
    assert solB.status == 1 and TB <= TB_bound * (1 + 1e-9), (TB, TB_bound)
    # Stage C: flow on beta until the leakage is at most eps; track the corridor.
    hist = []
    def leak(bb):
        _, _, _, (M, p0, pc, po) = loss_grad(v, bb)
        return po.max()
    def rhsC(t, bb):
        return -loss_grad(v, bb, need_beta=True)[2]
    evC = lambda t, bb: leak(bb) - eps
    evC.terminal = True
    solC = solve_ivp(rhsC, (0, 1e40), beta, events=evC, rtol=1e-9, atol=1e-12, method='LSODA', dense_output=True)
    assert solC.status == 1, solC.message
    TC = solC.t[-1]
    ts = np.concatenate([np.linspace(0, TC, 2000)])
    bs = solC.sol(ts)
    bmid = (bs[n] + bs[0]) / 2; Delta = bs[n] - bs[0]
    others = np.delete(bs, [0, n], axis=0)
    ok_b = np.all(np.diff(bmid) >= -1e-9)
    ok_D = np.all(np.abs(Delta) <= 1)
    ok_o = np.all(others <= 1e-9)
    TC_bound = 227 * n**5 * math.log(1.83 * n / eps) / (p**2 * (1 - p)**2 * a * a * eps)
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
    TD_bar = 5 * (2 + 9 * C0)**2 / (pi0 * g * g * a * a)
    return dict(n=n, p=p, a=a, eps=eps, alpha_B=alpha_B, TA=TA, TB=TB, TB_bound=TB_bound, TC=TC, TC_bound=TC_bound,
                b_end=float(bmid[-1]), Delta_max=float(np.abs(Delta).max()), others_max=float(others.max()),
                b_monotone=bool(ok_b), corridor=bool(ok_D and ok_o), TD=TD, TD_bar=TD_bar)


if __name__ == '__main__':
    for n, p, a in itertools.product([2, 4, 6], [.1, .25, .5, .9], [.05, -.05, .01]):
        r = run(n, p, a, sig=.1, g=.01)
        assert r['b_monotone'] and r['corridor'] and r['TC'] <= r['TC_bound'], r
        assert r['TD'] is not None and r['TD'] <= r['TD_bar'], r
        print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print('all cases: Stage B and C hitting times within their bounds, corridor invariant, margins positive in Stage D')
