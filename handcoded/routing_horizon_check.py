"""Check the staged-routing constants (Lemmas R-scale and R-D) without torch.

    python3 handcoded/routing_horizon_check.py

Enumerates all prompts for n=2,4. Uses the exact prior for numerical stages B/C,
and checks the scalar Stage A contraction bound separately. This is a finite
smoke check of the inequalities, not a substitute for the proof.
"""
import itertools
import math

import numpy as np


def logadd(x, y):
    top = max(x, y)
    return top + math.log(math.exp(x - top) + math.exp(y - top))


def check(n, p, head_width, sigma, sign):
    a = sign * sigma**4 / math.sqrt(head_width)
    g = sigma**2
    pi0 = min(p, 1 - p) / 2
    ustar = math.log((1 - p) / p)
    C0 = 2 * math.log(10 / pi0)
    upper = 5 * (2 + 9 * C0)**2 / (pi0 * g**2 * a**2)
    y = 625 / (pi0 * g**2 * a**2)
    log_Tp = 2 * (math.log(64 * y) + 2 * math.log(math.log(64 * y)))      # log T'
    B = max(C0, 2 * logadd(0., log_Tp))
    # Lemma R-scale: the explicit threshold implies the smallness conditions.
    Dp = C0 + 8 * math.log(40000 / pi0)
    if sigma <= .1 * head_width**.125 * (1 + math.log(140 * n * Dp))**-.25:
        assert sigma * (1 + abs(ustar)) > 1 or abs(a) * math.log(2.1 * n * B / (.015 * abs(a))) <= .002
    eps = .015 * abs(a) / B
    target = math.log(2.1 * n / eps)
    alpha_star = p * (1 - p) * abs(a) / (60 * n * target)
    assert abs(a) <= .1 and abs(a) * target <= .008
    assert sigma * (1 + abs(ustar)) <= 1
    patterns = np.asarray(list(itertools.product([-1., 1.], repeat=n)))
    letters = np.tile(patterns, (2, 1))
    starts = np.repeat([-1., 1.], len(patterns))
    states = starts[:, None] * np.column_stack([
        np.ones(len(letters)), np.cumprod(letters[:, :-1], axis=1)])
    weight = np.prod(np.where(letters < 0, p, 1 - p), axis=1) / 2
    bias = np.zeros(2 * n + 2)

    def features(t):
        offsets = n + t - np.arange(n)
        scores = a * states[:, t, None] * letters + bias[offsets]
        # Stable softmax including the count's own site.
        scale = np.maximum(bias[0], scores.max(axis=1))
        own = np.exp(bias[0] - scale)
        values = np.exp(scores - scale[:, None])
        denominator = own + values.sum(axis=1)
        return offsets, values / denominator[:, None], own / denominator

    # Conditional uniform Fourier coefficient in the current (c,m) pair.
    def coefficient(value, t):
        coeff = 0.
        for c, m in itertools.product([-1., 1.], repeat=2):
            mask = (states[:, t] == c) & (letters[:, t] == m)
            coeff += c * m * np.dot(weight[mask], value[mask]) / weight[mask].sum() / 4
        return coeff

    _, prob, own = features(1)
    M = prob.sum(axis=1)
    muM = sign * coefficient(M, 1)
    mucur = sign * coefficient(prob[:, 1] * own, 1)
    assert muM >= .34 * abs(a) / n**2 * (1 - 1e-6)
    assert mucur >= .08 * abs(a) / n**2
    etaB = alpha_star / (2 * p * (1 - p) * g**2 * muM)
    gate_gradient = 0.
    for t in range(1, n):
        _, prob, _ = features(t)
        y = states[:, t] * letters[:, t]
        residual = -y / (1 + np.exp(y * ustar * states[:, t]))
        gate_gradient += np.dot(weight, residual * prob.sum(axis=1)) / (n - 1)
    wr = -etaB * g**2 * gate_gradient
    alpha = sign * wr
    assert .5 * alpha_star <= alpha <= 1.5 * alpha_star
    gradient = np.zeros_like(bias)
    for t in range(1, n):
        offsets, prob, own = features(t)
        M = prob.sum(axis=1)
        y = states[:, t] * letters[:, t]
        residual = -y / (1 + np.exp(y * (ustar * states[:, t] + wr * M)))
        gradient[0] += wr * np.dot(weight, residual * (-M * own)) / (n - 1)
        gradient[offsets] += wr * np.sum(weight[:, None] * residual[:, None] * prob * own[:, None], axis=0) / (n - 1)
    etaC = target / (2 * p * (1 - p) * alpha * mucur)
    bias -= etaC * gradient
    assert bias[n] >= target - .1 and bias[0] >= target - .5
    delta = bias[n] - bias[0]
    assert abs(delta) <= 1
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    Mplus, Mminus = sigmoid(delta + a), sigmoid(delta - a)
    mid, d = (Mplus + Mminus) / 2, (Mplus - Mminus) / 2
    assert abs(d) >= .18 * abs(a) * (1 - 1e-6)
    worst = float('inf')
    for t in range(1, n):
        _, prob, _ = features(t)
        y = states[:, t] * letters[:, t]
        actual = prob.sum(axis=1)
        ideal = np.where(y > 0, Mplus, Mminus)
        assert np.max(np.abs(actual - ideal)) <= eps * 1.001
        worst = min(worst, np.min(y * (actual - mid) / d))
    assert worst >= .5
    # Stage D starts at (0, ustar/sigma, wr/g).
    distance2 = (C0 * mid / (sigma * d))**2 + (ustar / sigma)**2 + ((wr - C0 / d) / g)**2
    burn = distance2 / (.2 * pi0)
    assert burn <= upper
    assert 6 * B * eps / abs(d) <= .5
    # After T': on a log grid of times, the upper bound on L(v(t)) stays below the lower bound forced by a
    # nonpositive margin (Lemma R-D).
    for log_t in np.linspace(log_Tp, log_Tp + 200, 401):
        lt = logadd(0., log_t)
        upper_L = math.exp(-lt) + math.exp(2 * math.log(2 + 1.5 * 2 * lt / abs(d)) - math.log(2 * g * g) - log_t)
        lower_L = pi0 * math.log(2) * math.exp(-(lt + .011) / B)
        xi = eps * (2 + 6 * lt / abs(d) + math.sqrt(2) * g)
        assert 2 * xi <= (lt + .011) / B
        assert upper_L < lower_L
    grid = np.linspace(min(0, ustar), max(0, ustar), 101)
    ss = 1 / (1 + np.exp(-grid))
    assert np.min(ss * (1 - ss)) >= p * (1 - p) * (1 - 1e-12)
    return burn


if __name__ == '__main__':
    count = 0
    for args in itertools.product([2, 4], [.1, .25, .5, .9], [1, 8], [.02, .1], [-1, 1]):
        check(*args)
        count += 1
    print(f'{count} exact-population cases pass coefficient, leakage, separator, burn-in and after-T\' bounds.')
