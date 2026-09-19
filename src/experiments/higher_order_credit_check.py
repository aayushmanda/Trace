"""Independent symbolic and autodiff checks for the higher-order credit results.

Run with `uv run python -m src.experiments.higher_order_credit_check`.
Uses small shared tables with repeated gates. No training or data mutation.
"""
from __future__ import annotations

import json

import numpy as np
import sympy as sp
import torch


def product(matrices, k):
    result = np.eye(k)
    for matrix in matrices:
        result = result @ matrix
    return result


def variations(tables, delta, gates, source, target):
    """Occurrence/bridge formulas, independently of the autodiff computation."""
    k = tables.shape[-1]
    p = product([tables[g] for g in gates], k)[source, target]
    first = second = 0.0
    for t, g in enumerate(gates):
        term = [tables[h] if j != t else delta[g] for j, h in enumerate(gates)]
        first += product(term, k)[source, target]
        for r in range(t):
            term[r] = delta[gates[r]]
            second += 2 * product(term, k)[source, target]
            term[r] = tables[gates[r]]
    return p, first, second


def main():
    rng = np.random.default_rng(20260918)
    torch.set_default_dtype(torch.float64)
    k, m = 3, 2
    uniform = np.ones((k, k)) / k
    center = np.eye(k) - uniform
    true = np.stack([np.eye(k)[[1, 2, 0]], np.eye(k)[[1, 0, 2]]])
    tables = rng.uniform(0.2, 1.0, (m, k, k))
    tables /= tables.sum(axis=-1, keepdims=True)
    targets = rng.uniform(0.2, 1.0, tables.shape)
    targets /= targets.sum(axis=-1, keepdims=True)
    coverage = rng.uniform(0.1, 1.0, (m, k))
    coverage /= coverage.sum()
    errors = {name: 0.0 for name in ('first_variation', 'hessian', 'tied_row', 'scaling', 'mixing_row_credit')}
    omega = 0.37
    for depth in (2, 3, 5, 8):
        gates = [i % m for i in range(depth)]
        truth = product([true[g] for g in gates], k)
        delta = rng.normal(size=tables.shape)
        delta -= delta.mean(axis=-1, keepdims=True)
        for source in range(k):
            target = int(truth[source].argmax())
            p, first, second = variations(tables, delta, gates, source, target)
            h = torch.tensor(0.0, requires_grad=True)
            perturbed = torch.tensor(tables) + h * torch.tensor(delta)
            state = torch.eye(k)[source]
            for g in gates:
                state = state @ perturbed[g]
            loss = -state[target].log()
            d1 = torch.autograd.grad(loss, h, create_graph=True)[0]
            d2 = torch.autograd.grad(d1, h)[0]
            errors['first_variation'] = max(errors['first_variation'], abs(d1.item() + first / p))
            errors['hessian'] = max(errors['hessian'], abs(d2.item() - (first / p)**2 + second / p))
        # A single-row direction is feasible but not in the rule subspace.
        row = np.zeros_like(tables)
        row[0, 1, 2], row[0, 1, 0] = 1, -1
        mixed = np.broadcast_to(uniform, tables.shape).copy()
        answer_credit = sum(variations(mixed, row, gates, s, int(truth[s].argmax()))[1] * k for s in range(k)) / k
        errors['mixing_row_credit'] = max(errors['mixing_row_credit'], abs(answer_credit))
        # Nonuniform tables tied only in the compared row retain nonzero answer credit.
        tied = tables.copy()
        tied[0, 1, 0] = tied[0, 1, 2] = (tables[0, 1, 0] + tables[0, 1, 2]) / 2
        h = torch.tensor(0.0, requires_grad=True)
        pt = torch.tensor(tied) + h * torch.tensor(row)
        trace_loss = -(torch.tensor(coverage[:, :, None] * targets) * pt.log()).sum()
        answer_loss = 0.0
        answer_credit = 0.0
        for s in range(k):
            y = int(truth[s].argmax())
            state = torch.eye(k)[s]
            for g in gates:
                state = state @ pt[g]
            answer_loss = answer_loss - state[y].log() / k
            p, first, _ = variations(tied, row, gates, s, y)
            answer_credit += first / p / k
        combined = omega * trace_loss + (1 - omega) * answer_loss
        observed = torch.autograd.grad(combined, h)[0].item()
        expected = omega * coverage[0, 1] * (targets[0, 1, 0] - targets[0, 1, 2]) / tied[0, 1, 0] - (1 - omega) * answer_credit
        errors['tied_row'] = max(errors['tied_row'], abs(observed - expected))
        rule_delta = np.stack([center @ rng.normal(size=(k, k)) @ center for _ in range(m)])
        for gamma in (0.0, 0.1, 0.6):
            affine = uniform + gamma * (true - uniform)
            for s in range(k):
                y = int(truth[s].argmax())
                _, first, second = variations(affine, rule_delta, gates, s, y)
                _, reference_first, reference_second = variations(true, rule_delta, gates, s, y)
                errors['scaling'] = max(errors['scaling'], abs(first - gamma**(depth-1)*reference_first), abs(second - gamma**(depth-2)*reference_second))
    assert max(errors.values()) < 1e-10, errors

    # Exact multivariate polynomial check includes all mixed shared-table derivatives.
    x, y = sp.symbols('x y')
    u = sp.ones(3) / 3
    e0 = sp.Matrix([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
    e1 = sp.Matrix([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
    symbolic_depths = (2, 3, 4, 8)
    for depth in symbolic_depths:
        full, perturbation = sp.eye(3), sp.eye(3)
        for t in range(depth):
            e = x * e0 if t % 2 == 0 else y * e1
            full = (full * (u + e)).applyfunc(sp.expand)
            perturbation = (perturbation * e).applyfunc(sp.expand)
        assert (full - u - perturbation).applyfunc(sp.simplify) == sp.zeros(3)
        for entry in full - u:
            if entry != 0:
                assert all(sum(powers) == depth for powers, _ in sp.Poly(entry, x, y).terms())

    gamma = sp.symbols('gamma', positive=True)
    for states, depth in ((3, 2), (16, 3), (16, 8)):
        loss = sp.log(states) - sp.log(1 + (states - 1) * gamma**depth)
        curvature = -depth * (states-1) * gamma**(depth-2) * ((depth-1)-(states-1)*gamma**depth) / (1+(states-1)*gamma**depth)**2
        assert sp.simplify(sp.diff(loss, gamma, 2) - curvature) == 0
        for order in range(1, depth):
            assert sp.limit(sp.diff(loss, gamma, order), gamma, 0) == 0
        assert sp.limit(sp.diff(loss, gamma, depth), gamma, 0) == -(states-1)*sp.factorial(depth)

    # The full row-feasible Hessian is nonzero at mixing (outside C).
    v = np.array([0.1, -0.2, 0.1])
    marginal_delta = np.broadcast_to(np.outer(np.ones(k), v), tables.shape)
    mixed = np.broadcast_to(uniform, tables.shape)
    marginal_curvatures = []
    for depth in (3, 8):
        gates = [i % m for i in range(depth)]
        values = [variations(mixed, marginal_delta, gates, 0, y) for y in range(k)]
        curvature = np.mean([(first/p)**2 - second/p for p, first, second in values])
        assert np.isclose(curvature, k * (v @ v))
        marginal_curvatures.append(float(curvature))

    # Check the sufficient Hessian bound on an orthonormal basis of C^M.
    raw_basis = []
    for g in range(m):
        for i in range(k-1):
            for j in range(k-1):
                entry = np.zeros((m, k, k))
                entry[g] = np.outer(np.eye(k)[i]-np.eye(k)[-1], np.eye(k)[j]-np.eye(k)[-1])
                raw_basis.append(entry.ravel())
    basis = np.linalg.qr(np.array(raw_basis).T)[0].T.reshape(-1, m, k, k)
    basis_t = torch.tensor(basis)
    gates = [0, 1, 0]
    truth = product([true[g] for g in gates], k)
    def losses(coordinates):
        pt = torch.tensor(tables) + torch.einsum('n,ngij->gij', coordinates, basis_t)
        trace = -(torch.tensor(coverage[:, :, None]*targets) * pt.log()).sum()
        terminal, weighted_probability = 0.0, 0.0
        for s in range(k):
            target = int(truth[s].argmax())
            state = torch.eye(k)[s]
            for g in gates:
                state = state @ pt[g]
            base_p = product([tables[g] for g in gates], k)[s, target]
            terminal = terminal - state[target].log() / k
            weighted_probability = weighted_probability + state[target] / base_p / k
        return trace, terminal, weighted_probability
    zero = torch.zeros(len(basis))
    trace_hessian = torch.autograd.functional.hessian(lambda z: losses(z)[0], zero).numpy()
    answer_hessian = torch.autograd.functional.hessian(lambda z: losses(z)[1], zero).numpy()
    hessian = omega * trace_hessian + (1 - omega) * answer_hessian
    bridge = torch.autograd.functional.hessian(lambda z: losses(z)[2], zero).numpy()
    m_trace = np.min(coverage[:, :, None] * targets / tables**2)
    lambda_bridge = np.linalg.eigvalsh(bridge).max()
    lower_bound = omega*m_trace - (1-omega)*lambda_bridge
    minimum_eigenvalue = np.linalg.eigvalsh(hessian).min()
    assert minimum_eigenvalue >= lower_bound - 1e-10
    stable_weight = 0.99
    stable_bound = stable_weight*m_trace - (1-stable_weight)*lambda_bridge
    stable_eigenvalue = np.linalg.eigvalsh(stable_weight*trace_hessian + (1-stable_weight)*answer_hessian).min()
    assert stable_eigenvalue >= stable_bound > 0
    print(json.dumps({
        'seed': 20260918,
        'status': 'passed',
        'autodiff_depths': [2, 3, 5, 8],
        'symbolic_depths': list(symbolic_depths),
        'max_absolute_errors': errors,
        'full_feasible_hessian_counterexample': marginal_curvatures,
        'restricted_hessian_min_eigenvalue': float(minimum_eigenvalue),
        'sufficient_hessian_lower_bound': float(lower_bound),
        'positive_bound_case': {'trace_weight': stable_weight, 'lower_bound': float(stable_bound), 'min_eigenvalue': float(stable_eigenvalue)},
        'inflection_K16_D8': (7/15)**(1/8),
    }, indent=2))


if __name__ == '__main__':
    main()
