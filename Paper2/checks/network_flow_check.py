"""Gradient flow run through the matrix network, compared with the reduced flow.

The routing times of the paper come from the reduced two-variable flow (r^2, q), whose right-hand side equals the
network gradient (Proposition 3; checked pointwise in one_block.py). Here the flow itself is run through the network:
the four tied coordinates (q_s, q_d, r_1, r_2) follow -grad L computed by autograd through the explicit one-block
matrices (embeddings, attention, residual, four-unit ReLU MLP) on the exact population, from (0, 0, eps, eps), until
source attention reaches 1/2. The routing time is compared with the reduced flow of one_block.integrate.
The same is done for the causal model of Theorem 11: the flow averages correlation loss over the N supervised prefixes
of causal_credit.populations and stops when every prefix routes; it is compared with causal_credit.route.
Run: python Paper2/checks/network_flow_check.py
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from one_block import OneBlock, integrate, population
from causal_credit import network as causal_network, populations as causal_populations, route as causal_route

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
ETA = 0.1


def network_route(n, k, eps, loss):
    X, x, b, y, p = population(n, k, ETA)
    net = OneBlock(scores=(0., 0.), factors=(eps, eps))
    params = [net.qs, net.qd, net.r1, net.r2]

    def rhs(t, theta):
        with torch.no_grad():
            for prm, val in zip(params, theta):
                prm.copy_(torch.tensor(val))
        z, _ = net(x, X, b, n)
        L = (p * (-y * z / 2 if loss == 'correlation' else torch.nn.functional.softplus(-y * z))).sum()
        g = torch.autograd.grad(L, params)
        return [-float(gi) for gi in g]

    def routed(t, theta):
        return theta[0] - theta[1] - math.log(n - 1)
    routed.terminal, routed.direction = True, 1
    sol = solve_ivp(rhs, (0, 1e8), [0., 0., eps, eps], method='DOP853', rtol=1e-10, atol=1e-14, events=routed)
    if not len(sol.t_events[0]):
        raise RuntimeError((n, k, eps, loss, 'no route'))
    qs, qd, r1, r2 = sol.y[:, -1]
    return float(sol.t_events[0][0]), abs(r1 - r2), abs(qs + qd), int(sol.nfev)


def causal_prefixes(n, p, cancelled):
    """Explicit state sequences of every supervised prefix, with labels and masses (as in causal_credit.matrix_audit)."""
    population, _ = causal_populations(n, p, cancelled)
    batches = []
    for i, rows in population.items():
        total, x, b, y, weight = (torch.tensor(rows[:, j]) for j in range(5))
        states = torch.empty(len(rows), i)
        for row in range(len(rows)):
            positive = int(round((i - 1 + total[row].item() - x[row].item()) / 2))
            states[row, :positive] = 1
            states[row, positive:i - 1] = -1
            states[row, i - 1] = x[row]
        batches.append((states, b, y, weight))
    return batches


def causal_network_route(n, p, cancelled, eps):
    """Correlation-loss flow of (q_s, q_d, r_1, r_2) through the causal network, averaged over the N prefixes."""
    batches = causal_prefixes(n, p, cancelled)

    def rhs(t, theta):
        params = torch.tensor(theta, requires_grad=True)
        loss = sum(-(w * y * causal_network(st, b, params)).sum() for st, b, y, w in batches) / (2 * n)
        return [-float(g) for g in torch.autograd.grad(loss, params)[0]]

    def routed(t, theta):
        return theta[0] - theta[1] - math.log(2 * n - 1)
    routed.terminal, routed.direction = True, 1
    sol = solve_ivp(rhs, (0, 1e9), [0., 0., eps, eps], method='DOP853', rtol=1e-10, atol=1e-14, events=routed)
    if not len(sol.t_events[0]):
        raise RuntimeError((n, p, cancelled, eps, 'no route'))
    qs, qd, r1, r2 = sol.y[:, -1]
    return float(sol.t_events[0][0]), max(abs(r1 - r2), abs(qs + qd))


def main():
    records = []
    for k, eps, loss in itertools.product((1., 0.), (.03, .01, .003), ('correlation', 'ce')):
        n = 16
        t_net, dr, dq, nfev = network_route(n, k, eps, loss)
        t_ode = integrate(n, k, eps, loss)[0]
        rec = dict(n=n, kappa=k, eps=eps, loss=loss, t_network=t_net, t_reduced=t_ode,
                   rel_diff=abs(t_net - t_ode) / t_ode, final_r1_minus_r2=dr, final_qs_plus_qd=dq, rhs_evals=nfev)
        records.append(rec)
        print(f"kappa={k:+.0f} eps={eps:<6} {loss:12s} network {t_net:12.4f}  reduced {t_ode:12.4f}  "
              f"rel diff {rec['rel_diff']:.1e}", flush=True)
    causal = []
    for n, (name, p, cancelled), eps in itertools.product((8, 16), (('clean', 0., False), ('independent', .2, False),
                                                                      ('cancelling', .2, True)), (.03, .01)):
        t_net, drift = causal_network_route(n, p, cancelled, eps)
        t_ode = causal_route(n, p, cancelled, eps)
        rec = dict(n=n, regime=name, eta=p, eps=eps, t_network=t_net, t_reduced=t_ode,
                   rel_diff=abs(t_net - t_ode) / t_ode, symmetry_drift=drift)
        causal.append(rec)
        print(f"causal N={n:2d} {name:11s} eps={eps:<5} network {t_net:12.4f}  reduced {t_ode:12.4f}  "
              f"rel diff {rec['rel_diff']:.1e}", flush=True)
    summary = dict(runs=len(records), max_rel_diff=max(r['rel_diff'] for r in records),
                   max_symmetry_drift=max(max(r['final_r1_minus_r2'], r['final_qs_plus_qd']) for r in records),
                   causal_runs=len(causal), causal_max_rel_diff=max(r['rel_diff'] for r in causal),
                   causal_max_symmetry_drift=max(r['symmetry_drift'] for r in causal),
                   records=records, causal_records=causal)
    out = ROOT / 'checks' / 'results'
    out.mkdir(exist_ok=True)
    (out / 'network_flow_check.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('records')}, indent=2))


if __name__ == '__main__':
    main()
