"""Exact finite-state audit of causal, constant-rate error cancellation.

The DP state (prefix sum, final sign) is sufficient for both the error kernel
and the tied one-block attention read. No traces or gradients are sampled.
Run: python Paper2/checks/causal_credit.py
"""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def populations(n, p, cancelled):
    c = 1 - 2*p
    assert n >= 4 and 0 <= p < .5
    if cancelled:
        assert 2/(n-1) < min(2*p, c)
    mass = {(n, 1): .5, (-n, -1): .5}
    output = {}
    audit = {'max_moment_error': 0., 'max_local_error_error': 0.,
             'max_variance_error': 0., 'min_conditional_margin': 1.}
    for i in range(n, 2*n+1):
        variance = n*n+n-i
        rows = []
        next_mass = defaultdict(float)
        for (total, x), weight in mass.items():
            for b in (-1, 1):
                error = p + b*x*total/(2*variance) if cancelled else p
                assert 0 <= error < .5
                if cancelled:
                    assert error > 0
                audit['min_conditional_margin'] = min(audit['min_conditional_margin'], 1-2*error)
                for e in (-1, 1):
                    prob = weight*.5*(error if e == -1 else 1-error)
                    if prob == 0:
                        continue
                    y = b*x*e
                    rows.append((total, x, b, y, prob))
                    next_mass[(total+y, y)] += prob
        if i >= n+1:
            output[i] = np.array(rows, dtype=float)
            credit = 1 + sum(weight*x*(total-x) for (total, x), weight in mass.items())
            audit['max_moment_error'] = max(audit['max_moment_error'], abs(credit-(0 if cancelled else 1)))
            if cancelled:
                observed = sum(weight*total*total for (total, x), weight in mass.items())
                audit['max_variance_error'] = max(audit['max_variance_error'], abs(observed-variance))
            for x in (-1, 1):
                for b in (-1, 1):
                    subset = output[i][(output[i][:, 1] == x) & (output[i][:, 2] == b)]
                    wrong = subset[:, 3] != b*x
                    rate = subset[wrong, 4].sum()/subset[:, 4].sum()
                    audit['max_local_error_error'] = max(audit['max_local_error_error'], abs(rate-p))
        mass = next_mass
        assert abs(sum(mass.values())-1) < 1e-12
    assert audit['max_moment_error'] < 1e-10
    assert audit['max_local_error_error'] < 1e-12
    assert audit['max_variance_error'] < 1e-10
    return output, audit



def error_count_audit(n, p, cancelled):
    """Propagate (prefix sum, final sign, error count) before each instruction."""
    mass = {(n, 1, 0): .5, (-n, -1, 0): .5}
    history_error = 0.
    for i in range(n, 2*n+1):
        variance = n*n+n-i
        following = defaultdict(float)
        for (total, x, count), weight in mass.items():
            averaged_error = 0.
            for b in (-1, 1):
                error = p+b*x*total/(2*variance) if cancelled else p
                assert 0 < error < .5
                averaged_error += error/2
                for e in (-1, 1):
                    prob = weight*.5*(error if e == -1 else 1-error)
                    y = b*x*e
                    following[(total+y, y, count+int(e == -1))] += prob
            history_error = max(history_error, abs(averaged_error-p))
        mass = following
    observed = np.zeros(n+2)
    for (_, _, count), weight in mass.items():
        observed[count] += weight
    expected = np.array([math.comb(n+1, count)*p**count*(1-p)**(n+1-count)
                         for count in range(n+2)])
    discrepancy = float(np.max(np.abs(observed-expected)))
    wrong_tail = float(sum(weight for count, weight in enumerate(observed)
                           if count/(2*n) < p/4))
    bound = math.exp(-p*(n+1)/8)
    assert history_error < 1e-12
    assert discrepancy < 1e-12
    assert wrong_tail <= bound+1e-12
    return dict(max_history_error_rate_error=history_error,
                max_binomial_mass_error=discrepancy,
                probability_below_constant_fraction=wrong_tail,
                concentration_failure_bound=bound)


def network(states, b, parameters, bound=32.):
    qs, qd, r1, r2 = parameters.unbind()
    batch, length = states.shape
    embed = torch.zeros(batch, length+1, 7)
    embed[:, :length, 0] = states
    embed[:, :, 2] = 1
    embed[:, :length-1, 4] = 1
    embed[:, length-1, 3] = 1
    embed[:, length, 1] = b
    key = torch.stack((qs*0, qs*0, qs*0, qs, qd, qs*0, qs*0))
    query = embed[:, length] @ torch.tensor([0., 0., 1., 0., 0., 0., 0.])
    a = torch.softmax(query[:, None]*(embed[:, :length] @ key), dim=1)
    residual = embed[:, length].clone()
    residual[:, 5] += (a*states).sum(dim=1)
    rows = torch.zeros(4, 7)
    rows[:, 1] = bound*torch.tensor([1., 1., -1., -1.])
    rows[:, 5] = r1*torch.tensor([1., -1., 1., -1.])
    pre = residual @ rows.T - bound
    hidden = torch.where(pre > 0, pre, torch.where(pre < 0, pre*0, pre*.5))
    return residual[:, 6] + hidden @ (r2*torch.tensor([1., -1., -1., 1.]))


def coefficients(n, cancelled, s):
    i = np.arange(n+1, 2*n+1)
    a = 1/(1+(i-1)*np.exp(-s))
    if cancelled:
        return float(np.mean((i*a-1)/(i-1))), float(np.mean(i/(i-1)*a*(1-a)))
    return float(np.mean(a)), float(np.mean(a*(1-a)))


def matrix_audit(n, p, cancelled, population):
    maximum = 0.
    count = 0
    for gap, r in ((0., .01), (.2, .1), (math.log(2*n-1), .7)):
        params = torch.tensor([gap/2, -gap/2, r, r], requires_grad=True)
        loss = params.sum()*0
        for i, rows in population.items():
            total, x, b, y, weight = (torch.tensor(rows[:, j]) for j in range(5))
            states = torch.empty(len(rows), i)
            for row in range(len(rows)):
                # All decoys have equal scores. Their order cannot affect this network.
                positive = int(round((i-1+total[row].item()-x[row].item())/2))
                states[row, :positive] = 1
                states[row, positive:i-1] = -1
                states[row, i-1] = x[row]
            z = network(states, b, params)
            loss = loss - (weight*y*z).sum()/(2*n)
        loss.backward()
        h, hp = coefficients(n, cancelled, gap)
        c = 1-2*p
        target = np.array([c*r*r*hp/2, -c*r*r*hp/2, c*r*h/2, c*r*h/2])
        maximum = max(maximum, float(np.max(np.abs(-params.grad.numpy()-target))))
        count += 1
    assert maximum < 1e-12
    return maximum, count


def route(n, p, cancelled, u):
    c = 1-2*p
    gap = math.log(2*n-1)
    def rhs(time, state):
        squared, s = state
        h, hp = coefficients(n, cancelled, s)
        return [c*squared*h, c*squared*hp]
    def event(time, state):
        return state[1]-gap
    event.terminal = True
    event.direction = 1
    sol = solve_ivp(rhs, (0, 1e9), [u*u, 0.], events=event,
                    rtol=2e-10, atol=1e-15, method='DOP853')
    assert len(sol.t_events[0]) == 1
    time = float(sol.t_events[0][0])
    assert sol.y[0, -1] < 4*32**2
    if cancelled:
        lower = n/(2*math.e*c*u)*math.atan(1/u)
        upper = 12*n/c*(math.pi/u+4/math.e)
    else:
        lower = n/(3*math.e*c)*math.log(1+3/u**2)
        upper = 12*n/c*(math.log(1+u**-2)+1/math.e)
    assert lower <= time <= upper
    return time


def main():
    p = .2
    audit = []
    results = []
    clean_audit = []
    for n in (8, 16, 32):
        clean_population, clean_checks = populations(n, 0., False)
        clean_error, clean_configs = matrix_audit(n, 0., False, clean_population)
        clean_audit.append(dict(n=n, p=0., matrix_gradient_error=clean_error,
                                configurations=clean_configs, **clean_checks))
        for cancelled in (False, True):
            population, checks = populations(n, p, cancelled)
            error, configs = matrix_audit(n, p, cancelled, population)
            audit.append(dict(n=n, cancelled=cancelled, matrix_gradient_error=error,
                              configurations=configs, **checks,
                              **error_count_audit(n, p, cancelled)))
        for u in (.03, .01, .003, .001):
            clean = route(n, 0., False, u)
            fast = route(n, p, False, u)
            slow = route(n, p, True, u)
            rescale_error = abs((1-2*p)*fast/clean-1)
            assert rescale_error < 1e-7
            assert slow > fast > clean
            results.append(dict(n=n, p=p, u=u, clean_time=clean, fast_time=fast, cancelled_time=slow,
                                ratio=slow/fast, cancelled_clean_ratio=slow/clean,
                                reliability_rescale_relative_error=rescale_error,
                                expected_wrong_fraction=p*(n+1)/(2*n)))
    folder = ROOT/'checks/results'
    folder.mkdir(exist_ok=True)
    with (folder/'causal_credit.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    (folder/'causal_credit_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = [r'\begin{tabular}{@{}rr rrrr@{}}', r'\toprule',
             r'$N$ & $\varepsilon$ & clean & independent & structured & ratio to clean\\', r'\midrule']
    for row in results:
        lines.append(f"{row['n']} & {row['u']:g} & {row['clean_time']:,.1f} & {row['fast_time']:,.1f} & {row['cancelled_time']:,.1f} & {row['cancelled_clean_ratio']:.1f}"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'figures/causal_credit_table.tex').write_text('\n'.join(lines)+'\n')
    (folder/'causal_clean_audit.json').write_text(json.dumps(clean_audit, indent=2)+'\n')
    sweep = []
    reference_clean = next(row['clean_time'] for row in results if row['n'] == 16 and row['u'] == .01)
    reference_cancel = next((1-2*p)*row['cancelled_time'] for row in results if row['n'] == 16 and row['u'] == .01)
    for noise in (.1, .2, .3, .4):
        for cancelled in (False, True):
            time = route(16, noise, cancelled, .01)
            reference = reference_cancel if cancelled else reference_clean
            rescale_error = abs((1-2*noise)*time/reference-1)
            assert rescale_error < 1e-7
            sweep.append(dict(n=16, p=noise, u=.01, cancelled=cancelled, route_time=time,
                              scaled_time=(1-2*noise)*time, relative_rescale_error=rescale_error))
    with (folder/'causal_noise_sweep.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=sweep[0].keys())
        writer.writeheader(); writer.writerows(sweep)
    print(json.dumps(audit, indent=2))
    print('Clean matrix configurations:',sum(row['configurations'] for row in clean_audit))
    print('Three-regime causal routing runs:',3*len(results))
    print('Noise-rate sweep runs:',len(sweep))


if __name__ == '__main__':
    main()
