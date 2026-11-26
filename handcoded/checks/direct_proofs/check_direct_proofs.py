"""Finite exact-distribution checks for the new proof identities (not a formal proof)."""
from itertools import product
from math import comb, log, sqrt, tanh
from pathlib import Path
import re
import numpy as np

cases = cases_K2 = 0
for n in range(1, 8):
    bits = np.array(list(product((0, 1), repeat=n)))
    for K in (2, 3, 4, 6):
        alpha = np.exp(2j * np.pi / K)
        for p in (.1, .25, .49, .5, .75):
            weights = np.prod(np.where(bits == 1, p, 1-p), axis=1)
            target = np.eye(K)[bits.sum(axis=1) % K] - 1/K
            rho = np.array([1-p+p*alpha**j for j in range(1, K)])
            lam = np.max(np.abs(rho))
            spectrum = [comb(n, r)*np.sum(np.abs(rho)**(2*(n-r))*(1-np.abs(rho)**2)**r)/K for r in range(n+1)]
            assert abs(sum(spectrum)-(1-1/K)) < 1e-12
            for mask in range(1 << n):
                sites = [t for t in range(n) if mask >> t & 1]
                ids = bits[:, sites] @ (1 << np.arange(len(sites))) if sites else np.zeros(len(bits), dtype=int)
                mass = np.bincount(ids, weights=weights, minlength=1 << len(sites))
                cond = np.zeros((len(mass), K))
                np.add.at(cond, ids, weights[:, None]*target)
                cond /= mass[:, None]
                expected = np.sum(np.abs(rho)**(2*(n-len(sites))))/K
                assert np.max(np.abs(np.sum(cond**2, axis=1)-expected)) < 1e-12
                assert expected <= (1-1/K)*lam**(2*(n-len(sites))) + 1e-12
                # Arbitrary measurable test function: tower identity and contraction.
                h = np.sin(np.arange(len(mass)*K).reshape(len(mass), K)+.3)
                lhs = abs(np.sum(weights[:, None]*h[ids]*target))
                rhs = sqrt(np.sum(mass[:, None]*h*h))*lam**(n-len(sites))
                assert lhs <= rhs + 1e-12
                cases += 1
                cases_K2 += K == 2
print(f'PASS conditional contraction, tower identity, and target spectrum: {cases} support/distribution cases '
      f'({cases_K2} for counting modulo 2, the cases of Equation (9))')

cases = 0
for n in (2, 3, 8, 32):
    for p in (.01, .1, .25, .49):
        for K in (2, 3, 6):
            coefficients = []
            for t in range(2, n+1):
                m = n+2+t
                # N_t = the fixed query token + b_t + n-1 independent matches.
                emoment = 0.
                for b in (0, 1):
                    pb = p if b else 1-p
                    for k in range(n):
                        pk = comb(n-1, k)*p**k*(1-p)**(n-1-k)
                        nt = 1+b+k
                        emoment += pb*pk*(b-p)*(nt/m-nt**2/m**2)
                formula = p*(1-p)/m*(1-(3+2*(n-1)*p)/m)
                assert abs(emoment-formula) < 1e-13
                coefficients.append(emoment/K)
            coefficient = np.mean(coefficients)
            lower = p*(1-p)*min(1-2*p, .25)/(3*K*n)
            assert coefficient >= lower - 1e-14
            cases += 1
print(f'PASS explicit step-rule monomial coefficient and lower bound: {cases} cases (head scale factored out)')

# Corrected Phase III estimates; the old inequality fails strictly for positive small scales.
for asc in (1e-4, .01, .1, 1.):
    assert tanh(asc/16)**2 >= asc**2/1024
assert 12288 == 3*4*1024
assert 860160 == 12288*70
assert 98304 == 12288*8
assert 860160*log(2) < 6e5
assert 98304 < 1e5
print('PASS corrected tanh bound and all downstream Phase III constants')

folder = Path(__file__).resolve().parent
source = '\n'.join(f.read_text() for f in folder.glob('*.tex'))
labels = re.findall(r'\\label\{([^}]+)\}', source)
assert len(labels) == len(set(labels)), 'Duplicate labels'
refs = set()
for value in re.findall(r'\\(?:[Cc]ref|ref|eqref)\{([^}]+)\}', source):
    refs.update(value.split(','))
assert not (refs-set(labels)), f'Missing references: {refs-set(labels)}'
print('PASS all internal proof references resolve, with no duplicate labels')
