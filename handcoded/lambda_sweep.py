"""Plateau exit time against the horizon law, with lambda and n varied separately.

Setting of regime_both.py ((A1)-(A4), exact loss over all 2^n words and both start states), outcome
loss only. The query letter has probability p, so the moves are i.i.d. Bernoulli(p) and counting
mod 2 has contraction lambda = |1 - 2p|. Theorem 1(ii) keeps the iterates in the ball for a time of
order 1/(sigma^2 + delta_n), delta_n = lambda^n + sigma lambda^(n-1). Reports T(0.1) and
T(0.1) * (sigma^2 + delta_n) for every (p, n); the second column should not depend on (p, n).
    uv run python handcoded/lambda_sweep.py
"""
import json
import os

import regime_both as RB

if __name__ == "__main__":
    sigma, res = float(os.environ.get("SWEEP_SIGMA", "0.002")), []
    tilde = RB.theta_tilde(int(os.environ.get("REGIME_SEED", "0")))
    for p in (0.1, 0.15, 0.25, 0.35):
        RB.PM = p
        lam = abs(1 - 2 * p)
        for n in (4, 6, 8, 10, 12):
            r = RB.run(n, sigma, "outcome", tilde, max_steps=40000)
            delta = lam ** n + sigma * lam ** (n - 1)
            r.update(p=p, lam=lam, delta=delta)
            res.append(r)
            T = r.get("T", float("nan"))
            print(f"p={p:.2f} lambda={lam:.2f} n={n:2d}  T(0.1)={T:.3e}  lambda^-n={lam ** -n:.3e}"
                  f"  T*(sigma^2+delta_n)={T * (sigma ** 2 + delta):.3f}  switching {r['switch_start']}->{r.get('switch_end')}",
                  flush=True)
            os.makedirs("results/theorem_regime", exist_ok=True)
            json.dump(res, open("results/theorem_regime/lambda_sweep.json", "w"))
