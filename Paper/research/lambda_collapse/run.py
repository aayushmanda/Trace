"""Outcome exit time from the ball B_eps of the plateau theorem, for several contraction coefficients lambda.

Same protocol as Table 7 (handcoded/regime_both.py): two blocks of width 16, three heads of dimension 8, MLP width 16,
|b_m| >= 1, zero readout, eps = 0.002, exact full-batch float64 gradients over all words, update length 1e-6 eps ||theta~||,
exit from radius r_0 eps. Only the match probability p changes: lambda = 1 - 2p for K = 2.

    python Paper/research/lambda_collapse/run.py P SEED [N ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "handcoded"))
import regime_both as rb  # noqa: E402

p, seed = float(sys.argv[1]), int(sys.argv[2])
ns = [int(x) for x in sys.argv[3:]] or [4, 6, 8, 10, 12, 14]
rb.PM = p                                   # data() reads the module-level match probability
sigma, step_len = 0.002, 1e-6
tilde = rb.theta_tilde(seed)
tilde["R"].zero_()                          # zero readout, as in the theorem
r_ball = rb.ball_radius(tilde)
out = os.path.join(os.path.dirname(__file__), f"exit_p{p}_s{seed}.json")
res = []
for n in ns:
    r = rb.run(n, sigma, "outcome", tilde, eps=r_ball, step_len=step_len, max_steps=int(2e7), radii=[r_ball])
    r.update(seed=seed, p=p, lam=abs(1 - 2 * p), step_len=step_len, r_ball=r_ball)
    res.append(r)
    print(f"p={p} seed={seed} n={n} T_exit={r.get('T')} relu_flips={r['relu_flips']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
