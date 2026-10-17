"""Reduced routing--interaction model: exact population gradient descent.

One softmax head over N positions: the true source (displayed s_{t-1}), one stale
position (displayed s_{t-2}) and N-2 exchangeable decoys.  Readout
    z = m_c * X + (r1*r2) * beta * X,   X = sum_j alpha_j x_j,
a copy channel (one weight) and a rule channel (a product of two weights, so its
initial credit is one order higher in the initialization scale, as for an MLP
interaction).  Logistic loss on y in {+-1}.  Expectations are exact: we enumerate
x*, x_p, beta, the error, and the binomial decoy sum.

Laws (both: the displayed successor is wrong with probability eps given (x*, beta)):
  sym   : error independent of everything,           P(err) = eps
  stale : error only when the correct value differs from x_p, P(err | differ) = 2 eps,
          so a wrong step re-displays the stale state.
"""
import sys, math, itertools
import torch
torch.set_default_dtype(torch.float64)

def table(N, eps, law):
    M = N - 2
    rows = []
    for xs, xp, b in itertools.product([1, -1], repeat=3):
        correct = b * xs
        perr = eps if law == "sym" else (2 * eps if correct != xp else 0.0)
        for err, pe in ((False, 1 - perr), (True, perr)):
            if pe == 0: continue
            y = -correct if err else correct
            for k in range(M + 1):          # k decoys equal to +1
                S = 2 * k - M
                pk = math.comb(M, k) / 2 ** M
                rows.append((xs, xp, b, y, S, 0.125 * pe * pk))
    return torch.tensor(rows)

def run(N, eps, law, eps0=0.05, lr=1.0, steps=400000, check=200):
    T = table(N, eps, law)
    xs, xp, b, y, S, w = T.T
    p = {k: torch.zeros((), requires_grad=True) for k in ("xi_s", "xi_p", "xi_d", "mc")}
    r1 = torch.tensor(eps0, requires_grad=True); r2 = torch.tensor(eps0, requires_grad=True)
    params = list(p.values()) + [r1, r2]
    out = dict(N=N, law=law, eps=eps, T_route=None, min_alpha_s=1.0 / N, t_min=0, max_alpha_p=1.0 / N)
    for t in range(steps):
        logits = torch.stack([p["xi_s"], p["xi_p"], p["xi_d"] + math.log(N - 2)])
        a = torch.softmax(logits, 0)
        a_s, a_p, a_d = a[0], a[1], a[2] / (N - 2)
        X = a_s * xs + a_p * xp + a_d * S
        z = p["mc"] * X + r1 * r2 * b * X
        L = (w * torch.nn.functional.softplus(-y * z)).sum()
        for q in params:
            q.grad = None
        L.backward()
        with torch.no_grad():
            for q in params:
                q -= lr * q.grad
        if t % check == 0:
            A = a_s.item()
            if A < out["min_alpha_s"]: out["min_alpha_s"], out["t_min"] = A, t
            out["max_alpha_p"] = max(out["max_alpha_p"], a_p.item())
            if A >= 0.5:
                out["T_route"] = t; break
    out["final_alpha_s"] = a_s.item(); out["mc"] = p["mc"].item(); out["mr"] = (r1 * r2).item()
    return out

if __name__ == "__main__":
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    for N in (8, 16, 32, 64):
        for law in ("sym", "stale"):
            r = run(N, eps, law)
            print(f"N={N:4d} {law:5s} T_route={r['T_route']} min_alpha_s*N={r['min_alpha_s']*N:.3g} "
                  f"(t={r['t_min']}) max_alpha_p={r['max_alpha_p']:.3f} mc={r['mc']:.3f} mr={r['mr']:.3f}", flush=True)
