"""Rule-credit cancellation in the reduced model.

Decoys share one displayed value w (a coherent context) with E[x* w] = -r, and the
displayed successor is y = beta x* eta with eta independent of the context (so the
local law, the error rate and the Bayes predictor given the full context are those
of the symmetric law).  At uniform attention the rule credit is
    E[y beta X] = (1-2eps)(1 + (N-1)(-r)) / N = (1-2eps) kappa / N,
kappa = 1-(N-1)r.  kappa = 1 is the symmetric law; kappa = 0 cancels exactly.
Single shared readout z = m beta X with m = r1 r2 (rule channel only).
"""
import sys, math, itertools
import torch
torch.set_default_dtype(torch.float64)

def run(N, eps, kappa, eps0=0.01, lr=0.5, steps=3_000_000, check=10):
    r = (1 - kappa) / (N - 1)
    rows = []
    for xs, w, b in itertools.product([1, -1], repeat=3):
        pj = 0.25 * (1 - r * xs * w) * 0.5          # P(x*, w) with E[x* w] = -r; beta uniform
        for err, pe in ((False, 1 - eps), (True, eps)):
            y = -b * xs if err else b * xs
            rows.append((xs, w, b, y, pj * pe))
    xs, w, b, y, p = torch.tensor(rows).T
    xi_s = torch.zeros((), requires_grad=True); xi_d = torch.zeros((), requires_grad=True)
    r1 = torch.tensor(eps0, requires_grad=True); r2 = torch.tensor(eps0, requires_grad=True)
    ps = [xi_s, xi_d, r1, r2]
    for t in range(steps):
        a = torch.softmax(torch.stack([xi_s, xi_d + math.log(N - 1)]), 0)
        X = a[0] * xs + a[1] * w
        L = (p * torch.nn.functional.softplus(-y * r1 * r2 * b * X)).sum()
        for q in ps: q.grad = None
        L.backward()
        with torch.no_grad():
            for q in ps: q -= lr * q.grad
        if t % check == 0 and a[0].item() >= 0.5:
            return t
    return None

if __name__ == "__main__":
    eps = 0.1
    mode = sys.argv[1] if len(sys.argv) > 1 else "N"
    if mode == "N":            # Table (eps0 = 1e-2 rows)
        for N in (8, 16, 32, 64):
            res = {k: run(N, eps, k) for k in (1.0, 0.5, 0.1, 0.01, 0.0)}
            print(f"N={N:3d} " + " ".join(f"kappa={k}:T={v}" for k, v in res.items()), flush=True)
    elif mode == "eps0":       # Figure: N = 16
        for e0 in (0.03, 0.01, 0.003, 0.001):
            print(f"eps0={e0} kappa=1:T={run(16, eps, 1.0, eps0=e0)} kappa=0:T={run(16, eps, 0.0, eps0=e0)}", flush=True)
    elif mode == "kappa":      # Table (eps0 = 1e-3 rows)
        for N in (16, 32):
            for k in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0):
                print(f"N={N} kappa={k} T={run(N, eps, k, eps0=0.001)}", flush=True)
