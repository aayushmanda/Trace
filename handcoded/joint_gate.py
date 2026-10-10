"""Joint gradient flow of the reduced gate model from randomly scaled useful directions (counting modulo 2).

The directions come from the hand-coded gate of Proposition 2; only their magnitudes are random. Exact population over all
letter patterns and both start counts. At the position predicting s_t (t = 2..n) the gate head scores the letter at j by
a c_t m_j + beta_{n+t-j} and its own position by beta_0, with c_t = (-1)^{s_{t-1}} the displayed count and m_j = (-1)^{b_j}.
With M_t the gate's attention mass on the letters, the logit of y_t = (-1)^{s_t} is

    f_t = eps v_c + eps v_s c_t + eps g v_r M_t,      a = eps^2 q k / sqrt(d_h),

where q, k, g are the gate's query, key and value factors, (v_c, v_s) the count readout, v_r the gate readout and beta the
relative bias (offsets 0..2n; 0 is the own position, n the current letter). Every factor trains at once: no stages.
Initialization: q, k, g = eps * sign * U[1/2, 2]; readout and beta zero.

Logged along the trajectory: loss, smallest margin, leakage (largest attention outside own position and current letter),
a, the gate alignment a g v_r, the rule credit Phi_rule = E[(y_t - (1-2p) c_t) f_t / 2] with L_step = L_prior - Phi_rule,
the interference ratio <grad Phi, grad L_prior> / ||grad Phi||^2 (d Phi/dt = ||grad Phi||^2 (1 - ratio)), the beta-gradient on
the own, current-letter and distractor offsets, and the conserved quantities q^2 - k^2 and g^2 - v_r^2.

    python handcoded/joint_gate.py --n 4 6 8 --eps 0.3 --seeds 0 1 2 --out results/joint_gate/flow.json

Gradient flow is integrated in log time tau = log(1 + t) with LSODA, until every margin is positive or t = e^tau_max.
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[6])
ap.add_argument("--eps", type=float, nargs="+", default=[0.3])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--p", type=float, default=0.25)
ap.add_argument("--dh", type=int, default=8)
ap.add_argument("--tau_max", type=float, default=60.0, help="stop at t = e^tau_max")
ap.add_argument("--points", type=int, default=241, help="recorded points, uniform in tau")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(args.threads)
NAMES = ("q", "k", "g", "vc", "vs", "vr")


def setup(n, p):
    pats = torch.tensor(list(itertools.product([1., -1.], repeat=n)))            # m_j, -1 = match
    m = pats.repeat(2, 1)
    c0 = torch.tensor([1., -1.]).repeat_interleave(len(pats))
    w = torch.prod(torch.where(m < 0, p, 1 - p), 1) / 2
    c = torch.cat([c0[:, None], c0[:, None] * torch.cumprod(m[:, :-1], 1)], 1)    # c_t, t = 1..n
    y = c[:, 1:] * m[:, 1:]                                                       # (-1)^{s_t}, t = 2..n
    offs = torch.stack([n + t - torch.arange(n) for t in range(1, n)])            # offset of letter j from position t
    return m, c, w, y, offs


def unpack(x):
    th = dict(zip(NAMES, x[:6]))
    th["beta"] = x[6:]
    return th


def forward(x, n, D, eps):
    m, c, w, y, offs = D
    th = unpack(x)
    a = eps ** 2 * th["q"] * th["k"] / math.sqrt(args.dh)
    beta = th["beta"]
    sc = a * c[:, 1:, None] * m[:, None, :] + beta[offs][None]                   # (N, T, n) letter scores
    own = beta[0].expand(sc.shape[:2])[..., None]
    att = torch.softmax(torch.cat([own, sc], -1), -1)
    M = att[..., 1:].sum(-1)
    f = eps * th["vc"] + eps * th["vs"] * c[:, 1:] + eps * th["g"] * th["vr"] * M
    wt = w[:, None] / (n - 1)
    loss = (wt * torch.nn.functional.softplus(-y * f)).sum()
    phi = (wt * (y - (1 - 2 * args.p) * c[:, 1:]) * f / 2).sum()
    cur = att[:, torch.arange(n - 1), 1 + torch.arange(1, n)]                     # letter t sits at index t (1-based)
    leak = (1 - att[..., 0] - cur).max()
    return loss, phi, (y * f).min(), leak, a


def grad(x, n, D, eps):
    xt = torch.tensor(x, requires_grad=True)
    return torch.autograd.grad(forward(xt, n, D, eps)[0], xt)[0].numpy()


def init(n, eps, seed):
    gen = torch.Generator().manual_seed(seed)
    mag = 0.5 + 1.5 * torch.rand(3, generator=gen)
    sgn = torch.randint(2, (3,), generator=gen) * 2 - 1
    x = torch.zeros(6 + 2 * n + 1)
    x[:3] = eps * sgn * mag
    return x


def record(x, n, D, eps, t):
    xt = torch.tensor(x, requires_grad=True)
    loss, phi, minm, leak, a = forward(xt, n, D, eps)
    gl = torch.autograd.grad(loss, xt, retain_graph=True)[0]
    gphi = torch.autograd.grad(phi, xt)[0]
    gp2 = float((gphi * gphi).sum())
    gb = gl[6:].numpy()
    dist = np.delete(gb[1:], n - 1)                                               # offsets other than 0 and n
    th = unpack(x)
    return dict(t=float(t), loss=float(loss), phi=float(phi), min_margin=float(minm), leak=float(leak), a=float(a),
                gate_align=float(a) * float(th["g"]) * float(th["vr"]),
                interference=float((gphi * (gl + gphi)).sum()) / gp2 if gp2 > 0 else float("nan"),
                dL_dbeta_own=float(gb[0]), dL_dbeta_cur=float(gb[n]),
                dL_dbeta_dist_min=float(dist.min()), dL_dbeta_dist_max=float(dist.max()),
                beta_own=float(th["beta"][0]), beta_cur=float(th["beta"][n]),
                beta_dist_max=float(np.delete(th["beta"][1:], n - 1).max()),
                qk_inv=float(th["q"] ** 2 - th["k"] ** 2), gv_inv=float(th["g"] ** 2 - th["vr"] ** 2))


def run(n, eps, seed):
    D = setup(n, args.p)
    x0 = init(n, eps, seed).numpy()
    rhs = lambda tau, x: -math.exp(tau) * grad(x, n, D, eps)                       # dx/dtau, t = e^tau - 1
    def margin(tau, x):
        with torch.no_grad():
            return float(forward(torch.tensor(x), n, D, eps)[2])
    margin.terminal, margin.direction = True, 1
    taus = np.linspace(0, args.tau_max, args.points)
    sol = solve_ivp(rhs, (0, args.tau_max), x0, method="LSODA", t_eval=taus, events=margin, rtol=1e-8, atol=1e-12)
    traj = [record(sol.y[:, i], n, D, eps, math.expm1(sol.t[i])) for i in range(sol.y.shape[1])]
    solved = sol.status == 1
    end = sol.y_events[0][0] if solved else sol.y[:, -1]
    t_end = math.expm1(sol.t_events[0][0]) if solved else math.expm1(sol.t[-1])
    return dict(solved_at=t_end if solved else None, final=record(end, n, D, eps, t_end), traj=traj,
                message=sol.message, nfev=int(sol.nfev))


results = []
for n in args.n:
    for eps in args.eps:
        for seed in args.seeds:
            t0 = time.time()
            rec = run(n, eps, seed)
            rec.update(n=n, eps=eps, seed=seed, p=args.p, seconds=time.time() - t0)
            results.append(rec)
            f, f0 = rec["final"], rec["traj"][0]
            print(f"n={n:2d} eps={eps} seed={seed}: solved_at={rec['solved_at']} margin={f['min_margin']:+.3e} "
                  f"leak={f['leak']:.2e} a={f0['a']:+.2e}->{f['a']:+.2e} align={f['gate_align']:+.2e} "
                  f"|beta_cur|={f['beta_cur']:+.2f} inv drift qk={abs(f['qk_inv'] - f0['qk_inv']):.1e} "
                  f"gv={abs(f['gv_inv'] - f0['gv_inv']):.1e} ({rec['seconds']:.0f}s, nfev {rec['nfev']})", flush=True)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results, indent=1))
