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
ap.add_argument("--start", nargs="+", default=["A"])
ap.add_argument("--rho", type=float, nargs="+", default=[0.0])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--p", type=float, default=0.25)
ap.add_argument("--path", type=int, default=200)
ap.add_argument("--outcome_steps", type=int, default=20000)
ap.add_argument("--outcome_lr", type=float, default=1.0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default=None)
ap.add_argument("--max_nfev", type=int, default=60000, help="cap on Stage C right-hand-side evaluations")
ap.add_argument("--nuisance_keys", nargs="*", default=None, help="restrict the B/C nuisance to these blocks")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)

D, DH, W, H = 16, 8, 16, 2                  # residual, head, MLP width, heads per block
S0, S1, Q, R, BOS, ANS = range(6)
EC, ES, E0, EL, ER = range(5)               # residual directions of the scaffold
sig, p = args.sigma, args.p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def constants(n, K=2):
    """Constants of the staged routing proof (eq:R-Dconst) for |a| = sigma^4 / sqrt(d_h), g = sigma^2, K = 2."""
    a, g = sig**4 / math.sqrt(DH), sig**2
    pi0, ustar = min(p, 1 - p) / K, math.log((1 - p) / p)
    alpha_B = .01 * p * (1 - p) * a / n**2
    ell_p, L_A = math.log(1 / min(p, 1 - p)), math.log(4 * K / alpha_B**2)
    N_A = 3 * math.sqrt(K) * math.sqrt(2 * ell_p**2 + K * L_A**2)
    Nbar, Nbar0 = 7.84 * math.sqrt(K) / a, 1 + sig * N_A + K * alpha_B
    C_D = 2 * math.log(10 * (K - 1) / pi0)
    TD_bar = 5 * (Nbar0 + C_D * Nbar)**2 / (pi0 * g * g)
    R_D = 4.3 * math.sqrt(K - 1) * (Nbar0**2 + Nbar**2) / (pi0 * g * g)
    L_D = math.log(100 * (K - 1) * R_D)
    log_Tp = 2 * math.log(100 * R_D * L_D**2)
    B = max(C_D, 2 * (math.log(K - 1) + log_Tp + math.log1p(math.exp(-log_Tp))), Nbar0 + 1.5)
    eps = .01 * a / (math.sqrt(K) * B)                   # the leakage level zeta
    bstar = math.log(1.83 * n / eps)                     # Stage C ends once (beta_n + beta_0) / 2 reaches this
    alpha = p * (1 - p) * a / (60 * n * bstar)
    TB_bound = .011 * K / (g * g)
    TC_bound = 1134 * n**5 * math.log(1.83 * n / eps) / (p**2 * (1 - p)**2 * a * a * eps)
    # mu_M = hat M and mu_cur = hat Psi_cur at beta = 0, a > 0: condition on (c, m_t), average the other letters.
    others = np.array(list(itertools.product([1., -1.], repeat=n - 1)))
    wo = np.prod(np.where(others < 0, p, 1 - p), 1)
    muM = mucur = 0.
    for c, m in itertools.product([1., -1.], repeat=2):
        Z = 1 + math.exp(a * c * m) + np.exp(a * c * others).sum(1)
        muM += c * m * np.dot(wo, 1 - 1 / Z) / 4
        mucur += c * m * np.dot(wo, math.exp(a * c * m) / Z**2) / 4
    small = dict(a_le=a <= .1, ab=a * bstar <= .008, sig_u=sig * (1 + abs(ustar)) <= 1)
    return dict(a=a, g=g, pi0=pi0, ustar=ustar, C_D=C_D, TD_bar=TD_bar, B=B, eps=eps, bstar=bstar, alpha=alpha,
                alpha_B=alpha_B, TB_bound=TB_bound, TC_bound=TC_bound,
                muM=muM, mucur=mucur, etaB=alpha / (2 * p * (1 - p) * g * g * muM),
                etaC=bstar / (2 * p * (1 - p) * alpha * mucur), smallness=small)


def data(n, fmt):
    pats = torch.tensor(list(itertools.product([0, 1], repeat=n)))           # b_j = 1: letter q (a match)
    b = pats.repeat(2, 1)
    s0 = torch.arange(2).repeat_interleave(len(pats))
    w = (p ** b.sum(1) * (1 - p) ** (n - b.sum(1))).double() / 2
    s = (s0[:, None] + b.cumsum(1)) % 2                                     # s_1..s_n
    letters = torch.where(b == 1, Q, R)
    if fmt == "step":
        ids = torch.cat([torch.full((len(b), 1), BOS), letters, s0[:, None], s[:, :-1]], 1)
        pos = torch.arange(n + 2, 2 * n + 1)                                 # positions predicting s_2..s_n
        tgt = s[:, 1:]
    else:
        ids = torch.cat([torch.full((len(b), 1), BOS), letters, s0[:, None], torch.full((len(b), 1), ANS)], 1)
        pos = torch.tensor([n + 2])
        tgt = s[:, -1:]
    y = (1 - 2 * tgt).double()                                              # y = (-1)^{s_t}
    return ids.to(dev), pos.to(dev), y.to(dev), w.to(dev)


def init(n, start, rho, seed):
    gen = torch.Generator().manual_seed(seed)
    sign = lambda: float(torch.randint(2, (1,), generator=gen) * 2 - 1)
    AQ, AK, AV = sign(), sign(), sign()
    L = 2 * n + 2
    z = lambda *s: torch.zeros(*s)
    th = dict(wte=z(6, D), wpe=z(L, D), beta=z(L), v=z(3))
    for b in range(2):
        th[f"q{b}"], th[f"k{b}"], th[f"v{b}"] = z(H, DH, D), z(H, DH, D), z(H, D, D)
        th[f"in{b}"], th[f"out{b}"] = z(W, D), z(D, W)
        th[f"b{b}"] = sig * (torch.randint(2, (W,), generator=gen) * 2 - 1).double()
    th["wte"][S0, EC], th["wte"][S0, ES] = sig, sig
    th["wte"][S1, EC], th["wte"][S1, ES] = sig, -sig
    th["wte"][Q, E0], th["wte"][Q, EL] = sig, -sig
    th["wte"][R, E0], th["wte"][R, EL] = sig, sig
    if start == "sign_flip":
        AQ = -AQ
    th["q1"][0, 0, ES] = sig * AQ
    th["k1"][0, 0, E0 if start == "l0_seed" else EL] = sig * AK
    th["v1"][0, ER, E0] = sig * AV
    if start == "random_seed":
        for key in ("q1", "k1", "v1"):
            r = torch.randn(th[key][0].shape, generator=gen)
            th[key][0] = sig * r / r.norm()
    if start == "prev_offset":
        th["beta"][n + 1] = 2.0
    if start in ("B", "C"):
        fan = dict(wte=(D,), wpe=(D,), **{f"{k}{b}": (D,) for k in "qkv" for b in range(2)},
                   **{f"in{b}": (D,) for b in range(2)}, **{f"out{b}": (W,) for b in range(2)})
        if args.nuisance_keys:
            fan = {k: fan[k] for k in args.nuisance_keys}
        for key, (fi,) in fan.items():
            noise = sig * torch.randn(th[key].shape, generator=gen) / math.sqrt(fi)
            if start == "B":
                th[key] = th[key] + rho * noise
            else:
                th[key] = math.sqrt(1 - rho) * th[key] + math.sqrt(rho) * noise
    return {k: t.to(dev) for k, t in th.items()}, dict(AQ=AQ, AK=AK, AV=AV)


def forward(th, ids, n, full_readout=None):
    """Residual stream and gate-head attention. Gate head: segment mask {i} + letters at positions >= n+1."""
    N, L = ids.shape
    x = th["wte"][ids] + th["wpe"][:L]
    i = torch.arange(L, device=dev)
    causal = i[None, :] <= i[:, None]
    seg = (i[None, :] == i[:, None]) | ((i[:, None] >= n + 1) & (i[None, :] >= 1) & (i[None, :] <= n))
    rel = (i[:, None] - i[None, :]).clamp(min=0)
    gate_att = None
    for b in range(2):
        q = torch.einsum("nld,hed->nhle", x, th[f"q{b}"])
        k = torch.einsum("nld,hed->nhle", x, th[f"k{b}"])
        v = torch.einsum("nld,hed->nhle", x, th[f"v{b}"])
        sc = q @ k.transpose(-1, -2) / math.sqrt(DH)
        mask = causal.expand(H, L, L).clone()
        if b == 1:
            sc[:, 0] = sc[:, 0] + th["beta"][rel]
            mask[0] = causal & seg
        att = sc.masked_fill(~mask, -math.inf).softmax(-1)
        if b == 1:
            gate_att = att[:, 0]
        x = x + (att @ v).sum(1)
        x = x + torch.relu(x @ th[f"in{b}"].T + th[f"b{b}"]) @ th[f"out{b}"].T
    return x, gate_att


def features(th, ids, pos, n):
    h, att = forward(th, ids, n)
    return h[:, pos][..., [EC, ES, ER]], att[:, pos]                       # (N, T, 3), (N, T, L)


def loss_np(v, X, y, w):
    m = y * (X @ v)
    return float(np.sum(w * np.logaddexp(0, -m)))


def grad_hess(v, X, y, w):
    m = y * (X @ v)
    s = sigmoid(-m)
    gr = -(w * y * s) @ X
    hs = (X * (w * s * (1 - s))[:, None]).T @ X
    return gr, hs


def attention_stats(att, n):
    """att: (N, T, L) at positions predicting s_2..s_n; current letter of row t (t = 2..n) sits at position t."""
    T = att.shape[1]
    t = torch.arange(2, n + 1, device=dev)
    own = att[:, torch.arange(T, device=dev), n + t]
    cur = att[:, torch.arange(T, device=dev), t]
    leak = 1 - own - cur
    return dict(att_current=cur.mean().item(), att_own=own.mean().item(), leak_max=leak.max().item(),
                leak_mean=leak.mean().item())


def run_step(n, start, rho, seed, C):
    ids, pos, y, w = data(n, "step")
    th, signs = init(n, start, rho, seed)
    a_signed = signs["AQ"] * signs["AK"] * C["a"]                          # signs already include sign_flip
    rec = dict(n=n, start=start, rho=rho, seed=seed, sigma=sig, p=p, signs=signs, consts={k: C[k] for k in
               ("a", "g", "eps", "alpha_B", "TB_bound", "TC_bound", "TD_bar", "B")}, smallness=C["smallness"])
    T = n - 1
    yy = y.reshape(-1).cpu().numpy()
    ww = (w[:, None].expand(-1, T) / T).reshape(-1).cpu().numpy()
    with torch.no_grad():
        X, att = features(th, ids, pos, n)
    Xn = X.reshape(-1, 3).cpu().numpy()
    # Stage A: Newton on (v_c, v_s) to the prior.
    v = np.zeros(3)
    for _ in range(100):
        gr, hs = grad_hess(v, Xn, yy, ww)
        step = np.linalg.solve(hs[:2, :2], gr[:2])
        v[:2] -= step
        if np.abs(step).max() < 1e-15 * (1 + np.abs(v).max()):
            break
    rec["stageA_v"] = v.tolist()
    # Stage B: gradient flow on v_r from 0 until alpha = sign(a) g v_r reaches alpha_B.
    g_signed = signs["AV"] * C["g"]
    s_alpha = math.copysign(1, a_signed) * g_signed
    evB = lambda t, z: s_alpha * z[0] - C["alpha_B"]
    evB.terminal = True
    solB = solve_ivp(lambda t, z: [-grad_hess(np.array([v[0], v[1], z[0]]), Xn, yy, ww)[0][2]],
                     (0, 100 * C["TB_bound"]), [0.], events=evB, method="LSODA", rtol=1e-10, atol=1e-30)
    v[2] = solB.y[0, -1]
    rec["TB_flow"] = float(solB.t[-1]) if solB.status == 1 else None
    rec["alpha_over_alpha_B"] = float(s_alpha * v[2] / C["alpha_B"])
    # Stage C: gradient flow on beta through the full model until the leakage is at most eps.
    yt, wt, vt = torch.tensor(yy, device=dev), torch.tensor(ww, device=dev), torch.tensor(v, device=dev)
    class Cap(Exception):
        pass
    last = {"b": th["beta"].detach().cpu().numpy(), "t": 0., "k": 0}
    def rhsC(t, bb):
        last["k"] += 1
        if last["k"] > args.max_nfev:
            raise Cap
        last["b"], last["t"] = bb.copy(), t
        th["beta"] = torch.tensor(bb, device=dev, requires_grad=True)
        Xt, _ = features(th, ids, pos, n)
        lossC = (wt * torch.nn.functional.softplus(-yt * (Xt.reshape(-1, 3) @ vt))).sum()
        lossC.backward()
        return -th["beta"].grad.cpu().numpy()
    def evC(t, bb):
        th["beta"] = torch.tensor(bb, device=dev)
        with torch.no_grad():
            _, att = features(th, ids, pos, n)
        return attention_stats(att, n)["leak_max"] - .99 * C["eps"]
    evC.terminal = True
    t0C = time.time()
    if rec["TB_flow"] is not None:                       # the schedule enters Stage C only if Stage B stops
        try:
            solC = solve_ivp(rhsC, (0, 100 * C["TC_bound"]), th["beta"].detach().cpu().numpy(), events=evC,
                             method="LSODA", rtol=1e-8, atol=1e-10)
            th["beta"] = torch.tensor(solC.y[:, -1], device=dev)
            rec["TC_flow"] = float(solC.t[-1]) if solC.status == 1 else None
            rec["stageC_cap"], traj = False, solC.y
        except Cap:                                      # stopping rule not met within the evaluation cap
            th["beta"] = torch.tensor(last["b"], device=dev)
            rec["TC_flow"], rec["stageC_cap"] = None, True
            rec["stageC_t_reached"] = last["t"]
            traj = last["b"][:, None]
        rec["stageC_nfev"] = last["k"]
    else:
        rec["TC_flow"], rec["stageC_nfev"] = None, 0
        traj = th["beta"].detach().cpu().numpy()[:, None]
    rec["stageC_seconds"] = time.time() - t0C
    rec["C_b_monotone"] = bool(np.all(np.diff((traj[n] + traj[0]) / 2) >= -1e-9))
    rec["C_Delta_max"] = float(np.abs(traj[n] - traj[0]).max())
    rec["C_others_max"] = float(np.delete(traj, [0, n], axis=0).max())
    beta = th["beta"].cpu().numpy()
    with torch.no_grad():
        X, att = features(th, ids, pos, n)
    Xn = X.reshape(-1, 3).cpu().numpy()
    st = attention_stats(att, n)
    bgate = np.zeros_like(beta); bgate[0] = bgate[n] = 1
    rec.update(st, beta_n=float(beta[n]), beta_0=float(beta[0]), Delta=float(beta[n] - beta[0]),
               beta_other_max=float(np.delete(beta[:2 * n + 1], [0, n]).max()),
               align=float(beta @ bgate / (np.linalg.norm(beta) * np.linalg.norm(bgate) + 1e-300)))
    rec["leak_ok"] = st["leak_max"] <= C["eps"]
    rec["Delta_ok"] = abs(rec["Delta"]) <= 1
    # Stage D: gradient flow on v in log time tau = log(1 + t).
    margin = lambda vv: float(np.min(yy * (Xn @ vv)))
    d = .5 * math.tanh(a_signed / 2)
    vstar = np.array([-1 / (2 * d * sig), 0., 1 / (g_signed * d)])
    # Implicit Euler (proximal) steps on a geometric time grid: v_{k+1} = argmin L(v) + |v - v_k|^2 / (2 h_k),
    # solved by Newton; unconditionally stable, monotone in the loss, time resolution r - 1 = 5%.
    t_max, r = 100 * C["TD_bar"], 1.05
    _, hs = grad_hess(v, Xn, yy, ww)
    h, t_now, k = 1e-3 / np.linalg.eigvalsh(hs).max(), 0., 0
    tau_in, mins, vv = (0. if margin(v) > 0 else None), [], v.copy()
    t0 = time.time()
    while t_now < t_max:
        z = vv.copy()
        for _ in range(50):
            gr, hs = grad_hess(z, Xn, yy, ww)
            gz = gr + (z - vv) / h
            step = np.linalg.solve(hs + np.eye(3) / h, gz)
            z -= step
            if np.all(np.abs(step) <= 1e-13 * (np.abs(z) + 1)):
                break
        vv, t_now, h, k = z, t_now + h, h * r, k + 1
        mg = margin(vv)
        if tau_in is None and mg > 0:
            tau_in = t_now
        elif tau_in is not None:
            mins.append(mg)
    rec["stageD_steps"], rec["stageD_seconds"] = k, time.time() - t0
    rec["TD_flow"] = tau_in
    rec["margin_after_entry_min"] = float(min(mins)) if mins else None
    rec["stays_positive"] = bool(tau_in is not None and (not mins or min(mins) > 0))
    vend = vv
    rec["final_margin"], rec["final_loss"] = margin(vend), loss_np(vend, Xn, yy, ww)
    rec["final_acc"] = float(np.sum(ww * (yy * (Xn @ vend) > 0)))
    rec["prior_loss"] = -p * math.log(p) - (1 - p) * math.log(1 - p)
    rec["in_G"] = bool(rec["leak_ok"] and rec["Delta_ok"] and rec["stays_positive"])
    rec["T_enter_G"] = rec["TD_flow"] if rec["in_G"] else None
    # Straight path from (beta_C, v_end) to a rescaled theta*_gate (Proposition gate executor).
    if args.path:
        b_star = np.zeros_like(beta); b_star[0] = b_star[n] = math.log(10 * n / abs(a_signed))
        v_end_star = vstar * np.linalg.norm(vend) / np.linalg.norm(vstar)
        pm, pl = [], []
        for s in np.linspace(0, 1, args.path):
            th["beta"] = torch.tensor((1 - s) * beta + s * b_star, device=dev)
            with torch.no_grad():
                Xs, _ = features(th, ids, pos, n)
            Xs = Xs.reshape(-1, 3).cpu().numpy(); vs = (1 - s) * vend + s * v_end_star
            pm.append(float(np.min(yy * (Xs @ vs)))); pl.append(loss_np(vs, Xs, yy, ww))
        rec.update(path_min_margin=min(pm), path_max_loss=max(pl), path_end_margin=pm[-1],
                   path_start_margin=pm[0], path_barrier_free=bool(min(pm) > 0))
    return rec


def run_outcome(n, start, rho, seed):
    ids, pos, y, w = data(n, "outcome")
    th, signs = init(n, "B", rho, seed)                                    # rho = 0 is the scaffold
    th["readout"] = torch.zeros(D, device=dev)                             # full readout, zero (Corollary)
    th0 = {k: t.clone() for k, t in th.items()}
    if start == "outcome_gd":
        trained = [k for k in th if k not in ("beta", "v")]
    else:
        trained = ["beta", "readout"]
    for k in trained:
        th[k].requires_grad_(True)
    log = []
    for step in range(args.outcome_steps + 1):
        h, _ = forward(th, ids, n)
        f = h[:, pos[0]] @ th["readout"]
        lossv = (w * torch.nn.functional.softplus(-y[:, 0] * f)).sum()
        if step % max(args.outcome_steps // 20, 1) == 0:
            with torch.no_grad():
                dist = math.sqrt(sum(((th[k] - th0[k]) ** 2).sum().item() for k in trained))
                log.append(dict(step=step, loss_minus_log2=lossv.item() - math.log(2),
                                acc=(w * (((y[:, 0] * f) > 0) + .5 * (f == 0))).sum().item(),
                                logit_spread=(f.max() - f.min()).item(), dist_over_sigma=dist / sig))
        for k in trained:
            th[k].grad = None
        lossv.backward()
        with torch.no_grad():
            for k in trained:
                th[k] -= args.outcome_lr * th[k].grad
    return dict(n=n, start=start, rho=rho, seed=seed, sigma=sig, p=p, lr=args.outcome_lr, steps=args.outcome_steps,
                lam_n=abs(1 - 2 * p) ** n, log=log)


results = []
for n in args.n:
    C = constants(n)
    for start in args.start:
        for rho in (args.rho if start in ("B", "C", "outcome_gd") else [0.0]):
            for seed in args.seeds:
                t0 = time.time()
                rec = run_outcome(n, start, rho, seed) if start.startswith("outcome") else run_step(n, start, rho, seed, C)
                rec["seconds"] = time.time() - t0
                results.append(rec)
                if start.startswith("outcome"):
                    last = rec["log"][-1]
                    print(f"n={n} {start} rho={rho} seed={seed} acc={last['acc']:.4f} loss-log2={last['loss_minus_log2']:.2e} "
                          f"spread={last['logit_spread']:.2e} dist/sigma={last['dist_over_sigma']:.3f} ({rec['seconds']:.0f}s)",
                          flush=True)
                else:
                    print(f"n={n} {start} rho={rho} seed={seed} cur={rec['att_current']:.3f} own={rec['att_own']:.3f} "
                          f"leak={rec['leak_max']:.2e}/{C['eps']:.1e} Delta={rec['Delta']:+.3f} align={rec['align']:.3f} "
                          f"TD={rec['TD_flow']} inG={rec['in_G']} margin={rec['final_margin']:+.3f} "
                          f"path_min={rec.get('path_min_margin', float('nan')):+.3f} ({rec['seconds']:.0f}s)", flush=True)
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(json.dumps(results, default=float))
