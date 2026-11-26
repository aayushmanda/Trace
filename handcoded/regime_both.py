"""Exact float64 population GD exit times inside the outcome plateau (Theorem 1, Table 5 / tab:exit-times, Appendix G.1).

Computes exit times from the initialization ball B_eps (eps = 0.002) and macroscopic radius r = 0.1 (eps = 0.02)
on count mod 2 (p = 1/4, lambda = 1/2) over all 2^n words under both step and outcome objectives.

    REGIME_ZERO_READOUT=1 REGIME_STOP=ball REGIME_STEP=1e-6 REGIME_SEED=0 uv run python handcoded/regime_both.py 0.002
    REGIME_ZERO_READOUT=1 uv run python handcoded/regime_both.py 0.02
"""
import itertools
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, K, Wd, Hh, Dh, F, TMAX, PM = 2, 2, 16, 3, 8, 16, 40, 0.25
SEP, COLON = A + K, A + K + 1


def theta_tilde(seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    th = {"wte": rn(A + K + 3, Wd), "wpe": rn(TMAX, Wd), "R": rn(K, Wd)}
    for b in range(2):
        th[f"{b}.q"], th[f"{b}.k"], th[f"{b}.v"] = rn(Hh, Dh, Wd), rn(Hh, Dh, Wd), rn(Hh, Wd, Wd)
        th[f"{b}.mi"], th[f"{b}.mo"] = rn(F, Wd), rn(Wd, F)
        mb = rn(F)
        th[f"{b}.mb"] = torch.sign(mb) * (1 + mb.abs())
    return {k: v.to(dev) for k, v in th.items()}


def data(n):
    """All 2^n words and both start states, with their probabilities; query letter 0."""
    words = torch.tensor(list(itertools.product(range(A), repeat=n)))
    match = (words == 0).long()
    prob = (PM ** match.sum(1) * (1 - PM) ** (n - match.sum(1))).double() / K
    words, match, prob = words.repeat(K, 1), match.repeat(K, 1), prob.repeat(K)
    s0 = torch.arange(K).repeat_interleave(2 ** n)
    states = (s0[:, None] + match.cumsum(1)) % K                       # s_1..s_n
    head = torch.cat([(A + s0)[:, None], torch.zeros(len(words), 1, dtype=torch.long), words,
                      torch.full((len(words), 1), SEP)], 1)            # positions 0..n+2
    out_ids = torch.cat([head, torch.full((len(words), 1), COLON)], 1)  # ':' at n+3 predicts s_n
    step_ids = torch.cat([head, A + states[:, : n - 1]], 1)            # s_t at n+2+t, t = 1..n-1
    return dict(prob=prob.to(dev), out_ids=out_ids.to(dev), out_tgt=states[:, -1].to(dev),
                step_ids=step_ids.to(dev), step_tgt=states[:, 1:].to(dev))


def residual(p, ids, want_signs=False, signs=None):
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    switching = 0
    for b in range(2):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(Dh)).masked_fill(future, -math.inf)
        x = x + (att.softmax(-1) @ v).sum(1)
        pre = x @ p[f"{b}.mi"].T + p[f"{b}.mb"]
        if signs is not None:
            signs.append(pre.detach() > 0)
        if want_signs:
            on = pre > 0
            switching += int((on.flatten(0, 1).any(0) != on.flatten(0, 1).all(0)).sum())
        x = x + torch.relu(pre) @ p[f"{b}.mo"].T
    return (x, switching) if want_signs else x


def loss(p, d, mode, n, chunk=1 << 15, ref=None):
    """Weighted cross-entropy. With ref (a list), also compares every ReLU sign with ref: an empty list is filled with
    the current signs, and the return value is then (loss, number of pre-activations whose sign differs from ref)."""
    tot, flips, k = 0.0, 0, 0
    for i in range(0, len(d["prob"]), chunk):
        w = d["prob"][i:i + chunk]
        signs = [] if ref is not None else None
        if mode == "outcome":
            z = residual(p, d["out_ids"][i:i + chunk], signs=signs)[:, -1] @ p["R"].T
            ce = torch.nn.functional.cross_entropy(z, d["out_tgt"][i:i + chunk], reduction="none")
        else:                                            # positions n+3..2n+1 hold s_1..s_{n-1}, predict s_2..s_n
            z = residual(p, d["step_ids"][i:i + chunk], signs=signs)[:, n + 3:] @ p["R"].T
            ce = torch.nn.functional.cross_entropy(z.flatten(0, 1), d["step_tgt"][i:i + chunk].flatten(),
                                                   reduction="none").view(len(w), -1).mean(1)
        tot = tot + (w * ce).sum()
        if ref is not None:
            for s_ in signs:
                if len(ref) <= k:
                    ref.append(s_)
                else:
                    flips += int((s_ != ref[k]).sum())
                k += 1
    return (tot, flips) if ref is not None else tot


def switching(p, d, mode):
    with torch.no_grad():
        return residual(p, d["out_ids" if mode == "outcome" else "step_ids"][: 1 << 15], True)[1]


def ball_radius(tilde):
    """Theorem 3's ball radius r_0 = min(1/8, c_emb/4), in units of sigma * ||theta~||."""
    c_emb = float(torch.linalg.svdvals(tilde["wte"][A:A + K]).min())
    norm0 = math.sqrt(sum(float((v ** 2).sum()) for v in tilde.values()))
    return min(1 / 8, c_emb / 4) / norm0


def run(n, sigma, mode, tilde, eps=0.1, max_steps=20000, step_len=1e-3, radii=None):
    d = data(n)
    norm0 = math.sqrt(sum(float((v ** 2).sum()) for v in tilde.values()))
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    theta0 = {k: v.detach().clone() for k, v in p.items()}
    radii = sorted(set(radii or []) | {eps})
    ref, max_flips = [], 0
    L0, t, rec = None, 0.0, dict(n=n, sigma=sigma, mode=mode, switch_start=switching(p, d, mode), T_r={}, relu_flips={})
    for step in range(max_steps):
        L, flips = loss(p, d, mode, n, ref=ref)
        max_flips = max(max_flips, flips)
        grads = torch.autograd.grad(L, list(p.values()))
        gn = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if step == 0:
            L0, rec["grad0_over_sigma"] = float(L), gn / sigma
        dt = step_len * sigma * norm0 / gn
        with torch.no_grad():
            for v, g_ in zip(p.values(), grads):
                v -= dt * g_
        t += dt
        dist = math.sqrt(sum(float(((p[k] - theta0[k]) ** 2).sum()) for k in p)) / (sigma * norm0)
        for r in radii:
            if dist >= r and r not in rec["T_r"]:
                with torch.no_grad():
                    max_flips = max(max_flips, loss(p, d, mode, n, ref=ref)[1])   # signs at the exit point itself
                rec["T_r"][r], rec["relu_flips"][r] = t, max_flips
        if dist >= eps:
            with torch.no_grad():
                L1 = float(loss(p, d, mode, n))
            rec.update(T=t, steps=step + 1, drop_over_sigma2=(L0 - L1) / sigma ** 2,
                       gap_over_sigma2=(math.log(K) - L1) / sigma ** 2, switch_end=switching(p, d, mode))
            break
    return rec


if __name__ == "__main__":
    sigmas = [float(x) for x in sys.argv[1:]] or [0.02, 0.005, 0.002]
    seed = int(os.environ.get("REGIME_SEED", "0"))
    tilde, res = theta_tilde(seed), []
    zero_readout = os.environ.get("REGIME_ZERO_READOUT") == "1"
    step_len = float(os.environ.get("REGIME_STEP", "1e-3"))     # update length / (sigma ||theta~||)
    if zero_readout:                                     # Theorem 3 start: readout of theta~ is zero
        tilde["R"].zero_()
    os.makedirs("results/theorem_regime", exist_ok=True)
    for sigma in sigmas:
        for n in [int(x) for x in os.environ.get("REGIME_N", "4,6,8,10,12,14,16").split(",")]:
            for mode in ("step", "outcome"):
                # REGIME_STOP=ball stops at the radius of B_eps instead of r = 0.1 (to refine the small-radius exit time).
                stop = ball_radius(tilde) if os.environ.get("REGIME_STOP") == "ball" else 0.1
                r = run(n, sigma, mode, tilde, eps=stop, step_len=step_len,
                        max_steps=int(20000 * 1e-3 / step_len), radii=[ball_radius(tilde)])
                r["seed"] = seed
                r["zero_readout"] = zero_readout
                r["step_len"] = step_len
                res.append(r)
                delta = 0.5 ** n + sigma * 0.5 ** (n - 1)
                print(f"sigma={sigma} n={n:2d} {mode:8s} grad/sigma={r['grad0_over_sigma']:.3e}  T(0.1)={r.get('T', float('nan')):.3e}"
                      f"  grad/(sigma(sigma^2+delta_n))={r['grad0_over_sigma'] / (sigma ** 2 + delta):.3e}"
                      f"  T*(sigma^2+delta_n)={r.get('T', float('nan')) * (sigma ** 2 + delta):.2f}"
                      f"  switching {r['switch_start']}->{r.get('switch_end')}"
                      f"  T(r) {{{', '.join(f'{k:.2e}: {v:.3e}' for k, v in r['T_r'].items())}}}"
                      f"  relu flips {r['relu_flips']}", flush=True)
                json.dump(res, open(os.environ.get("REGIME_OUT", "results/theorem_regime/both.json"), "w"))
