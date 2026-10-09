"""Both losses inside the regime of Theorem 3 (outcome plateau), at horizons where the horizon term binds.

Assumptions (A1)-(A4): two blocks at fixed widths (16, three heads, MLP width 16), K = 2 value
logits, theta_0 = sigma * theta_tilde with every MLP bias |b~| >= 1, full-batch gradient descent
on the exact loss. To reach n = 16 exactly, the alphabet has two letters, the query letter with
probability 1/4, so the moves are i.i.d. with the same contraction lambda = 1/2 as four uniform
letters and the loss is a weighted sum over 2^n words and both start states.
  L_O: cross-entropy of the answer at the ':' position.
  L_S: cross-entropy of s_2..s_n under teacher forcing, averaged over t.
Each update moves the parameters by 1e-3 * sigma * ||theta~||; the training time is the sum of
the step sizes. Reports ||dL/dtheta(theta_0)|| / sigma and the time T(0.1) to move
0.1 * sigma * ||theta~||.
    uv run python handcoded/regime_both.py [sigma ...]
REGIME_ZERO_READOUT=1 zeroes the readout of theta~ (Theorem 3); other tensors are unchanged.
REGIME_STEP sets the update length in units of sigma*||theta~|| (default 1e-3).
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


def residual(p, ids, want_signs=False):
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
        if want_signs:
            on = pre > 0
            switching += int((on.flatten(0, 1).any(0) != on.flatten(0, 1).all(0)).sum())
        x = x + torch.relu(pre) @ p[f"{b}.mo"].T
    return (x, switching) if want_signs else x


def loss(p, d, mode, n, chunk=1 << 15):
    tot = 0.0
    for i in range(0, len(d["prob"]), chunk):
        w = d["prob"][i:i + chunk]
        if mode == "outcome":
            z = residual(p, d["out_ids"][i:i + chunk])[:, -1] @ p["R"].T
            ce = torch.nn.functional.cross_entropy(z, d["out_tgt"][i:i + chunk], reduction="none")
        else:                                            # positions n+3..2n+1 hold s_1..s_{n-1}, predict s_2..s_n
            z = residual(p, d["step_ids"][i:i + chunk])[:, n + 3:] @ p["R"].T
            ce = torch.nn.functional.cross_entropy(z.flatten(0, 1), d["step_tgt"][i:i + chunk].flatten(),
                                                   reduction="none").view(len(w), -1).mean(1)
        tot = tot + (w * ce).sum()
    return tot


def switching(p, d, mode):
    with torch.no_grad():
        return residual(p, d["out_ids" if mode == "outcome" else "step_ids"][: 1 << 15], True)[1]


def run(n, sigma, mode, tilde, eps=0.1, max_steps=20000, step_len=1e-3):
    d = data(n)
    norm0 = math.sqrt(sum(float((v ** 2).sum()) for v in tilde.values()))
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    theta0 = {k: v.detach().clone() for k, v in p.items()}
    L0, t, rec = None, 0.0, dict(n=n, sigma=sigma, mode=mode, switch_start=switching(p, d, mode))
    for step in range(max_steps):
        L = loss(p, d, mode, n)
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
                r = run(n, sigma, mode, tilde, step_len=step_len, max_steps=int(20000 * 1e-3 / step_len))
                r["seed"] = seed
                r["zero_readout"] = zero_readout
                r["step_len"] = step_len
                res.append(r)
                delta = 0.5 ** n + sigma * 0.5 ** (n - 1)
                print(f"sigma={sigma} n={n:2d} {mode:8s} grad/sigma={r['grad0_over_sigma']:.3e}  T(0.1)={r.get('T', float('nan')):.3e}"
                      f"  grad/(sigma(sigma^2+delta_n))={r['grad0_over_sigma'] / (sigma ** 2 + delta):.3e}"
                      f"  T*(sigma^2+delta_n)={r.get('T', float('nan')) * (sigma ** 2 + delta):.2f}"
                      f"  switching {r['switch_start']}->{r.get('switch_end')}", flush=True)
                json.dump(res, open(os.environ.get("REGIME_OUT", "results/theorem_regime/both.json"), "w"))
