"""Numerical checks of the credit theorems at initialization (tab:verify and tab:matched).

Setting of regime_both.py ((A1)-(A4), exact over all words and both start states) unless stated.

    uv run python handcoded/theory_checks/credit_checks.py rule          # prior / rule / outcome credit and sigma-exponents
    uv run python handcoded/theory_checks/credit_checks.py interaction   # outcome credit by degree against rule credit
    uv run python handcoded/theory_checks/credit_checks.py query         # rule credit, fixed against uniform query
    uv run python handcoded/theory_checks/credit_checks.py structured    # sigma^6 rule coefficient against its closed form
    uv run python handcoded/theory_checks/credit_checks.py setsums [n] [sigma1 sigma2 sigma3]   # Jacobian set-sums T_r

rule: the step target splits as  e_{s_t} - u = [Tbar^T e_{s_{t-1}} - u] + [e_{s_t} - Tbar^T e_{s_{t-1}}], the prior of s_t
  given the displayed state and the rule part. Reports || d/dtheta E <z_i, target> || for prior, rule and outcome.
interaction: splits the answer target y - u exactly into Efron-Stein components over the n letters and reports the outcome
  credit at degree r = 0..3 and at degree >= 2, against the rule credit; both need one attention score.
query: |Sigma| = 4 uniform letters (p = 1/4, lambda = 1/2). With a fixed query the leading rule coefficient is of order 1/n;
  averaging over a uniform query cancels it, leaving order 1/n^2.
structured: one explicit theta_tilde (a single block-2 head: query reads chi(s) = (-1)^s, key and value read
  phi = 1[letter = query]) where the sigma^6 rule coefficient is nonzero, against
  sigma^6 / sqrt(d_head) * w_k * p (1 - p) / m * (1 - (3 + 2 (n - 1) p) / m),  w_k = (chi(k - 1) - chi(k)) / K,
  averaged over t = 2..n with m = n + 2 + t positions in view.
setsums: T_r = sum_{|S|=r} ||J_S||_{L2} of the answer-logit Jacobian for one layer and for two blocks (four uniform letters,
  MLP biases bounded away from zero), estimated with random parameter directions. The paper proves exponent 2r in sigma for
  two blocks; measured 2, 6, 7, 11, 12 at r = 1..5.
"""
import itertools
import json
import math
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))  # handcoded/

import torch
from torch.func import jvp

import regime_both as RB

K, PM = RB.K, RB.PM


def _norm(grads):
    return math.sqrt(sum(float((x ** 2).sum()) for x in grads))


def _tbar(pm):
    eye = torch.eye(K, device=RB.dev)
    return eye, (1 - pm) * eye + pm * eye.roll(1, 1)                 # row s -> distribution of s + b


# --- rule: prior, rule and outcome credit --------------------------------------------------------------

def rule_credits(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye, Tbar = _tbar(PM)
    prev = d["step_ids"][:, n + 3:] - RB.A                            # displayed s_{t-1}, t = 2..n
    z = RB.residual(p, d["step_ids"])[:, n + 3:] @ p["R"].T
    prior_t, full_t = Tbar[prev] - 1 / K, eye[d["step_tgt"]] - 1 / K
    w = d["prob"][:, None, None]
    out = {}
    for name, tgt in (("prior", prior_t), ("rule", full_t - prior_t)):
        phi = (w * z * tgt).sum() / z.shape[1]
        out[name] = _norm(torch.autograd.grad(phi, list(p.values()), retain_graph=True))
    zo = RB.residual(p, d["out_ids"])[:, -1] @ p["R"].T
    phi = (d["prob"][:, None] * zo * (eye[d["out_tgt"]] - 1 / K)).sum()
    out["outcome"] = _norm(torch.autograd.grad(phi, list(p.values())))
    return out


def main_rule():
    res = []
    for seed in (0, 1, 2):
        tilde = RB.theta_tilde(seed)
        for n in (4, 6, 8, 10, 12, 14):
            row = {s: rule_credits(n, s, tilde) for s in (0.005, 0.01, 0.02)}
            expo = {k: math.log(row[0.02][k] / row[0.005][k]) / math.log(4) for k in ("prior", "rule", "outcome")}
            res.append(dict(seed=seed, n=n, credits=row, exponent=expo))
            c = row[0.02]
            print(f"draw {seed} n={n:2d} sigma=0.02: prior {c['prior']:.3e}  rule {c['rule']:.3e}  outcome {c['outcome']:.3e}"
                  f"   rule*n/sigma^6 {c['rule'] * n / 0.02 ** 6:.3g}"
                  f"   sigma-exponents prior {expo['prior']:.2f} rule {expo['rule']:.2f} outcome {expo['outcome']:.2f}", flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/rule_credit.json", "w"))


# --- interaction: outcome credit by degree --------------------------------------------------------------

def degree_parts(f, n):
    """f: (K * 2^n, K) target rows in RB.data() order. Returns the list over r of the degree-r part."""
    w = torch.tensor([PM, 1 - PM], device=f.device)                  # letter 0 is the query letter
    g = f.view(K, *([2] * n), K)                                       # start state, letters, logit
    parts = [g]                                                        # parts[r]: degree exactly r so far
    for t in range(n):
        ax = 1 + t
        new = [None] * (len(parts) + 1)
        for r, h in enumerate(parts):
            mean = (h * w.view(*([1] * ax), 2, *([1] * (n - t)))).sum(ax, keepdim=True).expand_as(h)
            new[r] = mean if new[r] is None else new[r] + mean
            new[r + 1] = h - mean
        parts = new
    return [p.reshape(-1, K) for p in parts]


def interaction_credits(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye = torch.eye(K, device=RB.dev)
    parts = degree_parts(eye[d["out_tgt"]] - 1 / K, n)
    zo = RB.residual(p, d["out_ids"])[:, -1] @ p["R"].T
    w = d["prob"][:, None]
    targets = {f"r{r}": parts[r] for r in range(min(4, n + 1))}
    targets["ge2"] = sum(parts[2:])
    targets["all"] = sum(parts)
    out = {}
    for name, tgt in targets.items():
        out[name] = _norm(torch.autograd.grad((w * zo * tgt).sum(), list(p.values()), retain_graph=True))
    out["rule"] = rule_credits(n, sigma, tilde)["rule"]
    return out


def main_interaction():
    lam, res = 1 - 2 * PM, []
    for seed in (0, 1, 2):
        tilde = RB.theta_tilde(seed)
        for n in (4, 6, 8, 10, 12, 14):
            row = {s: interaction_credits(n, s, tilde) for s in (0.005, 0.02)}
            expo = {k: math.log(row[0.02][k] / row[0.005][k]) / math.log(4) for k in row[0.02]}
            res.append(dict(seed=seed, n=n, credits=row, exponent=expo))
            c = row[0.02]
            print(f"draw {seed} n={n:2d} sigma=0.02: outcome all {c['all']:.2e} r0 {c['r0']:.2e} r1 {c['r1']:.2e}"
                  f" r2 {c['r2']:.2e} r3 {c['r3']:.2e} >=2 {c['ge2']:.2e} | rule {c['rule']:.2e}"
                  f" | rule/(>=2) {c['rule'] / c['ge2']:.3g}"
                  f" | (>=2)/(sigma^6 lam^(n-2)) {c['ge2'] / (0.02 ** 6 * lam ** (n - 2)):.3g}"
                  f" | exps r2 {expo['r2']:.2f} r3 {expo['r3']:.2f} >=2 {expo['ge2']:.2f} rule {expo['rule']:.2f}",
                  flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/interaction_credit.json", "w"))


# --- query: fixed against uniform query, four letters ---------------------------------------------------

Q_A = 4                                                              # letters in the query check
Q_SEP = Q_A + K


def query_data(n, queries):
    words = torch.tensor(list(itertools.product(range(Q_A), repeat=n)))
    rows, prob = [], []
    for q in queries:
        for s0 in range(K):
            match = (words == q).long()
            states = (s0 + match.cumsum(1)) % K
            head = torch.cat([torch.full((len(words), 1), Q_A + s0), torch.full((len(words), 1), q), words,
                              torch.full((len(words), 1), Q_SEP)], 1)
            rows.append((torch.cat([head, Q_A + states[:, : n - 1]], 1), states[:, 1:]))
            prob.append(torch.full((len(words),), 1.0 / (len(words) * K * len(queries))))
    ids = torch.cat([r[0] for r in rows]).to(RB.dev)
    tgt = torch.cat([r[1] for r in rows]).to(RB.dev)
    return ids, tgt, torch.cat(prob).to(RB.dev)


def query_rule_credit(n, sigma, tilde, queries):
    ids, tgt, prob = query_data(n, queries)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye, Tbar = _tbar(1 / Q_A)
    rule = eye[tgt] - Tbar[ids[:, n + 3:] - Q_A]
    phi, chunk = 0.0, 1 << 16
    for i in range(0, len(prob), chunk):
        z = RB.residual(p, ids[i:i + chunk])[:, n + 3:] @ p["R"].T
        phi = phi + (prob[i:i + chunk, None, None] * z * rule[i:i + chunk]).sum() / z.shape[1]
    return _norm(torch.autograd.grad(phi, list(p.values())))


def main_query():
    RB.A = Q_A                                       # token layout of theta_tilde: letters, values, SEP, ':', spare
    res = []
    for seed in (0, 1):
        tilde = RB.theta_tilde(seed)
        for n in (4, 5, 6, 7, 8):
            row = {}
            for name, qs in (("fixed", [0]), ("random", list(range(Q_A)))):
                r = {s: query_rule_credit(n, s, tilde, qs) for s in (0.01, 0.02)}
                row[name] = dict(R=r[0.02], expo=math.log(r[0.02] / r[0.01]) / math.log(2))
            res.append(dict(seed=seed, n=n, **row))
            f, rr = row["fixed"], row["random"]
            print(f"draw {seed} n={n}: fixed q  R={f['R']:.3e}  nR/s^6={n * f['R'] / 0.02 ** 6:.1f}  exp {f['expo']:.2f} | "
                  f"random q  R={rr['R']:.3e}  nR/s^6={n * rr['R'] / 0.02 ** 6:.2f}  n^2R/s^6={n * n * rr['R'] / 0.02 ** 6:.1f}"
                  f"  exp {rr['expo']:.2f}", flush=True)
    os.makedirs("results/theorem_regime", exist_ok=True)
    json.dump(res, open("results/theorem_regime/rule_credit_query.json", "w"))


# --- structured: the sigma^6 coefficient in closed form -------------------------------------------------

def structured_tilde():
    th = {k: torch.zeros_like(v) for k, v in RB.theta_tilde(0).items()}
    for s in range(K):
        th["wte"][RB.A + s, 0] = (-1.0) ** s          # chi(s) on coordinate 0
    th["wte"][0, 1] = 1.0                              # phi on coordinate 1 (query letter is letter 0)
    th["1.q"][0, 0, 0] = 1.0
    th["1.k"][0, 0, 1] = 1.0
    th["1.v"][0, 2, 1] = 1.0                           # value writes phi to coordinate 2
    for b in range(2):
        th[f"{b}.mb"][:] = -1.0                        # all MLP units off: the MLPs are zero maps
    return th


def structured_measured(n, sigma, tilde):
    d = RB.data(n)
    p = {k: (sigma * v).clone().requires_grad_() for k, v in tilde.items()}
    eye, Tbar = _tbar(PM)
    z = RB.residual(p, d["step_ids"])[:, n + 3:] @ p["R"].T
    rule = eye[d["step_tgt"]] - Tbar[d["step_ids"][:, n + 3:] - RB.A]
    phi = (d["prob"][:, None, None] * z * rule).sum() / z.shape[1]
    return torch.autograd.grad(phi, p["R"])[0][:, 2]   # derivative in R_{k,2}


def structured_closed_form(n, sigma):
    tot = 0.0
    for t in range(2, n + 1):
        m = n + 2 + t
        tot += PM * (1 - PM) / m * (1 - (3 + 2 * (n - 1) * PM) / m)   # the query token also has phi = 1
    w = [((-1.0) ** (k - 1) - (-1.0) ** k) / K for k in range(K)]
    return [sigma ** 6 / math.sqrt(RB.Dh) * wk * tot / (n - 1) for wk in w]


def main_structured():
    tilde = structured_tilde()
    for n in (2, 4, 6, 8, 10, 12):
        for sigma in (0.05, 0.02):
            got, want = structured_measured(n, sigma, tilde).tolist(), structured_closed_form(n, sigma)
            print(f"n={n:2d} sigma={sigma}: measured {got[0]:+.6e} {got[1]:+.6e}   closed form {want[0]:+.6e} {want[1]:+.6e}"
                  f"   ratio {got[0] / want[0]:.6f}", flush=True)


# --- setsums: Jacobian set-sums by degree ---------------------------------------------------------------

SS_A, SS_W, SS_H, SS_DH, SS_F, SS_DIRS = 4, 16, 3, 8, 16, 8


def setsum_models(n, seed=0):
    dev = RB.dev
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    words = torch.tensor(list(itertools.product(range(SS_A), repeat=n)))
    m = len(words)
    # one layer: query token, n letters, answer token
    ids1 = torch.cat([torch.full((m, 1), SS_A), words, torch.full((m, 1), SS_A + 1)], 1).to(dev)
    t1 = dict(E=rn(SS_A + 2, SS_W), P=rn(n + 2, SS_W), Wq=rn(SS_W, SS_W), Wk=rn(SS_W, SS_W), Wv=rn(SS_W, SS_W),
              R=rn(K, SS_W))

    def f1(p):
        x = p["E"][ids1] + p["P"][None]
        q = x[:, -1] @ p["Wq"].T; k = x @ p["Wk"].T; v = x @ p["Wv"].T
        a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(SS_W), -1)
        return (x[:, -1] + (a.unsqueeze(-1) * v).sum(1)) @ p["R"].T

    # two blocks: s0 q w1..wn SEP ':'
    vocab, T = SS_A + K + 3, n + 4
    ids2 = torch.cat([torch.full((m, 1), SS_A), torch.zeros(m, 1, dtype=torch.long), words,
                      torch.full((m, 1), SS_A + K), torch.full((m, 1), SS_A + K + 1)], 1).to(dev)
    t2 = {"wte": rn(vocab, SS_W), "wpe": rn(T, SS_W), "R": rn(K, SS_W)}
    for b in range(2):
        t2[f"{b}.q"], t2[f"{b}.k"], t2[f"{b}.v"] = rn(SS_H, SS_DH, SS_W), rn(SS_H, SS_DH, SS_W), rn(SS_H, SS_W, SS_W)
        t2[f"{b}.mi"], t2[f"{b}.mo"] = rn(SS_F, SS_W), rn(SS_W, SS_F)
        mb = rn(SS_F); t2[f"{b}.mb"] = torch.sign(mb) * (1 + mb.abs())
    future = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)

    def f2(p):
        x = p["wte"][ids2] + p["wpe"]
        for b in range(2):
            q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(SS_DH)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            x = x + torch.relu(x @ p[f"{b}.mi"].T + p[f"{b}.mb"]) @ p[f"{b}.mo"].T
        return x[:, -1] @ p["R"].T

    return ({k: v.to(dev) for k, v in t1.items()}, f1), ({k: v.to(dev) for k, v in t2.items()}, f2)


def setsums(tilde, f, n, sigma, seed=1):
    dev = RB.dev
    g = torch.Generator(device=dev).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.cat([torch.ones(SS_A, 1),
                                      torch.randn(SS_A, SS_A - 1, generator=torch.Generator().manual_seed(1))], 1))
    Q = (Q.T * torch.sign(Q.T[:, :1])).to(dev)            # rows orthonormal, row 0 constant
    idx = torch.tensor(list(itertools.product(range(SS_A), repeat=n)), device=dev)
    support = ((idx != 0).long() * (2 ** torch.arange(n, device=dev))).sum(1)   # set S as a bitmask
    th = {k: sigma * v for k, v in tilde.items()}
    e_S = torch.zeros(2 ** n, device=dev)
    for _ in range(SS_DIRS):
        d = {k: torch.randn(v.shape, generator=g, device=dev) for k, v in th.items()}
        c = jvp(f, (th,), (d,))[1].reshape((SS_A,) * n + (-1,))
        for ax in range(n):
            c = torch.tensordot(Q, c, dims=([1], [ax])).movedim(0, ax)
        e = (c.reshape(SS_A ** n, -1) ** 2).sum(1) / SS_A ** n / SS_DIRS
        e_S.index_add_(0, support, e)
    size = torch.tensor([bin(s).count("1") for s in range(2 ** n)], device=dev)
    return [float(e_S[size == r].sqrt().sum()) for r in range(n + 1)]


def main_setsums(argv):
    n = int(argv[0]) if argv else 6
    sig = [float(x) for x in argv[1:4]] if len(argv) > 3 else [0.1, 0.2, 0.3]
    (t1, f1), (t2, f2) = setsum_models(n)
    for name, tilde, f in (("one layer ", t1, f1), ("two blocks", t2, f2)):
        T = {s: setsums(tilde, f, n, s) for s in sig}
        for r in range(1, 6):
            e1 = math.log(T[sig[1]][r] / T[sig[0]][r]) / math.log(sig[1] / sig[0])
            e2 = math.log(T[sig[2]][r] / T[sig[1]][r]) / math.log(sig[2] / sig[1])
            print(f"{name} n={n} r={r}:  T_r(sigma={sig[0]})={T[sig[0]][r]:.3e}  sigma-exponent {e1:.2f}, {e2:.2f}", flush=True)


if __name__ == "__main__":
    cmds = {"rule": main_rule, "interaction": main_interaction, "query": main_query, "structured": main_structured}
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "setsums":
        main_setsums(rest)
    elif cmd in cmds:
        cmds[cmd]()
    else:
        raise SystemExit(__doc__)
