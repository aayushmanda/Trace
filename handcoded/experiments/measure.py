"""Measurements behind Silent Routing's Jacobian/ANOVA spectra and single-head untied routing (Appendices G.1-G.3).

    uv run python handcoded/experiments/measure.py spectrum task=count word_len=8 mod=2   # App. G.1 gradient decay
    uv run python handcoded/experiments/measure.py answers                                # App. G.2 Boolean-8 / register ANOVA
    uv run python handcoded/experiments/measure.py exact site=answer dirs=16 ckpt=ckpt/n8_block_s42/s42_outcome_step1000.pt
    uv run python handcoded/experiments/measure.py reach --n 6 8 --sigma 0.3 --seeds 0 1 --lr 2 --steps 150000 --out x.json

spectrum: interaction spectra of the logit Jacobian and the answer-dependent gradient against n, in the lettertrace.py
  architecture with the same theta_0 (key=value settings listed at the top of run_spectrum).
answers: exact answer spectra A_r of Boolean-8 (98.9% deg >= 2) and register mod 17 (99.9% deg >= 2) via DP.
exact: exact letter-order spectra W_r of logit Jacobians from forward-mode JVPs along outcome training (App. G.1).
reach: single-head untied routing from small initialization (App. G.3, 17/17 step vs 0/17 outcome); --format outcome
  runs the outcome baseline under the same protocol.
"""
import sys
from pathlib import Path

HANDCODED = Path(__file__).resolve().parents[1]
ROOT_DIR = HANDCODED.parent
for _p in (str(HANDCODED), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_spectrum(argv):
    __import__("sys").argv = ['measure.py spectrum'] + list(argv)
    
    import ast
    import json
    import math
    import sys
    
    import numpy as np
    import torch
    from torch.func import jacrev, vmap
    from torch.nn import functional as F
    
    # -----------------------------------------------------------------------------
    # config (key=value on the command line)
    
    task = "count"
    alphabet = 4
    word_len = 8
    mod = 2
    n_blocks = 2
    init_std = 0.02
    seed = 42                       # same seed as lettertrace.py => same theta_0
    rhos = [i / 10 for i in range(11)]
    pairs = 512                     # (x, x_rho) pairs per rho
    chunk = 32                      # examples per vmap'd Jacobian
    grad_prompts = 4096             # prompts for the measured gradients
    exact = True                    # enumerate every prompt for the gradients when feasible
    max_exact = 2_000_000           # largest prompt count enumerated exactly (n=10 needs 8.4M)
    trace_rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # wrong-trace gradient test (count, mod>=3)
    out = ""                        # optional json path
    ckpt = ""                       # load parameters from a lettertrace.py checkpoint
    site = "answer"                 # "answer" or "state" (see docstring)
    layout = "block"                # process rows: "block" (same prompt as outcome) or "stream"
    zero_readout = False            # init mode only: readout = 0, so predictions are exactly uniform
    grads_only = False              # skip C_J / W_r; only the two signals in fig:credit
    match_p = None                  # None => 1/alphabet. Else P(letter = query); architecture unchanged.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _cfg = {}
    for arg in sys.argv[1:]:
        key, value = arg.split("=", 1)
        assert key in ['task', 'alphabet', 'word_len', 'mod', 'n_blocks', 'init_std', 'seed', 'rhos', 'pairs', 'chunk', 'grad_prompts', 'exact', 'max_exact', 'trace_rhos', 'out', 'ckpt', 'site', 'layout', 'zero_readout', 'grads_only', 'match_p', 'device'], f"unknown setting {key}"
        _cfg[key] = value if key in ("task", "device", "out", "ckpt", "site", "layout") else ast.literal_eval(value)
    task = _cfg.get('task', task)
    alphabet = _cfg.get('alphabet', alphabet)
    word_len = _cfg.get('word_len', word_len)
    mod = _cfg.get('mod', mod)
    n_blocks = _cfg.get('n_blocks', n_blocks)
    init_std = _cfg.get('init_std', init_std)
    seed = _cfg.get('seed', seed)
    rhos = _cfg.get('rhos', rhos)
    pairs = _cfg.get('pairs', pairs)
    chunk = _cfg.get('chunk', chunk)
    grad_prompts = _cfg.get('grad_prompts', grad_prompts)
    exact = _cfg.get('exact', exact)
    max_exact = _cfg.get('max_exact', max_exact)
    trace_rhos = _cfg.get('trace_rhos', trace_rhos)
    out = _cfg.get('out', out)
    ckpt = _cfg.get('ckpt', ckpt)
    site = _cfg.get('site', site)
    layout = _cfg.get('layout', layout)
    zero_readout = _cfg.get('zero_readout', zero_readout)
    grads_only = _cfg.get('grads_only', grads_only)
    match_p = _cfg.get('match_p', match_p)
    device = _cfg.get('device', device)
    assert task in ("count", "first") and site in ("answer", "state") and layout in ("block", "stream")
    assert site == "answer" or task == "count", "site=state is defined for the count task"
    
    n, A = word_len, alphabet
    p_match = (1 / A) if match_p is None else float(match_p)
    assert 0.0 < p_match < 1.0, p_match
    V = mod if task == "count" else n + 1
    SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
    VOCAB, P, H = A + V + 3, 2 * n + 6, 3
    F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
    W = 2 * A + 4 * V + P + 3                   # slot widths of lettertrace.py
    
    # -----------------------------------------------------------------------------
    # the architecture of lettertrace.py, and the same theta_0
    
    
    def zero_params():
        p = {"wte": torch.zeros(VOCAB, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(VOCAB, W)}
        for b in range(n_blocks):
            p[f"{b}.q"], p[f"{b}.k"] = torch.zeros(H, P, W), torch.zeros(H, P, W)
            p[f"{b}.v"] = torch.zeros(H, W, W)
            p[f"{b}.mlp_in"], p[f"{b}.mlp_b"] = torch.zeros(F_WIDTH, W), torch.zeros(F_WIDTH)
            p[f"{b}.mlp_out"] = torch.zeros(W, F_WIDTH)
        return p
    
    
    def forward(p, ids):
        T = ids.shape[1]
        x = p["wte"][ids] + p["wpe"][:T]
        future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
        for b in range(n_blocks):
            q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
        return x @ p["readout"].T
    
    
    torch.manual_seed(seed)
    theta = {k: (torch.randn_like(t) * init_std).to(device) for k, t in zero_params().items()}
    step_label = "init"
    if ckpt:
        saved = torch.load(ckpt, map_location=device)
        cfg = saved["config"]
        for k_, v_ in dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks).items():
            assert cfg[k_] == v_, f"checkpoint has {k_}={cfg[k_]}, this run has {v_}"
        assert cfg.get("layout", layout) == layout, f"checkpoint has layout={cfg['layout']}, this run has {layout}"
        assert set(saved["params"]) == set(theta), "checkpoint tensors do not match the architecture"
        theta = {k: saved["params"][k].to(device).float() for k in theta}
        step_label = f"{saved['mode']} step {saved['step']}"
    elif zero_readout:
        theta["readout"].zero_()
    n_params = sum(t.numel() for t in theta.values())
    
    # -----------------------------------------------------------------------------
    # prompts: letters i.i.d. uniform (no conditioning on a match, as in the theory)
    
    rng = np.random.default_rng(seed + 1)
    
    
    def answer(s0, q, word):
        b = word == q[:, None]
        if task == "count":
            return (s0 + b.sum(1)) % mod
        return np.where(b.any(1), b.argmax(1) + 1, 0)
    
    
    def sample(m):
        q = rng.integers(A, size=m)
        word = rng.integers(A, size=(m, n))
        s0 = rng.integers(mod, size=m) if task == "count" else np.zeros(m, dtype=int)
        return s0, q, word
    
    
    def outcome_ids(s0, q, word):
        """Prompt up to ':'; the ':' position n+3 predicts the answer."""
        m = len(q)
        cols = [A + s0[:, None], q[:, None], word, np.full((m, 1), SEP), np.full((m, 1), COLON)]
        return torch.tensor(np.concatenate(cols, 1), device=device)
    
    
    # Positions that predict s_1..s_n in a process row.
    #   block:  s0 q w_1 .. w_n SEP s_1 .. s_n     position n+1+t predicts s_t (SEP for t = 1)
    #   stream: s0 q w_1 s_1 .. w_n s_n            position 2t (the letter w_t) predicts s_t
    PRED = np.arange(1, n + 1) + (n + 1 if layout == "block" else np.arange(1, n + 1))
    
    
    def trace_ids(s0, q, word, shown):
        """Process row with displayed states `shown` (m x k, k <= n), cut after the last state."""
        m, k = shown.shape
        if layout == "block":
            cols = [A + s0[:, None], q[:, None], word, np.full((m, 1), SEP), A + shown]
        else:
            cols = [A + s0[:, None], q[:, None]]
            for t in range(n):
                cols.append(word[:, t : t + 1])
                if t < k:
                    cols.append(A + shown[:, t : t + 1])
        return torch.tensor(np.concatenate(cols, 1), device=device)
    
    
    def process_rows(s0, q, word):
        """Clean process rows up to s_n; position PRED[t-1] predicts s_t."""
        b = word == q[:, None]
        states, s = [], s0.copy()
        for t in range(n):
            s = (s + b[:, t]) % mod if task == "count" else np.where((s == 0) & b[:, t], t + 1, s)
            states.append(s.copy())
        states = np.stack(states, 1)
        return trace_ids(s0, q, word, states), states
    
    def sample_state(m):
        """Prompts for site=state: displayed states are free inputs, uniform and independent."""
        s0, q, word = sample(m)
        shown = rng.integers(mod, size=(m, n - 1))
        return s0, q, word, shown
    
    
    def state_ids(s0, q, word, shown):
        """Process row with n-1 free displayed states; position PRED[-1] predicts
        s_n = s_{n-1} + [w_n = q]."""
        return trace_ids(s0, q, word, shown)
    
    
    def state_target(s0, q, word, shown):
        prev = shown[:, -1] if n > 1 else s0
        return (prev + (word[:, -1] == q)) % mod
    
    
    QPOS = n + 3 if site == "answer" else int(PRED[-1])
    draw = sample if site == "answer" else sample_state
    make_ids = (lambda d, w: outcome_ids(d[0], d[1], w)) if site == "answer" else \
               (lambda d, w: state_ids(d[0], d[1], w, d[3]))
    
    # -----------------------------------------------------------------------------
    # model side: C_J(rho) from per-example answer-logit Jacobians
    
    
    def value_logits(params, ids):
        return forward(params, ids[None])[0, QPOS, A : A + V]
    
    
    jac = vmap(jacrev(value_logits), in_dims=(None, 0))
    
    
    def flat_jacobian(ids):
        j = jac(theta, ids)
        return torch.cat([t.reshape(ids.shape[0], V, -1) for t in j.values()], 2)
    
    
    def stability(rho):
        """C_J(rho) and D(rho) = C_J(rho) - C_J(0) = sum_{r>=1} rho^r W_r. D uses one fresh
        word for both x_rho and x_0 (coupled), so the large order-0 energy cancels per pair."""
        c_vals, d_vals = [], []
        for start in range(0, pairs, chunk):
            m = min(chunk, pairs - start)
            d = draw(m)
            word = d[2]
            fresh = rng.integers(A, size=(m, n))
            word_rho = np.where(rng.random((m, n)) < rho, word, fresh)
            j = flat_jacobian(make_ids(d, word)).detach()
            j_rho = flat_jacobian(make_ids(d, word_rho)).detach()
            j_0 = flat_jacobian(make_ids(d, fresh)).detach()
            c_vals.append((j * j_rho).sum((1, 2)).double().cpu())
            d_vals.append((j * (j_rho - j_0)).sum((1, 2)).double().cpu())
        c, d = torch.cat(c_vals), torch.cat(d_vals)
        se = lambda v: v.std().item() / math.sqrt(len(v))
        return c.mean().item(), se(c), d.mean().item(), se(d)
    
    
    # -----------------------------------------------------------------------------
    # task side: exact answer spectrum A_r over the match bits
    
    
    def state_spectrum():
        """Target s_{n-1} + b_n with s_{n-1} fixed: only orders 0 and 1 (the last letter)."""
        p = p_match
        e0, e1 = np.eye(V)[0], np.eye(V)[1 % V]
        a0 = (1 - p) * e0 + p * e1 - 1 / V
        A_st = np.zeros(n + 1)
        A_st[0] = (a0 ** 2).sum()
        A_st[1] = p * (1 - p) * ((e1 - e0) ** 2).sum()
        return A_st
    
    
    def answer_spectrum():
        """Biased Fourier transform over b in {0,1}^n, b_t ~ Bern(p). The answer depends on
        the letters only through b, so its Efron-Stein energies over letters equal these."""
        assert n <= 18, "exact answer spectrum needs n <= 18"
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[::-1]) & 1
        y = answer(np.zeros(2 ** n, dtype=int), np.zeros(2 ** n, dtype=int), (1 - bits))  # q=0, letter 0 iff b=1
        f = np.eye(V)[y] - 1 / V
        f = f.reshape((2,) * n + (V,))
        c = math.sqrt(p_match * (1 - p_match))
        for t in range(n):
            f0, f1 = np.take(f, 0, axis=t), np.take(f, 1, axis=t)
            f = np.stack([(1 - p_match) * f0 + p_match * f1, c * (f1 - f0)], axis=t)
        energy = (f ** 2).sum(-1).reshape(-1)
        order = bits.sum(1)
        return np.array([energy[order == r].sum() for r in range(n + 1)])
    
    # -----------------------------------------------------------------------------
    # measured gradients at theta_0
    
    
    def all_prompts():
        """Every (s0, q, word) with its probability, or None if there are too many."""
        n_s0 = mod if task == "count" else 1
        total = n_s0 * A * A ** n
        if not exact or total > max_exact:
            return None
        idx = np.arange(total)
        word = (idx[:, None] // A ** np.arange(n)[::-1]) % A
        q = (idx // A ** n) % A
        s0 = idx // (A ** (n + 1)) if task == "count" else np.zeros(total, dtype=int)
        return s0, q, word
    
    
    PROMPTS = all_prompts()
    
    
    def grad_vec(make_scalar, m_total, batch=256):
        """Population gradient as one flat vector (exact when PROMPTS is set, else Monte Carlo)."""
        grads = []
        source = PROMPTS if PROMPTS is not None else None
        total = len(PROMPTS[0]) if PROMPTS is not None else m_total
        step = 4096 if PROMPTS is not None else batch
        for start in range(0, total, step):
            if PROMPTS is not None:
                args = tuple(a[start:start + step] for a in PROMPTS)
            else:
                args = sample(min(batch, m_total - start))
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            s = make_scalar(params, *args) * (len(args[0]) / total)
            grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
        return torch.stack(grads).sum(0)
    
    
    def grad_norm(make_scalar, m_total, batch=256):
        """Norm of the population gradient. Exact (every prompt) when feasible; otherwise
        a Monte Carlo mean over batches, with a standard error of the norm."""
        grads = []
        if PROMPTS is not None:
            total = len(PROMPTS[0])
            for start in range(0, total, 4096):
                sl = slice(start, start + 4096)
                params = {k: t.clone().requires_grad_() for k, t in theta.items()}
                s = make_scalar(params, *(a[sl] for a in PROMPTS)) * (len(PROMPTS[0][sl]) / total)
                grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
            return torch.stack(grads).sum(0).norm().item(), 0.0
        for start in range(0, m_total, batch):
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            s = make_scalar(params, *sample(min(batch, m_total - start)))
            grads.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, list(params.values()))]).double())
        g = torch.stack(grads)
        mean = g.mean(0)
        se = math.sqrt(((g - mean) ** 2).sum().item() / (len(g) * (len(g) - 1)))
        return mean.norm().item(), se
    
    
    def example_weight(q, word):
        """Density ratio of the Bern(p_match) letter law against the uniform enumeration.
        A uniform mean of (value * weight) is the expectation under that law. Weights are
        identically 1 when match_p is None."""
        if match_p is None:
            return None
        match = word == np.asarray(q)[:, None]
        nm = match.sum(1)
        logw = nm * math.log(A * p_match) + (n - nm) * math.log(A * (1.0 - p_match) / (A - 1))
        return torch.tensor(np.exp(logw), device=device, dtype=torch.float64)
    
    
    def weighted_mean(values, q, word):
        if match_p is None:
            return values.mean()
        w = example_weight(q, word)
        return (values.double() * w).sum() / values.shape[0]
    
    
    def outcome_theory(params, s0, q, word):             # E[J^T (e_y - u)], u uniform over values
        z = forward(params, outcome_ids(s0, q, word))[:, n + 3, A : A + V]
        e = torch.eye(V, device=device)[torch.tensor(answer(s0, q, word), device=device)]
        return weighted_mean(((e - 1 / V) * z).sum(1), q, word)
    
    
    def outcome_residual(params, s0, q, word):           # E[J^T (p_v - u)]: non-uniformity at init
        logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
        pv = logits.softmax(-1)[:, A : A + V].detach()
        return ((pv - 1 / V) * logits[:, A : A + V]).sum(1).mean()
    
    
    def outcome_full(params, s0, q, word):               # the actual answer-token loss
        logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
        return F.cross_entropy(logits, torch.tensor(A + answer(s0, q, word), device=device))
    
    
    def outcome_nonvalue(params, s0, q, word):          # E[J_nv^T p_nv]: mass on non-answer tokens
        logits = forward(params, outcome_ids(s0, q, word))[:, n + 3]
        p = logits.softmax(-1).detach()
        mask = torch.ones(VOCAB, device=device)
        mask[A : A + V] = 0
        return (p * mask * logits).sum(1).mean()
    
    
    def state_theory(params, s0, q, word, shown):        # E[J^T (e_target - u)] at site=state
        z = forward(params, state_ids(s0, q, word, shown))[:, QPOS, A : A + V]
        e = torch.eye(V, device=device)[torch.tensor(state_target(s0, q, word, shown), device=device)]
        return ((e - 1 / V) * z).sum(1).mean()
    
    
    def process_theory(params, s0, q, word):             # E_t E[J_t^T (e_{s_t} - u)], clean process rows
        ids, states = process_rows(s0, q, word)
        z = forward(params, ids)[:, PRED, A : A + V]
        e = torch.eye(V, device=device)[torch.tensor(states, device=device)]
        return weighted_mean(((e - 1 / V) * z).sum(-1).mean(-1), q, word)
    
    def corrupted_rows(s0, q, word, corrupt):
        """Process rows of a trace that is wrong at every step, and the true successor F_t of each
        displayed previous state (the rule applied to what the model actually sees)."""
        b = word == q[:, None]
        shown, succ, s = [], [], s0.copy()
        for t in range(n):
            f = (s + b[:, t]) % mod
            eps = np.ones_like(f) if corrupt == "shift" else rng.integers(1, mod, size=len(f))
            s = (f + eps) % mod
            succ.append(f)
            shown.append(s.copy())
        shown, succ = np.stack(shown, 1), np.stack(succ, 1)
        return trace_ids(s0, q, word, shown), succ
    
    
    def make_corrupt(corrupt):
        """E[J^T (R_F - u)] on corrupted contexts, with the step target replaced by its exact law
        given the context (Rao-Blackwell): scatter R_F = (K u - e_F)/(K-1), shift R_F = e_{F+1}."""
        def scalar(params, s0, q, word):
            ids, succ = corrupted_rows(s0, q, word, corrupt)
            z = forward(params, ids)[:, PRED, A : A + V]
            e = torch.eye(V, device=device)
            f = torch.tensor(succ, device=device)
            R = (1 - e[f]) / (V - 1) if corrupt == "scatter" else e[(f + 1) % V]
            return ((R - 1 / V) * z).sum(-1).mean()
        return scalar
    
    
    # -----------------------------------------------------------------------------
    # run
    
    lam = None
    if task == "count":
        _om = np.exp(2j * np.pi / mod)
        lam = max(abs(1 - p_match + p_match * _om ** j) for j in range(1, mod))
    print(f"task={task} A={A} n={n} p={p_match:g}" + (f" K={mod} lambda={lam:.4f}" if task == "count" else "") +
          f"  L={n_blocks} width={W} params={n_params:,}  site={site}  layout={layout}  weights: {step_label}"
          + (" (zero readout)" if zero_readout and not ckpt else ""))
    if match_p is not None and PROMPTS is not None:
        _q, _word = PROMPTS[1], PROMPTS[2]
        _w = example_weight(_q, _word)
        print(f"reweight mean {_w.mean().item():.6f} (1 if the Bern({p_match:g}) law is normalized)")
    
    A_r = answer_spectrum() if site == "answer" else state_spectrum()
    if task == "count" and site == "answer":
        om = np.exp(2j * np.pi / mod)
        g2 = [abs(1 - p_match + p_match * om ** j) ** 2 for j in range(1, mod)]
        closed = [math.comb(n, r) * sum(x ** (n - r) * (1 - x) ** r for x in g2) / mod for r in range(n + 1)]
        assert np.allclose(A_r, closed), "answer spectrum disagrees with the closed form"
    print("target spectrum A_r (exact):", np.round(A_r, 5).tolist(), f" total {A_r.sum():.4f}")
    
    C = None
    if grads_only:
        print("grads_only: skipping C_J / W_r")
    else:
        C = [stability(r) for r in rhos]
        print("\n  rho    C_J(rho)     s.e.      D(rho)=C_J(rho)-C_J(0)   s.e.")
        for r, (c_, cse, d_, dse) in zip(rhos, C):
            print(f"  {r:4.2f}   {c_:.4e}  {cse:.1e}   {d_:+.3e}              {dse:.1e}")
    
    W_r = a_ = b_ = hi_share = hi_share_se = best = best_upper = rho_best = None
    if C is not None:
        X = np.array([[r ** k for k in range(1, n + 1)] for r in rhos if r > 0])
        y = np.array([c[2] for r, c in zip(rhos, C) if r > 0])
        try:
            from scipy.optimize import nnls
            W_hi = nnls(X, y)[0]
        except ImportError:
            W_hi = np.clip(np.linalg.lstsq(X, y, rcond=None)[0], 0, None)
        W0 = C[0][0]
        W_r = np.concatenate([[W0], W_hi])
    
        rr = np.array([r for r in rhos if r > 0])
        dd = np.array([c[2] for r, c in zip(rhos, C) if r > 0])
        ww = 1 / np.maximum(np.array([c[3] for r, c in zip(rhos, C) if r > 0]), 1e-30)
        Xq = np.c_[rr, rr ** 2] * ww[:, None]
        coef = np.linalg.lstsq(Xq, dd * ww, rcond=None)[0]
        cov = np.linalg.inv(Xq.T @ Xq)
        a_, b_ = coef
        grad_f = np.array([-b_, a_]) / (a_ + b_) ** 2
        hi_share, hi_share_se = b_ / (a_ + b_), float(np.sqrt(grad_f @ cov @ grad_f))
        print("\nmodel spectrum W_r (W_0 measured, W_r>=1 fitted, descriptive):", [f"{w:.2e}" for w in W_r])
        print(f"share of Jacobian energy at order 0: {W0 / C[-1][0]:.4f}")
        print(f"letter-dependent energy: order 1 ~ {a_:.3e}, orders >= 2 ~ {b_:.3e}; "
              f"order >= 2 share {hi_share:+.2f} +- {hi_share_se:.2f} (weighted fit)")
    
        tail = lambda r: sum(A_r[k] * r ** -k for k in range(1, n + 1))
        cands = [(math.sqrt(W0 * A_r[0]) + math.sqrt(max(c[2], 0.0) * tail(r)),
                  math.sqrt(W0 * A_r[0]) + math.sqrt(max(c[2] + 2 * c[3], 0.0) * tail(r)), r)
                 for r, c in zip(rhos, C) if r > 0]
        best, best_upper, rho_best = min(cands)
        print(f"\nbound sqrt(W_0 A_0) + sqrt(D(rho) sum_r>=1 rho^-r A_r): {best:.3e}  "
              f"(with D+2se: {best_upper:.3e}; best rho = {rho_best:.2f})")
    
    mode = "exact over all prompts" if PROMPTS is not None else f"Monte Carlo, {grad_prompts} prompts"
    grads = {}
    if site == "answer":
        tgt = grad_vec(outcome_theory, grad_prompts)
        proc = grad_vec(process_theory, grad_prompts)
        grads = dict(grad_outcome=tgt.norm().item(), grad_process=proc.norm().item())
        print(f"gradients: {mode}")
        print(f"  answer-dependent  ||E[J_v^T(e_y-u)]|| (Theorem 1): {grads['grad_outcome']:.3e}")
        print(f"  process ||E_t E[J_t^T(e_st-u)]||:                  {grads['grad_process']:.3e}"
              f"   ratio process/outcome {grads['grad_process'] / max(grads['grad_outcome'], 1e-30):.1f}")
        if not grads_only:
            res = grad_vec(outcome_residual, grad_prompts)
            nv = grad_vec(outcome_nonvalue, grad_prompts)
            full = grad_vec(outcome_full, grad_prompts)
            err = (full - (res - tgt + nv)).norm().item() / max(full.norm().item(), 1e-30)
            grads.update(grad_residual=res.norm().item(), grad_nonvalue=nv.norm().item(),
                         grad_full=full.norm().item(), split_error=err)
            print(f"  answer-value residual ||E[J_v^T(p_v-u)]||:         {grads['grad_residual']:.3e}")
            print(f"  non-answer tokens ||E[J_nv^T p_nv]||:              {grads['grad_nonvalue']:.3e}")
            print(f"  full answer-token gradient:                        {grads['grad_full']:.3e}"
                  f"   (full = residual - dependent + non-answer; relative error {err:.1e})")
            print(f"  answer-dependent share of the full gradient:       {grads['grad_outcome'] / grads['grad_full']:.2e}")
    else:
        g_state, g_state_se = 0.0, 0.0
        vecs = []
        for start in range(0, grad_prompts, 256):
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            sc = state_theory(params, *sample_state(min(256, grad_prompts - start)))
            vecs.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(sc, list(params.values()))]).double())
        vecs = torch.stack(vecs)
        mean = vecs.mean(0)
        g_state = mean.norm().item()
        g_state_se = math.sqrt(((vecs - mean) ** 2).sum().item() / (len(vecs) * (len(vecs) - 1)))
        grads = dict(grad_state=g_state, grad_state_se=g_state_se)
        print(f"state-position ||E[J^T(e_target-u)]||: {g_state:.3e}  (s.e. {g_state_se:.1e}, Monte Carlo)")
    
    if out:
        payload = dict(task=task, A=A, n=n, K=mod, L=n_blocks, seed=seed, init_std=init_std,
                       match_p=p_match, lambda_theory=lam,
                       layout=layout, A_r=A_r.tolist(), gradients=mode, site=site,
                       weights=step_label, ckpt=ckpt, zero_readout=zero_readout,
                       grads_only=grads_only, **grads)
        if C is not None:
            payload.update(rhos=rhos, C_J=[c[0] for c in C], C_J_se=[c[1] for c in C],
                           D=[c[2] for c in C], D_se=[c[3] for c in C], W_r=W_r.tolist(),
                           bound=best, bound_upper=best_upper, rho_best=rho_best,
                           W1_fit=a_, Whi_fit=b_, hi_share=hi_share, hi_share_se=hi_share_se)
        with open(out, "w") as f:
            json.dump(payload, f, indent=1)
    
    # -----------------------------------------------------------------------------
    # wrong traces at initialization: is the net process gradient mu(rho) times the clean one?
    
    if task == "count" and mod >= 3 and site == "answer":
        g_clean, g_out_vec = proc, tgt
        cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()
        print("\nwrong traces at theta_0 (clean process gradient g_1; corrupted part g_R)")
        report = {}
        for corrupt in ("scatter", "shift"):
            g_R = grad_vec(make_corrupt(corrupt), grad_prompts)
            c = 1 / (mod - 1) if corrupt == "scatter" else 1.0
            print(f"  {corrupt}: cos(g_R, g_1) = {cos(g_R, g_clean):+.4f}  "
                  f"||g_R||/||g_1|| = {g_R.norm().item() / g_clean.norm().item():.4f}"
                  + (f"  (blind prediction: cos = -1, ratio = {1 / (mod - 1):.4f})" if corrupt == "scatter" else ""))
            rows = []
            for r in trace_rhos:
                g = r * g_clean + (1 - r) * g_R
                mu = r * (1 + c) - c
                rows.append(dict(rho=r, mu=mu, norm_ratio=g.norm().item() / g_clean.norm().item(),
                                 cos_clean=cos(g, g_clean), vs_clean_outcome=g.norm().item() / g_out_vec.norm().item()))
                print(f"    rho={r:.2f}  mu={mu:+.3f}  ||g_rho||/||g_1||={rows[-1]['norm_ratio']:.4f}  "
                      f"cos(g_rho,g_1)={rows[-1]['cos_clean']:+.4f}  vs clean outcome x{rows[-1]['vs_clean_outcome']:.1f}")
            report[corrupt] = dict(cos_R=cos(g_R, g_clean), ratio_R=g_R.norm().item() / g_clean.norm().item(), rows=rows)
        ratio = g_out_vec.norm().item() / g_clean.norm().item()
        print(f"  noisy CoT beats clean outcome (scatter, blind) once rho > (1+(K-1)r)/K = {(1 + (mod - 1) * ratio) / mod:.4f},"
              f" r = ||g_O||/||g_1|| = {ratio:.2e}")
        if out:
            with open(out) as f:
                record = json.load(f)
            record["wrong_traces"] = report
            with open(out, "w") as f:
                json.dump(record, f, indent=1)

def run_answers(argv):
    __import__("sys").argv = ['measure.py answers'] + list(argv)
    """Answer spectra of the validation tasks (Table 1 lambda and the validation-spectra appendix).
    
        python handcoded/spectrum.py job=answers
    
    Exact answer spectrum A_r = sum_{|S|=r} E||a_S||^2 for a reversible task with
    uniform s0, by dynamic programming over second moments (Lemma B.2):
      t not in S: Sigma -> P^T Sigma P;  t in S: Sigma -> E[T^T Sigma T] - P^T Sigma P.
    The answer may be a readout of the state (register machine reports x only)."""
    import numpy as np, itertools
    def mixing_rate(Ts, ws):
        """lambda of Eq. (2): largest singular value of the averaged move on mean-zero vectors."""
        K = Ts[0].shape[0]
        P = sum(w * T for w, T in zip(ws, Ts))
        C = np.eye(K) - np.ones((K, K)) / K
        return np.linalg.svd(C @ P @ C, compute_uv=False)[0]
    def spectrum(Ts, ws, n, R):
        K = Ts[0].shape[0]
        P = sum(w * T for w, T in zip(ws, Ts))
        Sig = [np.zeros((K, K)) for _ in range(n + 1)]
        Sig[0] = (np.eye(K) - np.ones((K, K)) / K) / K          # E[(e_s0-u)(e_s0-u)^T]
        for t in range(n):
            new = [np.zeros((K, K)) for _ in range(n + 1)]
            for r in range(t + 1):
                S_ = Sig[r]
                avg = P.T @ S_ @ P
                new[r] += avg
                new[r + 1] += sum(w * T.T @ S_ @ T for w, T in zip(ws, Ts)) - avg
            Sig = new
        return np.array([np.trace(R.T @ S_ @ R) for S_ in Sig])   # answer = R^T(e_y - u)
    # Boolean-8
    NB = 4; K = 16
    def bits(v): return [int(b) for b in f"{v:0{NB}b}"]
    def val(b): return int("".join(map(str, b)), 2)
    def apply(s, g):
        r = s.copy(); op = g[0]
        if op == "x": r[g[1]] ^= 1
        elif op == "c": r[g[2]] ^= r[g[1]]
        elif op == "s": r[g[1]], r[g[2]] = r[g[2]], r[g[1]]
        else: r[g[3]] ^= r[g[1]] & r[g[2]]
        return r
    law = [(("x", i), .25 / 4) for i in range(NB)]
    for op in "cs": law += [((op, a, b), .25 / 12) for a, b in itertools.permutations(range(NB), 2)]
    law += [(("t", a, b, c), .25 / 24) for a, b, c in itertools.permutations(range(NB), 3)]
    Ts, ws = [], []
    for g, w in law:
        T = np.zeros((K, K))
        for s in range(K): T[s, val(apply(bits(s), g))] = 1
        Ts.append(T); ws.append(w)
    print("boolean lambda %.4f" % mixing_rate(Ts, ws))
    A = spectrum(Ts, ws, 8, np.eye(K))
    print("boolean D=8  total %.4f (1-1/K=%.4f)" % (A.sum(), 1 - 1 / K), " A_r:", np.round(A, 4).tolist(),
          " share at r>=D/2: %.3f" % (A[4:].sum() / A.sum()), " share r<=1: %.4f" % (A[:2].sum() / A.sum()))
    # register, answer = x register
    M = 17; K = M * M
    def reg(x, y, i): return {"a": ((x + y) % M, y), "b": (x, (x + y) % M), "c": (y, x), "d": ((x + 1) % M, y), "e": (x, (y + 1) % M)}[i]
    Ts = []
    for i in "abcde":
        T = np.zeros((K, K))
        for x in range(M):
            for y in range(M):
                a, b = reg(x, y, i); T[x * M + y, a * M + b] = 1
        Ts.append(T)
    R = np.zeros((K, M))
    for x in range(M):
        for y in range(M): R[x * M + y, x] = 1
    print("register lambda %.4f" % mixing_rate(Ts, [.2] * 5))
    A = spectrum(Ts, [.2] * 5, 16, R)
    print("register D=16 x-readout total %.4f (1-1/17=%.4f)" % (A.sum(), 1 - 1 / 17), " A_r:", np.round(A, 4).tolist(),
          " share r>=8: %.3f" % (A[8:].sum() / A.sum()), " share r<=1: %.4f" % (A[:2].sum() / A.sum()),
          " mean order %.2f" % ((np.arange(17) * A).sum() / A.sum()))
    return


def run_exact(argv):
    __import__("sys").argv = ['measure.py exact'] + list(argv)
    
    import ast, math, sys
    import numpy as np
    import torch
    from torch.func import jvp
    from torch.nn import functional as F
    
    ckpt = ""
    site = "answer"          # answer | every_k
    every_k = 2
    shown_draws = 2
    dirs = 16
    seed = 0
    init_std = 0.02          # init scale when no ckpt is given
    init_seed = 42           # seed of theta_0 when no ckpt is given
    n, A, mod, n_blocks = 8, 4, 2, 2
    device = "cuda"
    _cfg = {}
    for a in sys.argv[1:]:
        k, v = a.split("=", 1)
        assert k in ['ckpt', 'site', 'every_k', 'shown_draws', 'dirs', 'seed', 'init_std', 'init_seed', 'n', 'A', 'mod', 'n_blocks', 'device'], f"unknown setting {k}"
        _cfg[k] = v if k in ("ckpt", "device", "site") else ast.literal_eval(v)
    ckpt = _cfg.get('ckpt', ckpt)
    site = _cfg.get('site', site)
    every_k = _cfg.get('every_k', every_k)
    shown_draws = _cfg.get('shown_draws', shown_draws)
    dirs = _cfg.get('dirs', dirs)
    seed = _cfg.get('seed', seed)
    init_std = _cfg.get('init_std', init_std)
    init_seed = _cfg.get('init_seed', init_seed)
    n = _cfg.get('n', n)
    A = _cfg.get('A', A)
    mod = _cfg.get('mod', mod)
    n_blocks = _cfg.get('n_blocks', n_blocks)
    device = _cfg.get('device', device)
    torch.set_default_dtype(torch.float64)
    
    V = mod
    SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
    VOCAB, P, H = A + V + 3, 2 * n + 6, 3
    F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
    W = 2 * A + 4 * V + P + 3
    QPOS = n + 3
    
    
    def forward(p, ids):  # copied from handcoded/spectrum.py
        T = ids.shape[1]
        x = p["wte"][ids] + p["wpe"][:T]
        future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
        for b in range(n_blocks):
            q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
            k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
            v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
            att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -math.inf)
            x = x + (att.softmax(-1) @ v).sum(1)
            x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
        return x @ p["readout"].T
    
    
    if ckpt:
        saved = torch.load(ckpt, map_location=device)
        cfg = saved["config"]
        assert (cfg["task"], cfg["word_len"], cfg["alphabet"], cfg["mod"], cfg["n_blocks"]) == ("count", n, A, mod, n_blocks), cfg
        theta = {k: t.to(device).double() for k, t in saved["params"].items()}
        label = f"{saved['mode']} step {saved['step']}"
    else:  # theta_0 of seed 42, as in spectrum.py
        torch.manual_seed(init_seed)
        shapes = {"wte": (VOCAB, W), "wpe": (P, W), "readout": (VOCAB, W)}
        for b in range(n_blocks):
            shapes |= {f"{b}.q": (H, P, W), f"{b}.k": (H, P, W), f"{b}.v": (H, W, W),
                       f"{b}.mlp_in": (F_WIDTH, W), f"{b}.mlp_b": (F_WIDTH,), f"{b}.mlp_out": (W, F_WIDTH)}
        theta = {k: (torch.randn(s, dtype=torch.float32) * init_std).double().to(device) for k, s in shapes.items()}
        label = f"init std={init_std} seed={init_seed}"
    
    words = torch.tensor((np.arange(A ** n)[:, None] // A ** np.arange(n)[::-1]) % A, device=device)
    
    
    def ids_for(s0, q):
        m = len(words)
        return torch.cat([torch.full((m, 1), A + s0, device=device), torch.full((m, 1), q, device=device),
                          words, torch.full((m, 1), SEP, device=device), torch.full((m, 1), COLON, device=device)], 1)
    
    
    def logits(p, ids):
        return forward(p, ids)[:, QPOS, A:A + V]
    
    
    # orthonormal basis for L2(uniform on A letters): column 0 constant
    Qb = torch.linalg.qr(torch.cat([torch.ones(A, 1), torch.randn(A, A - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
    Qb = (Qb * math.sqrt(A) * torch.sign(Qb[0, 0])).to(device)   # E_x[phi_i phi_j] = delta_ij
    ORDER = (words != 0).sum(1)   # word index doubles as the multi-index of coefficients
    
    
    def order_energy(f):
        """f: (A^n, V) values on the grid -> energy per order under the uniform law."""
        c = f.reshape((A,) * n + (V,))
        for t in range(n):
            c = torch.tensordot(c, Qb, dims=([t], [0])) / A      # coefficient <f, phi_i>
            c = c.movedim(-1, t)
        e = (c ** 2).sum(-1).reshape(-1)
        return torch.zeros(n + 1, device=device).index_add_(0, ORDER, e)
    
    
    def answer_onehot(s0, q):
        y = (s0 + (words == q).sum(1)) % mod
        return torch.eye(V, device=device)[y]
    
    
    if site == "answer":
        gen = torch.Generator(device=device).manual_seed(seed)
        Wr = torch.zeros(n + 1, device=device)
        Ar = torch.zeros(n + 1, device=device)
        grad_phi = {k: torch.zeros_like(t) for k, t in theta.items()}
        combos = [(s0, q) for s0 in range(mod) for q in range(A)]
        for s0, q in combos:
            ids = ids_for(s0, q)
            Ar += order_energy(answer_onehot(s0, q) - 1 / V) / len(combos)
            for _ in range(dirs):
                g = {k: torch.randn(t.shape, generator=gen, device=device) for k, t in theta.items()}
                _, jg = jvp(lambda p: logits(p, ids), (theta,), (g,))
                Wr += order_energy(jg) / (len(combos) * dirs)
            p = {k: t.clone().requires_grad_() for k, t in theta.items()}
            phi = ((answer_onehot(s0, q) - 1 / V) * logits(p, ids)).sum(1).mean() / len(combos)
            gs = torch.autograd.grad(phi, list(p.values()))
            print(f"s0={s0} q={q} per-combo ||grad Phi|| {len(combos)*math.sqrt(sum((t**2).sum().item() for t in gs)):.4e}")
            for k, gk in zip(p, gs):
                grad_phi[k] += gk
    
        Wr, Ar = Wr.cpu().numpy(), Ar.cpu().numpy()
        gphi = math.sqrt(sum((t ** 2).sum().item() for t in grad_phi.values()))
        Wt = Wr.sum()
        print(f"== {label}")
        print("W_r / W :", " ".join(f"{x:.3e}" for x in Wr / Wt), f"  (W = {Wt:.4e})")
        print("A_r     :", " ".join(f"{x:.3e}" for x in Ar), f"  (sum = {Ar.sum():.4f})")
        print(f"order-0 share {Wr[0] / Wt:.6f}   H_0/W {Wr[1:].sum() / Wt:.3e}   H_1/W {Wr[2:].sum() / Wt:.3e}")
        print(f"||grad Phi|| exact {gphi:.4e}   sum_r sqrt(A_r W_r) {np.sqrt(Ar * Wr).sum():.4e}")
        for qq in range(3):
            lo = math.sqrt(Ar[:qq + 1].sum() * Wr[:qq + 1].sum()); hi = math.sqrt(Ar[qq + 1:].sum() * Wr[qq + 1:].sum())
            print(f"  q={qq}: sqrt(A<=q W<=q) {lo:.3e}   sqrt(A>q H_q) {hi:.3e}   bound {lo + hi:.3e}")
        raise SystemExit(0)
    
    # every-k site: position n+1+j predicts v_j = s_{jk} from the displayed v_{j-1} (free input).
    assert saved["config"].get("every_k", 1) == every_k if ckpt else True
    k = every_k; NV = n // k
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    
    
    def state_logits(p, ids, pos):
        return forward(p, ids)[:, pos, A:A + V]
    
    
    def ids_state(s0, q, shown):
        m = len(words)
        cols = [torch.full((m, 1), A + s0, device=device), torch.full((m, 1), q, device=device), words,
                torch.full((m, 1), SEP, device=device)]
        cols += [torch.full((m, 1), A + int(v), device=device) for v in shown]
        return torch.cat(cols, 1)
    
    
    def subset_energy(f):
        """energy per coefficient multi-index (A^n,) summed over V"""
        c = f.reshape((A,) * n + (V,))
        for t in range(n):
            c = torch.tensordot(c, Qb, dims=([t], [0])) / A
            c = c.movedim(-1, t)
        return (c ** 2).sum(-1).reshape(-1)
    
    
    rows = []
    for j in range(1, NV + 1):
        win = list(range((j - 1) * k, j * k))                 # letter axes the target depends on
        nz = (words != 0)
        in_win_only = nz[:, [t for t in range(n) if t not in win]].sum(1) == 0
        win_order = nz[:, win].sum(1)
        masks = {"win_order%d" % r: (in_win_only & (win_order == r)) for r in range(k + 1)}
        masks["outside"] = ~in_win_only
        acc = {m: 0.0 for m in masks}; tot = 0.0; a_acc = {m: 0.0 for m in masks}
        count = 0
        for s0 in range(mod):
            for q in range(A):
                for _ in range(shown_draws if j > 1 else 1):
                    shown = rng.integers(mod, size=j - 1)
                    prev = shown[-1] if j > 1 else s0
                    ids = ids_state(s0, q, shown)
                    pos = n + 2 + (j - 1)
                    y = (prev + (words[:, win] == q).sum(1)) % mod
                    ea = subset_energy(torch.eye(V, device=device)[y] - 1 / V)
                    for m, mk in masks.items():
                        a_acc[m] += ea[mk].sum().item()
                    for _ in range(dirs):
                        g = {kk: torch.randn(t.shape, generator=gen, device=device) for kk, t in theta.items()}
                        _, jg = jvp(lambda p: state_logits(p, ids, pos), (theta,), (g,))
                        e = subset_energy(jg)
                        tot += e.sum().item() / dirs
                        for m, mk in masks.items():
                            acc[m] += e[mk].sum().item() / dirs
                    count += 1
        rows.append((j, {m: v / tot for m, v in acc.items()}, {m: v / count for m, v in a_acc.items()}, tot / count))
    print(f"== {label}  every_k={k}")
    for j, wsh, a, wt in rows:
        print(f"v{j}: W share " + " ".join(f"{m}={v:.3e}" for m, v in wsh.items()) + f"  | A " + " ".join(f"{m}={v:.3e}" for m, v in a.items() if v > 1e-12) + f"  | W={wt:.3e}")


def run_reach(argv):
    __import__("sys").argv = ['measure.py reach'] + list(argv)
    import argparse
    import json
    import math
    import time
    from pathlib import Path
    
    import torch
    import torch.nn.functional as F
    
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[6])
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--A", type=int, default=4)        # alphabet; letter 0 is the query
    p.add_argument("--sigma", type=float, nargs="+", default=[0.3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--format", choices=("step", "outcome"), default="step")
    p.add_argument("--armijo", type=int, default=0,
                   help="after solving, up to this many full-batch Armijo updates; record when 4L/pi_min < 1")
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    H = 3
    TRAINED = ("q", "k", "rb", "readout")
    
    
    def run(n, K, A, sigma, seed):
        torch.manual_seed(seed)
        SEP, VOCAB, P = A + K, A + K + 1, 2 * n + 6
        W = 2 * A + 4 * K + P + 3                          # residual width of lettertrace.py
        FW = max(A, 3 * K * (n + 1), 2 * n + 1)            # MLP width of lettertrace.py
    
        def sample(m):
            w = torch.randint(A, (m, n))
            b = (w == 0).long()
            s0 = torch.randint(K, (m,))
            s = (s0[:, None] + b.cumsum(1)) % K                                   # s_1..s_n
            ids = torch.cat([A + s0[:, None], torch.zeros(m, 1, dtype=torch.long), w,
                             torch.full((m, 1), SEP), A + s[:, :-1]], 1)          # s0 q w SEP s_1..s_{n-1}
            if args.format == "outcome":                                          # s0 q w SEP -> s_n
                return ids[:, :n + 3].to(dev), s[:, -1:].to(dev)
            return ids.to(dev), s.to(dev)
    
        x_tr, y_tr = sample(args.batch)
        x_ev, y_ev = sample(4 * args.batch)
        T = x_tr.shape[1]                                                         # 2n + 2
        pred = torch.arange(n + 2, 2 * n + 2, device=dev)                         # positions predicting s_1..s_n
        if args.format == "outcome":
            pred = torch.tensor([n + 2], device=dev)                              # SEP predicts s_n
    
        g = lambda fan, *s: sigma * torch.randn(*s, device=dev) / math.sqrt(fan)     # variance sigma^2 / fan-in
        th = {"wte": g(W, VOCAB, W), "wpe": g(W, P, W), "readout": g(W, K, W)}
        for b in range(2):
            th[f"{b}.q"], th[f"{b}.k"], th[f"{b}.v"] = g(W, H, P, W), g(W, H, P, W), g(W, H, W, W)
            th[f"{b}.rb"] = torch.zeros(H, P, device=dev)
            th[f"{b}.mlp_in"], th[f"{b}.mlp_out"] = g(W, FW, W), g(FW, W, FW)
            sign = torch.randint(2, (FW,), device=dev) * 2 - 1
            th[f"{b}.mlp_b"] = sigma * sign * (1 + torch.rand(FW, device=dev))   # |b~_m| in [1, 2]
        params = [t.requires_grad_() for k, t in th.items() if k.split(".")[-1] in TRAINED]
    
        rel = (torch.arange(T, device=dev)[:, None] - torch.arange(T, device=dev)[None, :]).clamp(min=0)
        future = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)
    
        def forward(ids):
            x = th["wte"][ids] + th["wpe"][:T]
            for b in range(2):
                q = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.q"])
                k = torch.einsum("btw,hdw->bhtd", x, th[f"{b}.k"])
                v = torch.einsum("btw,hvw->bhtv", x, th[f"{b}.v"])
                att = q @ k.transpose(-1, -2) / math.sqrt(P) + th[f"{b}.rb"][:, rel]
                x = x + (att.masked_fill(future, -math.inf).softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ th[f"{b}.mlp_in"].T + th[f"{b}.mlp_b"]) @ th[f"{b}.mlp_out"].T
            return x[:, pred] @ th["readout"].T                                   # K value logits
    
        def margin(ids, y):
            with torch.no_grad():
                z = forward(ids)
                gold = z.gather(-1, y[..., None]).squeeze(-1)
                other = z.scatter(-1, y[..., None], float("-inf")).max(-1).values
                return (gold - other).min().item(), (z.argmax(-1) == y).all(1).double().mean().item()
    
        log = []
        for step in range(args.steps + 1):
            loss = F.cross_entropy(forward(x_tr).reshape(-1, K), y_tr.reshape(-1))
            if step % args.every == 0:
                m, exact = margin(x_ev, y_ev)
                log.append(dict(step=step, loss=loss.item(), margin=m, exact=exact,
                                rb=max(th[f"{b}.rb"].abs().max().item() for b in range(2)),
                                readout=th["readout"].norm().item()))
                solved = next((r["step"] for r in log if r["margin"] > 0), None)
                if solved is not None and (args.armijo or step >= solved + 5000):   # hand over to Armijo, or stop
                    break
            for q in params:
                q.grad = None
            loss.backward()
            with torch.no_grad():
                for q in params:
                    q -= args.lr * q.grad
        basin = []
        if args.armijo and any(r["margin"] > 0 for r in log):
            # Theorem 4 on the training set: items are (prompt, position) with weight 1/(N n), so pi_min = 1/(N n).
            pi_min = 1.0 / (x_tr.shape[0] * n)
            lossf = lambda: F.cross_entropy(forward(x_tr).reshape(-1, K), y_tr.reshape(-1))
            eta_bar, entered = 100.0, None
            for j in range(args.armijo):
                loss = lossf()
                for q in params:
                    q.grad = None
                loss.backward()
                g2 = sum((q.grad ** 2).sum() for q in params).item()
                ratio = 4 * loss.item() / pi_min
                if entered is None and ratio < 1:
                    entered = j
                m, _ = margin(x_ev, y_ev)
                basin.append(dict(update=j, loss=loss.item(), ratio=ratio, heldout_margin=m))
                if entered is not None and j >= entered + 100:
                    break
                eta = eta_bar
                with torch.no_grad():
                    old = [q.detach().clone() for q in params]
                    while True:                                   # Armijo: L(theta - eta g) <= L - eta/2 |g|^2
                        for q, o in zip(params, old):
                            q.copy_(o - eta * q.grad)
                        if lossf().item() <= loss.item() - eta / 2 * g2 or eta < 1e-8:
                            break
                        eta /= 2
            basin = dict(entered=entered, trace=basin)
        return log, basin
    
    
    results = []
    for n in args.n:
        for sigma in args.sigma:
            for seed in args.seeds:
                t0 = time.time()
                log, basin = run(n, args.K, args.A, sigma, seed)
                solved = next((r["step"] for r in log if r["margin"] > 0), None)
                last = log[-1]
                print(f"n={n:3d} sigma={sigma:<6} seed={seed} solved_at={solved} final loss {last['loss']:.4f} "
                      f"margin {last['margin']:+.3f} exact {last['exact']:.3f} max|rb| {last['rb']:.2f} "
                      f"basin_entered_after={basin.get('entered') if basin else None} Armijo updates "
                      f"({time.time() - t0:.0f}s)", flush=True)
                results.append(dict(n=n, K=args.K, A=args.A, sigma=sigma, seed=seed, lr=args.lr, log=log, basin=basin))
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(json.dumps(results))


def run_bbh(argv):
    __import__("sys").argv = ['measure.py bbh'] + list(argv)
    import ast
    import json
    import math
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    from src.shuffled import parse_bbh, sample, prompt, answer  # noqa: E402
    
    model_name = "Qwen/Qwen2.5-0.5B"
    contexts = 10
    seed = 2001
    out = ""
    _cfg = {}
    for a in sys.argv[1:]:
        k, v = a.split("=", 1)
        assert k in ['model_name', 'contexts', 'seed', 'out'], f"unknown setting {k}"
        _cfg[k] = v if k in ("model_name", "out") else ast.literal_eval(v)
    model_name = _cfg.get('model_name', model_name)
    contexts = _cfg.get('contexts', contexts)
    seed = _cfg.get('seed', seed)
    out = _cfg.get('out', out)
    K, n, V = 3, 3, 6                     # objects, moves, values per move (pair x name order)
    dev = "cuda"
    torch.manual_seed(seed)               # LoRA A initialization
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                                             bias="none", target_modules="all-linear"))
    model.eval()
    params = [p for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name]
    assert all(float(p.abs().max()) == 0.0 for p in params), "LoRA B must start at zero"
    P = sum(p.numel() for p in params)
    letters = [tok("\nAnswer: (" + chr(65 + j) + ")", add_special_tokens=False)["input_ids"][4] for j in range(K)]
    suffix = tok("\nAnswer: (", add_special_tokens=False)["input_ids"]
    
    # orthonormal basis of L2(uniform on V values), column 0 constant
    Qb = torch.linalg.qr(torch.cat([torch.ones(V, 1), torch.randn(V, V - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
    Qb = (Qb * math.sqrt(V) * torch.sign(Qb[0, 0])).to(dev)
    grid = np.array([(i // V ** 2, (i // V) % V, i % V) for i in range(V ** n)])
    ORDER = torch.tensor((grid != 0).sum(1), device=dev)
    
    
    def degree_energy(f):
        """f: (V^n, ...) on the grid -> energy per degree under the uniform law."""
        c = f.reshape((V,) * n + (-1,))
        for t in range(n):
            c = torch.tensordot(c, Qb, dims=([t], [0])) / V
            c = c.movedim(-1, t)
        e = (c ** 2).sum(-1).reshape(-1)
        return torch.zeros(n + 1, device=dev, dtype=e.dtype).index_add_(0, ORDER, e)
    
    
    scenes, _ = parse_bbh(K)
    rng = np.random.default_rng(seed)
    results = []
    for c in range(contexts):
        base = sample(K, n, rng, scenes)
        J = torch.zeros(V ** n, K, P, device=dev)
        target = torch.zeros(V ** n, K, device=dev)
        for i, (m0, m1, m2) in enumerate(grid):
            ex = dict(base, moves=[int(m // 2) for m in (m0, m1, m2)], flips=[bool(m % 2) for m in (m0, m1, m2)])
            ids = tok(prompt(ex, K), add_special_tokens=True)["input_ids"] + suffix
            z = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1, letters]
            for k in range(K):
                g = torch.autograd.grad(z[k], params, retain_graph=k < K - 1)
                J[i, k] = torch.cat([x.reshape(-1) for x in g])
            target[i, ord(answer(ex, K)[1]) - 65] = 1.0
        Wr = degree_energy(J).double().cpu().numpy()
        Ar = degree_energy(target - 1.0 / K).double().cpu().numpy()
        gphi = (torch.einsum("ikp,ik->p", J, target - 1.0 / K) / V ** n).norm().item()
        results.append(dict(W_r=Wr.tolist(), A_r=Ar.tolist(), grad_phi=gphi))
        print(f"context {c}: W_r/W = {np.round(Wr / Wr.sum(), 12).tolist()}  W={Wr.sum():.4e}  "
              f"A_r = {np.round(Ar, 6).tolist()}  |grad Phi| = {gphi:.3e}", flush=True)
        del J
    
    Wm = np.mean([r["W_r"] for r in results], 0)
    summary = dict(model=model_name, seed=seed, contexts=contexts, lora_B_params=P,
                   W_r_mean=Wm.tolist(), share=(Wm / Wm.sum()).tolist(), per_context=results)
    print("mean share by degree:", np.round(Wm / Wm.sum(), 12).tolist())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=1))



if __name__ == "__main__":
    COMMANDS = {"spectrum": run_spectrum, "answers": run_answers, "exact": run_exact, "reach": run_reach, "bbh": run_bbh}
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2:])
