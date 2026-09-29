"""
Process vs outcome supervision on letter tasks, one file, only torch.

A prompt is a start value s0, a query letter q and a word of n letters.
    count: s_t = (s_{t-1} + [w_t == q]) mod m      every step is a permutation (A2)
    first: s_t = index of the first q in w_1..w_t   absorbing once found (breaks A2)
Process supervision trains on s_1..s_n, outcome supervision on s_n only. Both exact
solutions are written by hand into ONE architecture, a random theta_0 in that
architecture is trained under each loss, and everything is scored by greedy
free-running generation on held-out words.

    python handcoded/lettertrace.py task=count word_len=12 mod=2 steps=8000
    python handcoded/lettertrace.py task=first word_len=12 steps=8000
    python handcoded/lettertrace.py task=count rho=0.5 corrupt=shift mod=3
    python handcoded/lettertrace.py modes=outcome steps=32000 save_every=1000 save_dir=ckpt/n8_long
    python handcoded/lettertrace.py save_every=500 save_dir=ckpt/n8

Import configure, make_corpus, init_theta, handcoded_params, train_mode, and
generate from this file. handcoded/lettertrace_train.ipynb trains both losses
from one initialization and replays them against the two handcoded programs.

Rows (both formats share the prompt  s0 q w_1 .. w_n SEP):
    outcome:  s0 q w_1 .. w_n SEP  : s_n EOS
    process:  s0 q w_1 .. w_n SEP  s_1 .. s_n : s_n EOS
Both models see the same prompt; process writes the intermediate states before the answer.
(layout=stream, which interleaves letters and states, s0 q w_1 s_1 .. w_n s_n : s_n EOS,
is kept as an ablation; there the letters are fed in during evaluation.)
Block 0 of every solution marks each letter with [w_t == q]. Block 1 either steps once
per emitted value (process) or reads all n marks at the answer (outcome).
"""
import ast
import math
import os
import random
import sys

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# config: every name below can be overridden as key=value on the command line,
# or by configure(**overrides) before calling the training functions.

TASKS = ("count", "first", "shell", "first_cup", "lights", "lights_parity")
SHELL_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
H = 3
_STRING_KEYS = ("device", "task", "corrupt", "layout", "save_dir", "tag", "modes", "opt")
DEFAULTS = dict(
    task="count",          # count, first, shell, first_cup, lights, lights_parity
    alphabet=4,            # A letters; P(w_t == q) = 1/A
    word_len=12,           # n, the horizon
    mod=2,                 # m, count task only
    rho=1.0,               # fraction of clean process traces, count task only
    corrupt="scatter",     # scatter | shift | slip (one wrong step, then local-correct)
    slip_at=0,             # slip: 0 = random position, else 1..n
    layout="block",        # process rows: "block" (same prompt as outcome) or "stream"
    n_blocks=2,
    every_k=1,             # process: supervise s_k, s_{2k}, ... (n must be divisible by k)
    opt="adamw",           # "adamw" or "sgd"
    train_size=20_000,
    test_size=1_000,
    steps=8_000,
    batch_size=256,
    lr=1e-3,
    init_std=0.02,
    seed=42,
    save_every=0,          # 0 = no checkpoints; else write every this many steps
    save_dir="ckpt",
    tag="",                # checkpoint prefix; empty => <mode>_step<it>.pt
    modes="process,outcome",   # which losses to train, comma-separated
    eval_every=0,          # 0: eight evaluations per run; else evaluate every this many steps
    clean_only=False,      # control: drop the corrupted traces after the rho draw, train on the clean ones
    gap=0.0,               # handcoded routing score gap; 0 => max(16, 3 log(8(n+3))), exact for every n
    perturb=(),            # handcoded robustness: absolute Gaussian noise std on every weight entry
    perturb_draws=5,       # noise draws per std
    handcoded_only=False,  # stop after evaluating the handcoded solutions
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def configure(**overrides):
    """Set the module globals the training functions read, then rebuild derived sizes."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown setting {sorted(unknown)}")
    g = globals()
    for key, value in DEFAULTS.items():
        g[key] = overrides[key] if key in overrides else value
    assert task in TASKS and corrupt in ("scatter", "shift", "slip") and n_blocks >= 2
    assert layout in ("stream", "block") and opt in ("adamw", "sgd")
    if task == "shell" or task == "first_cup":
        g["alphabet"], g["mod"] = 10, 5
    elif task == "lights":
        g["alphabet"], g["mod"] = 4, 16
    elif task == "lights_parity":
        g["alphabet"], g["mod"] = 4, 2
    g["train_modes"] = tuple(m.strip() for m in modes.split(",") if m.strip())
    assert train_modes and set(train_modes) <= {"process", "outcome"}, (
        f"modes must be process and/or outcome (got {modes!r})")
    assert task == "count" or rho == 1.0, "corruption is defined for the count task"
    assert every_k >= 1 and word_len % every_k == 0, "every_k must divide word_len"
    assert every_k == 1 or layout == "block", "every-k is defined for the block layout"
    random.seed(seed)
    torch.manual_seed(seed)

    g["n"], g["A"] = word_len, alphabet
    g["V"] = n + 1 if task == "first" else mod
    g["SEP"], g["COLON"], g["EOS"] = A + V, A + V + 1, A + V + 2
    g["VOCAB"] = A + V + 3
    g["N_TRACE"] = n // every_k
    g["P"] = 2 * n + 6                               # longest row: prompt n+3, process continuation n+3
    g["STREAM"] = layout == "stream"
    g["F_WIDTH"] = max(A, 3 * V * (n + 1), 2 * n + 1)
    start = 0
    for name, size in dict(TL=A, TV=V, POS=P, QRY=A, MATCH=1, PREV=V, MT=1, CNT=1, S0=V, OUT=V).items():
        g[name] = slice(start, start + size)
        start += size
    g["W"] = start
    g["GAP"] = gap if gap else max(16.0, 3 * math.log(8 * (n + 3)))
    g["C"] = GAP * math.sqrt(P)                      # routing score gap GAP: softmax leak per position ~ e^-GAP


def step(s, t, letter, q):
    """The gold transition after letter t (1-indexed)."""
    if task == "count":
        return (s + (letter == q)) % mod
    if task == "first":
        return s if s else (t if letter == q else 0)
    if task == "shell":
        a, b = SHELL_PAIRS[letter]
        return b if s == a else a if s == b else s
    if task == "first_cup":
        if t > 1:
            return s
        a, b = SHELL_PAIRS[letter]
        return min(a, b)
    if task == "lights":
        return s ^ (1 << letter)
    if task == "lights_parity":
        return s ^ 1
    raise ValueError(task)


def make_example(rng):
    while True:
        q, word = rng.randrange(A), tuple(rng.randrange(A) for _ in range(n))
        if task != "first" or q in word:
            break
    if task == "count":
        s0 = rng.randrange(mod)
    elif task in ("shell", "lights", "lights_parity"):
        s0 = rng.randrange(mod)
    else:
        s0 = 0
    states, s = [], s0
    for t, c in enumerate(word, 1):
        s = step(s, t, c, q)
        states.append(s)
    return s0, q, word, states


def corrupt_trace(ex, rng):
    """Wrong at every step (scatter/shift), or one slip then locally correct."""
    s0, q, word, _ = ex
    shown, s = [], s0
    j = slip_at if (corrupt == "slip" and slip_at) else rng.randrange(1, n + 1)
    for t, c in enumerate(word, 1):
        right = step(s, t, c, q)
        if corrupt == "slip":
            s = rng.choice([v for v in range(mod) if v != right]) if t == j else right
        elif corrupt == "shift":
            s = (right + 1) % mod
        else:
            s = rng.choice([v for v in range(mod) if v != right])
        shown.append(s)
    return shown


def shown_traces(examples, rng):
    """Process targets, nested in rho.

    One u_i and one wrong trace are drawn per example, before rho is consulted.
    The example is clean iff u_i < rho. Raising rho therefore only turns corrupt
    traces into clean ones: every clean trace at a smaller rho stays clean, and a
    trace that stays corrupt is byte-identical. Train and test streams are not this rng.
    """
    if task != "count" or rho == 1.0:
        return [ex[3] for ex in examples]
    uniforms = [rng.random() for _ in examples]
    wrong = [corrupt_trace(ex, rng) for ex in examples]
    return [states if u < rho else bad for (_, _, _, states), u, bad in zip(examples, uniforms, wrong)]


def prompt(ex):
    s0, q, word, _ = ex
    return [A + s0, q, *word, SEP]


def row(ex, mode, shown=None):
    """Full training row: prompt, trace (process only), then ': s_n EOS'."""
    s0, q, word, states = ex
    tail = [COLON, A + states[-1], EOS]
    values = [A + v for v in (shown or states)]
    if mode == "process" and every_k > 1:
        values = values[every_k - 1::every_k]
    if mode == "outcome":
        return prompt(ex) + tail
    if STREAM:
        return [A + s0, q] + [t for pair in zip(word, values) for t in pair] + tail
    return prompt(ex) + values + tail


def supervised(mode):
    """Row positions whose token is a training target."""
    n_val = n if every_k == 1 else N_TRACE
    length = n + 6 if mode == "outcome" else (2 * n + 5 if STREAM else n + n_val + 6)
    if mode == "process" and STREAM:
        return [2 * t + 1 for t in range(1, n + 1)] + [2 * n + 2, 2 * n + 3, 2 * n + 4]
    return list(range(n + 3, length))


def encode(rows, mode):
    """Teacher-forced (inputs, targets); unsupervised positions are ignored in the loss."""
    ids = torch.tensor(rows, device=device)
    inputs, targets = ids[:, :-1], torch.full_like(ids[:, 1:], -100)
    keep = torch.tensor(supervised(mode), device=device)
    targets[:, keep - 1] = ids[:, keep]
    return inputs, targets

# -----------------------------------------------------------------------------
# the architecture: L residual blocks, H softmax heads, one ReLU MLP, no LayerNorm.
# Residual slots: tok letter | tok value | pos | query | match | prev | mt | cnt | s0 | out


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

# -----------------------------------------------------------------------------
# the handcoded solutions, as values of the same tensors


def at(sl, i=0):
    return sl.start + i


def route(p, b, h, table, fallback):
    """Head h of block b: position i attends to table[i] (list = uniform), else to fallback."""
    p[f"{b}.k"][h, :, POS] = torch.eye(P)
    for i in range(P):
        for j in table.get(i, [fallback]):
            p[f"{b}.q"][h, j, at(POS, i)] = C


def base(letters):
    """Embeddings and block 0: every letter position gets MATCH = [w_t == q]."""
    p = zero_params()
    p["wte"][:A, TL] = torch.eye(A)
    p["wte"][A : A + V, TV] = torch.eye(V)
    p["wpe"][:, POS] = torch.eye(P)
    route(p, 0, 0, {i: [1] for i in letters}, fallback=0)           # copy q to letter positions
    p["0.v"][0][QRY, TL] = torch.eye(A)
    for a in range(A):
        p["0.mlp_in"][a, at(TL, a)] = p["0.mlp_in"][a, at(QRY, a)] = 1
        p["0.mlp_b"][a] = -1.5
        p["0.mlp_out"][at(MATCH), a] = 2
    return p


def readout(p, colon_at, eos_at):
    p["readout"][A : A + V, OUT] = 20 * torch.eye(V)
    p["readout"][COLON, at(POS, colon_at)] = 20
    p["readout"][EOS, at(POS, eos_at)] = 20
    return p


def process_solution():
    """The position predicting s_t reads s_{t-1} and MATCH at w_t, emits s_t."""
    if STREAM:      # w_t at 2t predicts s_t; s_{t-1} at 2t-1 (s0 at 0)
        letter = {t: 2 * t for t in range(1, n + 1)}
        pred = {2 * t: t for t in range(1, n + 1)}
        prev = {2 * t: 0 if t == 1 else 2 * t - 1 for t in range(1, n + 1)}
        last, colon_at = 2 * n + 1, 2 * n + 1
    else:           # s_{t-1} at n+1+t (SEP for t = 1) predicts s_t; w_t at 1+t
        letter = {t: 1 + t for t in range(1, n + 1)}
        pred = {n + 1 + t: t for t in range(1, n + 1)}
        prev = {i: 0 if t == 1 else i for i, t in pred.items()}
        last, colon_at = 2 * n + 2, 2 * n + 2
    p = base(letter.values())
    route(p, 1, 0, {i: [j] for i, j in prev.items()}, fallback=1)
    route(p, 1, 1, {i: [letter[t]] for i, t in pred.items()}, fallback=0)
    route(p, 1, 2, {last + 1: [last]}, fallback=1)                   # the answer copies s_n
    p["1.v"][0][PREV, TV] = torch.eye(V)
    p["1.v"][1][MT, MATCH] = torch.eye(1)
    p["1.v"][2][OUT, TV] = torch.eye(V)

    units = []                      # (inputs, bias, output value)
    if task == "count":
        for j in range(V):
            units.append(({at(PREV, j): 1, at(MT): -1}, -0.5, j))
            units.append(({at(PREV, j): 1, at(MT): 1}, -1.5, (j + 1) % mod))
    else:
        units += [({at(PREV, j): 1}, -0.5, j) for j in range(1, V)]    # found: keep it
        units.append(({at(PREV, 0): 1, at(MT): -1}, -0.5, 0))          # not yet
        for i, t in pred.items():                                       # found at step t
            units.append(({at(PREV, 0): 1, at(MT): 1, at(POS, i): 1}, -2.5, t))
    for u, (inputs, bias, value) in enumerate(units):
        for w, c in inputs.items():
            p["1.mlp_in"][u, w] = c
        p["1.mlp_b"][u] = bias
        p["1.mlp_out"][at(OUT, value), u] = 2
    return readout(p, colon_at=colon_at, eos_at=last + 2)


def outcome_solution():
    """The ':' position n+3 reads all n marks at once and writes s_n."""
    p = base(range(2, n + 2))
    ans = n + 3
    if task == "count":
        route(p, 1, 0, {ans: list(range(2, n + 2))}, fallback=0)       # uniform over the word
        route(p, 1, 1, {ans: [0]}, fallback=1)
        p["1.v"][0][CNT, MATCH] = n                                     # mean mark * n = count
        p["1.v"][1][S0, TV] = torch.eye(V)
        gate, u = n + 2, 0          # triangle bump r(x-c+1) - 2r(x-c) + r(x-c-1), gated on s0 = j
        for j in range(V):
            for c in range(n + 1):
                for offset, coef in ((1, 1), (0, -2), (-1, 1)):
                    p["1.mlp_in"][u, at(CNT)] = 1
                    p["1.mlp_in"][u, at(S0, j)] = gate
                    p["1.mlp_b"][u] = offset - c - gate
                    p["1.mlp_out"][at(OUT, (j + c) % mod), u] = coef
                    u += 1
    else:
        # score = C_hit * MATCH - beta * position: the earliest marked position wins
        beta = C
        k, q = p["1.k"][0], p["1.q"][0]
        k[0, at(MATCH)] = 1
        k[1, POS] = -torch.arange(P, dtype=torch.float)
        k[2, at(POS, 1)] = 1                                            # fallback: the query, copies nothing
        q[0, at(POS, ans)] = beta * (n + 2) + C
        q[1, at(POS, ans)] = beta
        for i in range(P):
            if i != ans:
                q[2, at(POS, i)] = C                                    # elsewhere: attend to 1
        for i in range(2, n + 2):
            p["1.v"][0][at(OUT, i - 1), at(POS, i)] = 1                 # word position i is t = i-1
        p["1.v"][0][at(OUT, 0), at(POS, 0)] = 1                         # no match: position 0 wins -> "not found"
    return readout(p, colon_at=n + 2, eos_at=n + 4)

# -----------------------------------------------------------------------------
# free-running evaluation on clean gold continuations


@torch.no_grad()
def generate(p, examples, mode):
    """Greedy continuation and the gold continuation. Stream layout feeds the letters in."""
    def process_values(e):
        vals = [A + v for v in e[3]]
        return vals[every_k - 1::every_k] if every_k > 1 else vals
    gold = [process_values(e) * (mode == "process") + [COLON, A + e[3][-1], EOS] for e in examples]
    if mode == "process" and STREAM:
        ids, out = torch.tensor([[A + e[0], e[1]] for e in examples], device=device), []
        for t in range(n):
            letters = torch.tensor([[e[2][t]] for e in examples], device=device)
            ids = torch.cat([ids, letters], dim=1)
            ids = torch.cat([ids, forward(p, ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            out.append(ids[:, -1:])
        for _ in range(3):
            ids = torch.cat([ids, forward(p, ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            out.append(ids[:, -1:])
        rows = torch.cat(out, dim=1).tolist()
    else:
        ids = torch.tensor([prompt(e) for e in examples], device=device)
        for _ in range(len(gold[0])):
            ids = torch.cat([ids, forward(p, ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
        rows = ids[:, n + 3 :].tolist()
    return rows, gold


@torch.no_grad()
def free_run(p, examples, mode):
    """Greedy generation; in the stream layout the letters are fed in, the model writes values."""
    rows, gold = generate(p, examples, mode)
    exact = sum(r == g for r, g in zip(rows, gold)) / len(gold)
    answer = sum(COLON in r and r.index(COLON) + 1 < len(r) and r[r.index(COLON) + 1] == g[-2]
                 for r, g in zip(rows, gold)) / len(gold)
    n_val = n if every_k == 1 else N_TRACE
    steps_ok = (sum(a == b for r, g in zip(rows, gold) for a, b in zip(r[:n_val], g[:n_val])) / (n_val * len(gold))
                if mode == "process" else float("nan"))
    return exact, answer, steps_ok


def make_corpus():
    """Train words, the traces supervision sees, and held-out test words."""
    rng_train = random.Random(seed)
    rng_corrupt = random.Random(seed + 10_000)
    rng_test = random.Random(seed + 20_000)
    train = [make_example(rng_train) for _ in range(train_size)]
    shown = shown_traces(train, rng_corrupt)
    seen, test, tries = {tuple(prompt(e)) for e in train}, [], 0
    while len(test) < test_size:
        e, tries = make_example(rng_test), tries + 1
        assert tries < 100 * test_size, "prompt space too small for held-out test words"
        if tuple(prompt(e)) not in seen:
            test.append(e)
    clean = sum(s == e[3] for e, s in zip(train, shown))
    print(f"corpus  train={len(train)}  clean={clean}/{len(train)} ({clean / len(train):.3f})  "
          f"test={len(test)}  (streams: train seed, corrupt seed+10000, test seed+20000)")
    if clean_only:   # the clean subset of this exact corpus (corrupted traces are wrong at every step)
        keep = [i for i, (e, sh) in enumerate(zip(train, shown)) if sh == e[3]]
        train, shown = [train[i] for i in keep], [shown[i] for i in keep]
        print(f"clean_only: training on the {len(train)} clean traces of the rho={rho} corpus")
    return train, shown, test


def init_theta():
    """One random initialization in the shared architecture. Both losses clone it."""
    return {k: torch.randn_like(t) * init_std for k, t in zero_params().items()}


def handcoded_params():
    """The two exact programs, on `device`. Count and first-occurrence only, every_k = 1."""
    assert task in ("count", "first") and every_k == 1
    sols = {"process": process_solution(), "outcome": outcome_solution()}
    return {m: {k: t.to(device) for k, t in p.items()} for m, p in sols.items()}


@torch.no_grad()
def mean_loss(params, mode, examples, shown=None):
    """Mean token cross-entropy of a fixed parameter dict. Handcoded programs are flat lines."""
    if shown is None:
        shown = [e[3] for e in examples]
    inputs, targets = encode([row(e, mode, s) for e, s in zip(examples, shown)], mode)
    total, count = 0.0, 0
    for i in range(0, inputs.shape[0], batch_size):
        logit = forward(params, inputs[i:i + batch_size])
        tgt = targets[i:i + batch_size]
        total += F.cross_entropy(logit.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum").item()
        count += int((tgt != -100).sum())
    return total / max(count, 1)


def train_mode(mode, theta_0, train, shown, test, keep_state=False):
    """Train one clone of theta_0. Returns (params, history). History has a loss every step."""
    p = {k: t.clone().to(device).requires_grad_() for k, t in theta_0.items()}
    if opt == "sgd":
        optimizer = torch.optim.SGD(p.values(), lr=lr, momentum=0.0)
    else:
        optimizer = torch.optim.AdamW(p.values(), lr=lr, weight_decay=0.0)
    inputs, targets = encode([row(e, mode, s) for e, s in zip(train, shown)], mode)
    batches = torch.Generator().manual_seed(seed)
    history = []
    tick = eval_every if eval_every else max(1, steps // 8)
    bar = tqdm(range(1, steps + 1), desc=mode, dynamic_ncols=True)
    for it in bar:
        index = torch.randint(len(train), (batch_size,), generator=batches).to(device)
        logits = forward(p, inputs[index])
        loss = F.cross_entropy(logits.flatten(0, 1), targets[index].flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.values(), 1.0)
        optimizer.step()
        rec = dict(step=it, loss=loss.item())
        postfix = {"loss": f"{loss.item():.3f}"}
        if it % tick == 0 or it == steps:
            exact, answer, steps_ok = free_run(p, test, mode)
            rec.update(exact=exact, answer=answer, steps_ok=steps_ok)
            postfix["exact"] = f"{exact:.3f}"
            line = (f"{mode:8s} step {it:5d}  loss {loss.item():.4f}  test exact {exact:.3f}  "
                    f"answer {answer:.3f}" + (f"  step acc {steps_ok:.3f}" if mode == "process" else ""))
            tqdm.write(line)
            if keep_state:
                rec["params"] = {k: t.detach().cpu().clone() for k, t in p.items()}
        bar.set_postfix(postfix, refresh=False)
        history.append(rec)
        if save_every and (it % save_every == 0 or it == steps):
            name = f"{tag}_{mode}_step{it}.pt" if tag else f"{mode}_step{it}.pt"
            path = os.path.join(save_dir, name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "config": dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks,
                               layout=layout, seed=seed, init_std=init_std,
                               every_k=every_k, opt=opt, batch_size=batch_size),
                "params": {k: t.detach().cpu() for k, t in p.items()},
                "mode": mode, "step": it,
            }, path)
            tqdm.write(f"wrote {path}")
    return p, history


def main():
    over = {}
    for arg in sys.argv[1:]:
        key, value = arg.split("=", 1)
        assert key in DEFAULTS, f"unknown setting {key}"
        over[key] = value if key in _STRING_KEYS else ast.literal_eval(value)
    configure(**over)
    train, shown, test = make_corpus()
    theta_0 = init_theta()
    print(f"task={task} layout={layout} A={A} n={n}" + (f" m={mod} rho={rho} corrupt={corrupt}" if task == "count" else "")
          + (f" every_k={every_k}" if every_k > 1 else "")
          + f"  L={n_blocks} opt={opt} batch={batch_size} heads={H} d_ff={F_WIDTH} width={W}"
          + f"  params={sum(t.numel() for t in theta_0.values()):,}")
    if task == "count" and mod == 2:
        print(f"answer bias given everything but the count: (1-2/A)^n = {(1 - 2 / A) ** n:.2e}")
    if task in ("count", "first") and every_k == 1:
        for mode, p in handcoded_params().items():
            exact, answer, _ = free_run(p, test, mode)
            print(f"handcoded {mode:8s}  gap {GAP:.2f}  exact {exact:.3f}  answer {answer:.3f}")
            for sigma in perturb:
                noise_gen = torch.Generator(device="cpu").manual_seed(seed + 30_000)
                worst = 1.0
                for _ in range(perturb_draws):
                    noisy = {k: t + sigma * torch.randn(t.shape, generator=noise_gen).to(t.device)
                             for k, t in p.items()}
                    worst = min(worst, free_run(noisy, test, mode)[0])
                print(f"handcoded {mode:8s}  noise std {sigma:g}  worst exact over {perturb_draws} draws {worst:.3f}")
    if handcoded_only:
        return
    for mode in train_modes:
        train_mode(mode, theta_0, train, shown, test)


configure()

if __name__ == "__main__":
    main()