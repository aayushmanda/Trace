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

Rows (outcome is the same in both layouts):
    outcome:          s0 q w_1 .. w_n SEP : s_n EOS
    process, stream:  s0 q w_1 s_1 w_2 s_2 .. w_n s_n : s_n EOS   letters are given, not trained
    process, block:   s0 q w_1 .. w_n SEP s_1 .. s_n : s_n EOS
Both formats see the same letters; process adds the intermediate targets. In the stream
layout every process step is local: the letter is the current token, s_{t-1} the previous.
Block 0 of every solution marks each letter with [w_t == q]. Block 1 either steps once
per emitted value (process) or reads all n marks at the answer (outcome).
"""
import ast
import math
import random
import sys

import torch
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# config: every name below can be overridden as key=value on the command line

task = "count"          # "count" or "first"
alphabet = 4            # A letters; P(w_t == q) = 1/A
word_len = 12           # n, the horizon
mod = 2                 # m, count task only
rho = 1.0               # fraction of clean process traces, count task only
corrupt = "scatter"     # corrupted steps: "scatter" = uniform wrong value, "shift" = off by one
layout = "stream"       # process rows: "stream" (letter, value, letter, ...) or "block"
n_blocks = 2
train_size = 20_000
test_size = 1_000
steps = 8_000
batch_size = 256
lr = 1e-3
init_std = 0.02
seed = 42
modes = "process,outcome"   # which losses to train, e.g. "process" for a reliability sweep
compiled = False            # torch.compile the training forward pass
eval_every = 0              # 0: eight evaluations per run
out = ""                    # if set, append one CSV row per trained loss to this file
save_every = 0              # if > 0, save parameters every this many steps (and at step 0)
save_dir = "ckpt"           # where checkpoints go
zero_readout = False        # start from exactly uniform predictions (readout = 0)
device = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    assert key in globals(), f"unknown setting {key}"
    globals()[key] = value if key in ("device", "task", "corrupt", "layout", "modes", "out", "save_dir") else ast.literal_eval(value)
assert task in ("count", "first") and corrupt in ("scatter", "shift") and n_blocks >= 2
assert layout in ("stream", "block")
assert task == "count" or rho == 1.0, "corruption is defined for the count task"
random.seed(seed)
torch.manual_seed(seed)

# -----------------------------------------------------------------------------
# the task

n, A = word_len, alphabet
V = mod if task == "count" else n + 1       # value tokens; for "first", 0 = not found yet
SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
VOCAB = A + V + 3
P = 2 * n + 6                               # longest row: prompt n+3, process continuation n+3


def step(s, t, letter, q):
    """The gold transition after letter t (1-indexed)."""
    if task == "count":
        return (s + (letter == q)) % mod
    return s if s else (t if letter == q else 0)


def make_example(rng):
    while True:
        q, word = rng.randrange(A), tuple(rng.randrange(A) for _ in range(n))
        if task == "count" or q in word:
            break
    s0 = rng.randrange(mod) if task == "count" else 0
    states, s = [], s0
    for t, c in enumerate(word, 1):
        s = step(s, t, c, q)
        states.append(s)
    return s0, q, word, states


def shown_trace(ex, rng):
    """The trace a process example displays: clean with prob rho, else wrong at every step."""
    s0, q, word, states = ex
    if rng.random() < rho:
        return states
    shown, s = [], s0
    for t, c in enumerate(word, 1):
        right = step(s, t, c, q)
        s = (right + 1) % mod if corrupt == "shift" else rng.choice([v for v in range(mod) if v != right])
        shown.append(s)
    return shown


STREAM = layout == "stream"


def prompt(ex):
    s0, q, word, _ = ex
    return [A + s0, q, *word, SEP]


def row(ex, mode, shown=None):
    """Full training row: prompt, trace (process only), then ': s_n EOS'."""
    s0, q, word, states = ex
    tail = [COLON, A + states[-1], EOS]
    values = [A + v for v in (shown or states)]
    if mode == "outcome":
        return prompt(ex) + tail
    if STREAM:
        return [A + s0, q] + [t for pair in zip(word, values) for t in pair] + tail
    return prompt(ex) + values + tail


def supervised(mode):
    """Row positions whose token is a training target."""
    length = n + 6 if mode == "outcome" else (2 * n + 5 if STREAM else 2 * n + 6)
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

H = 3
F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
_sizes = dict(TL=A, TV=V, POS=P, QRY=A, MATCH=1, PREV=V, MT=1, CNT=1, S0=V, OUT=V)
_start = 0
for _name, _size in _sizes.items():
    globals()[_name] = slice(_start, _start + _size)
    _start += _size
W = _start


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

C = 16 * math.sqrt(P)       # routing score: softmax leak per position ~ e^-16


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
        k[2, at(POS, 0)] = 1
        q[0, at(POS, ans)] = beta * (n + 2) + C
        q[1, at(POS, ans)] = beta
        for i in range(P):
            if i != ans:
                q[2, at(POS, i)] = C                                    # elsewhere: attend to 0
        for i in range(2, n + 2):
            p["1.v"][0][at(OUT, i - 1), at(POS, i)] = 1                 # word position i is t = i-1
    return readout(p, colon_at=n + 2, eos_at=n + 4)

# -----------------------------------------------------------------------------
# free-running evaluation on clean gold continuations


@torch.no_grad()
def free_run(p, examples, mode):
    """Greedy generation; in the stream layout the letters are fed in, the model writes values."""
    gold = [[A + v for v in e[3]] * (mode == "process") + [COLON, A + e[3][-1], EOS] for e in examples]
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
    exact = sum(r == g for r, g in zip(rows, gold)) / len(gold)
    answer = sum(COLON in r and r.index(COLON) + 1 < len(r) and r[r.index(COLON) + 1] == g[-2]
                 for r, g in zip(rows, gold)) / len(gold)
    steps_ok = (sum(a == b for r, g in zip(rows, gold) for a, b in zip(r[:n], g[:n])) / (n * len(gold))
                if mode == "process" else float("nan"))
    return exact, answer, steps_ok

# -----------------------------------------------------------------------------
# data with held-out words, then one random theta_0 trained under each loss

rng = random.Random(seed)
train = [make_example(rng) for _ in range(train_size)]
shown = [shown_trace(e, rng) for e in train]
seen, test, tries = {tuple(prompt(e)) for e in train}, [], 0
while len(test) < test_size:
    e, tries = make_example(rng), tries + 1
    assert tries < 100 * test_size, "prompt space too small for held-out test words"
    if tuple(prompt(e)) not in seen:
        test.append(e)

solutions = {"process": process_solution(), "outcome": outcome_solution()}
solutions = {m: {k: t.to(device) for k, t in p.items()} for m, p in solutions.items()}
theta_0 = {k: torch.randn_like(t) * init_std for k, t in zero_params().items()}
if zero_readout:
    theta_0["readout"].zero_()
CONFIG = dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks, layout=layout,
              rho=rho, corrupt=corrupt, seed=seed, init_std=init_std, zero_readout=zero_readout)


def save(p, mode, it):
    import os
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{task}_n{n}_K{mod}_A{A}_L{n_blocks}_rho{rho}_{corrupt}_seed{seed}"
    torch.save(dict(params={k: t.detach().cpu() for k, t in p.items()}, config=CONFIG, mode=mode, step=it),
               f"{save_dir}/{tag}_{mode}_step{it}.pt")

print(f"task={task} layout={layout} A={A} n={n}" + (f" m={mod} rho={rho} corrupt={corrupt}" if task == "count" else "")
      + f"  L={n_blocks} heads={H} d_ff={F_WIDTH} width={W}"
      + f"  params={sum(t.numel() for t in theta_0.values()):,}")
if task == "count" and mod == 2:
    print(f"answer bias given everything but the count: (1-2/A)^n = {(1 - 2 / A) ** n:.2e}")
for mode, p in solutions.items():
    exact, answer, _ = free_run(p, test, mode)
    print(f"handcoded {mode:8s}  exact {exact:.3f}  answer {answer:.3f}")

torch.set_float32_matmul_precision("high")
fwd = torch.compile(forward) if compiled else forward       # fixed shapes: compiles once
every = eval_every or max(1, steps // 8)
results = []
for mode in modes.split(","):
    p = {k: t.clone().to(device).requires_grad_() for k, t in theta_0.items()}
    optimizer = torch.optim.AdamW(p.values(), lr=lr, weight_decay=0.0)
    inputs, targets = encode([row(e, mode, s) for e, s in zip(train, shown)], mode)
    batches = torch.Generator().manual_seed(seed)
    escape = None
    if save_every:
        save(p, mode, 0)
    for it in range(1, steps + 1):
        index = torch.randint(len(train), (batch_size,), generator=batches).to(device)
        logits = fwd(p, inputs[index])
        loss = F.cross_entropy(logits.flatten(0, 1), targets[index].flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.values(), 1.0)
        optimizer.step()
        if save_every and (it % save_every == 0 or it == steps):
            save(p, mode, it)
        if it % every == 0 or it == steps:
            exact, answer, steps_ok = free_run(p, test, mode)
            if escape is None and exact >= 0.5:
                escape = it
            print(f"{mode:8s} step {it:5d}  loss {loss.item():.4f}  test exact {exact:.3f}  "
                  f"answer {answer:.3f}" + (f"  step acc {steps_ok:.3f}" if mode == "process" else ""))
    results.append(dict(task=task, layout=layout, A=A, n=n, m=mod, rho=rho, corrupt=corrupt,
                        L=n_blocks, steps=steps, lr=lr, seed=seed, mode=mode, exact=exact,
                        answer=answer, step_acc=steps_ok, escape=escape if escape else ""))

if out:
    import csv
    import os
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        if new:
            writer.writeheader()
        writer.writerows(results)