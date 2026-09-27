"""Does s_n depend on displayed s_{n-1} or on earlier letters?

Block process prefix: s0 q w1..wn SEP s1..s_{n-1}. Predict s_n.
Conditions keep the local pair (s_{n-1}, w_n) or break it.
"""
import ast, json, math, random, sys
from pathlib import Path

import torch
from torch.nn import functional as F

ckpt = ""
examples = 2048
batch = 256
out = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    assert key in globals(), key
    globals()[key] = value if key in ("ckpt", "device", "out") else ast.literal_eval(value)
assert ckpt, "ckpt= is required"

saved = torch.load(ckpt, map_location=device)
cfg = saved["config"]
task, n, A, mod = cfg["task"], cfg["word_len"], cfg.get("alphabet", 4), cfg["mod"]
n_blocks = cfg["n_blocks"]
assert task == "count"
V, SEP, H, P = mod, A + mod, 3, 2 * n + 6


def forward(p, ids):
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    for b in range(n_blocks):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -1e9)
        x = x + (att.softmax(-1) @ v).sum(1)
        x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
    return x @ p["readout"].T


theta = {k: t.to(device).float() for k, t in saved["params"].items()}
rng = random.Random(0)


def gold_states(s0, q, word):
    s, out = s0, []
    for c in word:
        s = (s + (c == q)) % mod
        out.append(s)
    return out


def prefix(s0, q, word, states, *, prev=None, letters=None, early=None):
    w = list(letters if letters is not None else word)
    st = list(states)
    if early is not None:
        st = list(early) + st[len(early):]
    if prev is not None:
        st = st[:-2] + [prev] + st[-1:]
    return [A + s0, q, *w, SEP, *[A + v for v in st[:-1]]]


def score(prefs, targets):
    acc = []
    for i in range(0, len(prefs), batch):
        ids = torch.tensor(prefs[i:i + batch], device=device)
        gold = torch.tensor(targets[i:i + batch], device=device)
        pred = forward(theta, ids)[:, -1, A:A + V].argmax(-1)
        acc.append((pred == gold).float())
    return torch.cat(acc).mean().item()


pairs = []
for _ in range(examples):
    q = rng.randrange(A)
    word = [rng.randrange(A) for _ in range(n)]
    s0 = rng.randrange(mod)
    states = gold_states(s0, q, word)
    local = states[-1]
    pairs.append((s0, q, word, states, local))

conds = {}
# clean
conds["clean"] = score([prefix(*p[:4]) for p in pairs], [p[4] for p in pairs])
# random displayed s_{n-1}
conds["rand_prev"] = score(
    [prefix(*p[:4], prev=rng.randrange(mod)) for p in pairs], [p[4] for p in pairs])
# random earlier displayed states s_1..s_{n-2}, keep s_{n-1}
conds["rand_early"] = score(
    [prefix(*p[:4], early=[rng.randrange(mod) for _ in range(n - 2)]) for p in pairs],
    [p[4] for p in pairs])
# random letters except w_n
conds["rand_early_letters"] = score(
    [prefix(*p[:4], letters=[rng.randrange(A) for _ in range(n - 1)] + [p[2][-1]]) for p in pairs],
    [p[4] for p in pairs])
# random w_n, keep displayed s_{n-1}
conds["rand_wn"] = score(
    [prefix(*p[:4], letters=p[2][:-1] + [rng.randrange(A)]) for p in pairs],
    [p[4] for p in pairs])
# vs local rule after rand_prev (should follow displayed prev if local)
local_after_flip = []
prefs = []
for p in pairs:
    prev = rng.randrange(mod)
    local_after_flip.append((prev + (p[2][-1] == p[1])) % mod)
    prefs.append(prefix(*p[:4], prev=prev))
conds["rand_prev_vs_local"] = score(prefs, local_after_flip)

rec = dict(ckpt=ckpt, step=saved.get("step"), mode=saved.get("mode"), n=n, K=mod, **conds)
print(json.dumps(rec, indent=2))
if out:
    Path(out).write_text(json.dumps(rec, indent=2))
