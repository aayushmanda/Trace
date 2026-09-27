"""Flip-the-first-step intervention (Proposition 3).

Block layout: the letters are in the prompt, then SEP, then displayed states.
We keep (q, word) fixed and only change the first displayed state s1 to be
consistent or inconsistent with the true rule, then score the next-state
prediction against the local rule on the displayed s1.
"""
import ast
import json
import math
import random
import sys
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
    assert key in globals(), f"unknown setting {key}"
    globals()[key] = value if key in ("ckpt", "device", "out") else ast.literal_eval(value)
assert ckpt, "ckpt=... is required"

saved = torch.load(ckpt, map_location=device)
cfg = saved["config"]
task, n, A, mod = cfg["task"], cfg["word_len"], cfg.get("alphabet", 4), cfg["mod"]
n_blocks = cfg["n_blocks"]
layout = cfg.get("layout", "block")
assert task == "count"

V = mod
H = 3
P = 2 * n + 6
SEP = A + V


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


def make_pair():
    q = rng.randrange(A)
    word = [rng.randrange(A) for _ in range(n)]
    s0 = rng.randrange(mod)
    s1_clean = (s0 + (word[0] == q)) % mod
    s1_flip = (s1_clean + 1) % mod
    local_clean = (s1_clean + (word[1] == q)) % mod
    local_flip = (s1_flip + (word[1] == q)) % mod
    if layout == "block":
        pref_clean = [A + s0, q, *word, SEP, A + s1_clean]
        pref_flip = [A + s0, q, *word, SEP, A + s1_flip]
    else:
        pref_clean = [A + s0, q, word[0], A + s1_clean, word[1]]
        pref_flip = [A + s0, q, word[0], A + s1_flip, word[1]]
    return pref_clean, pref_flip, local_clean, local_flip


pairs = [make_pair() for _ in range(examples)]


@torch.no_grad()
def score(prefs, targets):
    ids = torch.tensor(prefs, device=device)
    tgt = torch.tensor(targets, device=device)
    acc, margin = [], []
    for i in range(0, len(prefs), batch):
        logits = forward(theta, ids[i:i + batch])[:, -1, A:A + V]
        pred = logits.argmax(-1)
        gold = tgt[i:i + batch]
        acc.append((pred == gold).float())
        top = logits.gather(1, gold[:, None]).squeeze(1)
        other = logits.clone()
        other[torch.arange(len(gold), device=device), gold] = -1e9
        margin.append(top - other.max(-1).values)
    return torch.cat(acc).mean().item(), torch.cat(margin).mean().item()


clean_acc, clean_m = score([p[0] for p in pairs], [p[2] for p in pairs])
flip_local_acc, flip_local_m = score([p[1] for p in pairs], [p[3] for p in pairs])
flip_gold_acc, flip_gold_m = score([p[1] for p in pairs], [p[2] for p in pairs])
rec = dict(
    ckpt=ckpt, step=saved.get("step"), mode=saved.get("mode"),
    n=n, K=mod, layout=layout,
    after_correct_first_local_acc=clean_acc, after_correct_first_margin=clean_m,
    after_flipped_first_local_acc=flip_local_acc, after_flipped_first_local_margin=flip_local_m,
    after_flipped_first_gold_acc=flip_gold_acc, after_flipped_first_gold_margin=flip_gold_m,
)
print(json.dumps(rec, indent=2))
if out:
    Path(out).write_text(json.dumps(rec, indent=2))
