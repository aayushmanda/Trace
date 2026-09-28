"""Init residual with full-context J vs J restricted to the local triple.

The displayed triple (prev state, query, letter) is blind. Replacing every other
displayed state with a dummy token makes J_t a function of that triple plus the
Z-independent prompt. If the 0.21-0.25 gap is blindness, the local residual
collapses and the full-context one does not.
"""
import ast, json, random, sys
from pathlib import Path

import torch
from torch.nn import functional as F

word_len = 8
alphabet = 4
mod = 3
rho = 0.5
n_blocks = 2
init_std = 0.02
seed = 42
train_size = 20_000
grad_batch = 1024
out = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    assert key in globals(), key
    globals()[key] = value if key in ("device", "out") else ast.literal_eval(value)

n, A, V = word_len, alphabet, mod
SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
P, H = 2 * n + 6, 3
F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
W = 2 * A + 4 * V + P + 3  # TL+QRY + TV+PREV+S0+OUT + POS + MATCH+MT+CNT
mu = rho * (1 + 1 / (mod - 1)) - 1 / (mod - 1)
random.seed(seed)
torch.manual_seed(seed)


def step(s, letter, q):
    return (s + (letter == q)) % mod


def make_example(rng):
    q, word = rng.randrange(A), tuple(rng.randrange(A) for _ in range(n))
    s0, s, states = rng.randrange(mod), None, []
    s = s0
    for letter in word:
        s = step(s, letter, q)
        states.append(s)
    return s0, q, word, states


def corrupt_trace(ex, rng):
    s0, q, word, _ = ex
    shown, s = [], s0
    for letter in word:
        right = step(s, letter, q)
        s = rng.choice([v for v in range(mod) if v != right])
        shown.append(s)
    return shown


def row(ex, shown):
    s0, q, word, states = ex
    return [A + s0, q, *word, SEP, *[A + v for v in shown], COLON, A + states[-1], EOS]


STATE_POS = list(range(n + 3, 2 * n + 3))
ANS_POS = [2 * n + 4]
ALL_POS = STATE_POS + ANS_POS
DUMMY = A + 0


def encode(rows, keep):
    ids = torch.tensor(rows, device=device)
    inputs, targets = ids[:, :-1], torch.full_like(ids[:, 1:], -100)
    idx = torch.tensor(keep, device=device)
    targets[:, idx - 1] = ids[:, idx]
    return inputs, targets


def local_rows(rows, t):
    """Keep only the previous displayed state; dummy the rest."""
    out = [list(r) for r in rows]
    for r in out:
        for i, pos in enumerate(STATE_POS):
            if i != t - 1:
                r[pos] = DUMMY
    return out


def zero_params():
    p = {"wte": torch.zeros(A + V + 3, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(A + V + 3, W)}
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
        att = (q @ k.transpose(-1, -2) / (P ** 0.5)).masked_fill(future, -float("inf"))
        x = x + (att.softmax(-1) @ v).sum(1)
        x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
    return x @ p["readout"].T


def kway_targets(targets):
    out = targets.clone()
    mask = targets != -100
    out = out.masked_fill(~mask, 0) - A
    return out.clamp(min=0, max=V - 1).masked_fill(~mask, -100)


def theory_grad(p, inputs, gold_ids):
    for t in p.values():
        if t.grad is not None:
            t.grad = None
        t.requires_grad_(True)
    logits = forward(p, inputs)[:, :, A:A + V]
    mask = gold_ids != -100
    gold_v = kway_targets(gold_ids).clamp(min=0)
    e = torch.eye(V, device=device)[gold_v]
    u = torch.full_like(e, 1 / V)
    loss = (((e - u) * logits).sum(-1) * mask).sum() / inputs.shape[0]
    loss.backward()
    return torch.cat([t.grad.detach().flatten() for t in p.values()])


def residual(g_rho, g_clean, g_ans):
    combo = mu * g_clean + g_ans
    return (g_rho - combo).norm().item() / (g_rho.norm().item() + 1e-12)


rng = random.Random(seed)
train = [make_example(rng) for _ in range(train_size)]
uniforms = [rng.random() for _ in train]
wrong = [corrupt_trace(ex, rng) for ex in train]
shown = [st if u < rho else bad for (_, _, _, st), u, bad in zip(train, uniforms, wrong)]
gold_rows = [row(e, e[3]) for e in train]
mix_rows = [row(e, s) for e, s in zip(train, shown)]

theta = {k: (torch.randn_like(t) * init_std).to(device).requires_grad_() for k, t in zero_params().items()}
take = torch.randperm(len(train), device=device)[:grad_batch]
take_list = take.tolist()
mix_b = [mix_rows[i] for i in take_list]
gold_b = [gold_rows[i] for i in take_list]

inp_mix, tgt_rho = encode(mix_b, ALL_POS)
inp_gold, tgt_clean = encode(gold_b, STATE_POS)
_, tgt_ans = encode(mix_b, ANS_POS)
g_rho = theory_grad(theta, inp_mix, tgt_rho)
g_clean = theory_grad(theta, inp_gold, tgt_clean)
g_ans = theory_grad(theta, inp_mix, tgt_ans)
resid_full = residual(g_rho, g_clean, g_ans)

g_rho_l = g_clean_l = g_ans_l = None
for t in range(n):
    inp_m, tgt_m = encode(local_rows(mix_b, t), [STATE_POS[t]])
    inp_g, tgt_g = encode(local_rows(gold_b, t), [STATE_POS[t]])
    gr = theory_grad(theta, inp_m, tgt_m)
    gc = theory_grad(theta, inp_g, tgt_g)
    ga = theory_grad(theta, inp_m, tgt_ans) if t == 0 else None
    g_rho_l = gr if g_rho_l is None else g_rho_l + gr
    g_clean_l = gc if g_clean_l is None else g_clean_l + gc
    if ga is not None:
        g_ans_l = ga
resid_local = residual(g_rho_l, g_clean_l, g_ans_l)

# First-step only: context is (s0, q, w1), always independent of Z.
inp_m1, tgt_m1 = encode(mix_b, [STATE_POS[0]])
inp_g1, tgt_g1 = encode(gold_b, [STATE_POS[0]])
resid_s1 = residual(theory_grad(theta, inp_m1, tgt_m1), theory_grad(theta, inp_g1, tgt_g1), g_ans)

payload = dict(
    seed=seed, rho=rho, mu=mu, n=n, K=mod,
    residual_full=resid_full, residual_local=resid_local, residual_s1=resid_s1,
)
print(json.dumps(payload, indent=2), flush=True)
if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
