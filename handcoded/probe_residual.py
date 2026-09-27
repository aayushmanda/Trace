"""Linear probe for Z, and the three-gradient residual of Thm. 3, along training.

Trains count K=3, rho=0.5 scatter, block process, then at each checkpoint:
  residual = ||g_rho - mu g_clean - (1-mu) g_unif - g_ans|| / ||g_rho||
  probe    = linear readout of last-layer x at s1 vs s2 predicting Z (clean vs corrupt)
"""
import ast, json, math, os, random, sys
from pathlib import Path

import torch
from torch.nn import functional as F

task = "count"
alphabet = 4
word_len = 8
mod = 3
rho = 0.5
corrupt = "scatter"
n_blocks = 2
steps = 16000
eval_every = 2000
batch_size = 256
lr = 1e-3
init_std = 0.02
seed = 42
train_size = 20000
test_size = 2000
grad_batch = 1024
save_dir = ""
out = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    assert key in globals(), key
    globals()[key] = value if key in ("device", "corrupt", "save_dir", "out") else ast.literal_eval(value)

n, A, V = word_len, alphabet, mod
SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
P, H = 2 * n + 6, 3
F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
_sizes = dict(TL=A, TV=V, POS=P, QRY=A, MATCH=1, PREV=V, MT=1, CNT=1, S0=V, OUT=V)
_start = 0
for _name, _size in _sizes.items():
    globals()[_name] = slice(_start, _start + _size)
    _start += _size
W = _start
c = 1 / (mod - 1)
mu = rho * (1 + c) - c
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


def encode(rows, keep):
    ids = torch.tensor(rows, device=device)
    inputs, targets = ids[:, :-1], torch.full_like(ids[:, 1:], -100)
    idx = torch.tensor(keep, device=device)
    targets[:, idx - 1] = ids[:, idx]
    return inputs, targets


STATE_POS = list(range(n + 3, 2 * n + 3))          # s1..sn
ANS_POS = [2 * n + 4]                               # answer value after colon
ALL_POS = STATE_POS + ANS_POS                       # theorem: states + answer, no EOS


def zero_params():
    p = {"wte": torch.zeros(A + V + 3, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(A + V + 3, W)}
    for b in range(n_blocks):
        p[f"{b}.q"], p[f"{b}.k"] = torch.zeros(H, P, W), torch.zeros(H, P, W)
        p[f"{b}.v"] = torch.zeros(H, W, W)
        p[f"{b}.mlp_in"], p[f"{b}.mlp_b"] = torch.zeros(F_WIDTH, W), torch.zeros(F_WIDTH)
        p[f"{b}.mlp_out"] = torch.zeros(W, F_WIDTH)
    return p


def hidden(p, ids):
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
    return x


def forward(p, ids):
    return hidden(p, ids) @ p["readout"].T


def theory_grad(p, inputs, gold_ids):
    """Label-dependent gradient E[J^T(e-u)] in the K-way simplex (Thm. 1/2)."""
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


def kway_targets(targets):
    """Map token ids A+v to class v; keep -100 ignore."""
    out = targets.clone()
    mask = targets != -100
    out = out.masked_fill(~mask, 0) - A
    return out.clamp(min=0, max=V - 1).masked_fill(~mask, -100)


def grad_vec(p, inputs, targets, uniform=False):
    """K-way state CE, mean over batch, sum over positions, so the three terms add."""
    for t in p.values():
        if t.grad is not None:
            t.grad = None
        t.requires_grad_(True)
    logits = forward(p, inputs)[:, :, A:A + V]
    B = inputs.shape[0]
    if uniform:
        logp = F.log_softmax(logits, dim=-1)
        mask = targets != -100
        loss = -(logp.mean(-1) * mask).sum() / B
    else:
        loss = F.cross_entropy(logits.flatten(0, 1), kway_targets(targets).flatten(),
                               ignore_index=-100, reduction="sum") / B
    loss.backward()
    return torch.cat([t.grad.detach().flatten() for t in p.values()])


def probe_acc(p, rows, z, pos):
    """Linear probe of last-layer x[:, pos] for Z. Fit on first half, test on second."""
    with torch.no_grad():
        ids = torch.tensor(rows, device=device)
        feats, labs = [], []
        for i in range(0, len(rows), 256):
            feats.append(hidden(p, ids[i:i + 256][:, :pos + 1])[:, -1].float())
            labs.append(z[i:i + 256])
        X = torch.cat(feats)
        y = torch.cat(labs).float()
    n_tr = len(X) // 2
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=device)], 1)
    Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=device)], 1)
    w = torch.linalg.lstsq(Xtr, ytr[:, None]).solution
    pred = (Xte @ w).squeeze(1) > 0.5
    return (pred == (yte > 0.5)).float().mean().item()


rng_tr, rng_c, rng_te = random.Random(seed), random.Random(seed + 10_000), random.Random(seed + 20_000)
train = [make_example(rng_tr) for _ in range(train_size)]
uniforms = [rng_c.random() for _ in train]
wrong = [corrupt_trace(ex, rng_c) for ex in train]
shown = [st if u < rho else bad for (_, _, _, st), u, bad in zip(train, uniforms, wrong)]
z_train = torch.tensor([float(u < rho) for u in uniforms], device=device)
test = [make_example(rng_te) for _ in range(test_size)]
u_te = [rng_te.random() for _ in test]
wrong_te = [corrupt_trace(ex, rng_te) for ex in test]
shown_te = [st if u < rho else bad for (_, _, _, st), u, bad in zip(test, u_te, wrong_te)]
z_te = torch.tensor([float(u < rho) for u in u_te], device=device)

gold_rows = [row(e, e[3]) for e in train]
mix_rows = [row(e, s) for e, s in zip(train, shown)]
test_rows = [row(e, s) for e, s in zip(test, shown_te)]
inp_mix, tgt_rho = encode(mix_rows, ALL_POS)
inp_gold, tgt_clean = encode(gold_rows, STATE_POS)
_, tgt_unif = encode(gold_rows, STATE_POS)
_, tgt_ans = encode(gold_rows, ANS_POS)

theta = {k: (torch.randn_like(t) * init_std).to(device).requires_grad_() for k, t in zero_params().items()}
opt = torch.optim.AdamW(theta.values(), lr=lr, weight_decay=0.0)
g = torch.Generator().manual_seed(seed)
records = []
print(f"seed={seed} rho={rho} mu={mu:.3f} K={mod} n={n}", flush=True)
last_loss = float("nan")


def evaluate(it):
    take = torch.randperm(len(train), device=device)[:grad_batch]
    g_rho = grad_vec(theta, inp_mix[take], tgt_rho[take])
    g_clean = grad_vec(theta, inp_gold[take], tgt_clean[take])
    g_unif = grad_vec(theta, inp_gold[take], tgt_unif[take], uniform=True)
    g_ans = grad_vec(theta, inp_mix[take], tgt_ans[take])
    combo = mu * g_clean + (1 - mu) * g_unif + g_ans
    combo_thm = mu * g_clean + g_ans
    resid = (g_rho - combo).norm().item() / (g_rho.norm().item() + 1e-12)
    resid_thm = (g_rho - combo_thm).norm().item() / (g_rho.norm().item() + 1e-12)
    p1 = probe_acc(theta, test_rows, z_te, pos=n + 2)
    p2 = probe_acc(theta, test_rows, z_te, pos=n + 3)
    g_rho_lab = theory_grad(theta, inp_mix[take], tgt_rho[take])
    g_cl_lab = theory_grad(theta, inp_gold[take], tgt_clean[take])
    g_ans_lab = theory_grad(theta, inp_mix[take], tgt_ans[take])
    combo_lab = mu * g_cl_lab + g_ans_lab
    resid_lab = (g_rho_lab - combo_lab).norm().item() / (g_rho_lab.norm().item() + 1e-12)
    rec = dict(step=it, loss=last_loss, residual=resid, residual_thm=resid_thm,
               residual_lab=resid_lab, probe_s1=p1, probe_s2=p2,
               g_rho=g_rho.norm().item(), g_clean=g_clean.norm().item(),
               g_unif=g_unif.norm().item(), g_ans=g_ans.norm().item(), mu=mu)
    records.append(rec)
    print(f"step {it:5d}  residual {resid:.3f}  thm {resid_thm:.3f}  "
          f"lab {resid_lab:.3f}  probe_s1 {p1:.3f}  probe_s2 {p2:.3f}", flush=True)


evaluate(0)
for it in range(1, steps + 1):
    idx = torch.randint(len(train), (batch_size,), generator=g).to(device)
    loss = F.cross_entropy(forward(theta, inp_mix[idx]).flatten(0, 1), tgt_rho[idx].flatten(), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(theta.values(), 1.0)
    opt.step()
    last_loss = float(loss.detach())
    if it % eval_every and it != steps:
        continue
    evaluate(it)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"config": dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks,
                                   layout="block", seed=seed),
                    "params": {k: t.detach().cpu() for k, t in theta.items()},
                    "mode": "process", "step": it},
                   os.path.join(save_dir, f"s{seed}_process_step{it}.pt"))

payload = dict(seed=seed, rho=rho, mu=mu, records=records)
print(json.dumps(payload, indent=2))
if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
