"""Does credit cancellation slow learning in a standard GPT-2-style Transformer?

Exploratory control outside the paper's one-block model. A 2-layer pre-LN decoder
(learned token + positional embeddings, multi-head causal attention, GELU MLP,
GPT-2 init) is trained from scratch with AdamW on fresh samples of the anti-copy
laws D_kappa (Def. Dkappa) and the clean mixture D_lambda (cor:dilution).

Sequence: tokens x_1..x_N (state tokens 0:+1, 1:-1), then instruction b (2:+1, 3:-1).
The prediction is read at the instruction position (index N); cross-entropy on y only
(two output classes). Source x_N sits at index N-1.

    CUDA_VISIBLE_DEVICES=3 uv run python Paper2/checks/standard_transformer.py \
        --init-std 0.02 --seeds 0 1 2 3 4 5 6 7 --out Paper2/checks/results/standard_transformer/std0.02.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LAWS = {"D1": ("kappa", 1.0), "D0": ("kappa", 0.0), "D-1": ("kappa", -1.0), "Dlam0.1": ("lambda", 0.1)}


def sample(law, n, batch, eta, gen, device):
    """Return tokens (batch, n+1), labels y in {0,1} (0 means +1), clean labels."""
    kind, val = LAWS[law]
    x = torch.randint(0, 2, (batch, n), generator=gen, device=device) * 2 - 1
    if kind == "kappa":
        m = torch.full((batch,), int(1 - val), device=device)
    else:  # fraction lambda from D_1 (M=0), rest from D_0 (M=1)
        m = (torch.rand(batch, generator=gen, device=device) >= val).long()
    ranks = torch.rand(batch, n - 1, generator=gen, device=device).argsort(1).argsort(1)
    anti = ranks < m[:, None]
    dec = x[:, : n - 1]
    x[:, : n - 1] = torch.where(anti, -x[:, n - 1 : n].expand_as(dec), dec)
    b = torch.randint(0, 2, (batch,), generator=gen, device=device) * 2 - 1
    xi = torch.where(torch.rand(batch, generator=gen, device=device) < eta, -1, 1)
    clean = b * x[:, n - 1]
    y = clean * xi
    tok = torch.cat([(x < 0).long(), (2 + (b < 0).long())[:, None]], 1)
    return tok, (y < 0).long(), (clean < 0).long()


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in self.c_attn(x).chunk(3, -1))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        a = s.softmax(-1)
        return self.c_proj((a @ v).transpose(1, 2).reshape(B, T, C)), a


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y, a = self.attn(self.ln1(x))
        x = x + y
        return x + self.mlp(self.ln2(x)), a


class GPT(nn.Module):
    def __init__(self, T, d, h, L, std):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(4, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2, bias=False)
        # GPT-2 init: N(0, std) weights, zero biases, LN at (1, 0), residual projections std/sqrt(2L).
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif ".ln" in name or name.startswith("ln_f"):
                nn.init.ones_(p)
            elif name.endswith("c_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=std / math.sqrt(2 * L))
            else:
                nn.init.normal_(p, std=std)

    def forward(self, tok):
        x = self.wte(tok) + self.wpe(torch.arange(tok.shape[1], device=tok.device))
        atts = []
        for blk in self.blocks:
            x, a = blk(x)
            atts.append(a[:, :, -1, :])  # attention from the prediction position: (B, H, T)
        return self.head(self.ln_f(x[:, -1])), atts


def run(law, seed, a):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    model = GPT(a.n + 1, a.d, a.heads, a.layers, a.init_std).to(dev)
    if a.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    egen = torch.Generator(device=dev).manual_seed(999)
    etok, ey, eclean = sample(law, a.n, a.eval_batch, a.eta, egen, dev)
    curve, hit = [], None
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                z, atts = model(etok)
                acc = (z.argmax(-1) == eclean).float().mean().item()
                loss = F.cross_entropy(z, ey).item()
                src = [at[:, :, a.n - 1].mean(0).tolist() for at in atts]  # per layer, per head
            model.train()
            curve.append(dict(step=step, clean_acc=acc, loss=loss, src_att=src))
            if hit is None and acc >= 0.99:
                hit = step
                if a.stop_after_hit and step >= a.min_steps:
                    break
        if step == a.steps:
            break
        tok, y, _ = sample(law, a.n, a.batch, a.eta, gen, dev)
        z, _ = model(tok)
        loss = F.cross_entropy(z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return dict(law=law, seed=seed, hit_step=hit, final=curve[-1], curve=curve)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--laws", nargs="+", default=list(LAWS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--opt", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batch", type=int, default=8192)
    p.add_argument("--stop-after-hit", action="store_true")
    p.add_argument("--min-steps", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for law in a.laws:
        for seed in a.seeds:
            t0 = time.time()
            r = run(law, seed, a)
            runs.append(r)
            print(f"{law:8s} seed={seed} hit={r['hit_step']} final_acc={r['final']['clean_acc']:.4f} "
                  f"loss={r['final']['loss']:.4f} src_att={[max(h) for h in r['final']['src_att']]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            out.write_text(json.dumps(dict(config=vars(a), runs=runs)))


if __name__ == "__main__":
    main()
