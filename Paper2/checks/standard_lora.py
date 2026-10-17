"""Rank-8 LoRA on pretrained Qwen2.5-0.5B on a text encoding of the anti-copy laws.

Prompt (token ids built from single-token pieces so positions are fixed):
    "Signs:" s_1 "," s_2 "," ... "," s_N " Flip:" b " Next:"   ->   y
where each s_j, b, y is the token "+" or "-". The loss is cross-entropy restricted to
the two answer logits {"+","-"} at the last position. LoRA conventions follow src/lora.py
(all-linear, alpha = 2r, dropout 0, AdamW, lr 3e-4, weight decay 0).
Attention to the source token (s_N) from the last position is read with eager attention.

    CUDA_VISIBLE_DEVICES=3 uv run python Paper2/checks/standard_lora.py \
        --out Paper2/checks/results/standard_transformer/lora_qwen05.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from standard_transformer import LAWS, sample  # noqa: E402


def single(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, (s, ids)
    return ids[0]


def build(tok, n):
    pre = tok("Signs:", add_special_tokens=False)["input_ids"]
    flip = tok(" Flip:", add_special_tokens=False)["input_ids"]
    nxt = tok(" Next:", add_special_tokens=False)["input_ids"]
    plus, minus, comma = single(tok, "+"), single(tok, "-"), single(tok, ",")
    # Check that the concatenation tokenizes to the same ids (no merges across pieces).
    text = "Signs:" + ",".join(["+", "-"] * (n // 2)) + " Flip:-" + " Next:"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = []
    for j in range(n):
        body += [plus if j % 2 == 0 else minus] + ([comma] if j < n - 1 else [])
    manual = pre + body + flip + [minus] + nxt
    print("tokenization consistent:", ids == manual, flush=True)
    return dict(pre=pre, flip=flip, nxt=nxt, plus=plus, minus=minus, comma=comma)


def encode(t, sym, n, dev):
    """t: (B, n+1) tokens from sample(); 0/2 means +1, 1/3 means -1."""
    B = t.shape[0]
    pm = torch.tensor([sym["plus"], sym["minus"]], device=dev)
    xs = pm[t[:, :n]]
    b = pm[t[:, n] - 2]
    cols = [torch.tensor(sym["pre"], device=dev).expand(B, -1)]
    src_pos = None
    for j in range(n):
        if j == n - 1:
            src_pos = sum(c.shape[1] for c in cols)
        cols.append(xs[:, j : j + 1])
        if j < n - 1:
            cols.append(torch.full((B, 1), sym["comma"], device=dev))
    cols += [torch.tensor(sym["flip"], device=dev).expand(B, -1), b[:, None],
             torch.tensor(sym["nxt"], device=dev).expand(B, -1)]
    return torch.cat(cols, 1), src_pos


def logits2(model, ids, sym, attn=False):
    out = model(input_ids=ids, output_attentions=attn)
    z = out.logits[:, -1, [sym["plus"], sym["minus"]]].float()  # class 0 = "+", 1 = "-"
    return z, out.attentions


def run(law, seed, a, tok, sym):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=a.rank, lora_alpha=2 * a.rank,
                                             lora_dropout=0.0, bias="none", target_modules="all-linear")).to(dev)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    egen = torch.Generator(device=dev).manual_seed(999)
    et, ey, ec = sample(law, a.n, a.eval_batch, a.eta, egen, dev)
    eids, src = encode(et, sym, a.n, dev)
    curve, hit = [], None
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            model.eval()
            accs, losses, att = [], [], None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, a.eval_batch, 256):
                    z, atts = logits2(model, eids[i : i + 256], sym, attn=(i == 0))
                    accs.append((z.argmax(-1) == ec[i : i + 256]).float())
                    losses.append(F.cross_entropy(z, ey[i : i + 256], reduction="none"))
                    if i == 0:  # per layer, max over heads, mean over examples
                        att = [float(A[:, :, -1, src].float().mean(0).max()) for A in atts]
                        uni = [float(A[:, :, -1, src - 2].float().mean(0).max()) for A in atts]
            model.train()
            acc = torch.cat(accs).mean().item()
            curve.append(dict(step=step, clean_acc=acc, loss=torch.cat(losses).mean().item(),
                              src_att_maxhead=att, prev_decoy_att_maxhead=uni))
            print(f"  {law} s{seed} step {step} acc {acc:.3f} loss {curve[-1]['loss']:.3f} "
                  f"src_att(max layer) {max(att):.3f}", flush=True)
            if hit is None and acc >= 0.99:
                hit = step
        if step == a.steps:
            break
        t, y, _ = sample(law, a.n, a.batch, a.eta, gen, dev)
        ids, _ = encode(t, sym, a.n, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = logits2(model, ids, sym)
        loss = F.cross_entropy(z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    del model, opt
    torch.cuda.empty_cache()
    return dict(law=law, seed=seed, hit_step=hit, final=curve[-1], curve=curve)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--laws", nargs="+", default=list(LAWS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    sym = build(tok, a.n)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for law in a.laws:
        for seed in a.seeds:
            t0 = time.time()
            r = run(law, seed, a, tok, sym)
            runs.append(r)
            print(f"{law:8s} seed={seed} hit={r['hit_step']} final_acc={r['final']['clean_acc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            out.write_text(json.dumps(dict(config=vars(a), runs=runs)))


if __name__ == "__main__":
    main()
