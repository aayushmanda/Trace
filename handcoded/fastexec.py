
from __future__ import annotations

import argparse
import random
import re
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# config

@dataclass(frozen=True)
class Cfg:
    # model
    embedding: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.0
    # optimisation
    steps: int = 8_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_seed: int = 12345
    # runtime
    eval_batch_size: int = 256
    compile: object = False   # False, True, or a torch.compile mode
    bf16: bool = True
    log_every: int = 50

    @staticmethod
    def of(args, **over) -> "Cfg":
        """Accept the notebook's SimpleNamespace TRAIN_ARGS (or another Cfg)."""
        if isinstance(args, Cfg):
            return replace(args, **over) if over else args
        fields = Cfg.__dataclass_fields__
        kw = {k: getattr(args, k) for k in fields if hasattr(args, k)}
        kw.update(over)
        return Cfg(**kw)


def set_seed(seed: int):
    """Identical to compare_supervision.set_seed -- same RNG draws, same init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# data: strings in, three GPU tensors out

TARGETS: dict[str, Callable] = {
    "outcome":      lambda i: f" : {i.gold}\n",
    "answer_first": lambda i: f" : {i.gold} ; {i.correct_trace}\n",
    "process":      lambda i: f" {i.correct_trace} : {i.gold}\n",
    "corrupted":    lambda i: f" {i.wrong_trace} : {i.gold}\n",
}


def stride_target(depth: int, k: int) -> Callable:
    """Keep every k-th `gate>state` item; k >= depth is plain OUTCOME."""
    def build(inst):
        if k >= depth:
            return f" : {inst.gold}\n"
        steps = inst.correct_trace.split()
        assert len(steps) == depth and depth % k == 0, (len(steps), depth, k)
        return f" {' '.join(steps[k - 1::k])} : {inst.gold}\n"
    return build


def _lut(tokenizer) -> np.ndarray:
    table = np.full(256, -1, dtype=np.int64)
    for ch, i in tokenizer.stoi.items():
        table[ord(ch)] = i
    return table


class Split:
    """x, y, mask on the device -- uint8/bool, cast per batch. Nothing else."""

    __slots__ = ("x", "y", "mask", "width", "n")

    def __init__(self, x, y, mask):
        self.x, self.y, self.mask = x, y, mask
        self.n, self.width = x.shape

    def batch(self, ix):
        return self.x[ix].long(), self.y[ix].long(), self.mask[ix].float()


def pack(instances: Sequence, task, target_of: Callable, device) -> Split:
    """Tokenize a split in one vectorised pass. Same bytes as SupervisionDataset."""
    tokenizer = task.tokenizer
    if tokenizer.vocab_size > 256:
        raise ValueError("uint8 storage requires vocab_size <= 256")
    prompts = [inst.prompt for inst in instances]
    fulls = [p + target_of(inst) for p, inst in zip(prompts, instances)]

    ids = _lut(tokenizer)[np.frombuffer("".join(fulls).encode("ascii"), dtype=np.uint8)]
    if (ids < 0).any():
        raise ValueError("unknown character for the task tokenizer")

    plen = np.fromiter(map(len, prompts), np.int64, len(prompts))
    flen = np.fromiter(map(len, fulls), np.int64, len(fulls))
    if flen.max() > task.block_size:
        raise ValueError(f"{task.name}: {flen.max()} tokens exceeds block_size={task.block_size}")

    width = int(flen.max()) - 1                      # after the x/y shift
    seq = np.full((len(fulls), width + 1), tokenizer.pad_id, dtype=np.uint8)
    seq[np.arange(width + 1)[None, :] < flen[:, None]] = ids   # ragged fill, row-major

    col = np.arange(width)[None, :]
    live = col < (flen - 1)[:, None]                 # positions a short row really has
    pad = tokenizer.pad_id
    x = np.where(live, seq[:, :-1], pad).astype(np.uint8)
    y = np.where(live, seq[:, 1:], pad).astype(np.uint8)
    mask = live & (col + 1 >= plen[:, None])
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
    return Split(to(x), to(y), to(mask))


def batches(n: int, batch_size: int, seed: int) -> Iterable[torch.Tensor]:
    """The exact index stream of DataLoader(shuffle=True, drop_last=True, generator=g).

    Two draws per epoch are not ours: the loader takes a worker base seed, and
    RandomSampler draws one permutation it uses and one it throws away (the
    `[:num_samples % n]` tail, empty here). Reproducing them is what makes a run
    of this file bit-comparable with a run of compare_supervision.
    """
    g = torch.Generator().manual_seed(seed)
    scratch = torch.empty((), dtype=torch.int64)
    while True:
        scratch.random_(generator=g)
        perm = torch.randperm(n, generator=g)
        torch.randperm(n, generator=g)
        for start in range(0, n - batch_size + 1, batch_size):
            yield perm[start:start + batch_size]


# -----------------------------------------------------------------------------
# model: src.model.GPTModel, written out flat, plus a KV cache
# Module creation order and parameter names match GPTModel exactly, so the
# state_dicts are interchangeable and a shared seed gives identical weights.

class Attn(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head, self.head_size = n_head, n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout_p = dropout

    def _qkv(self, x):
        B, T, _ = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        return [t.view(B, T, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v)]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def prefill(self, x, kc, vc, keep):
        B, T, C = x.shape
        q, k, v = self._qkv(x)
        kc[:, :, :T], vc[:, :, :T] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def step(self, x, kc, vc, slot, keep):
        """One token against the whole cache; `keep` masks the slots not written."""
        B, _, C = x.shape
        q, k, v = self._qkv(x)
        kc.index_copy_(2, slot.view(1), k)
        vc.index_copy_(2, slot.view(1), v)
        y = F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep)
        return self.proj(y.transpose(1, 2).reshape(B, 1, C))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = Attn(n_embd, n_head, dropout)
        self.ffwd = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, pad_id, n_embd=128, n_head=4, n_layer=2, dropout=0.0):
        super().__init__()
        self.block_size, self.pad_id = block_size, pad_id
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        per_token = F.cross_entropy(
            logits.reshape(B * T, -1).float(), targets.reshape(B * T),
            reduction="none", ignore_index=self.pad_id).view(B, T)
        if mask is None:
            return logits, per_token.mean()
        return logits, (per_token * mask).sum() / mask.sum().clamp(min=1)

    # -- greedy decoding with a KV cache -------------------------------------
    # Prompts are LEFT-padded so that every prompt length in the split decodes in
    # one batch: each row carries its own position offset, and the pad slots are
    # masked out of attention. That turns "one generate() call per distinct
    # prompt length" into one call, which matters because each call is
    # max_new_tokens sequential steps no matter how few rows it has.

    def _cache(self, B, width, device, dtype):
        sa = self.blocks[0].sa
        shape = (B, sa.n_head, width, sa.head_size)
        return [(torch.zeros(shape, device=device, dtype=dtype),
                 torch.zeros(shape, device=device, dtype=dtype)) for _ in self.blocks]

    def _trunk(self, x, pos_ids, cache, slot, keep):
        x = x + self.position_embedding_table(pos_ids)
        for block, (kc, vc) in zip(self.blocks, cache):
            h = block.ln1(x)
            if slot is None:
                h = block.sa.prefill(h, kc, vc, keep)
            else:
                h = block.sa.step(h, kc, vc, slot, keep)
            x = x + h
            x = x + block.ffwd(block.ln2(x))
        return self.lm_head(self.ln_f(x))[:, -1, :]

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, stop_id=None, greedy=True,
                 lengths=None, poll=16):
        """Greedy decode. Returns the prompt with max_new_tokens appended.

        `lengths[b]` is how much of row b is real prompt (the rest is left pad);
        omit it for equal-length prompts, which is GPTModel.generate's contract.
        Rows that have emitted `stop_id` are pinned to it, so everything up to
        the first newline is what the row would have produced on its own.
        """
        assert greedy, "fastexec decodes greedily"
        B, Tp = idx.shape
        device, dtype = idx.device, self.lm_head.weight.dtype
        width = Tp + max_new_tokens
        if width > self.block_size:
            raise ValueError(f"{Tp}+{max_new_tokens} exceeds block_size={self.block_size}")
        if lengths is None:
            lengths = torch.full((B,), Tp, dtype=torch.long, device=device)
        offset = (Tp - lengths).unsqueeze(1)                     # left pad per row

        span = torch.arange(width, device=device)
        real = span[:Tp].unsqueeze(0) >= offset                  # [B, Tp] non-pad prompt
        pos_ids = (span[:Tp].unsqueeze(0) - offset).clamp_min(0)

        cache = self._cache(B, width, device, dtype)
        causal = span[:Tp].unsqueeze(1) >= span[:Tp].unsqueeze(0)   # [Tp, Tp] query >= key
        keep = (causal.unsqueeze(0) & real.unsqueeze(1))[:, None]   # [B,1,Tp,Tp]
        logits = self._trunk(self.token_embedding_table(idx), pos_ids, cache, None, keep)

        done = torch.zeros(B, 1, dtype=torch.bool, device=device)
        out = [idx]
        for i in range(max_new_tokens):
            token = logits.argmax(dim=-1, keepdim=True)
            if stop_id is not None:
                token = torch.where(done, torch.full_like(token, stop_id), token)
                done |= token == stop_id
            out.append(token)
            if i + 1 == max_new_tokens:
                break
            # polling costs a host sync, so do it rarely rather than every token
            if stop_id is not None and i % poll == poll - 1 and bool(done.all()):
                out.append(token.expand(B, max_new_tokens - i - 1))
                break
            slot = span[Tp + i]
            keep = ((span >= offset) & (span <= Tp + i))[:, None, None]  # [B,1,1,width]
            logits = self._trunk(self.token_embedding_table(token),
                                 lengths.unsqueeze(1) + i, cache, slot, keep)
        return torch.cat(out, dim=1)


def build_model(task, cfg: Cfg, device) -> GPT:
    return GPT(vocab_size=task.tokenizer.vocab_size, block_size=task.block_size,
               pad_id=task.tokenizer.pad_id, n_embd=cfg.embedding, n_head=cfg.heads,
               n_layer=cfg.layers, dropout=cfg.dropout).to(device)


# -----------------------------------------------------------------------------
# the loop

def train(task, instances, mode, seed, args, device, split: Split | None = None,
          target_of: Callable | None = None, checkpoints=(), at_checkpoint=None,
          desc: str | None = None):
    """Train one model. Returns (model, final_loss).

    `split` lets a caller pack the data once and reuse it across seeds.
    `at_checkpoint(model, step, loss)` runs at every step in `checkpoints`, which
    is how the sweeps read out mid-run without a second training pass.
    """
    cfg = Cfg.of(args)
    target_of = target_of or TARGETS[mode]
    if split is None:
        split = pack(instances, task, target_of, device)

    set_seed(seed)
    model = build_model(task, cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            fused=(device.type == "cuda"))
    fwd = model
    if cfg.compile and device.type == "cuda":
        mode = cfg.compile if isinstance(cfg.compile, str) else "default"
        # Train on the compiled wrapper; generate/eval keep eager `model`.
        fwd = torch.compile(model, mode=mode, dynamic=False)
    bf16 = cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    stream = batches(split.n, cfg.batch_size, cfg.batch_seed)
    params = list(model.parameters())
    checkpoints = set(checkpoints)

    model.train()
    loss = torch.zeros((), device=device)
    bar = tqdm(range(1, cfg.steps + 1), desc=desc or f"{task.name}/{mode}/seed={seed}", leave=False)
    for step in bar:
        x, y, m = split.batch(next(stream).to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            _, loss = fwd(x, targets=y, mask=m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % cfg.log_every == 0:                     # the only host sync
            bar.set_postfix(loss=f"{loss.item():.4f}")
        if step in checkpoints:
            at_checkpoint(model, step, float(loss.detach()))
            model.train()
    bar.close()
    return model, float(loss.detach())


# -----------------------------------------------------------------------------
# evaluation

def extract(text: str):
    """sweep_ratio.extract_generation, inlined so this file stands alone."""
    line = text.split("\n", 1)[0]
    if ":" not in line:
        return None, None, False
    trace_text, answer_text = line.rsplit(":", 1)
    match = re.search(r"\d+", answer_text)
    return trace_text.strip(), (match.group(0) if match else None), True


@torch.inference_mode()
def evaluate(model, task, instances, condition, args, device,
             gold_trace_of: Callable | None = None, desc="eval"):
    """Free-running greedy evaluation. Metric keys match sweep_ratio.evaluate.

    `condition` is "outcome" (answer only, 8 new tokens) or anything else (the
    whole trace). `gold_trace_of(inst)` supplies the expected trace; it defaults
    to the valid trace, and the stride sweep passes the strided one.
    """
    model.eval()
    gold_trace_of = gold_trace_of or (lambda inst: inst.correct_trace)
    cfg = Cfg.of(args)
    tokenizer = task.tokenizer
    max_new = 8 if condition == "outcome" else task.max_new_tokens

    lut = _lut(tokenizer)
    prompts = [inst.prompt for inst in instances]
    ids = lut[np.frombuffer("".join(prompts).encode("ascii"), dtype=np.uint8)]
    plen = np.fromiter(map(len, prompts), np.int64, len(prompts))

    answer = trace = step_hits = step_total = colons = total = 0
    bar = tqdm(total=len(instances), desc=desc, leave=False)
    for start in range(0, len(instances), cfg.eval_batch_size):
        stop = min(start + cfg.eval_batch_size, len(instances))
        rows, lens = instances[start:stop], plen[start:stop]
        wide = int(lens.max())
        # left-pad to a common width so every prompt length decodes in one batch
        block = np.full((len(rows), wide), tokenizer.pad_id, dtype=np.int64)
        flat = ids[plen[:start].sum():plen[:stop].sum()]
        block[np.arange(wide)[None, :] >= (wide - lens)[:, None]] = flat
        ctx = torch.from_numpy(block).to(device)
        out = model.generate(ctx, max_new_tokens=max_new, stop_id=tokenizer.newline_id,
                             greedy=True, lengths=torch.from_numpy(lens).to(device))
        for generated, inst in zip(out[:, wide:].tolist(), rows):
            predicted_trace, predicted_answer, saw_colon = extract(tokenizer.decode(generated))
            colons += int(saw_colon)
            answer += int(predicted_answer == inst.gold)
            if condition != "outcome":
                gold = gold_trace_of(inst)
                trace += int(predicted_trace == gold)
                got = [] if predicted_trace is None else predicted_trace.split()
                want = gold.split()
                step_hits += sum(a == b for a, b in zip(got, want))
                step_total += len(want)
            total += 1
        bar.update(len(rows))
    bar.close()
    return {
        "answer_accuracy": answer / total,
        "exact_trace_accuracy": None if condition == "outcome" else trace / total,
        "trace_step_accuracy": None if condition == "outcome" else step_hits / max(step_total, 1),
        "colon_rate": colons / total,
    }


def run(task, train_instances, val_instances, mode, seed, args, device,
        condition=None, split=None, target_of=None, gold_trace_of=None, desc=None):
    """train + evaluate + free the model. The unit the sweeps actually call."""
    model, loss = train(task, train_instances, mode, seed, args, device,
                        split=split, target_of=target_of, desc=desc)
    metrics = evaluate(model, task, val_instances,
                       condition or ("outcome" if mode == "outcome" else "process"),
                       args, device, gold_trace_of=gold_trace_of,
                       desc=f"{desc or mode} eval")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, "mode": mode, "loss": loss, **metrics}


# -----------------------------------------------------------------------------
# checks against the repository implementation

def _demo(depth=8, n=512, seed=0):
    from src.registry import TASKS
    task = TASKS[f"boolean_circuit_{depth}"]
    state = random.getstate()
    random.seed(seed)
    seen, items = set(), []
    while len(items) < n:
        inst = task.sample()
        if inst.prompt not in seen:
            seen.add(inst.prompt)
            items.append(inst)
    random.setstate(state)
    return task, items


def selftest(device=None):
    import compare_supervision as cs
    import sweep_ratio as sr
    from src.model import GPTModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, items = _demo()
    args = Cfg(steps=20, batch_size=32, eval_batch_size=64, compile=False, bf16=False)
    ok = lambda name, cond: print(f"  {'PASS' if cond else 'FAIL'}  {name}") or bool(cond)
    results = []

    # 1. same construction order => same weights from the same seed
    set_seed(7)
    mine = build_model(task, args, device)
    set_seed(7)
    theirs = cs.build_model(task, argparse.Namespace(
        embedding=args.embedding, heads=args.heads, layers=args.layers, dropout=args.dropout), device)
    same = (set(mine.state_dict()) == set(theirs.state_dict()) and
            all(torch.equal(mine.state_dict()[k], theirs.state_dict()[k]) for k in mine.state_dict()))
    results.append(ok("identical init to src.model.GPTModel", same))

    # 2. same tokens, same mask -- on the trimmed width
    for mode in ("outcome", "process", "corrupted", "answer_first"):
        split = pack(items, task, TARGETS[mode], device)
        ref = cs.SupervisionDataset(items, task, mode)
        w = split.width
        results.append(ok(f"pack == SupervisionDataset [{mode}]",
                          torch.equal(split.x.cpu().long(), ref.x[:, :w].long()) and
                          torch.equal(split.y.cpu().long(), ref.y[:, :w].long()) and
                          torch.equal(split.mask.cpu().float(), ref.mask[:, :w]) and
                          not ref.mask[:, w:].any()))

    # 3. same loss from the same weights (trimmed vs padded)
    split = pack(items, task, TARGETS["process"], device)
    ref = cs.SupervisionDataset(items, task, "process")
    x, y, m = split.batch(torch.arange(64, device=device))
    xr, yr, mr = (ref.x[:64].long().to(device), ref.y[:64].long().to(device), ref.mask[:64].float().to(device))
    with torch.no_grad():
        l_new = mine(x, targets=y, mask=m)[1]
        l_old = theirs(xr, targets=yr, mask=mr)[1]
    results.append(ok(f"masked loss matches (|d|={abs(float(l_new - l_old)):.2e})",
                      abs(float(l_new - l_old)) < 2e-5))

    # 4. KV-cache decode gives the same metrics as the repository decoder
    for condition in ("outcome", "process"):
        a = evaluate(mine, task, items[:128], condition, args, device)
        b = sr.evaluate(theirs, task, items[:128], "outcome" if condition == "outcome" else "mixed_process",
                        argparse.Namespace(eval_batch_size=args.eval_batch_size), device)
        results.append(ok(f"evaluate == sweep_ratio.evaluate [{condition}]",
                          all(a[k] == b[k] for k in b)))

    # 5. same minibatch order as DataLoader(shuffle=True, drop_last=True)
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(args.batch_seed)
    loader = DataLoader(torch.arange(len(items)), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, generator=g)
    stream = batches(len(items), args.batch_size, args.batch_seed)
    results.append(ok("batch order == DataLoader",
                      all(torch.equal(a, next(stream)) for a in loader)))

    # 6. same gradient from the same weights on the same batch. (Weight equality
    #    after N Adam steps is not a criterion anyone can meet: fp32 reduction
    #    order is nondeterministic, and Adam normalises the difference up. We
    #    print the repo's own run-to-run drift next to ours so the number means
    #    something.)
    mine.zero_grad(); theirs.zero_grad()
    mine(x, targets=y, mask=m)[1].backward()
    theirs(xr, targets=yr, mask=mr)[1].backward()
    worst = max(float((pa.grad - pb.grad).abs().max())
                for pa, pb in zip(mine.parameters(), theirs.parameters()))
    scale = max(float(pa.grad.abs().max()) for pa in mine.parameters())
    results.append(ok(f"gradients agree on a shared batch (max |dg|={worst:.2e}, |g|={scale:.2e})",
                      worst < 1e-5 * max(scale, 1.0)))

    # 7. and end to end, a 20-step run lands where the repository's lands
    opt_args = argparse.Namespace(
        batch_seed=args.batch_seed, batch_size=args.batch_size, workers=0, steps=args.steps,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        embedding=args.embedding, heads=args.heads, layers=args.layers, dropout=args.dropout)
    m_new, _ = train(task, items, "process", 11, args, device, desc="selftest/new")
    m_old, _ = cs.train_one(task, items, "process", 11, opt_args, device)
    m_old2, _ = cs.train_one(task, items, "process", 11, opt_args, device)
    drift = lambda p, q: max(float((p.state_dict()[k] - q.state_dict()[k]).abs().max())
                             for k in p.state_dict())
    ours, theirs = drift(m_new, m_old), drift(m_old, m_old2)
    results.append(ok(f"20 steps: ours-vs-repo {ours:.1e} against repo-vs-repo {theirs:.1e}",
                      ours < 1e-2))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


def bench(device=None, steps=400, depth=8, n=20_000, runs=2):
    """Time a small sweep the way a sweep actually runs: several models in one
    process, so the first model pays for torch.compile and the rest do not."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import compare_supervision as cs
    import sweep_ratio as sr
    task, items = _demo(depth=depth, n=n)
    val = items[:2_000]
    old = argparse.Namespace(batch_seed=12345, batch_size=128, eval_batch_size=256,
                             workers=4, steps=steps, lr=3e-4, weight_decay=0.0, grad_clip=1.0,
                             embedding=128, heads=4, layers=2, dropout=0.0)
    new = Cfg(steps=steps)

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t

    def free(m):
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"depth={depth} n={n} steps={steps} runs={runs} device={device}")
    for mode, condition in (("outcome", "outcome"), ("process", "mixed_process")):
        for r in range(runs):
            (m, _), t_old = timed(lambda: cs.train_one(task, items, mode, 2001 + r, old, device))
            _, e_old = timed(lambda: sr.evaluate(m, task, val, condition, old, device))
            free(m)
            split = pack(items, task, TARGETS[mode], device)
            (m, _), t_new = timed(lambda: train(task, items, mode, 2001 + r, new, device, split=split))
            _, e_new = timed(lambda: evaluate(m, task, val, mode, new, device))
            free(m)
            print(f"  {mode:8s} run{r}  train {t_old:6.1f}s -> {t_new:5.1f}s ({t_old / t_new:4.1f}x)"
                  f"   eval {e_old:5.1f}s -> {e_new:4.1f}s ({e_old / e_new:4.1f}x)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--steps", type=int, default=400)
    a = p.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.bench:
        bench(steps=a.steps)
