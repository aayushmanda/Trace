"""The learner: model, training loop, and the measurements taken on it.

One module holds the three things that define a single run -- the architecture,
the optimization, and the evaluation -- because they are what a `RunSpec` in
sweeps.py turns into. Nothing here knows what a sweep is.

The evaluation deserves a note. The headline number in the original experiment
was *end-to-end*: generate from the bare prompt and parse an answer out of
whatever came back. That metric confounds four things -- emitting a well-formed
trace, reaching the delimiter, computing the answer, and fitting inside the
token budget. `evaluate()` therefore also reports **answer forcing**: the trace
is supplied in context up to " : " and only the answer is produced. Comparing
the two separates "the model cannot compute the answer" from "the model could
not get to the answer in time", and the forced-with-a-corrupted-trace variant
separates both from "the model never reads the trace".
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config, ModelConfig, OptimConfig
from data import Pool, expand, teacher_forcing_rows
from tasks import Instance, Task, get_task


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """A small decoder-only transformer, sized for one Task."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        pad_id: int,
        cfg: ModelConfig,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(block_size, cfg.n_embd)
        self.blocks = nn.Sequential(
            *[Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    @classmethod
    def for_task(cls, task: Task, cfg: Config) -> "GPT":
        return cls(
            vocab_size=task.tokenizer.vocab_size,
            block_size=task.block_size,
            pad_id=task.tokenizer.pad_id,
            cfg=cfg.model,
            label_smoothing=cfg.optim.label_smoothing,
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            return logits, None

        loss_per_token = F.cross_entropy(
            logits.view(B * T, -1),
            targets.reshape(B * T),
            reduction="none",
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
        ).view(B, T)

        loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, stop_id: Optional[int] = None):
        """Greedy decoding for a batch of *equal-length* contexts.

        Equal length matters: every row then occupies the same absolute
        positions, so a batched decode is token-for-token identical to decoding
        the rows one at a time. Callers bucket by context length to guarantee it.
        """
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_id is not None:
                finished |= idx_next.squeeze(1) == stop_id
                if bool(finished.all()):
                    break
        return idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total_steps: int, cfg: OptimConfig) -> float:
    """Linear warmup then cosine decay to `min_learning_rate`."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > total_steps:
        return cfg.min_learning_rate
    decay_ratio = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


def train(
    pool: Pool,
    task: Task,
    cfg: Config,
    seed: int,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Trains a fresh model on `pool`. Returns (model, mean loss over last 10%).

    Batch indices are drawn from an explicitly seeded generator rather than the
    global RNG, so the sequence of minibatches depends only on `seed` and not on
    how many random numbers model initialisation happened to consume.
    """
    steps = cfg.optim.steps if steps is None else steps
    batch_size = cfg.optim.batch_size if batch_size is None else batch_size

    set_seed(seed)
    model = GPT.for_task(task, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
    )

    generator = torch.Generator(device=pool.device).manual_seed(seed)
    tail_from = int(steps * 0.9)
    tail_losses: List[float] = []

    model.train()
    for step in range(steps):
        lr = lr_at(step, steps, cfg.optim)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb, mb = pool.batch(batch_size, generator=generator)
        _, loss = model(xb, yb, mb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        optimizer.step()

        if step >= tail_from:
            tail_losses.append(float(loss.detach()))

    return model, (sum(tail_losses) / len(tail_losses) if tail_losses else float("nan"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """What one trained model scored on the held-out set.

    free_acc         end-to-end: generate from the bare prompt, parse an answer.
                     This is the metric the original curves were drawn with, and
                     it confounds trace generation with answer computation.
    forced_acc       ANSWER FORCING: the clean trace is given up to " : " and
                     only the answer is generated. Isolates the answer-position
                     computation from trace generation and the token budget.
    forced_wrong_acc the same with a corrupted trace. forced_acc >> this means
                     the model is genuinely conditioning on the trace it reads;
                     forced_acc ~= this means it is not.
    answer_nll       teacher-forced negative log-likelihood of the gold answer
                     tokens. Continuous, generation-free, and the most sensitive
                     of the four inside the transition.
    answer_tf_acc    teacher-forced exact match: every gold answer token is the
                     argmax. Same forward pass as answer_nll.
    no_answer_rate   fraction of free generations that produced nothing
                     parseable; truncation_rate, the subset that never emitted a
                     newline inside the budget. Both quantify how much of a
                     free_acc deficit is a generation-length artefact.
    """

    free_acc: Optional[float] = None
    forced_acc: Optional[float] = None
    forced_wrong_acc: Optional[float] = None
    answer_nll: Optional[float] = None
    answer_tf_acc: Optional[float] = None
    no_answer_rate: Optional[float] = None
    truncation_rate: Optional[float] = None
    train_loss: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@torch.no_grad()
def _greedy_probe(
    model: GPT,
    task: Task,
    instances: Sequence[Instance],
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    first_match: bool,
):
    """Greedy decode from `task.context(inst, mode)`; returns rate triple.

    Contexts are bucketed by token length so every row in a batch sits at the
    same absolute positions -- decoding is then identical to the unbatched loop.
    """
    tok = task.tokenizer
    buckets: Dict[int, List[tuple]] = {}
    for inst in instances:
        context = task.context(inst, mode)
        ids = tok.encode(context)
        buckets.setdefault(len(ids), []).append((ids, context, inst.gold))

    correct = no_answer = truncated = 0
    for items in buckets.values():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            ctx = torch.tensor(
                [ids for ids, _, _ in chunk], dtype=torch.long, device=model.lm_head.weight.device
            )
            out = model.generate(ctx, max_new_tokens=max_new_tokens, stop_id=tok.newline_id)

            for row, (_, context, gold) in zip(out.tolist(), chunk):
                tail = tok.decode(row)[len(context) :]
                truncated += int("\n" not in tail)
                pred = task.extract_answer(tail, first=first_match)
                no_answer += int(pred is None)
                correct += int(pred == gold)

    n = len(instances)
    return correct / n, no_answer / n, truncated / n


@torch.no_grad()
def _teacher_forced_probe(
    model: GPT, task: Task, instances: Sequence[Instance], mode: str, batch_size: int
):
    """(mean answer NLL, exact-match rate) with the context supplied verbatim.

    One forward pass per batch, no generation: this is the answer-position
    measurement the token budget cannot distort.
    """
    device = model.lm_head.weight.device
    tokens, bounds = teacher_forcing_rows(task, instances, mode, device)

    nll_total, nll_tokens, exact = 0.0, 0, 0
    for start in range(0, len(instances), batch_size):
        x, y, mask = expand(
            tokens[start : start + batch_size],
            bounds[start : start + batch_size],
            task.tokenizer.pad_id,
        )
        logits, _ = model(x)
        # Raw NLL: no label smoothing, so the number is a likelihood, not a loss.
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)

        nll_total += float((nll * mask).sum())
        nll_tokens += int(mask.sum())

        hit = (logits.argmax(dim=-1) == y) | (mask == 0)
        exact += int(hit.all(dim=1).sum())

    return nll_total / max(nll_tokens, 1), exact / len(instances)


@torch.no_grad()
def evaluate(
    model: GPT,
    task,
    instances: Sequence[Instance],
    cfg: Config,
    trace_in_context: bool = True,
) -> Metrics:
    """All configured measurements on one trained model.

    `trace_in_context=False` is condition C: the model was trained on
    prompt -> answer, so there is no trace to force and the teacher-forced
    probe conditions on the prompt alone.
    """
    task = get_task(task)
    was_training = model.training
    model.eval()

    ev = cfg.evaluation
    m = Metrics()
    # Answers are short; a couple of tokens of slack is enough to catch overrun.
    answer_budget = max(len(i.gold) for i in instances) + 3

    if ev.free:
        m.free_acc, m.no_answer_rate, m.truncation_rate = _greedy_probe(
            model, task, instances, "free", ev.batch_size, task.max_new_tokens, first_match=False
        )

    if ev.forced and trace_in_context:
        m.forced_acc, _, _ = _greedy_probe(
            model, task, instances, "forced_correct", ev.batch_size, answer_budget, first_match=True
        )
    if ev.forced_wrong and trace_in_context:
        m.forced_wrong_acc, _, _ = _greedy_probe(
            model, task, instances, "forced_wrong", ev.batch_size, answer_budget, first_match=True
        )

    if ev.teacher_forced:
        m.answer_nll, m.answer_tf_acc = _teacher_forced_probe(
            model,
            task,
            instances,
            "forced_correct" if trace_in_context else "direct",
            ev.batch_size,
        )

    if was_training:
        model.train()
    return m
