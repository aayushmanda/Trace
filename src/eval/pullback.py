"""Parameter-space pullback of local-rule credit (Corollary 34)."""
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.data.boolean_circuit_tasks import _apply_gate, _bits, _state_text, make_boolean_circuit_sampler
from src.data.dataclass import ANSWER_SEP
from src.data.sample import sample_instances
from src.eval.induced_rule import (
    GATES, K, M, STATES, TOK, TRUE, U, build_dataset, credit_norms,
    free_running_accuracy, induced_rules, parse_prompt, probe_contexts,
    rule_recovery, scales,
)
from src.models.gpt import GPTModel
from src.training.io import append_rows
from src.training.loop import train_with_checkpoints
from src.training.optim import make_loader, make_optimizer
from src.training.seed import configure_device, maybe_compile, set_seed

PI = np.eye(K) - U


def surrogate_credit(Phat, instances, depth):
    P = np.clip(Phat, 1e-12, None)
    P = P / P.sum(axis=2, keepdims=True)
    G = np.zeros((M, K, K))
    n = 0
    for inst in instances:
        s0, gates = parse_prompt(inst.prompt, depth)
        gi = [GATES.index(g) for g in gates]
        y = int(inst.gold, 2)
        fwd = [np.eye(K)[s0]]
        for t in range(depth):
            fwd.append(fwd[-1] @ P[gi[t]])
        bwd = [None] * (depth + 1)
        bwd[depth] = np.eye(K)[y]
        for t in range(depth - 1, -1, -1):
            bwd[t] = P[gi[t]] @ bwd[t + 1]
        p_y = float(fwd[0] @ bwd[0])
        if p_y <= 1e-12:
            continue
        n += 1
        for t in range(1, depth + 1):
            G[gi[t - 1]] += np.outer(fwd[t - 1], bwd[t]) / p_y
    G /= max(n, 1)
    return np.einsum("ij,mjk,kl->mil", PI, G, PI)


def pullback_grad(model, depth, device, credit, chunk=256):
    model.zero_grad(set_to_none=True)
    ctxs, index = probe_contexts(depth, 1, ["x0", "c01", "s23", "t012", "x3", "c30"])
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    buckets = {}
    for ci, ctx in enumerate(ctxs):
        buckets.setdefault(len(ctx), []).append(ci)
    for idxs in buckets.values():
        for i in range(0, len(idxs), chunk):
            sub = idxs[i:i + chunk]
            rows = []
            for ci in sub:
                cids = TOK.encode(ctxs[ci])
                rows.extend(cids + c for c in cand)
            T = len(rows[0])
            data = torch.tensor(rows, dtype=torch.long, device=device)
            logits, _ = model(data)
            logp = F.log_softmax(logits[:, T - L - 1:T - 1, :].float(), dim=-1)
            tgt = data[:, T - L:T]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1).view(len(sub), K)
            P = torch.softmax(lp, dim=-1)
            w = torch.tensor(
                np.stack([credit[gi, s] for gi, s in (index[ci] for ci in sub)]),
                dtype=P.dtype, device=device,
            )
            (P * w).sum().backward()
    return torch.cat([p.grad.detach().reshape(-1).float() for p in model.parameters() if p.grad is not None])


def outcome_grad(model, instances, device, batch=128):
    model.zero_grad(set_to_none=True)
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    buckets = {}
    for i, inst in enumerate(instances):
        buckets.setdefault(len(inst.prompt), []).append(i)
    for idxs in buckets.values():
        for i in range(0, len(idxs), batch):
            sub = idxs[i:i + batch]
            rows, gold = [], []
            for j in sub:
                cids = TOK.encode(instances[j].prompt + ANSWER_SEP)
                rows.extend(cids + c for c in cand)
                gold.append(int(instances[j].gold, 2))
            T = len(rows[0])
            data = torch.tensor(rows, dtype=torch.long, device=device)
            logits, _ = model(data)
            logp = F.log_softmax(logits[:, T - L - 1:T - 1, :].float(), dim=-1)
            tgt = data[:, T - L:T]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1).view(len(sub), K)
            loss = F.cross_entropy(lp, torch.tensor(gold, device=device), reduction="sum")
            (loss / len(instances)).backward()
    g = torch.cat([p.grad.detach().reshape(-1).float() for p in model.parameters() if p.grad is not None])
    return -g


def process_grad(model, instances, depth, device, batch=64):
    model.zero_grad(set_to_none=True)
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    ctxs, golds = [], []
    for inst in instances:
        s0, gates = parse_prompt(inst.prompt, depth)
        st = _bits(s0)
        body = ""
        for g in gates:
            ctxs.append(f"{inst.prompt} {body}{g}>")
            st = _apply_gate(st, g)
            golds.append(int(_state_text(st), 2))
            body += f"{g}>{_state_text(st)} "
    n = len(ctxs)
    buckets = {}
    for i, c in enumerate(ctxs):
        buckets.setdefault(len(c), []).append(i)
    for idxs in buckets.values():
        for i in range(0, len(idxs), batch):
            sub = idxs[i:i + batch]
            rows = [TOK.encode(ctxs[j]) + c for j in sub for c in cand]
            T = len(rows[0])
            data = torch.tensor(rows, dtype=torch.long, device=device)
            logits, _ = model(data)
            logp = F.log_softmax(logits[:, T - L - 1:T - 1, :].float(), dim=-1)
            tgt = data[:, T - L:T]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1).view(len(sub), K)
            tt = torch.tensor([golds[j] for j in sub], device=device)
            (F.cross_entropy(lp, tt, reduction="sum") / n).backward()
    g = torch.cat([p.grad.detach().reshape(-1).float() for p in model.parameters() if p.grad is not None])
    return -g


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def run(args):
    import time
    device = torch.device(args.device)
    configure_device(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    sampler = make_boolean_circuit_sampler(args.depth)
    block = 40 + 12 * args.depth
    set_seed(args.seed)
    train = sample_instances(sampler, args.train_size, seed=1000 + args.depth)
    val = sample_instances(sampler, args.val_size, seed=7000 + args.depth, exclude={i.prompt for i in train})
    probe = val[:args.probe_size]
    rng = random.Random(2000 + args.seed)
    fmts = ["trace" if rng.random() < 0.5 else "direct" for _ in train]
    ds = build_dataset(train, fmts, block)
    loader = make_loader(ds, args, device)
    model = GPTModel(
        vocab_size=TOK.vocab_size, block_size=block, pad_id=TOK.pad_id,
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, dropout=0.0,
    ).to(device)
    model = maybe_compile(model, device, enabled=getattr(args, "compile", None))
    opt = make_optimizer(model, args, device)
    ckpts = sorted(set(args.checkpoints))
    rows = []

    def on_checkpoint(step, model, _loss):
        t0 = time.time()
        Phat, on_set = induced_rules(model, args.depth, device, step=1)
        gam, eps = scales(Phat)
        credit = surrogate_credit(Phat, probe, args.depth)
        ctrue = np.einsum("ij,mjk->mik", PI, TRUE - U[None])
        ctrue *= np.linalg.norm(credit) / (np.linalg.norm(ctrue) + 1e-30)
        rc = np.random.default_rng(0).normal(size=credit.shape)
        rc = np.einsum("ij,mjk,kl->mil", PI, rc, PI)
        rc *= np.linalg.norm(credit) / (np.linalg.norm(rc) + 1e-30)
        gp = pullback_grad(model, args.depth, device, credit)
        gt = pullback_grad(model, args.depth, device, ctrue)
        gr = pullback_grad(model, args.depth, device, rc)
        go = outcome_grad(model, probe, device)
        gq = process_grad(model, probe, args.depth, device)
        cred_mean, _ = credit_norms(Phat, probe, args.depth)
        acc = free_running_accuracy(model, probe, args.depth, device, "both")
        row = dict(
            depth=args.depth, seed=args.seed, step=step, free_answer_acc=acc,
            eps_rule_hat=eps, gamma_hat=gam, rule_recovery=rule_recovery(Phat),
            credit_mean=cred_mean, state_on_set_mass=on_set,
            cos_true_outcome=cosine(gt, go), cos_true_process=cosine(gt, gq),
            cos_random_outcome=cosine(gr, go), cos_random_process=cosine(gr, gq),
            cos_surrogate_outcome=cosine(gp, go), cos_outcome_process=cosine(go, gq),
            norm_outcome=float(go.norm()), norm_process=float(gq.norm()),
            seconds=round(time.time() - t0, 1),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.zero_grad(set_to_none=True)

    train_with_checkpoints(model, loader, opt, device, ckpts, on_checkpoint, grad_clip=1.0)
    append_rows(args.out, rows)
