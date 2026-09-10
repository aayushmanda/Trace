"""Parameter-space pullback of local-rule credit (Corollary 34).

Training-side gradients are the **exact serialized LM objective**: full-vocabulary
autoregressive CE on the continuation, with the same prompt/continuation mask as
`ContinuationDataset` / `GPTModel.forward`. The readout Jacobian `J^T W` is still
taken through the induced-state tables (16 valid successors).
"""
import json
import random

import numpy as np
import torch

from src.data.boolean_circuit_tasks import make_boolean_circuit_sampler
from src.data.datasets import encode_pair
from src.eval.induced_rule import (
    GATES, K, M, TOK, TRUE, U, build_dataset, credit_norms,
    free_running_accuracy, induced_rules, parse_prompt, probe_contexts,
    render, rule_recovery, scales,
)
from src.models.gpt import GPTModel
from src.training.io import append_rows
from src.training.loop import train_with_checkpoints
from src.training.optim import make_loader, make_optimizer
from src.training.seed import configure_device, maybe_compile, set_seed

PI = np.eye(K) - U


def _flat_grad(model):
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
        else:
            parts.append(p.grad.detach().reshape(-1).float().cpu())
    if not parts:
        raise RuntimeError("no parameters")
    return torch.cat(parts)


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def relative_grad_error(a, b):
    """||a/||a|| − b/||b||||; scale-free disagreement of two parameter gradients."""
    an = a / (a.norm() + 1e-30)
    bn = b / (b.norm() + 1e-30)
    return float((an - bn).norm())


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
    """J^T vec(W): gradient of Σ ⟨W_g, P̂_g⟩ through the 16-way induced readout."""
    import torch.nn.functional as F

    model.zero_grad(set_to_none=True)
    ctxs, index = probe_contexts(depth, 1, ["x0", "c01", "s23", "t012", "x3", "c30"])
    from src.eval.induced_rule import STATES
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
    return _flat_grad(model)


def serialized_lm_grad(model, instances, formats, device, block_size, batch=32):
    """−∇_θ of full-vocab continuation CE (prompt tokens unmasked / zero loss)."""
    assert len(instances) == len(formats)
    model.zero_grad(set_to_none=True)
    n_mask = 0.0
    chunks = []
    for inst, fmt in zip(instances, formats):
        _prompt, target = render(inst, fmt)
        x, y, mask = encode_pair(TOK, inst.prompt, target, block_size)
        chunks.append((x, y, mask))
        n_mask += sum(mask)
    n_mask = max(n_mask, 1.0)
    vocab = model.vocab_size
    for i in range(0, len(chunks), batch):
        sub = chunks[i:i + batch]
        x = torch.tensor([r[0] for r in sub], dtype=torch.long, device=device)
        y = torch.tensor([r[1] for r in sub], dtype=torch.long, device=device)
        mask = torch.tensor([r[2] for r in sub], dtype=torch.float32, device=device)
        logits, loss = model(x, targets=y, mask=mask)
        if logits.shape[-1] != vocab:
            raise RuntimeError(f"expected full vocab {vocab}, got {logits.shape[-1]}")
        if logits.shape[-1] == K:
            raise RuntimeError("serialized LM grad must not collapse to 16-way state CE")
        (loss * mask.sum() / n_mask).backward()
    return -_flat_grad(model)


def outcome_grad(model, instances, device, block_size=None, batch=32):
    block_size = block_size or model.block_size
    return serialized_lm_grad(model, instances, ["direct"] * len(instances), device, block_size, batch)


def process_grad(model, instances, depth, device, block_size=None, batch=16):
    del depth
    block_size = block_size or model.block_size
    return serialized_lm_grad(model, instances, ["trace"] * len(instances), device, block_size, batch)


def lm_loss_value(model, instances, formats, device, block_size, batch=32):
    total = 0.0
    n_mask = 0.0
    with torch.no_grad():
        for i in range(0, len(instances), batch):
            insts = instances[i:i + batch]
            fmts = formats[i:i + batch]
            xs, ys, masks = [], [], []
            for inst, fmt in zip(insts, fmts):
                _p, target = render(inst, fmt)
                x, y, mask = encode_pair(TOK, inst.prompt, target, block_size)
                xs.append(x); ys.append(y); masks.append(mask)
            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            mask = torch.tensor(masks, dtype=torch.float32, device=device)
            _, loss = model(x, targets=y, mask=mask)
            total += float(loss.detach()) * float(mask.sum())
            n_mask += float(mask.sum())
    return total / max(n_mask, 1.0)


def fd_relative_error(model, instances, formats, device, block_size, grad, eta=1e-3):
    """|ΔL − ⟨∇L, η u⟩| / |ΔL| along the normalized (−grad) direction. ∇L = −grad."""
    direction = grad / (grad.norm() + 1e-30)
    L0 = lm_loss_value(model, instances, formats, device, block_size)
    offset, used = 0, []
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            delta = direction[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            p.add_(delta, alpha=eta)
            used.append((p, delta))
            offset += n
    L1 = lm_loss_value(model, instances, formats, device, block_size)
    with torch.no_grad():
        for p, delta in used:
            p.add_(delta, alpha=-eta)
    actual = L1 - L0
    pred = eta * float((-grad).dot(direction))
    return abs(actual - pred) / (abs(actual) + 1e-12)


def measure_pullback(model, probe, depth, device, block_size, Phat=None, fd=True):
    if Phat is None:
        Phat, _ = induced_rules(model, depth, device, step=1)
    credit = surrogate_credit(Phat, probe, depth)
    ctrue = np.einsum("ij,mjk->mik", PI, TRUE - U[None])
    ctrue *= np.linalg.norm(credit) / (np.linalg.norm(ctrue) + 1e-30)
    rc = np.random.default_rng(0).normal(size=credit.shape)
    rc = np.einsum("ij,mjk,kl->mil", PI, rc, PI)
    rc *= np.linalg.norm(credit) / (np.linalg.norm(rc) + 1e-30)
    gp = pullback_grad(model, depth, device, credit)
    gt = pullback_grad(model, depth, device, ctrue)
    gr = pullback_grad(model, depth, device, rc)
    go = outcome_grad(model, probe, device, block_size)
    gq = process_grad(model, probe, depth, device, block_size)
    row = dict(
        cos_true_outcome=cosine(gt, go),
        cos_true_process=cosine(gt, gq),
        cos_random_outcome=cosine(gr, go),
        cos_random_process=cosine(gr, gq),
        cos_surrogate_outcome=cosine(gp, go),
        cos_outcome_process=cosine(go, gq),
        rel_grad_err_outcome=relative_grad_error(gt, go),
        rel_grad_err_process=relative_grad_error(gt, gq),
        rel_grad_err_random=relative_grad_error(gr, go),
        norm_outcome=float(go.norm()),
        norm_process=float(gq.norm()),
        pullback_objective="serialized_lm_ce",
    )
    if fd:
        row["rel_grad_err_fd_outcome"] = fd_relative_error(
            model, probe, ["direct"] * len(probe), device, block_size, go,
        )
    model.zero_grad(set_to_none=True)
    return row


def run(args):
    import time
    device = torch.device(args.device)
    configure_device(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    sampler = make_boolean_circuit_sampler(args.depth)
    block = 40 + 12 * args.depth
    set_seed(args.seed)
    train = sample_instances_safe(args, sampler)
    val = sample_instances_safe(args, sampler, val=True, train=train)
    probe = val[:args.probe_size]
    rng = random.Random(2000 + args.seed)
    frac = getattr(args, "trace_fraction", 0.5)
    fmts = ["trace" if rng.random() < frac else "direct" for _ in train]
    ds = build_dataset(train, fmts, block)
    loader = make_loader(ds, args, device)
    model = GPTModel(
        vocab_size=TOK.vocab_size, block_size=block, pad_id=TOK.pad_id,
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, dropout=0.0,
    ).to(device)
    train_model = maybe_compile(model, device, enabled=getattr(args, "compile", None))
    opt = make_optimizer(model, args, device)
    ckpts = sorted(set(args.checkpoints))
    rows = []

    def on_checkpoint(step, model, _loss):
        t0 = time.time()
        Phat, on_set = induced_rules(model, args.depth, device, step=1)
        gam, eps = scales(Phat)
        pull = measure_pullback(model, probe, args.depth, device, block, Phat=Phat)
        cred_mean, _ = credit_norms([Phat] * args.depth, probe, args.depth)
        acc = free_running_accuracy(model, probe, args.depth, device, "both")
        row = dict(
            depth=args.depth, seed=args.seed, step=step, free_answer_acc=acc,
            eps_rule_hat=eps, gamma_hat=gam, rule_recovery=rule_recovery(Phat),
            credit_mean=cred_mean, state_on_set_mass=on_set,
            **pull, seconds=round(time.time() - t0, 1),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.zero_grad(set_to_none=True)

    train_with_checkpoints(model, loader, opt, device, ckpts, on_checkpoint, grad_clip=1.0,
                           train_model=train_model)
    append_rows(args.out, rows)


def sample_instances_safe(args, sampler, val=False, train=None):
    from src.data.sample import sample_instances
    if val:
        return sample_instances(sampler, args.val_size, seed=7000 + args.depth,
                                exclude={i.prompt for i in train})
    return sample_instances(sampler, args.train_size, seed=1000 + args.depth)
