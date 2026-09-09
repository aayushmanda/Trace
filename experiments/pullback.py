"""Does the Transformer's outcome gradient route through its induced rules?

The factorization theorem lives on transition matrices.  A network has none, but
it has induced kernels Phat_g(theta).  If the model's relevant computation
passes through them, the chain rule says the executor-relevant part of the
parameter gradient is the pullback

    sum_g J_g(theta)^T vec( Rule(-grad_{Phat_g} L_prod) ),
    J_g(theta) = d vec Phat_g / d theta,

which is exactly grad_theta of the scalar  S = sum_g < Rule(-grad L_prod), Phat_g >.
One backward pass gives it.  We compare that vector against the true parameter
gradient of the outcome loss on the same examples.  A large positive cosine says
the two agree in parameter space; a small one says the induced-rule description,
however good at the function level, does not carry the gradient.
"""
import argparse, csv, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from induced_rule import (  # noqa: E402
    GATES, K, M, STATES, TOK, TRUE, U, ANSWER_SEP, build_dataset, credit_norms,
    free_running_accuracy, induced_rules, parse_prompt, probe_contexts,
    rule_recovery, sample_instances, scales, set_seed, split,
)
from boolean_circuit_tasks import (  # noqa: E402
    _apply_gate, _bits, _state_text, make_boolean_circuit_sampler,
)
from model import GPTModel  # noqa: E402

PI = np.eye(K) - U


def surrogate_credit(Phat, instances, depth):
    """Rule(-grad_{P_g} L_prod) for every gate, from the induced tables."""
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
    return np.einsum("ij,mjk,kl->mil", PI, G, PI)   # Rule(.) = Pi G Pi


def pullback_grad(model, depth, device, credit, chunk=256):
    """grad_theta sum_g < credit_g, Phat_g(theta) >, accumulated in chunks."""
    model.zero_grad(set_to_none=True)
    ctxs, index = probe_contexts(depth, 1, ["x0", "c01", "s23", "t012", "x3", "c30"])
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    buckets = {}
    for ci, ctx in enumerate(ctxs):
        buckets.setdefault(len(ctx), []).append(ci)
    for _, idxs in buckets.items():
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
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1)
            lp = lp.view(len(sub), K)
            P = torch.softmax(lp, dim=-1)                      # renormalized rows
            w = torch.tensor(
                np.stack([credit[gi, s] for gi, s in (index[ci] for ci in sub)]),
                dtype=P.dtype, device=device)
            (P * w).sum().backward()
    return torch.cat([p.grad.detach().reshape(-1).float()
                      for p in model.parameters() if p.grad is not None])


def outcome_grad(model, instances, device, batch=128):
    """-grad_theta of the outcome loss, answers scored over the 16 states."""
    model.zero_grad(set_to_none=True)
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    buckets = {}
    for i, inst in enumerate(instances):
        buckets.setdefault(len(inst.prompt), []).append(i)
    total = 0
    for _, idxs in buckets.items():
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
            total += len(sub)
    g = torch.cat([p.grad.detach().reshape(-1).float()
                   for p in model.parameters() if p.grad is not None])
    return -g


def process_grad(model, instances, depth, device, batch=64):
    """-grad_theta of the teacher-forced local rule loss.

    L_proc = -(1/ND) sum_{i,t} log Phat(s_t | s_{t-1}, g_t), scored over the
    sixteen complete successor states, so it is the same object the induced
    kernel is read from.
    """
    model.zero_grad(set_to_none=True)
    cand = [TOK.encode(c) for c in STATES]
    L = len(cand[0])
    ctxs, golds = [], []
    for inst in instances:
        s0, gates = parse_prompt(inst.prompt, depth)
        st = _bits(s0)
        body = ""
        for t, g in enumerate(gates):
            ctxs.append(f"{inst.prompt} {body}{g}>")
            st = _apply_gate(st, g)
            golds.append(int(_state_text(st), 2))
            body += f"{g}>{_state_text(st)} "
    n = len(ctxs)
    buckets = {}
    for i, c in enumerate(ctxs):
        buckets.setdefault(len(c), []).append(i)
    for _, idxs in buckets.items():
        for i in range(0, len(idxs), batch):
            sub = idxs[i:i + batch]
            rows = []
            for j in sub:
                cids = TOK.encode(ctxs[j])
                rows.extend(cids + c for c in cand)
            T = len(rows[0])
            data = torch.tensor(rows, dtype=torch.long, device=device)
            logits, _ = model(data)
            logp = F.log_softmax(logits[:, T - L - 1:T - 1, :].float(), dim=-1)
            tgt = data[:, T - L:T]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1).view(len(sub), K)
            tt = torch.tensor([golds[j] for j in sub], device=device)
            (F.cross_entropy(lp, tt, reduction="sum") / n).backward()
    g = torch.cat([p.grad.detach().reshape(-1).float()
                   for p in model.parameters() if p.grad is not None])
    return -g


def append_rows(path, rows):
    """Append to a CSV, refusing to write rows whose schema differs.

    Appending a wider row dict under a narrower existing header silently
    corrupts the file, so check the header and divert to a sibling file
    instead of writing a row that will not parse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if path.exists():
        with path.open(newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing is not None and existing != fields:
            path = path.with_name(f"{path.stem}__schema{len(fields)}{path.suffix}")
            print(f"schema differs from the existing file; writing {path}", flush=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return path


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def run(args):
    device = torch.device(args.device)
    sampler = make_boolean_circuit_sampler(args.depth)
    block = 40 + 12 * args.depth
    set_seed(args.seed)
    train = sample_instances(sampler, args.train_size, seed=1000 + args.depth)
    val = sample_instances(sampler, args.val_size, seed=7000 + args.depth,
                           exclude={i.prompt for i in train})
    probe = val[:args.probe_size]

    rng = random.Random(2000 + args.seed)
    fmts = ["trace" if rng.random() < 0.5 else "direct" for _ in train]
    ds = build_dataset(train, fmts, block)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(12345))

    model = GPTModel(vocab_size=TOK.vocab_size, block_size=block, pad_id=TOK.pad_id,
                     n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
                     dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    ckpts = sorted(set(args.checkpoints))
    rows, it = [], iter(loader)

    for step in range(0, max(ckpts) + 1):
        if step in ckpts:
            model.eval()
            t0 = time.time()
            Phat, on_set = induced_rules(model, args.depth, device, step=1)
            gam, eps = scales(Phat)
            credit = surrogate_credit(Phat, probe, args.depth)
            # the true-executor direction, pulled back into parameter space:
            # this is the direction Theorem 5 says the process gradient reveals
            # at first order and the outcome gradient does not
            ctrue = np.einsum("ij,mjk->mik", PI, TRUE - U[None])
            ctrue *= np.linalg.norm(credit) / (np.linalg.norm(ctrue) + 1e-30)
            # control: a random credit field of the same norm in the same subspace
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
            row = dict(depth=args.depth, seed=args.seed, step=step,
                       free_answer_acc=acc, eps_rule_hat=eps, gamma_hat=gam,
                       rule_recovery=rule_recovery(Phat), credit_mean=cred_mean,
                       state_on_set_mass=on_set,
                       cos_true_outcome=cosine(gt, go),
                       cos_true_process=cosine(gt, gq),
                       cos_random_outcome=cosine(gr, go),
                       cos_random_process=cosine(gr, gq),
                       cos_surrogate_outcome=cosine(gp, go),
                       cos_outcome_process=cosine(go, gq),
                       norm_outcome=float(go.norm()),
                       norm_process=float(gq.norm()),
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.train()
            model.zero_grad(set_to_none=True)
        if step == max(ckpts):
            break
        try:
            x, y, m = next(it)
        except StopIteration:
            it = iter(loader); x, y, m = next(it)
        x = x.to(device, torch.long); y = y.to(device, torch.long)
        m = m.to(device, torch.float32)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, targets=y, mask=m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    append_rows(args.out, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--probe-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[0, 500, 2000, 4000])
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--out", default="results/pullback.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
