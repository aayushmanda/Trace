"""Induced local-rule readout of a trained Transformer.

    Phat_g[s, s'] = P_theta(s' | s, g)
"""
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.boolean_circuit_tasks import N_BITS, _apply_gate, _bits, _state_text, make_boolean_circuit_sampler
from src.data.dataclass import ANSWER_SEP, GLOBAL_TOKENIZER
from src.data.datasets import ContinuationDataset
from src.data.sample import sample_instances
from src.models.gpt import GPTModel
from src.training.io import append_rows
from src.training.loop import train_with_checkpoints
from src.training.optim import make_loader, make_optimizer
from src.training.progress import progress
from src.training.seed import configure_device, maybe_compile, set_seed

K = 2 ** N_BITS
TOK = GLOBAL_TOKENIZER
STATES = [f"{v:0{N_BITS}b}" for v in range(K)]


def gate_alphabet():
    gates = [f"x{k}" for k in range(N_BITS)]
    gates += [f"c{i}{j}" for i, j in permutations(range(N_BITS), 2)]
    gates += [f"s{i}{j}" for i, j in permutations(range(N_BITS), 2)]
    gates += [f"t{a}{b}{c}" for a, b, c in permutations(range(N_BITS), 3)]
    return gates


GATES = gate_alphabet()
M = len(GATES)
DEFAULT_FILLERS = (
    ("x0", "c01", "s23", "t012", "x3", "c30"),
    ("x1", "c12", "s30", "t123", "x2", "c23"),
    ("x2", "c23", "s01", "t201", "x0", "c12"),
)


def true_table(g):
    T = np.zeros((K, K))
    for s in range(K):
        nxt = int(_state_text(_apply_gate(_bits(s), g)), 2)
        T[s, nxt] = 1.0
    return T


TRUE = np.stack([true_table(g) for g in GATES])
U = np.full((K, K), 1.0 / K)


def render(inst, fmt):
    if fmt == "direct":
        return inst.prompt, f"{ANSWER_SEP}{inst.gold}\n"
    return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n"


def build_dataset(instances, formats, block_size):
    targets = [render(inst, fmt)[1] for inst, fmt in zip(instances, formats)]
    return ContinuationDataset(instances, TOK, block_size, targets)


@torch.inference_mode()
def score_candidates(model, contexts, candidates, device, batch=1024):
    cand_ids = [TOK.encode(c) for c in candidates]
    L = len(cand_ids[0])
    assert all(len(c) == L for c in cand_ids)
    n_cand = len(cand_ids)
    out = np.zeros((len(contexts), n_cand))
    buckets = {}
    for ci, ctx in enumerate(contexts):
        buckets.setdefault(len(ctx), []).append(ci)
    for idxs in progress(list(buckets.values()), desc="induced-rule readout", leave=False):
        rows = []
        for ci in idxs:
            cids = TOK.encode(contexts[ci])
            for c in cand_ids:
                rows.append(cids + c)
        T = len(rows[0])
        data = torch.tensor(rows, dtype=torch.long)
        vals = torch.empty(len(rows), dtype=torch.float64)
        for i in range(0, len(rows), batch):
            chunk = data[i:i + batch].to(device)
            logits, _ = model(chunk)
            logp = F.log_softmax(logits[:, T - L - 1:T - 1, :].float(), dim=-1)
            tgt = chunk[:, T - L:T]
            vals[i:i + batch] = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1).double().cpu()
        vals = vals.view(len(idxs), n_cand).numpy()
        for k, ci in enumerate(idxs):
            out[ci] = vals[k]
    return out


def renormalize(logp):
    m = logp.max(axis=1, keepdims=True)
    w = np.exp(logp - m)
    P = w / w.sum(axis=1, keepdims=True)
    on_set = np.exp(logp).sum(axis=1)
    return P, on_set


def _inverse_gate(g):
    for s in range(K):
        if int(_state_text(_apply_gate(_apply_gate(_bits(s), g), g)), 2) != s:
            return None
    return g


INVERSE = {g: _inverse_gate(g) for g in GATES}


def probe_contexts(depth, step, filler):
    ctxs, index = [], []
    for gi, g in enumerate(GATES):
        for s in range(K):
            gates = [filler[(step - 1 + k) % len(filler)] for k in range(depth)]
            gates[step - 1] = g
            cur = _bits(s)
            for back in range(step - 2, -1, -1):
                inv = INVERSE[gates[back]]
                cur = _apply_gate(cur, inv) if inv else cur
            s0 = cur
            prompt = f"i{_state_text(s0)};u{''.join(gates)}"
            body, st = "", s0
            for t in range(step - 1):
                st = _apply_gate(st, gates[t])
                body += f"{gates[t]}>{_state_text(st)} "
            assert int(_state_text(st), 2) == s
            ctxs.append(f"{prompt} {body}{g}>")
            index.append((gi, s))
    return ctxs, index


def induced_rules(model, depth, device, step=1, filler=None):
    filler = filler or ["x0", "c01", "s23", "t012", "x3", "c30"]
    ctxs, index = probe_contexts(depth, step, filler)
    logp = score_candidates(model, ctxs, STATES, device)
    P, on_set = renormalize(logp)
    Phat = np.zeros((M, K, K))
    for row, (gi, s) in enumerate(index):
        Phat[gi, s] = P[row]
    return Phat, float(on_set.mean())


def split(Phat):
    Mx = Phat - U[None]
    c = Mx.mean(axis=1)
    return c, Mx - c[:, None, :]


def scales(Phat):
    c, Fc = split(Phat)
    gam = float(max(np.linalg.norm(c[g]) for g in range(M)))
    eps = float(max(np.linalg.norm(Fc[g], 2) for g in range(M)))
    return gam, eps


def rule_recovery(Phat):
    return float((Phat.argmax(axis=2) == TRUE.argmax(axis=2)).mean())


def parse_prompt(prompt, depth):
    head, tail = prompt.split(";u")
    s0 = int(head[1:], 2)
    widths = {"x": 2, "c": 3, "s": 3, "t": 4}
    gates, i = [], 0
    while i < len(tail):
        n = widths[tail[i]]
        gates.append(tail[i:i + n])
        i += n
    assert len(gates) == depth, (gates, depth)
    return s0, gates


def _tables_seq(tables, depth):
    if isinstance(tables, (list, tuple)):
        if len(tables) != depth:
            raise ValueError(f"need one induced table per step, got {len(tables)} for D={depth}")
        return list(tables)
    raise TypeError("composition/credit require a per-step sequence of P̂^{(t)}, not one table reused")


def composition(tables, s0, gates):
    """e_{s0}^T P̂^{(1)}_{g1} ⋯ P̂^{(D)}_{gD} with a distinct table at each position."""
    v = np.zeros(K)
    v[s0] = 1.0
    seq = _tables_seq(tables, len(gates))
    for t, g in enumerate(gates):
        v = v @ seq[t][GATES.index(g)]
    return v


def table_row_tv(A, B):
    return float(0.5 * np.abs(A - B).sum(axis=-1).mean())


@torch.inference_mode()
def terminal_distribution(model, instances, depth, device):
    ctxs = [inst.prompt + ANSWER_SEP for inst in instances]
    return renormalize(score_candidates(model, ctxs, STATES, device))


def delta_comp(model, instances, Phat, depth, device):
    Q, on_set = terminal_distribution(model, instances, depth, device)
    tv, gold_gap, prod_acc, model_acc = [], [], [], []
    for row, inst in enumerate(instances):
        s0, gates = parse_prompt(inst.prompt, depth)
        v = composition(Phat, s0, gates)
        y = int(inst.gold, 2)
        tv.append(0.5 * np.abs(Q[row] - v).sum())
        gold_gap.append(abs(Q[row][y] - v[y]))
        prod_acc.append(float(v.argmax() == y))
        model_acc.append(float(Q[row].argmax() == y))
    return dict(
        delta_comp_tv=float(np.mean(tv)),
        delta_comp_gold=float(np.mean(gold_gap)),
        product_answer_acc=float(np.mean(prod_acc)),
        model_answer_acc=float(np.mean(model_acc)),
        answer_on_set_mass=float(on_set.mean()),
    )


def credit_norms(Phat, instances, depth, scale=1.0, mode="actual"):
    c, Fc = split(Phat)
    if mode == "actual":
        P = Phat.copy()
    elif mode == "conditional":
        P = U[None] + scale * Fc
    elif mode == "proportional":
        P = U[None] + scale * (c[:, None, :] + Fc)
    else:
        raise ValueError(mode)
    P = np.clip(P, 1e-12, None)
    P = P / P.sum(axis=2, keepdims=True)
    Pi = np.eye(K) - U
    out = []
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
        if p_y <= 0:
            continue
        for t in range(1, depth + 1):
            q, b = fwd[t - 1], bwd[t]
            out.append(np.linalg.norm(Pi @ q) * np.linalg.norm(Pi @ b) / p_y)
    return float(np.mean(out)), float(np.max(out))


def credit_exponent(Phat, instances, depth, mode, lambdas=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)):
    _, Fc = split(Phat)
    base = float(max(np.linalg.norm(Fc[g], 2) for g in range(M)))
    if base <= 0:
        return dict(exponent_fit="", exponent_r2="", credit_by_lambda="", eps_by_lambda="")
    xs, ys = [], []
    for lam in lambdas:
        mean, _ = credit_norms(Phat, instances, depth, scale=lam, mode=mode)
        xs.append(lam * base)
        ys.append(mean)
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    resid = ly - (slope * lx + intercept)
    ss = 1.0 - float((resid ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30))
    return {
        f"exponent_fit_{mode}": float(slope),
        f"exponent_r2_{mode}": ss,
        f"credit_by_lambda_{mode}": json.dumps({f"{l:.5g}": float(v) for l, v in zip(lambdas, ys)}),
        f"eps_by_lambda_{mode}": json.dumps({f"{l:.5g}": float(v) for l, v in zip(lambdas, xs)}),
    }


@torch.inference_mode()
def free_running_accuracy(model, instances, depth, device, condition, batch=256):
    model.eval()
    ok = 0
    budget = 8 if condition == "outcome" else 10 * depth + 12
    buckets = {}
    for inst in instances:
        buckets.setdefault(len(inst.prompt), []).append(inst)
    for rows in buckets.values():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            ctx = torch.tensor([TOK.encode(inst.prompt) for inst in chunk], dtype=torch.long, device=device)
            out = model.generate(ctx, max_new_tokens=budget, stop_id=TOK.newline_id, greedy=True)
            for row, inst in enumerate(chunk):
                text = TOK.decode(out[row, ctx.shape[1]:].tolist()).split("\n")[0]
                if ":" not in text:
                    continue
                ans = text.rsplit(":", 1)[1].strip()[:N_BITS]
                ok += int(ans == inst.gold)
    return ok / len(instances)


def _formats(args, n):
    import random
    rng = random.Random(2000 + args.seed)
    if args.condition == "outcome":
        return ["direct"] * n
    if args.condition == "process":
        return ["trace"] * n
    return ["trace" if rng.random() < args.trace_fraction else "direct" for _ in range(n)]


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
    ds = build_dataset(train, _formats(args, len(train)), block)
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
        probe_steps = args.probe_steps or [args.probe_step]
        probe_steps = sorted({max(1, min(args.depth, int(s))) for s in probe_steps})
        acc = free_running_accuracy(model, probe, args.depth, device, args.condition)
        if getattr(args, "ckpt_dir", None) and step == max(ckpts):
            dest = Path(args.ckpt_dir)
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "step": step},
                dest / f"{args.condition}_D{args.depth}_s{args.seed}_step{step}.pt",
            )
        pull_row = {}
        if getattr(args, "with_pullback", False) and step in {0, max(ckpts)}:
            from src.eval.pullback import (  # noqa: PLC0415
                PI, cosine, outcome_grad, process_grad, pullback_grad, surrogate_credit,
            )
            Phat1, _ = induced_rules(model, args.depth, device, step=1)
            credit = surrogate_credit(Phat1, probe, args.depth)
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
            pull_row = dict(
                cos_true_outcome=cosine(gt, go),
                cos_true_process=cosine(gt, gq),
                cos_random_outcome=cosine(gr, go),
                cos_random_process=cosine(gr, gq),
                cos_surrogate_outcome=cosine(gp, go),
                cos_outcome_process=cosine(go, gq),
            )
            model.zero_grad(set_to_none=True)
        for pstep in probe_steps:
            Phat, on_set = induced_rules(model, args.depth, device, step=pstep)
            gam, eps = scales(Phat)
            rec = rule_recovery(Phat)
            dc = delta_comp(model, probe, Phat, args.depth, device)
            cred_mean, cred_max = credit_norms(Phat, probe, args.depth)
            if step == max(ckpts) and pstep == probe_steps[0]:
                expo = dict(exponent_target=args.depth - 1)
                for mode in ("conditional", "proportional"):
                    expo.update(credit_exponent(Phat, probe, args.depth, mode))
            else:
                expo = dict(exponent_target=args.depth - 1)
                for mode in ("conditional", "proportional"):
                    expo.update({
                        f"exponent_fit_{mode}": "", f"exponent_r2_{mode}": "",
                        f"credit_by_lambda_{mode}": "", f"eps_by_lambda_{mode}": "",
                    })
            row = dict(
                condition=args.condition, trace_fraction=args.trace_fraction,
                depth=args.depth, seed=args.seed, probe_step=pstep, step=step,
                free_answer_acc=acc, gamma_hat=gam, eps_rule_hat=eps,
                rule_recovery=rec, credit_mean=cred_mean, credit_max=cred_max,
                state_on_set_mass=on_set, **dc, **expo, **pull_row,
                probe_seconds=round(time.time() - t0, 1),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if getattr(args, "dump_tables", None) and step == max(ckpts):
                dest = Path(args.dump_tables)
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest.with_name(f"{dest.stem}_step{pstep}{dest.suffix}"), Phat)

    train_with_checkpoints(model, loader, opt, device, ckpts, on_checkpoint, grad_clip=1.0,
                           train_model=train_model)
    append_rows(args.out, rows)
    return rows
