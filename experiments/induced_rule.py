"""Measure the induced local rule of a trained Transformer.

Defines, for a trained model theta, the induced kernel

    Phat_g[s, s'] = P_theta(s' | s, g)

read at a trace-format decision position, and reports

    gamma_hat     = max_g || c_g ||_2            (marginal scale)
    eps_rule_hat  = max_g || F_g ||_op           (conditional scale)
    delta_comp    = discrepancy between the model's own terminal answer
                    distribution and the product of its induced kernels

together with the conditional credit norm predicted by the factorization
theorem.  Everything here is a readout of the network, not a parameter.
"""
import argparse, csv, json, math, random, sys, time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from boolean_circuit_tasks import (  # noqa: E402
    N_BITS, _apply_gate, _bits, _state_text, make_boolean_circuit_sampler,
)
from dataclass import GLOBAL_TOKENIZER, ANSWER_SEP  # noqa: E402
from model import GPTModel  # noqa: E402

K = 2 ** N_BITS
TOK = GLOBAL_TOKENIZER
STATES = [f"{v:0{N_BITS}b}" for v in range(K)]


def gate_alphabet():
    """The 52 legal gate strings, in a fixed order."""
    gates = [f"x{k}" for k in range(N_BITS)]
    gates += [f"c{i}{j}" for i, j in permutations(range(N_BITS), 2)]
    gates += [f"s{i}{j}" for i, j in permutations(range(N_BITS), 2)]
    gates += [f"t{a}{b}{c}" for a, b, c in permutations(range(N_BITS), 3)]
    return gates


GATES = gate_alphabet()
M = len(GATES)


def true_table(g):
    """The exact permutation matrix T_g."""
    T = np.zeros((K, K))
    for s in range(K):
        nxt = int(_state_text(_apply_gate(_bits(s), g)), 2)
        T[s, nxt] = 1.0
    return T


TRUE = np.stack([true_table(g) for g in GATES])          # (M, K, K)
U = np.full((K, K), 1.0 / K)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def render(inst, fmt):
    if fmt == "direct":
        return inst.prompt, f"{ANSWER_SEP}{inst.gold}\n"
    return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n"


def build_dataset(instances, formats, block_size):
    """Supervised continuation positions only; prompt carries no loss."""
    width = block_size - 1
    x = torch.full((len(instances), width), TOK.pad_id, dtype=torch.uint8)
    y = torch.full((len(instances), width), TOK.pad_id, dtype=torch.uint8)
    mask = torch.zeros((len(instances), width), dtype=torch.bool)
    for row, (inst, fmt) in enumerate(zip(instances, formats)):
        prompt, target = render(inst, fmt)
        pids = TOK.encode(prompt)
        full = pids + TOK.encode(target)
        if len(full) > block_size:
            raise ValueError(f"{len(full)} tokens exceeds block_size={block_size}")
        seq = torch.tensor(full, dtype=torch.uint8)
        n = len(full) - 1
        x[row, :n] = seq[:-1]
        y[row, :n] = seq[1:]
        mask[row, len(pids) - 1:n] = True
    return torch.utils.data.TensorDataset(x, y, mask)


def sample_instances(sampler, n, seed, exclude=None):
    state = random.getstate()
    random.seed(seed)
    seen = set() if exclude is None else set(exclude)
    out, tries = [], 0
    while len(out) < n:
        inst = sampler()
        tries += 1
        if tries > 200 * n:
            raise RuntimeError(f"prompt space too small for {n} unique instances")
        if inst.prompt in seen:
            continue
        seen.add(inst.prompt)
        out.append(inst)
    random.setstate(state)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@torch.inference_mode()
def score_candidates(model, contexts, candidates, device, batch=1024):
    """Sum log P(candidate | context) over candidate tokens.

    Returns (len(contexts), len(candidates)) log-probabilities, unnormalized
    over the candidate set: the model may place mass outside it.
    """
    cand_ids = [TOK.encode(c) for c in candidates]
    L = len(cand_ids[0])
    assert all(len(c) == L for c in cand_ids)
    n_cand = len(cand_ids)
    out = np.zeros((len(contexts), n_cand))
    # Gate strings differ in width, so contexts differ in length.  Absolute
    # position embeddings make left padding unsafe; bucket by length instead.
    buckets = {}
    for ci, ctx in enumerate(contexts):
        buckets.setdefault(len(ctx), []).append(ci)
    for _, idxs in buckets.items():
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
    """Softmax over the candidate set; also return the mass that was on it."""
    m = logp.max(axis=1, keepdims=True)
    w = np.exp(logp - m)
    P = w / w.sum(axis=1, keepdims=True)
    on_set = np.exp(logp).sum(axis=1)
    return P, on_set


# --------------------------------------------------------------------------
# the induced rule
# --------------------------------------------------------------------------
def probe_contexts(depth, step, filler):
    """Contexts placing (s, g) at transition `step` with a gold prefix."""
    ctxs, index = [], []
    for gi, g in enumerate(GATES):
        for s in range(K):
            gates = [filler[(step - 1 + k) % len(filler)] for k in range(depth)]
            gates[step - 1] = g
            # walk a gold prefix backwards is not needed: choose s0 so the
            # state entering `step` equals s by construction.
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
            assert int(_state_text(st), 2) == s, "prefix must arrive at s"
            ctxs.append(f"{prompt} {body}{g}>")
            index.append((gi, s))
    return ctxs, index


def _inverse_gate(g):
    """x, c, s, t are all involutions on four bits."""
    for s in range(K):
        if int(_state_text(_apply_gate(_apply_gate(_bits(s), g), g)), 2) != s:
            return None
    return g


INVERSE = {g: _inverse_gate(g) for g in GATES}


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
    """P - U = 1 c^T + F, with F in the conditional subspace."""
    Mx = Phat - U[None]
    c = Mx.mean(axis=1)                       # (M, K) column means
    Fc = Mx - c[:, None, :]
    return c, Fc


def scales(Phat):
    c, Fc = split(Phat)
    gam = float(max(np.linalg.norm(c[g]) for g in range(M)))
    eps = float(max(np.linalg.norm(Fc[g], 2) for g in range(M)))
    return gam, eps


def rule_recovery(Phat):
    """Fraction of (g, s) pairs whose arg-max destination is the true one."""
    pred = Phat.argmax(axis=2)
    true = TRUE.argmax(axis=2)
    return float((pred == true).mean())


# --------------------------------------------------------------------------
# composition error and credit
# --------------------------------------------------------------------------
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


def composition(Phat, s0, gates):
    v = np.zeros(K); v[s0] = 1.0
    for g in gates:
        v = v @ Phat[GATES.index(g)]
    return v


@torch.inference_mode()
def terminal_distribution(model, instances, depth, device):
    ctxs = [inst.prompt + ANSWER_SEP for inst in instances]
    logp = score_candidates(model, ctxs, STATES, device)
    return renormalize(logp)


def delta_comp(model, instances, Phat, depth, device):
    """Model's own terminal answer law against the product of induced kernels."""
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
    """|| Rule(-grad_{P_t} l_out) ||_F on the induced tables.

    mode "actual"        the tables as read out (scale is ignored)
    mode "conditional"   P = U + scale * F, dropping the marginal part, so the
                         tables are doubly stochastic: exactly the hypothesis of
                         the factorization theorem's depth bound
    mode "proportional"  P = U + scale * (P - U), shrinking marginal and
                         conditional parts together, which is what the trained
                         model looks like on the way to uniform
    """
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
            q = fwd[t - 1]; b = bwd[t]
            out.append(np.linalg.norm(Pi @ q) * np.linalg.norm(Pi @ b) / p_y)
    return float(np.mean(out)), float(np.max(out))


def credit_exponent(Phat, instances, depth, mode,
                    lambdas=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)):
    """Rescale the model's own learned rule directions toward uniform.

    Scaling leaves the geometry intact and moves eps_rule exactly linearly, so a
    log-log fit of credit against eps_rule recovers the exponent the
    factorization theorem predicts.  Under "conditional" the tables stay doubly
    stochastic and the predicted slope is D - 1.  Under "proportional" the
    marginal part shrinks only as lambda, so the bound's marginal term dominates
    at late positions and the slope is expected to fall toward 1.  The gap
    between the two is the content of the marginal-control assumption.
    """
    c, Fc = split(Phat)
    base = float(max(np.linalg.norm(Fc[g], 2) for g in range(M)))
    if base <= 0:
        return dict(exponent_fit="", exponent_r2="", credit_by_lambda="", eps_by_lambda="")
    xs, ys = [], []
    for lam in lambdas:
        mean, _ = credit_norms(Phat, instances, depth, scale=lam, mode=mode)
        xs.append(lam * base); ys.append(mean)
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    resid = ly - (slope * lx + intercept)
    ss = 1.0 - float((resid ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30))
    return {f"exponent_fit_{mode}": float(slope),
            f"exponent_r2_{mode}": ss,
            f"credit_by_lambda_{mode}": json.dumps(
                {f"{l:.5g}": float(v) for l, v in zip(lambdas, ys)}),
            f"eps_by_lambda_{mode}": json.dumps(
                {f"{l:.5g}": float(v) for l, v in zip(lambdas, xs)})}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.inference_mode()
def free_running_accuracy(model, instances, depth, device, condition, batch=256):
    """Greedy generation from the prompt alone."""
    model.eval()
    ok = 0
    budget = 8 if condition == "outcome" else 10 * depth + 12
    buckets = {}
    for inst in instances:
        buckets.setdefault(len(inst.prompt), []).append(inst)
    for rows in buckets.values():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            ctx = torch.tensor([TOK.encode(inst.prompt) for inst in chunk],
                               dtype=torch.long, device=device)
            out = model.generate(ctx, max_new_tokens=budget,
                                 stop_id=TOK.newline_id, greedy=True)
            for row, inst in enumerate(chunk):
                text = TOK.decode(out[row, ctx.shape[1]:].tolist()).split("\n")[0]
                if ":" not in text:
                    continue
                ans = text.rsplit(":", 1)[1].strip()[:N_BITS]
                ok += int(ans == inst.gold)
    return ok / len(instances)


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
    if args.condition == "outcome":
        fmts = ["direct"] * len(train)
    elif args.condition == "process":
        fmts = ["trace"] * len(train)
    else:  # both formats, so trace and direct queries are both in distribution
        fmts = ["trace" if rng.random() < args.trace_fraction else "direct"
                for _ in train]

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
            Phat, on_set = induced_rules(model, args.depth, device, step=args.probe_step)
            gam, eps = scales(Phat)
            rec = rule_recovery(Phat)
            dc = delta_comp(model, probe, Phat, args.depth, device)
            cred_mean, cred_max = credit_norms(Phat, probe, args.depth)
            acc = free_running_accuracy(model, probe, args.depth, device, args.condition)
            if step == max(ckpts):
                expo = dict(exponent_target=args.depth - 1)
                for mode in ("conditional", "proportional"):
                    expo.update(credit_exponent(Phat, probe, args.depth, mode))
            else:
                expo = dict(exponent_target=args.depth - 1)
                for mode in ("conditional", "proportional"):
                    expo.update({f"exponent_fit_{mode}": "", f"exponent_r2_{mode}": "",
                                 f"credit_by_lambda_{mode}": "", f"eps_by_lambda_{mode}": ""})
            row = dict(condition=args.condition,
                       trace_fraction=args.trace_fraction,
                       depth=args.depth, seed=args.seed,
                       step=step, free_answer_acc=acc, gamma_hat=gam,
                       eps_rule_hat=eps, rule_recovery=rec,
                       credit_mean=cred_mean, credit_max=cred_max,
                       state_on_set_mass=on_set, **dc, **expo,
                       probe_seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.dump_tables and step == max(ckpts):
                np.save(args.dump_tables, Phat)
            model.train()
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--condition", default="both",
                   choices=["both", "outcome", "process"])
    p.add_argument("--trace-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--train-size", type=int, default=100_000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--probe-size", type=int, default=400)
    p.add_argument("--probe-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[0, 250, 500, 1000, 2000, 4000, 6000])
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--out", default="results/induced_rule.csv")
    p.add_argument("--dump-tables", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
