"""Shared trainable kernel executor: one A ∈ R^{M×K×K}, P_g = softmax(A_g).

The same parameters define outcome composition CE and process local CE.
Init near uniform (scale ε), vary ε and D; report projected true-rule gradient,
escape time, and success vs D.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training.io import write_csv
from src.training.seed import configure_device, set_seed

ROOT = Path(__file__).resolve().parents[2]


def true_perm(m, k, generator):
    return torch.stack([torch.randperm(k, generator=generator) for _ in range(m)])


def population(depth, k, m, seed, cap=2048):
    total = k * m ** depth
    g = torch.Generator().manual_seed(9999 + seed)
    if total <= cap:
        s0 = torch.arange(k).repeat_interleave(m ** depth)
        gates = torch.tensor(list(itertools.product(range(m), repeat=depth))).repeat(k, 1)
        return s0, gates
    s0 = torch.randint(0, k, (cap,), generator=g)
    gates = torch.randint(0, m, (cap, depth), generator=g)
    return s0, gates


def rollout_gold(true_gates, s0, gates):
    s = s0.clone()
    nxt = []
    for t in range(gates.shape[1]):
        s = true_gates[gates[:, t], s]
        nxt.append(s.clone())
    return s, torch.stack(nxt, dim=1)


def process_loss(A, s0, gates, next_states):
    logp = F.log_softmax(A, dim=-1)
    loss = 0.0
    s = s0
    for t in range(gates.shape[1]):
        lp = logp[gates[:, t], s]
        gold = next_states[:, t]
        loss = loss + -lp.gather(1, gold[:, None]).squeeze(1).mean()
        s = gold
    return loss / gates.shape[1]


def outcome_loss(A, s0, gates, gold, k):
    P = torch.softmax(A, dim=-1)
    q = F.one_hot(s0, k).to(dtype=P.dtype)
    for t in range(gates.shape[1]):
        q = torch.einsum("nk,nkl->nl", q, P[gates[:, t]])
    return -torch.log(q.gather(1, gold[:, None]).squeeze(1) + 1e-12).mean()


def rule_direction(true_gates, k):
    m = true_gates.shape[0]
    W = torch.zeros(m, k, k)
    for g in range(m):
        for s in range(k):
            W[g, s, int(true_gates[g, s])] = 1.0
            W[g, s] -= 1.0 / k
    return W


def projected_grad(A, loss, W):
    A.grad = None
    loss.backward(retain_graph=True)
    return float((A.grad * W).sum())


def recovery_pct(A, true_gates):
    return 100.0 * (A.argmax(dim=-1) == true_gates).float().mean().item()


def run_shared_trial(depth, eps, seed, k=8, m=4, lr=80.0, max_steps=2000, device="cpu"):
    set_seed(1000 * seed + depth)
    g = torch.Generator().manual_seed(1000 * seed + depth)
    true_gates = true_perm(m, k, g)
    s0, gates = population(depth, k, m, seed)
    gold, next_states = rollout_gold(true_gates, s0, gates)
    W = rule_direction(true_gates, k).to(device)
    s0, gates, gold, next_states = s0.to(device), gates.to(device), gold.to(device), next_states.to(device)
    true_gates = true_gates.to(device)

    def init_A():
        torch.manual_seed(2000 * seed + 7 * depth)
        return (eps * torch.randn(m, k, k, device=device)).requires_grad_(True)

    rows = {}
    for name, loss_fn in (
        ("outcome", lambda A: outcome_loss(A, s0, gates, gold, k)),
        ("process", lambda A: process_loss(A, s0, gates, next_states)),
    ):
        A = init_A()
        L = loss_fn(A)
        proj = projected_grad(A, L, W)
        A.grad = None
        opt = torch.optim.SGD([A], lr=lr)
        chance = 100.0 / k
        escape, censored = max_steps, True
        rec = recovery_pct(A.detach(), true_gates)
        for step in range(1, max_steps + 1):
            opt.zero_grad(set_to_none=True)
            loss_fn(A).backward()
            opt.step()
            if step <= 50 or step % 25 == 0:
                rec = recovery_pct(A.detach(), true_gates)
                if rec > chance + 5.0:
                    escape, censored = step, False
                    break
        rows[name] = dict(
            depth=depth, eps=eps, seed=seed, loss=name, k=k, m=m,
            proj_true_rule=proj, escape_step=escape, censored=censored,
            success=not censored, final_recovery=rec,
        )
    return rows["outcome"], rows["process"]


def run(args):
    from src.training.config import load_yaml
    cfg = load_yaml(args.config) if getattr(args, "config", None) else {}
    smoke = bool(getattr(args, "smoke", False))
    device = getattr(args, "device", None) or cfg.get("device") or "cpu"
    configure_device(device, compile=False)
    if smoke:
        depths = [2, 3]
        epss = [0.4]
        seeds = [1]
        max_steps = 40
        k, m, lr = 8, 4, 80.0
    else:
        depths = list(cfg.get("depths", [2, 3, 4, 6]))
        epss = list(cfg.get("eps", [0.6, 0.4, 0.28, 0.2]))
        seeds = list(cfg.get("seeds", [1, 2, 3]))
        max_steps = int(cfg.get("max_steps", 4000))
        k, m, lr = int(cfg.get("k", 8)), int(cfg.get("m", 4)), float(cfg.get("lr", 80.0))
    out = Path(getattr(args, "out", None) or cfg.get("out") or ROOT / "results/revision/shared_kernel.csv")
    records = []
    for depth in depths:
        for eps in epss:
            for seed in seeds:
                o, p = run_shared_trial(depth, eps, seed, k=k, m=m, lr=lr, max_steps=max_steps, device=device)
                records.extend([o, p])
                print(json.dumps({"outcome": o, "process": p}), flush=True)
    write_csv(out, records)
    print("wrote", out)
    return records
