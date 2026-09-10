"""Projected simplex kernel flow (theory-matched diagnostic; not a T2 proof).

Distinct from `src/eval/escape_time.py`, which parameterizes P_g = softmax(A_g)
on unconstrained logits. Here each row of P_g lives on the probability simplex,
gradient steps are centered onto the row-tangent space, then Euclidean-projected
back to the simplex.

Tracks mixing radius r(t) = max_g ||P_g - U||_2 and a discrete Dini ratio
max(0, Delta r / Delta t) / r^{D-1}. That ratio is an empirical C-hat, not a theorem.
"""
from __future__ import annotations

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
    import itertools
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


def project_simplex(v):
    """Euclidean projection of the last axis onto {x ≥ 0, 1ᵀx = 1}."""
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1)
    rho = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    theta = (cssv - 1.0) / rho
    keep = u > theta
    k = keep.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = theta.gather(-1, k - 1)
    return (v - tau).clamp(min=0.0)


def row_tangent(grad):
    """Project a (M, K, K) Euclidean gradient onto {G : G 1 = 0} per row."""
    return grad - grad.mean(dim=-1, keepdim=True)


def mixing_radius(P):
    k = P.shape[-1]
    u = torch.full((k, k), 1.0 / k, device=P.device, dtype=P.dtype)
    r = torch.tensor(0.0, device=P.device, dtype=P.dtype)
    for g in range(P.shape[0]):
        r = torch.maximum(r, torch.linalg.matrix_norm(P[g] - u, ord=2))
    return float(r)


def outcome_loss(P, s0, gates, gold, k):
    q = F.one_hot(s0, k).to(dtype=P.dtype)
    for t in range(gates.shape[1]):
        q = torch.einsum("nk,nkl->nl", q, P[gates[:, t]])
    return -torch.log(q.gather(1, gold[:, None]).squeeze(1) + 1e-12).mean()


def process_loss(P, s0, gates, next_states):
    logp = torch.log(P.clamp(min=1e-12))
    loss = 0.0
    s = s0
    for t in range(gates.shape[1]):
        lp = logp[gates[:, t], s]
        gold = next_states[:, t]
        loss = loss + -lp.gather(1, gold[:, None]).squeeze(1).mean()
        s = gold
    return loss / gates.shape[1]


def init_near_uniform(m, k, eps, device):
    u = torch.full((m, k, k), 1.0 / k, device=device)
    noise = eps * torch.randn(m, k, k, device=device)
    return project_simplex(u + noise)


def projected_step(P, loss, lr):
    grad = torch.autograd.grad(loss, P, retain_graph=False, create_graph=False)[0]
    update = row_tangent(grad)
    with torch.no_grad():
        P2 = project_simplex(P.detach() - lr * update)
    return P2.requires_grad_(True), update


def run_projected_trial(depth, eps, seed, k=4, m=2, lr=0.05, max_steps=40, device="cpu"):
    """Euler projected flow of outcome CE. Returns per-step rows; not a T2 proof."""
    set_seed(1000 * seed + depth)
    g = torch.Generator().manual_seed(1000 * seed + depth)
    true_gates = true_perm(m, k, g).to(device)
    s0, gates = population(depth, k, m, seed)
    gold, next_states = rollout_gold(true_gates, s0, gates)
    s0, gates, gold, next_states = s0.to(device), gates.to(device), gold.to(device), next_states.to(device)
    torch.manual_seed(2000 * seed + 7 * depth)
    P = init_near_uniform(m, k, eps, device).requires_grad_(True)
    rows = []
    r_prev = mixing_radius(P.detach())
    for step in range(max_steps):
        L = outcome_loss(P, s0, gates, gold, k)
        Lp = process_loss(P.detach(), s0, gates, next_states)
        P, _ = projected_step(P, L, lr)
        r = mixing_radius(P.detach())
        dr = (r - r_prev) / max(lr, 1e-12)
        dplus = max(0.0, dr)
        denom = max(r_prev ** (depth - 1), 1e-12)
        rows.append(dict(
            parameterization="projected_simplex",
            not_a_proof=True,
            depth=depth, eps=eps, seed=seed, step=step, k=k, m=m, lr=lr,
            r=r, r_prev=r_prev, dini_plus=dplus, discrete_dr=dr,
            c_hat=dplus / denom, outcome_loss=float(L.detach()),
            process_loss=float(Lp.detach()),
        ))
        r_prev = r
    return rows


def run(args):
    from src.training.config import load_yaml
    cfg = load_yaml(args.config) if getattr(args, "config", None) else {}
    smoke = bool(getattr(args, "smoke", False))
    device = getattr(args, "device", None) or cfg.get("device") or "cpu"
    configure_device(device, compile=False)
    if smoke:
        depths, epss, seeds = [2], [0.3], [1]
        max_steps, k, m, lr = 20, 4, 2, 0.05
        out = Path(getattr(args, "out", None) or ROOT / "results/paper_revision_v2/e4_projected_kernel/smoke.csv")
    else:
        depths = list(cfg.get("depths", [2]))
        epss = list(cfg.get("eps", [0.3]))
        seeds = list(cfg.get("seeds", [1]))
        max_steps = int(cfg.get("max_steps", 30))
        k, m, lr = int(cfg.get("k", 4)), int(cfg.get("m", 2)), float(cfg.get("lr", 0.05))
        out = Path(getattr(args, "out", None) or cfg.get("out") or ROOT / "results/paper_revision_v2/e4_projected_kernel/smoke.csv")
    records = []
    for depth in depths:
        for eps in epss:
            for seed in seeds:
                rows = run_projected_trial(depth, eps, seed, k=k, m=m, lr=lr, max_steps=max_steps, device=device)
                records.extend(rows)
                print(json.dumps({"depth": depth, "eps": eps, "seed": seed,
                                  "r0": rows[0]["r"], "rT": rows[-1]["r"],
                                  "max_c_hat": max(r["c_hat"] for r in rows),
                                  "not_a_proof": True}), flush=True)
    write_csv(out, records)
    persist = out.with_name(out.stem + "_persist.json")
    persist.write_text(json.dumps({
        "experiment": "e4_projected_kernel",
        "parameterization": "projected_simplex",
        "not_a_proof": True,
        "csv": str(out),
        "n_rows": len(records),
        "note": "Discrete Dini ratios do not prove T2. See Paper/T2_CANDIDATE.md.",
    }, indent=2) + "\n")
    print("wrote", out)
    return records
