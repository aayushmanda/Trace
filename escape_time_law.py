import itertools
import json
import os
from pathlib import Path
import numpy as np
import torch

# This machine has few cores and may be shared with other jobs; these tensors
# are tiny (a handful of KxK matrices), so extra threads only add contention,
# not speed.
torch.set_num_threads(min(2, os.cpu_count() or 1))

K, M = 8, 4  # states, gates -- smaller than the paper's K=16, M=52 purely so
# the population K*M^D stays cheap to materialize on a shared, CPU-only,
# 4-core machine; M*K = 32 gate-state cells still gives a recovery statistic
# stable enough that initialization noise alone cannot trigger a false
# "escape" (chance = 100/8 = 12.5%), and the persistence check below guards
# the rest of the way.
MAX_POPULATION = 4096  # cap on rows materialized per step (see full_population)

# Per depth, the eps grid and step budget are chosen so the *worst-case*
# predicted exponent (D-2) at the smallest eps in the grid stays inside a few
# thousand updates: at D=3,4 this is true of the same wide grid down to
# eps=0.07 (worst case (1/0.07)^2 ~ 200 updates); at D=6 the worst case is
# quartic, so we use a narrower grid with a higher floor.
EPS_GRIDS = {
    3: [0.60, 0.40, 0.28, 0.20, 0.14, 0.10, 0.07],
    4: [0.60, 0.40, 0.28, 0.20, 0.14, 0.10, 0.07],
    6: [0.55, 0.40, 0.30, 0.22, 0.16],
}
MAX_STEPS = {3: 20_000, 4: 20_000, 6: 10_000}  # safety caps
DEPTHS = [3, 4, 6]
SEEDS = [1, 2, 3]
LR = 80.0  # plain SGD on the population loss, not Adam -- see module docstring.
# This is a large learning rate, deliberately: the population loss is exact
# (noiseless) every step, so there is nothing to overshoot into except the
# loss surface's own curvature, and a small LR would spend most of the
# budget re-deriving the eps^(D-1) prefactor of Theorem 3 rather than
# measuring the eps-scaling of escape time. LR=250 was tried and rejected:
# it occasionally let D=6 "escape" in a single step, i.e. it was large
# enough to jump straight past the very trapping region the experiment
# measures.
CHANCE_MARGIN = 5.0  # gate recovery must exceed chance + this many points


def full_population(depth: int, seed: int):
    """Enumerate every (start state, gate sequence) pair exactly, unless that
    population is too large to materialize cheaply every step (D=6: K*M^D =
    24,576 rows), in which case we draw one large fixed subsample and reuse
    it, unchanged, at every step -- still zero sampling noise across steps,
    just a coarser (but fixed) population."""
    total = K * M**depth
    if total <= MAX_POPULATION:
        s0 = torch.arange(K).repeat_interleave(M**depth)
        gate_idx = torch.tensor(list(itertools.product(range(M), repeat=depth))).repeat(K, 1)
        return s0, gate_idx
    g = torch.Generator().manual_seed(9999 + seed)
    s0 = torch.randint(0, K, (MAX_POPULATION,), generator=g)
    gate_idx = torch.randint(0, M, (MAX_POPULATION, depth), generator=g)
    return s0, gate_idx


def conditional_marginal_split(E: torch.Tensor):
    """E: (M,K,K) with E.sum(-1) == 0 (row sums zero, Lemma 3). Returns (F, c):
    F is the doubly-centered conditional part (in C), c the marginal part (in R)."""
    c = E.mean(dim=-2)  # (M,K): column means -- c^T = (1/K) 1^T X per gate
    F = E - c.unsqueeze(-2)
    return F, c


def gate_recovery_pct(logits: torch.Tensor, true_gates: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)  # (M,K)
    return 100.0 * (pred == true_gates).float().mean().item()


def run_trial(depth: int, eps: float, seed: int):
    g = torch.Generator().manual_seed(1000 * seed + depth)
    true_gates = torch.stack([torch.randperm(K, generator=g) for _ in range(M)])  # (M,K)

    torch.manual_seed(2000 * seed + 7 * depth)
    logits = (eps * torch.randn(M, K, K)).requires_grad_(True)

    s0, gate_idx = full_population(depth, seed)
    with torch.no_grad():
        s = s0.clone()
        for t in range(depth):
            s = true_gates[gate_idx[:, t], s]
        gold = s  # (N,) terminal state for every population row, computed once

    opt = torch.optim.SGD([logits], lr=LR)
    chance = 100.0 / K
    phi_trace, gamma_trace = [], []
    max_steps = MAX_STEPS[depth]
    pending_escape_step = None  # candidate escape step awaiting confirmation at the next check

    for step in range(1, max_steps + 1):
        P = torch.softmax(logits, dim=-1)  # (M,K,K)
        q = torch.eye(K)[s0]  # (N,K)
        for t in range(depth):
            q = torch.einsum("nk,nkl->nl", q, P[gate_idx[:, t]])
        loss = -torch.log(q.gather(1, gold[:, None]).squeeze(1) + 1e-12).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step <= 200 or step % 50 == 0:
            with torch.no_grad():
                U = torch.full((K, K), 1.0 / K)
                E = torch.softmax(logits, dim=-1) - U
                F, c = conditional_marginal_split(E)
                phi = torch.linalg.matrix_norm(F, ord=2).max().item()
                gamma = c.norm(dim=-1).max().item()
                phi_trace.append((step, phi))
                gamma_trace.append((step, gamma))
                recovery = gate_recovery_pct(logits.detach(), true_gates)

            above = recovery > chance + CHANCE_MARGIN
            if above and pending_escape_step is not None:
                # confirmed at two consecutive checks: not an init-noise fluke
                return {
                    "depth": depth, "eps": eps, "seed": seed,
                    "escape_step": pending_escape_step, "censored": False,
                    "phi0": phi_trace[0][1], "phi_at_escape": phi,
                    "gamma_at_escape": gamma,
                }
            pending_escape_step = step if above else None

    with torch.no_grad():
        recovery = gate_recovery_pct(logits.detach(), true_gates)
    return {
        "depth": depth, "eps": eps, "seed": seed,
        "escape_step": max_steps, "censored": True,
        "phi0": phi_trace[0][1], "phi_at_escape": phi_trace[-1][1],
        "gamma_at_escape": gamma_trace[-1][1],
        "final_recovery": recovery,
    }


def fit_exponent(rows):
    """log(escape) ~ a * log(1/eps) + b, using only non-censored trials."""
    xs = np.log(1.0 / np.array([r["eps"] for r in rows]))
    ys = np.log(np.array([r["escape_step"] for r in rows]))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def main():
    results = []
    print(f"{'D':>2} {'eps':>6} {'seed':>4} {'escape':>9} {'censored':>9} {'phi0':>7} {'gamma@esc':>10}")
    for depth in DEPTHS:
        for eps in EPS_GRIDS[depth]:
            for seed in SEEDS:
                r = run_trial(depth, eps, seed)
                results.append(r)
                print(
                    f"{depth:2d} {eps:6.3f} {seed:4d} {r['escape_step']:9d} "
                    f"{str(r['censored']):>9} {r['phi0']:7.4f} {r['gamma_at_escape']:10.4f}"
                )

    summary = {}
    print("\n--- fitted escape-time exponents a(D):  T(eps) ~ eps^-a(D) ---")
    print(f"{'D':>2} {'a_hat':>8} {'(D-2)/2':>9} {'D-2':>6} {'n_escaped':>10} {'n_censored':>11}")
    for depth in DEPTHS:
        rows = [r for r in results if r["depth"] == depth]
        escaped = [r for r in rows if not r["censored"]]
        censored = [r for r in rows if r["censored"]]
        if len(escaped) >= 2:
            slope, intercept = fit_exponent(escaped)
        else:
            slope, intercept = float("nan"), float("nan")
        lo, hi = (depth - 2) / 2, depth - 2
        summary[depth] = {
            "a_hat": slope, "lower_bound": lo, "upper_bound": hi,
            "n_escaped": len(escaped), "n_censored": len(censored),
        }
        print(f"{depth:2d} {slope:8.3f} {lo:9.3f} {hi:6.3f} {len(escaped):10d} {len(censored):11d}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "escape_time_law.json", "w") as f:
        json.dump({"rows": results, "summary": summary}, f, indent=2)
    print(f"\nsaved: {out_dir / 'escape_time_law.json'}")

    slopes = [summary[d]["a_hat"] for d in DEPTHS if not np.isnan(summary[d]["a_hat"])]
    if len(slopes) >= 2:
        depths_used = [d for d in DEPTHS if not np.isnan(summary[d]["a_hat"])]
        growth_slope = np.polyfit(depths_used, slopes, 1)[0]
        print(f"a(D) vs D linear slope = {growth_slope:.3f}  (mechanism predicts > 0; a slope near 0 would falsify it)")


if __name__ == "__main__":
    main()
