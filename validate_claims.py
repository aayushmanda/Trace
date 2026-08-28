#!/usr/bin/env python3
"""
validate_claims.py

Small claim-validation runner for aayushmanda/Trace.

Tests
-----
1. Local positive-margin fraction versus exact rollout.
2. Homogeneous/iid rollout prediction A_local^D and a union-bound lower bound.
3. Same-checkpoint prefix survival across rollout depth.
4. rho -> one-step margin drift at a fixed checkpoint using the SGD statement
   in the paper.

The script reuses mechanism_diagnostics.py for task serialization, datasets,
training construction, and free-running evaluation.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from mechanism_diagnostics import (
    SequenceDataset,
    build_model,
    build_teacher_queries,
    configure_runtime,
    evaluate_free,
    generate_unique,
    make_optimizer,
    nested_clean_flags,
    set_seed,
)
from src.registry import TASKS


@torch.inference_mode()
def transition_margins(model, task, instances, batch_size, device):
    """M_t = minimum correct-vs-best-wrong token margin for a successor state."""
    _, queries = build_teacher_queries(task, instances)
    depth = len(instances[0].correct_trace.split())
    groups = defaultdict(list)
    for t, context, target in queries:
        groups[(len(context), len(target))].append((t, context, target))

    margins_by_step = [[] for _ in range(depth)]
    model.eval()

    for (context_len, target_len), rows in groups.items():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            context = torch.tensor([r[1] for r in batch], dtype=torch.long, device=device)
            target = torch.tensor([r[2] for r in batch], dtype=torch.long, device=device)

            seq = torch.cat([context, target], dim=1)
            logits, _ = model(seq[:, :-1])
            z = logits[:, context_len - 1:context_len - 1 + target_len]

            correct = z.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            wrong = z.clone()
            wrong.scatter_(-1, target.unsqueeze(-1), float("-inf"))
            token_margin = correct - wrong.max(dim=-1).values
            state_margin = token_margin.min(dim=1).values.cpu().numpy()

            for i, (t, _, _) in enumerate(batch):
                margins_by_step[t].append(float(state_margin[i]))

    per_step_positive = np.asarray(
        [np.mean(np.asarray(m) > 0) for m in margins_by_step], dtype=float
    )
    all_margins = np.concatenate([np.asarray(m, dtype=float) for m in margins_by_step])
    return all_margins, per_step_positive


def empirical_survival(free):
    """Recover empirical P(E_t) and epsilon_t from first-failure frequencies."""
    first = np.asarray(free["first_error_rate"], dtype=float)
    survival = 1.0 - np.cumsum(first)

    eps = np.full_like(survival, np.nan)
    prev = 1.0
    for t in range(len(survival)):
        eps[t] = first[t] / prev if prev > 0 else np.nan
        prev = survival[t]

    product = np.cumprod(np.nan_to_num(1.0 - eps, nan=1.0))
    return survival, eps, product


def train_one(task, rho, seed, train_instances, val_instances, assignment_order, args, device):
    flags = nested_clean_flags(len(train_instances), rho, assignment_order)
    dataset = SequenceDataset(train_instances, task, "mixed", clean_flags=flags)

    generator = torch.Generator().manual_seed(args.batch_seed + seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)

    cfg = SimpleNamespace(
        embedding=args.embedding,
        heads=args.heads,
        layers=args.layers,
        dropout=0.0,
        lr=args.lr,
        weight_decay=0.0,
        deterministic=args.deterministic,
    )

    set_seed(seed)
    model = build_model(task, cfg, device)
    optimizer = make_optimizer(model, cfg, device)

    probe_step = args.probe_step if args.probe_step > 0 else max(1, args.steps // 2)
    probe_state = None

    for step in range(1, args.steps + 1):
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)

        model.train()
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        mask = mask.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if abs(rho - args.probe_base_rho) < 1e-12 and step == probe_step:
            probe_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    free = evaluate_free(
        model, task, val_instances, False, args.eval_batch_size, device
    )
    margin_instances = val_instances[: min(args.margin_size, len(val_instances))]
    margins, per_step_positive = transition_margins(
        model, task, margin_instances, args.eval_batch_size, device
    )

    local_positive = float(np.mean(margins > 0))
    depth = len(per_step_positive)
    survival, eps, product = empirical_survival(free)

    row = {
        "task": task.name,
        "rho": rho,
        "seed": seed,
        "depth": depth,
        "local_positive_margin": local_positive,
        "observed_exact": float(free["exact_trace_accuracy"]),
        "iid_prediction": float(local_positive ** depth),
        "stepwise_prediction": float(np.prod(per_step_positive)),
        "union_lower_bound": float(max(0.0, 1.0 - depth * (1.0 - local_positive))),
        "survival_product_error": float(np.max(np.abs(survival - product))),
        "mean_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
    }

    for d in [1, 2, 4, 8, 12, 16, 20]:
        if d <= depth:
            row[f"survival_d{d}"] = float(survival[d - 1])

    del dataset, loader, optimizer
    return model, probe_state, row


def one_step_margin_probe(task, checkpoint, probe_instances, eval_instances, args, device):
    """
    Hold theta fixed, vary clean/corrupt minibatch composition, take one SGD step,
    and measure the change in held-out clean margin.
    """
    cfg = SimpleNamespace(
        embedding=args.embedding,
        heads=args.heads,
        layers=args.layers,
        dropout=0.0,
        lr=args.lr,
        weight_decay=0.0,
        deterministic=args.deterministic,
    )

    clean = SequenceDataset(probe_instances, task, "process")
    corrupt = SequenceDataset(probe_instances, task, "corrupted")
    B = min(args.probe_batch_size, len(probe_instances))
    order = np.random.default_rng(args.assignment_seed).permutation(B)

    baseline_model = build_model(task, cfg, device)
    baseline_model.load_state_dict(checkpoint)
    before, _ = transition_margins(
        baseline_model, task, eval_instances, args.eval_batch_size, device
    )
    before_mean = float(before.mean())
    del baseline_model

    rows = []
    for rho in args.probe_rhos:
        model = build_model(task, cfg, device)
        model.load_state_dict(checkpoint)

        n_clean = round(rho * B)
        clean_rows = order[:n_clean]

        x = corrupt.x[:B].clone()
        y = corrupt.y[:B].clone()
        mask = corrupt.mask[:B].clone()

        if len(clean_rows):
            idx = torch.tensor(clean_rows, dtype=torch.long)
            x[idx] = clean.x[idx]
            y[idx] = clean.y[idx]
            mask[idx] = clean.mask[idx]

        x = x.to(device, dtype=torch.long)
        y = y.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.float32)

        # Proposition 3 is stated for SGD, so use one small SGD probe step.
        optimizer = torch.optim.SGD(model.parameters(), lr=args.probe_lr)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, mask=mask)
        loss.backward()
        optimizer.step()

        after, _ = transition_margins(
            model, task, eval_instances, args.eval_batch_size, device
        )
        after_mean = float(after.mean())

        rows.append({
            "rho": rho,
            "before_margin": before_mean,
            "after_margin": after_mean,
            "delta_margin": after_mean - before_mean,
            "probe_loss": float(loss.detach()),
        })
        del model, optimizer

    x = np.asarray([r["rho"] for r in rows], dtype=float)
    y = np.asarray([r["delta_margin"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - yhat) ** 2) / max(denom, 1e-30)
    return rows, float(slope), float(r2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--seed", type=int, default=2001)

    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--train-size", type=int, default=100000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--margin-size", type=int, default=256)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)

    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--assignment-seed", type=int, default=777)
    p.add_argument("--batch-seed", type=int, default=12345)

    p.add_argument("--probe-base-rho", type=float, default=0.8)
    p.add_argument("--probe-step", type=int, default=0, help="0 means half of --steps")
    p.add_argument("--probe-rhos", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-lr", type=float, default=1e-4)

    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, default=Path("results/claim_validation.csv"))
    args = p.parse_args()

    if args.task not in TASKS:
        raise SystemExit(f"Unknown task: {args.task}")
    if args.probe_base_rho not in args.rhos:
        raise SystemExit("--probe-base-rho must be included in --rhos")

    configure_runtime(args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TASKS[args.task]

    train = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {inst.prompt for inst in train}
    val = generate_unique(task, args.val_size, args.val_seed, train_prompts)
    assignment = np.random.default_rng(args.assignment_seed).permutation(len(train))

    rows = []
    probe_checkpoint = None

    for rho in args.rhos:
        print(f"\n[{task.name}] rho={rho:.2f}, seed={args.seed}, steps={args.steps}")

        model, checkpoint, row = train_one(
            task, rho, args.seed, train, val, assignment, args, device
        )
        rows.append(row)

        if checkpoint is not None:
            probe_checkpoint = checkpoint

        print(
            f"local+={row['local_positive_margin']:.4f}  "
            f"exact={row['observed_exact']:.4f}  "
            f"iid={row['iid_prediction']:.4f}  "
            f"step-prod={row['stepwise_prediction']:.4f}  "
            f"survival-id-error={row['survival_product_error']:.2e}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved rollout validation: {args.out}")

    if probe_checkpoint is not None:
        probe_train = train[: max(args.probe_batch_size, 256)]
        probe_eval = val[: min(args.margin_size, len(val))]

        drift, slope, r2 = one_step_margin_probe(
            task, probe_checkpoint, probe_train, probe_eval, args, device
        )

        drift_path = args.out.with_name(args.out.stem + "_margin_drift.csv")
        with drift_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(drift[0].keys()))
            writer.writeheader()
            writer.writerows(drift)

        print(f"Saved margin-drift probe: {drift_path}")
        print(f"rho -> Delta margin slope = {slope:.6g}, R^2 = {r2:.4f}")

        for row in drift:
            print(
                f"  rho={row['rho']:.2f}: "
                f"Delta margin={row['delta_margin']:+.6e}"
            )


if __name__ == "__main__":
    main()