
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.boolean_circuit_tasks import _apply_gate
from src.model import GPTModel
from src.tokenizer import CharTokenizer


STATE_SYMBOLS = "ABCDEFGHIJKLMNOP"
TARGETED_RHOS = {
    1: [0.00, 0.35, 0.45, 0.49, 0.50, 0.51, 0.55, 0.65, 1.00],
    3: [0.00, 0.15, 0.22, 0.24, 0.25, 0.26, 0.28, 0.35, 1.00],
    15: [0.00, 0.03, 0.05, 0.06, 0.0625, 0.065, 0.075, 0.10, 1.00],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_determinism(allow_nondeterministic: bool) -> None:
    if allow_nondeterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def all_boolean_gates() -> list[str]:
    gates = [f"x{i}" for i in range(4)]
    gates += [f"c{i}{j}" for i in range(4) for j in range(4) if i != j]
    gates += [f"s{i}{j}" for i in range(4) for j in range(4) if i != j]
    gates += [f"t{i}{j}{k}" for i, j, k in itertools.permutations(range(4), 3)]
    return gates


def int_to_bits(value: int) -> list[int]:
    return [int(bit) for bit in f"{value:04b}"]


def bits_to_int(bits: list[int]) -> int:
    return int("".join(map(str, bits)), 2)


def state_symbol(value: int) -> str:
    return STATE_SYMBOLS[value]


def next_state_value(state: int, gate: str) -> int:
    return bits_to_int(_apply_gate(int_to_bits(state), gate))


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer(STATE_SYMBOLS + "xcst0123;>")


class LocalExampleDataset(Dataset):
    def __init__(self, contexts: list[str], labels: list[str], tokenizer: CharTokenizer):
        self.encoded = [torch.tensor(tokenizer.encode(z), dtype=torch.long) for z in contexts]
        self.target_ids = torch.tensor([tokenizer.stoi[y] for y in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int):
        return self.encoded[index], self.target_ids[index]


def collate_local(batch, pad_id: int):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    padded = torch.full((len(xs), int(lengths.max())), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    return padded, lengths, torch.stack(ys)


def build_base_contexts(max_contexts: int, context_seed: int):
    rows = []
    for state in range(16):
        for gate in all_boolean_gates():
            rows.append((f"{state_symbol(state)};{gate}>", next_state_value(state, gate)))
    rng = random.Random(context_seed)
    rng.shuffle(rows)
    if max_contexts > 0:
        rows = rows[: min(max_contexts, len(rows))]
    return rows


def support_shifts(support_seed: int) -> list[int]:
    """Return one global ordering of 15 fixed-point-free class shifts."""
    shifts = list(range(1, 16))
    random.Random(support_seed).shuffle(shifts)
    return shifts


def wrong_support(clean_value: int, support_size: int, support_seed: int) -> list[int]:
    """Construct nested, equally structured wrong-target supports.

    Each alternative is a global cyclic permutation of the valid class. Taking
    prefixes of one shift ordering ensures W_1(z) subset W_3(z) subset W_15(z).
    """
    if not 1 <= support_size <= 15:
        raise ValueError("support_size must be in [1, 15]")
    shifts = support_shifts(support_seed)[:support_size]
    return [(clean_value + shift) % 16 for shift in shifts]


def build_condition(
    base_contexts,
    rho: float,
    support_size: int,
    repeats: int,
    assignment_seed: int,
    support_seed: int,
):
    """Build exact nested clean assignments with balanced corrupted targets.

    Each context has one fixed permutation of its replica indices. The first
    round(rho * repeats) ranks are clean. Remaining ranks cycle through the
    nested K-target support, so corrupted counts differ by at most one and are
    exactly uniform whenever their total is divisible by K.
    """
    contexts, labels, context_ids = [], [], []
    counts = [defaultdict(int) for _ in base_contexts]

    for cid, (z, clean_value) in enumerate(base_contexts):
        support = wrong_support(clean_value, support_size, support_seed)
        order = list(range(repeats))
        random.Random(assignment_seed + 1_000_003 * cid).shuffle(order)
        n_clean = round(rho * repeats)
        values = [None] * repeats

        for rank, rep in enumerate(order):
            values[rep] = clean_value if rank < n_clean else support[rank % support_size]

        for rep, value in enumerate(values):
            contexts.append(z)
            labels.append(state_symbol(value))
            context_ids.append(cid)
            counts[cid][value] += 1

    return contexts, labels, context_ids, counts


def build_model(tokenizer: CharTokenizer, args, device: torch.device):
    max_context_len = max(
        len(f"{state_symbol(state)};{gate}>")
        for state in range(16)
        for gate in all_boolean_gates()
    )
    return GPTModel(
        vocab_size=tokenizer.vocab_size,
        block_size=max_context_len + 1,
        pad_id=tokenizer.pad_id,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        dropout=args.dropout,
    ).to(device)


def autocast_context(device: torch.device, precision: str):
    use_bf16 = precision == "bf16" or (
        precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), "bf16"
    return nullcontext(), "fp32"


def train_one(base_contexts, rho, support_size, run_seed, tokenizer, args, device):
    assignment_seed = args.assignment_seed + run_seed
    support_seed = args.support_seed + run_seed
    batch_seed = args.batch_seed + run_seed
    contexts, labels, _, counts = build_condition(
        base_contexts=base_contexts,
        rho=rho,
        support_size=support_size,
        repeats=args.repeats,
        assignment_seed=assignment_seed,
        support_seed=support_seed,
    )

    dataset = LocalExampleDataset(contexts, labels, tokenizer)
    generator = torch.Generator().manual_seed(batch_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=partial(collate_local, pad_id=tokenizer.pad_id),
    )

    set_seed(run_seed)
    model = build_model(tokenizer, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    iterator = iter(loader)
    losses = []
    actual_precision = None
    model.train()

    for step in range(args.steps):
        try:
            x, lengths, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, lengths, y = next(iterator)

        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        amp_context, actual_precision = autocast_context(device, args.precision)
        with amp_context:
            logits, _ = model(x)
            last_logits = logits[torch.arange(x.size(0), device=device), lengths - 1]
            loss = F.cross_entropy(last_logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))

        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            tail = float(np.mean(losses[-min(100, len(losses)) :]))
            print(
                f"K={support_size:2d} rho={rho:7.4f} seed={run_seed} "
                f"step={step + 1:5d}/{args.steps} loss={tail:.5f}"
            )

    seeds = {
        "run_seed": run_seed,
        "support_seed_actual": support_seed,
        "assignment_seed_actual": assignment_seed,
        "batch_seed_actual": batch_seed,
    }
    train_loss = float(np.mean(losses[-min(100, len(losses)) :]))
    return model, counts, train_loss, actual_precision, seeds


@torch.inference_mode()
def evaluate_condition(model, base_contexts, counts, tokenizer, args, device):
    model.eval()
    state_ids = [tokenizer.stoi[symbol] for symbol in STATE_SYMBOLS]

    clean_correct = 0
    state_restricted_clean_correct = 0
    empirical_target_clean = 0
    model_correct_given_target_clean = 0
    wrong_when_target_clean = 0
    barrier_violations = 0
    non_state_predictions = 0

    state_margins, all_vocab_margins, clean_nlls = [], [], []
    kls, barriers, wrong_gaps = [], [], []
    a_values, b_values, qmax_values, realized_rhos = [], [], [], []

    for start in range(0, len(base_contexts), args.eval_batch_size):
        batch = base_contexts[start : start + args.eval_batch_size]
        encoded = [torch.tensor(tokenizer.encode(z), dtype=torch.long) for z, _ in batch]
        lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
        x = torch.full((len(encoded), int(lengths.max())), tokenizer.pad_id, dtype=torch.long)
        for i, ids in enumerate(encoded):
            x[i, : len(ids)] = ids

        x = x.to(device)
        lengths_dev = lengths.to(device)
        logits, _ = model(x)
        last_logits = logits[
            torch.arange(x.size(0), device=device), lengths_dev - 1
        ].float().cpu()
        log_probs = torch.log_softmax(last_logits, dim=-1)
        full_preds = torch.argmax(last_logits, dim=-1)
        state_preds = torch.tensor(state_ids)[
            torch.argmax(last_logits[:, state_ids], dim=-1)
        ]

        for j, ((_, clean_value), logit_row, log_r, pred, state_pred) in enumerate(
            zip(batch, last_logits, log_probs, full_preds, state_preds)
        ):
            cid = start + j
            clean_id = tokenizer.stoi[state_symbol(clean_value)]
            is_clean = int(pred) == clean_id
            is_state_clean = int(state_pred) == clean_id
            clean_correct += int(is_clean)
            state_restricted_clean_correct += int(is_state_clean)
            non_state_predictions += int(int(pred) not in state_ids)

            wrong_state_ids = [state_id for state_id in state_ids if state_id != clean_id]
            state_margin = float(logit_row[clean_id] - logit_row[wrong_state_ids].max())
            all_wrong = logit_row.clone()
            all_wrong[clean_id] = -torch.inf
            all_vocab_margin = float(logit_row[clean_id] - all_wrong.max())
            state_margins.append(state_margin)
            all_vocab_margins.append(all_vocab_margin)
            clean_nlls.append(float(-log_r[clean_id]))

            total = sum(counts[cid].values())
            p = torch.zeros(tokenizer.vocab_size, dtype=torch.float64)
            for value, count in counts[cid].items():
                p[tokenizer.stoi[state_symbol(value)]] = count / total

            a = float(p[clean_id])
            b = max(float(p[state_id]) for state_id in wrong_state_ids)
            corrupt_mass = 1.0 - a
            qmax = b / corrupt_mass if corrupt_mass > 0 else 0.0
            a_values.append(a)
            b_values.append(b)
            qmax_values.append(qmax)
            realized_rhos.append(a)

            support_mask = p > 0
            kl = float(
                (
                    p[support_mask]
                    * (torch.log(p[support_mask]) - log_r.double()[support_mask])
                ).sum()
            )
            kls.append(kl)

            if a > b:
                empirical_target_clean += 1
                model_correct_given_target_clean += int(is_clean)
                if b == 0.0:
                    barrier = a * math.log(2.0)
                else:
                    barrier = a * math.log((2.0 * a) / (a + b)) + b * math.log(
                        (2.0 * b) / (a + b)
                    )
                barriers.append(barrier)

                if not is_clean:
                    wrong_when_target_clean += 1
                    wrong_gaps.append(kl - barrier)
                    if kl + args.barrier_tol < barrier:
                        barrier_violations += 1

    n = len(base_contexts)
    margins = np.asarray(state_margins, dtype=np.float64)
    return {
        "clean_accuracy": clean_correct / n,
        "state_restricted_clean_accuracy": state_restricted_clean_correct / n,
        "positive_state_margin_fraction": float(np.mean(margins > 0)),
        "mean_state_margin": float(np.mean(margins)),
        "median_state_margin": float(np.median(margins)),
        "std_state_margin": float(np.std(margins, ddof=1)) if n > 1 else 0.0,
        "mean_all_vocab_margin": float(np.mean(all_vocab_margins)),
        "mean_clean_nll": float(np.mean(clean_nlls)),
        "non_state_prediction_rate": non_state_predictions / n,
        "empirical_target_clean_fraction": empirical_target_clean / n,
        "clean_accuracy_given_empirical_target_clean": (
            model_correct_given_target_clean / empirical_target_clean
            if empirical_target_clean
            else float("nan")
        ),
        "wrong_when_empirical_target_clean": wrong_when_target_clean,
        "mean_kl_to_empirical_target": float(np.mean(kls)),
        "mean_decision_barrier": float(np.mean(barriers)) if barriers else float("nan"),
        "min_wrong_kl_minus_barrier": float(min(wrong_gaps)) if wrong_gaps else float("nan"),
        "barrier_violations": barrier_violations,
        "mean_realized_rho": float(np.mean(realized_rhos)),
        "mean_empirical_qmax": float(np.mean(qmax_values)),
        "max_empirical_qmax": float(np.max(qmax_values)),
        "mean_target_clean_mass": float(np.mean(a_values)),
        "mean_strongest_wrong_mass": float(np.mean(b_values)),
    }


def configuration_id(args, context_count: int) -> str:
    payload = {
        "contexts": context_count,
        "repeats": args.repeats,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "embedding": args.embedding,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "context_seed": args.context_seed,
        "support_seed": args.support_seed,
        "assignment_seed": args.assignment_seed,
        "batch_seed": args.batch_seed,
        "precision": args.precision,
        "allow_nondeterministic": args.allow_nondeterministic,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def zero_crossing(points: list[tuple[float, float]]):
    points = sorted(points)
    violations = sum(points[i + 1][1] < points[i][1] for i in range(len(points) - 1))
    for rho, margin in points:
        if margin == 0.0:
            return rho, "exact", violations
    for (rho0, margin0), (rho1, margin1) in zip(points, points[1:]):
        if margin0 < 0.0 < margin1:
            crossing = rho0 + (-margin0) * (rho1 - rho0) / (margin1 - margin0)
            return crossing, "interpolated", violations
    if points and points[0][1] > 0:
        return float("nan"), "below_grid", violations
    if points and points[-1][1] < 0:
        return float("nan"), "above_grid", violations
    return float("nan"), "no_negative_to_positive_crossing", violations


def estimate_boundaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["support_size"]), int(row["run_seed"]))].append(
            (float(row["rho"]), float(row["mean_state_margin"]))
        )

    output = []
    for (support_size, run_seed), points in sorted(grouped.items()):
        estimate, status, violations = zero_crossing(points)
        output.append(
            {
                "support_size": support_size,
                "run_seed": run_seed,
                "rho_star_theory": 1.0 / (support_size + 1),
                "rho_star_empirical": estimate,
                "status": status,
                "margin_monotonicity_violations": violations,
            }
        )
    return output


def aggregate_boundaries(boundaries: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in boundaries:
        estimate = float(row["rho_star_empirical"])
        if np.isfinite(estimate):
            grouped[int(row["support_size"])].append(estimate)

    output = []
    for support_size in sorted({int(row["support_size"]) for row in boundaries}):
        estimates = np.asarray(grouped[support_size], dtype=np.float64)
        theory = 1.0 / (support_size + 1)
        output.append(
            {
                "support_size": support_size,
                "qmax_theory": 1.0 / support_size,
                "rho_star_theory": theory,
                "successful_seed_estimates": len(estimates),
                "rho_star_empirical_mean": (
                    float(np.mean(estimates)) if len(estimates) else float("nan")
                ),
                "rho_star_empirical_std": (
                    float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
                ),
                "mean_absolute_error_from_theory": (
                    float(np.mean(np.abs(estimates - theory)))
                    if len(estimates)
                    else float("nan")
                ),
            }
        )
    return output


def grouped_metric(rows, metric: str):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["support_size"])][float(row["rho"])].append(float(row[metric]))
    return grouped


def plot_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = grouped_metric(rows, "clean_accuracy")
    margins = grouped_metric(rows, "mean_state_margin")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, support_size in enumerate(sorted(accuracy)):
        color = colors[index % len(colors)]
        rhos = sorted(accuracy[support_size])
        acc_means = [100.0 * np.mean(accuracy[support_size][rho]) for rho in rhos]
        acc_stds = [
            100.0 * np.std(accuracy[support_size][rho], ddof=1)
            if len(accuracy[support_size][rho]) > 1
            else 0.0
            for rho in rhos
        ]
        margin_means = [np.mean(margins[support_size][rho]) for rho in rhos]
        margin_stds = [
            np.std(margins[support_size][rho], ddof=1)
            if len(margins[support_size][rho]) > 1
            else 0.0
            for rho in rhos
        ]
        theory = 1.0 / (support_size + 1)
        label = rf"$K={support_size}$"
        axes[0].errorbar(rhos, acc_means, yerr=acc_stds, marker="o", capsize=3, color=color, label=label)
        axes[1].errorbar(rhos, margin_means, yerr=margin_stds, marker="o", capsize=3, color=color, label=label)
        axes[0].axvline(theory, color=color, linestyle="--", alpha=0.55)
        axes[1].axvline(theory, color=color, linestyle="--", alpha=0.55)

    axes[0].set_xlabel(r"Local-target reliability $\rho$")
    axes[0].set_ylabel("Valid-target greedy accuracy (%)")
    axes[0].set_ylim(-2, 102)
    axes[1].set_xlabel(r"Local-target reliability $\rho$")
    axes[1].set_ylabel(r"Valid-target margin $\Delta_K(\rho)$")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict], boundaries: list[dict], boundary_summary: list[dict]) -> None:
    print("\nSUMMARY")
    print("=" * 104)
    accuracy = grouped_metric(rows, "clean_accuracy")
    margins = grouped_metric(rows, "mean_state_margin")
    kls = grouped_metric(rows, "mean_kl_to_empirical_target")

    for support_size in sorted(accuracy):
        theory = 1.0 / (support_size + 1)
        print(f"\nK={support_size} | qmax=1/{support_size} | theoretical rho*={theory:.6f}")
        print("rho       clean accuracy       state margin         KL(target || model)")
        for rho in sorted(accuracy[support_size]):
            acc = np.asarray(accuracy[support_size][rho])
            margin = np.asarray(margins[support_size][rho])
            kl = np.asarray(kls[support_size][rho])
            acc_sd = np.std(acc, ddof=1) if len(acc) > 1 else 0.0
            margin_sd = np.std(margin, ddof=1) if len(margin) > 1 else 0.0
            print(
                f"{rho:7.4f}   {100*np.mean(acc):7.2f} +/- {100*acc_sd:6.2f}   "
                f"{np.mean(margin):9.4f} +/- {margin_sd:8.4f}   {np.mean(kl):10.6f}"
            )

    print("\nEMPIRICAL MARGIN CROSSINGS")
    print("K    seed    theory       estimate      status                         violations")
    for row in boundaries:
        estimate = float(row["rho_star_empirical"])
        estimate_text = f"{estimate:.6f}" if np.isfinite(estimate) else "      nan"
        print(
            f"{int(row['support_size']):2d}   {int(row['run_seed']):4d}   "
            f"{float(row['rho_star_theory']):.6f}   {estimate_text:>10s}   "
            f"{str(row['status']):28s}   {int(row['margin_monotonicity_violations'])}"
        )

    print("\nBOUNDARY SUMMARY ACROSS SEEDS")
    print("K    theory       empirical mean +/- std      mean absolute error")
    for row in boundary_summary:
        print(
            f"{int(row['support_size']):2d}   {float(row['rho_star_theory']):.6f}   "
            f"{float(row['rho_star_empirical_mean']):.6f} +/- "
            f"{float(row['rho_star_empirical_std']):.6f}      "
            f"{float(row['mean_absolute_error_from_theory']):.6f}"
        )


def condition_grid(args) -> dict[int, list[float]]:
    if args.rhos is not None:
        return {support_size: sorted(set(args.rhos)) for support_size in args.supports}
    missing = [support_size for support_size in args.supports if support_size not in TARGETED_RHOS]
    if missing:
        raise ValueError(f"No targeted default grid for supports {missing}; provide --rhos explicitly")
    return {support_size: TARGETED_RHOS[support_size] for support_size in args.supports}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched competitor-support experiment with resumable multi-seed output."
    )
    parser.add_argument("--supports", nargs="+", type=int, default=[1, 3, 15])
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=None,
        help="Shared rho grid. If omitted, use a targeted grid around 1/(K+1) for each K.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    parser.add_argument("--max-contexts", type=int, default=832)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--context-seed", type=int, default=501)
    parser.add_argument("--support-seed", type=int, default=701)
    parser.add_argument("--assignment-seed", type=int, default=901)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--barrier-tol", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/competitor_support.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for support_size in args.supports:
        if not 1 <= support_size <= 15:
            raise ValueError(f"support size must be in [1,15], got {support_size}")
    grid = condition_grid(args)
    for support_size, rhos in grid.items():
        for rho in rhos:
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"rho must be in [0,1], got {rho} for K={support_size}")

    configure_determinism(args.allow_nondeterministic)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer()
    base_contexts = build_base_contexts(args.max_contexts, args.context_seed)
    config_id = configuration_id(args, len(base_contexts))

    rows = read_csv(args.output) if args.resume else []
    if rows and any(row.get("config_id") != config_id for row in rows):
        raise ValueError(
            f"Existing {args.output} was produced by another configuration. "
            "Choose a new --output or restore the original arguments."
        )
    completed = {
        (int(row["support_size"]), float(row["rho"]), int(row["run_seed"]))
        for row in rows
    }

    total_runs = sum(len(grid[support_size]) * len(args.seeds) for support_size in args.supports)
    print("=" * 96)
    print("MATCHED COMPETITOR-SUPPORT EXPERIMENT")
    print("=" * 96)
    print(f"device={device} config_id={config_id}")
    print(f"contexts={len(base_contexts)} repeats={args.repeats} steps={args.steps}")
    print(f"supports={args.supports} seeds={args.seeds} total_runs={total_runs}")
    for support_size in args.supports:
        print(f"K={support_size:2d}: rho*={1/(support_size+1):.6f}, grid={grid[support_size]}")
    if completed:
        print(f"resume=True: {len(completed)} completed runs loaded")

    for support_size in args.supports:
        theory = 1.0 / (support_size + 1)
        for rho in grid[support_size]:
            for run_seed in args.seeds:
                key = (support_size, float(rho), run_seed)
                if key in completed:
                    print(f"skip completed K={support_size} rho={rho:.6f} seed={run_seed}")
                    continue

                print("\n" + "-" * 96)
                print(f"K={support_size} rho={rho:.6f} theory_rho*={theory:.6f} seed={run_seed}")
                model, counts, train_loss, actual_precision, seeds = train_one(
                    base_contexts, rho, support_size, run_seed, tokenizer, args, device
                )
                metrics = evaluate_condition(model, base_contexts, counts, tokenizer, args, device)
                row = {
                    "config_id": config_id,
                    "support_size": support_size,
                    "qmax_theory": 1.0 / support_size,
                    "rho_star_theory": theory,
                    "rho": rho,
                    **seeds,
                    "contexts": len(base_contexts),
                    "repeats": args.repeats,
                    "steps": args.steps,
                    "precision": actual_precision,
                    "train_loss": train_loss,
                    **metrics,
                }
                rows.append(row)
                completed.add(key)
                write_csv(rows, args.output)

                print(
                    f"accuracy={100*metrics['clean_accuracy']:.2f}% | "
                    f"margin={metrics['mean_state_margin']:.5f} | "
                    f"KL={metrics['mean_kl_to_empirical_target']:.6f} | "
                    f"rho_realized={metrics['mean_realized_rho']:.6f} | "
                    f"qmax_realized={metrics['mean_empirical_qmax']:.6f}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    boundaries = estimate_boundaries(rows)
    boundary_summary = aggregate_boundaries(boundaries)
    boundary_path = args.output.with_name(args.output.stem + "_boundaries.csv")
    boundary_summary_path = args.output.with_name(
        args.output.stem + "_boundary_summary.csv"
    )
    plot_path = args.output.with_name(args.output.stem + "_curves.png")
    write_csv(boundaries, boundary_path)
    write_csv(boundary_summary, boundary_summary_path)
    plot_results(rows, plot_path)
    print_summary(rows, boundaries, boundary_summary)

    total_violations = sum(int(float(row["barrier_violations"])) for row in rows)
    print(f"\nSaved runs:      {args.output}")
    print(f"Saved boundaries: {boundary_path}")
    print(f"Saved summary:    {boundary_summary_path}")
    print(f"Saved figure:    {plot_path}")
    print(f"Barrier violations: {total_violations}")
    if total_violations:
        raise RuntimeError(
            "Observed a numerical violation of the exact KL barrier. Inspect the run before changing tolerance."
        )


if __name__ == "__main__":
    main()