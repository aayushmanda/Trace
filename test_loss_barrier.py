
import argparse
import csv
import itertools
from functools import partial
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.boolean_circuit_tasks import _apply_gate
from src.model import GPTModel
from src.tokenizer import CharTokenizer


STATE_SYMBOLS = "ABCDEFGHIJKLMNOP"  # 16 atomic state tokens


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_boolean_gates():
    gates = [f"x{i}" for i in range(4)]
    gates += [f"c{i}{j}" for i in range(4) for j in range(4) if i != j]
    gates += [f"s{i}{j}" for i in range(4) for j in range(4) if i != j]
    gates += [f"t{i}{j}{k}" for i, j, k in itertools.permutations(range(4), 3)]
    return gates


def int_to_bits(value: int):
    return [int(b) for b in f"{value:04b}"]


def bits_to_int(bits):
    return int("".join(map(str, bits)), 2)


def state_symbol(value: int) -> str:
    return STATE_SYMBOLS[value]


def next_state_value(state: int, gate: str) -> int:
    return bits_to_int(_apply_gate(int_to_bits(state), gate))


def build_tokenizer():
    chars = STATE_SYMBOLS + "xcst0123;>"
    return CharTokenizer(chars)


class LocalExampleDataset(Dataset):
    def __init__(self, contexts, labels, tokenizer):
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.encoded = [torch.tensor(tokenizer.encode(z), dtype=torch.long) for z in contexts]
        self.target_ids = torch.tensor([tokenizer.stoi[y] for y in labels], dtype=torch.long)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.target_ids[idx]


def collate_local(batch, pad_id):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    return padded, lengths, torch.stack(ys)


def build_base_contexts(max_contexts: int, context_seed: int):
    rows = []
    for state in range(16):
        for gate in all_boolean_gates():
            clean = next_state_value(state, gate)
            rows.append((f"{state_symbol(state)};{gate}>", clean))

    rng = random.Random(context_seed)
    rng.shuffle(rows)
    if max_contexts > 0:
        rows = rows[: min(max_contexts, len(rows))]
    return rows


def wrong_support(clean_value: int, support_size: int, context_index: int, support_seed: int):
    if not 1 <= support_size <= 15:
        raise ValueError("support_size must be in [1, 15]")
    candidates = [v for v in range(16) if v != clean_value]
    rng = random.Random(support_seed + 100_003 * context_index + 97 * support_size)
    return rng.sample(candidates, support_size)


def build_condition(base_contexts, rho, support_size, repeats, assignment_seed, support_seed):
    """
    Create repeated identical contexts with hard labels.

    Each replica gets a fixed U~Uniform[0,1]. It is clean iff U<rho, making the
    clean set nested as rho increases. Corrupted labels are fixed in advance and
    cycle through a shuffled m-element support, making Q_z approximately uniform.
    """
    contexts, labels, context_ids = [], [], []
    counts = [defaultdict(int) for _ in base_contexts]

    for cid, (z, clean_value) in enumerate(base_contexts):
        support = wrong_support(clean_value, support_size, cid, support_seed)
        local_rng = random.Random(assignment_seed + 1_000_003 * cid)

        uniforms = [local_rng.random() for _ in range(repeats)]
        corrupt_values = []
        while len(corrupt_values) < repeats:
            block = support.copy()
            local_rng.shuffle(block)
            corrupt_values.extend(block)
        corrupt_values = corrupt_values[:repeats]

        for rep in range(repeats):
            value = clean_value if uniforms[rep] < rho else corrupt_values[rep]
            symbol = state_symbol(value)
            contexts.append(z)
            labels.append(symbol)
            context_ids.append(cid)
            counts[cid][value] += 1

    return contexts, labels, context_ids, counts


def build_model(tokenizer, args, device):
    max_context_len = max(len(f"{state_symbol(s)};{g}>") for s in range(16) for g in all_boolean_gates())
    return GPTModel(
        vocab_size=tokenizer.vocab_size,
        block_size=max_context_len + 1,
        pad_id=tokenizer.pad_id,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        dropout=args.dropout,
    ).to(device)


def train_one(base_contexts, rho, support_size, seed, tokenizer, args, device):
    contexts, labels, _, counts = build_condition(
        base_contexts=base_contexts,
        rho=rho,
        support_size=support_size,
        repeats=args.repeats,
        assignment_seed=args.assignment_seed,
        support_seed=args.support_seed,
    )
    dataset = LocalExampleDataset(contexts, labels, tokenizer)
    generator = torch.Generator().manual_seed(args.batch_seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        collate_fn=partial(collate_local, pad_id=tokenizer.pad_id),
    )

    set_seed(seed)
    model = build_model(tokenizer, args, device)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    iterator = iter(loader)
    losses = []
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

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                last_logits = logits[torch.arange(x.size(0), device=device), lengths - 1]
                loss = torch.nn.functional.cross_entropy(last_logits, y)
        else:
            logits, _ = model(x)
            last_logits = logits[torch.arange(x.size(0), device=device), lengths - 1]
            loss = torch.nn.functional.cross_entropy(last_logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach()))
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            tail = np.mean(losses[-min(100, len(losses)):])
            print(
                f"support={support_size:2d} rho={rho:5.3f} seed={seed} "
                f"step={step+1:5d}/{args.steps} loss={tail:.4f}"
            )

    return model, counts, float(np.mean(losses[-min(100, len(losses)):]))


@torch.inference_mode()
def evaluate_condition(model, base_contexts, counts, tokenizer, args, device):
    model.eval()
    clean_state_ids = [tokenizer.stoi[s] for s in STATE_SYMBOLS]

    clean_correct = 0
    pop_correct_contexts = 0
    model_correct_given_pop = 0
    wrong_when_pop_clean = 0
    barrier_violations = 0
    kls, barriers, wrong_gaps = [], [], []
    a_values, b_values = [], []

    for start in range(0, len(base_contexts), args.eval_batch_size):
        batch = base_contexts[start : start + args.eval_batch_size]
        encoded = [torch.tensor(tokenizer.encode(z), dtype=torch.long) for z, _ in batch]
        lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
        max_len = int(lengths.max())
        x = torch.full((len(encoded), max_len), tokenizer.pad_id, dtype=torch.long)
        for i, ids in enumerate(encoded):
            x[i, : len(ids)] = ids

        x = x.to(device)
        lengths_dev = lengths.to(device)
        logits, _ = model(x)
        last_logits = logits[torch.arange(x.size(0), device=device), lengths_dev - 1]
        log_probs = torch.log_softmax(last_logits.float(), dim=-1).cpu()
        preds = torch.argmax(last_logits, dim=-1).cpu()

        for j, ((_, clean_value), log_r, pred) in enumerate(zip(batch, log_probs, preds)):
            cid = start + j
            clean_id = tokenizer.stoi[state_symbol(clean_value)]
            is_clean = int(pred) == clean_id
            clean_correct += int(is_clean)

            total = sum(counts[cid].values())
            p = torch.zeros(tokenizer.vocab_size, dtype=torch.float64)
            for value, count in counts[cid].items():
                p[tokenizer.stoi[state_symbol(value)]] = count / total

            a = float(p[clean_id])
            wrong_ps = [float(p[sid]) for sid in clean_state_ids if sid != clean_id]
            b = max(wrong_ps)
            a_values.append(a)
            b_values.append(b)

            support_mask = p > 0
            kl = float((p[support_mask] * (torch.log(p[support_mask]) - log_r.double()[support_mask])).sum())
            kls.append(kl)

            if a > b:
                pop_correct_contexts += 1
                model_correct_given_pop += int(is_clean)

                if b == 0.0:
                    barrier = a * math.log(2.0)
                else:
                    barrier = (
                        a * math.log((2.0 * a) / (a + b))
                        + b * math.log((2.0 * b) / (a + b))
                    )
                barriers.append(barrier)

                if not is_clean:
                    wrong_when_pop_clean += 1
                    wrong_gaps.append(kl - barrier)
                    if kl + args.barrier_tol < barrier:
                        barrier_violations += 1

    n = len(base_contexts)
    result = {
        "clean_accuracy": clean_correct / n,
        "population_clean_fraction": pop_correct_contexts / n,
        "clean_accuracy_given_population_clean": (
            model_correct_given_pop / pop_correct_contexts if pop_correct_contexts else float("nan")
        ),
        "wrong_when_population_clean": wrong_when_pop_clean,
        "mean_kl": float(np.mean(kls)),
        "mean_barrier": float(np.mean(barriers)) if barriers else float("nan"),
        "min_wrong_kl_minus_barrier": float(min(wrong_gaps)) if wrong_gaps else float("nan"),
        "barrier_violations": barrier_violations,
        "mean_a": float(np.mean(a_values)),
        "mean_b": float(np.mean(b_values)),
    }
    return result


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["support_size"])][float(row["rho"])].append(float(row["clean_accuracy"]))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for support_size in sorted(grouped):
        rhos = sorted(grouped[support_size])
        means = [np.mean(grouped[support_size][rho]) * 100 for rho in rhos]
        stds = [
            (np.std(grouped[support_size][rho], ddof=1) * 100 if len(grouped[support_size][rho]) > 1 else 0.0)
            for rho in rhos
        ]
        rho_star = 1.0 / (support_size + 1)
        label = f"m={support_size}, theory rho*={rho_star:.3f}"
        ax.errorbar(rhos, means, yerr=stds, marker="o", capsize=3, label=label)

    ax.set_xlabel("Trace reliability rho")
    ax.set_ylabel("Clean local greedy accuracy (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows):
    print("\nSUMMARY")
    print("=" * 90)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["support_size"])][float(row["rho"])].append(row)

    for support_size in sorted(grouped):
        rho_star = 1.0 / (support_size + 1)
        print(f"\nsupport m={support_size} | qmax=1/{support_size} | theoretical rho*={rho_star:.4f}")
        print("rho     clean_acc      pop_clean      mean_KL       mean_B      violations")
        for rho in sorted(grouped[support_size]):
            rs = grouped[support_size][rho]
            acc = np.mean([float(r["clean_accuracy"]) for r in rs])
            pop = np.mean([float(r["population_clean_fraction"]) for r in rs])
            kl = np.mean([float(r["mean_kl"]) for r in rs])
            finite_bs = [float(r["mean_barrier"]) for r in rs if np.isfinite(float(r["mean_barrier"]))]
            b = np.mean(finite_bs) if finite_bs else float("nan")
            violations = sum(int(r["barrier_violations"]) for r in rs)
            print(f"{rho:5.3f}   {100*acc:8.2f}%      {100*pop:8.2f}%     {kl:9.5f}   {b:9.5f}      {violations}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supports", nargs="+", type=int, default=[1, 3, 15],
                        help="Number m of possible corrupted targets. Use 1,3,15 for rho*=1/2,1/4,1/16.")
    parser.add_argument("--rhos", nargs="+", type=float,
                        default=[0.03, 0.06, 0.08, 0.15, 0.25, 0.35, 0.50, 0.60, 0.80])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    parser.add_argument("--max-contexts", type=int, default=832,
                        help="832 uses all 16 states x 52 gate strings; use smaller for smoke tests.")
    parser.add_argument("--repeats", type=int, default=64,
                        help="Repeated hard-label observations per exact local context.")
    parser.add_argument("--steps", type=int, default=3000)
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
    parser.add_argument("--context-seed", type=int, default=501)
    parser.add_argument("--support-seed", type=int, default=701)
    parser.add_argument("--assignment-seed", type=int, default=901)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--barrier-tol", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/local_loss_barrier.csv")
    args = parser.parse_args()

    for rho in args.rhos:
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0,1], got {rho}")
    for m in args.supports:
        if not 1 <= m <= 15:
            raise ValueError(f"support size must be in [1,15], got {m}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = build_tokenizer()
    base_contexts = build_base_contexts(args.max_contexts, args.context_seed)

    print("=" * 90)
    print("LOCAL LOSS-BARRIER / CORRUPTION-GEOMETRY TEST")
    print("=" * 90)
    print(f"device={device}")
    print(f"contexts={len(base_contexts)} repeats={args.repeats} steps={args.steps}")
    print(f"supports={args.supports}")
    print(f"rhos={args.rhos}")
    print(f"seeds={args.seeds}")
    print("Theory: qmax=1/m and rho*=1/(m+1) under matched contexts.")

    rows = []
    for support_size in args.supports:
        rho_star = 1.0 / (support_size + 1)
        for rho in args.rhos:
            for seed in args.seeds:
                print("\n" + "-" * 90)
                print(f"m={support_size} rho={rho:.3f} theory_rho*={rho_star:.4f} seed={seed}")

                model, counts, train_loss = train_one(
                    base_contexts, rho, support_size, seed, tokenizer, args, device
                )
                metrics = evaluate_condition(model, base_contexts, counts, tokenizer, args, device)

                row = {
                    "support_size": support_size,
                    "qmax_theory": 1.0 / support_size,
                    "rho_star_theory": rho_star,
                    "rho": rho,
                    "seed": seed,
                    "contexts": len(base_contexts),
                    "repeats": args.repeats,
                    "steps": args.steps,
                    "train_loss": train_loss,
                    **metrics,
                }
                rows.append(row)

                print(
                    f"clean_acc={100*metrics['clean_accuracy']:.2f}% | "
                    f"pop_clean={100*metrics['population_clean_fraction']:.2f}% | "
                    f"KL={metrics['mean_kl']:.5f} | B={metrics['mean_barrier']:.5f} | "
                    f"barrier_violations={metrics['barrier_violations']}"
                )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    output = Path(args.output)
    write_csv(rows, output)
    plot_path = output.with_name(output.stem + "_accuracy.png")
    plot_accuracy(rows, plot_path)
    print_summary(rows)

    total_violations = sum(int(r["barrier_violations"]) for r in rows)
    print(f"\nSaved CSV:  {output}")
    print(f"Saved plot: {plot_path}")
    print(f"Total numerical barrier violations: {total_violations}")

    if total_violations:
        raise RuntimeError(
            "Observed a numerical violation of the exact KL barrier. "
            "Increase --barrier-tol slightly only after checking the implementation."
        )


if __name__ == "__main__":
    main()
