import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.model import GPTModel
from src.registry import TASKS
from sweep_ratio import generate_unique


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_boolean_step(step):
    gate, state = step.rsplit(">", 1)
    if len(state) != 4 or any(c not in "01" for c in state):
        raise ValueError(f"Bad Boolean step: {step!r}")
    return gate, state


def build_wrong_states(instances, seed):
    """One fixed locally-invalid 4-bit successor per (example, transition)."""
    rng = np.random.default_rng(seed)
    out = []
    for inst in instances:
        row = []
        for step in inst.correct_trace.split():
            _, clean_state = split_boolean_step(step)
            clean = int(clean_state, 2)
            offset = int(rng.integers(1, 16))
            row.append(f"{(clean + offset) % 16:04b}")
        out.append(row)
    return out


def build_reliability_scores(n, depth, trace_seed, step_seed):
    """Fixed nested uniforms reused across rho values."""
    trace_scores = np.random.default_rng(trace_seed).random(n)
    step_scores = np.random.default_rng(step_seed).random((n, depth))
    return trace_scores, step_scores


class CanonicalContextDataset(Dataset):
    def __init__(self, instances, task, mode, rho, trace_scores, step_scores, wrong_states):
        if mode not in {"trace", "step"}: raise ValueError(mode)
        if not 0.0 <= rho <= 1.0: raise ValueError("rho must be in [0,1]")
        tokenizer = task.tokenizer
        width = task.block_size - 1
        if tokenizer.vocab_size > 256: raise ValueError("uint8 storage requires tokenizer.vocab_size <= 256")

        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)
        clean_count = total_transitions = 0

        for row, inst in enumerate(instances):
            steps = inst.correct_trace.split(); depth = len(steps)
            if mode == "trace":
                valid = np.full(depth, trace_scores[row] < rho, dtype=bool)
            else:
                valid = np.asarray(step_scores[row, :depth] < rho, dtype=bool)
            clean_count += int(valid.sum()); total_transitions += depth

            # IMPORTANT: x comes from the fully clean canonical continuation in BOTH modes.
            continuation = f" {inst.correct_trace} : {inst.gold}\n"
            prompt_ids = tokenizer.encode(inst.prompt)
            full = prompt_ids + tokenizer.encode(continuation)
            if len(full) > task.block_size:
                raise ValueError(f"{task.name}: {len(full)} tokens exceeds block_size={task.block_size}")

            seq = torch.tensor(full, dtype=torch.uint8)
            length = len(full) - 1
            self.x[row, :length] = seq[:-1]
            self.y[row, :length] = seq[1:]
            self.mask[row, len(prompt_ids) - 1:length] = True

            # Only state TARGET LABELS change. Inputs stay clean, including previous state bits.
            cursor = 1
            for t, step in enumerate(steps):
                step_start = continuation.find(step, cursor)
                if step_start < 0: raise RuntimeError(f"Could not locate {step!r} in continuation")
                gate, clean_state = split_boolean_step(step)
                state_start = step_start + len(gate) + 1  # +1 for '>'
                if not valid[t]:
                    wrong = wrong_states[row][t]
                    wrong_ids = tokenizer.encode(wrong)
                    clean_ids = tokenizer.encode(clean_state)
                    if len(wrong_ids) != len(clean_ids): raise RuntimeError("state tokenization length mismatch")
                    y_start = len(prompt_ids) + state_start - 1
                    self.y[row, y_start:y_start + len(wrong_ids)] = torch.tensor(wrong_ids, dtype=torch.uint8)
                cursor = step_start + len(step)

        self.realized_local_reliability = clean_count / max(total_transitions, 1)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], self.mask[i]


def build_model(task, args, device):
    return GPTModel(vocab_size=task.tokenizer.vocab_size, block_size=task.block_size,
        pad_id=task.tokenizer.pad_id, n_embd=args.embedding, n_head=args.heads,
        n_layer=args.layers, dropout=args.dropout).to(device)


def make_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


@torch.inference_mode()
def evaluate_free(model, task, instances, batch_size, device):
    model.eval(); buckets = {}
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets.setdefault(len(ids), []).append((ids, inst))

    answer_ok = exact_ok = step_ok = total_steps = colon_ok = total = 0
    for rows in buckets.values():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            context = torch.tensor([ids for ids, _ in batch], dtype=torch.long, device=device)
            prompt_len = context.shape[1]
            out = model.generate(context, max_new_tokens=task.max_new_tokens,
                stop_id=task.tokenizer.newline_id, greedy=True)
            for ids, (_, inst) in zip(out.tolist(), batch):
                tail = task.tokenizer.decode(ids[prompt_len:])
                line = tail.split("\n", 1)[0]
                emitted_colon = ":" in line; colon_ok += int(emitted_colon)
                answer = task.extract_answer(line) if emitted_colon else None
                answer_ok += int(answer == inst.gold)
                trace = line.rsplit(":", 1)[0].strip() if emitted_colon else None
                exact_ok += int(trace == inst.correct_trace)
                gold_steps = inst.correct_trace.split(); pred_steps = [] if trace is None else trace.split()
                step_ok += sum(t < len(pred_steps) and pred_steps[t] == gold for t, gold in enumerate(gold_steps))
                total_steps += len(gold_steps); total += 1
    return {"answer_accuracy": answer_ok / total, "exact_trace_accuracy": exact_ok / total,
        "trace_step_accuracy": step_ok / total_steps, "colon_rate": colon_ok / total}


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True); exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row)


def train_one(task, dataset, mode, rho, seed, args, device):
    set_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.batch_seed))
    it = iter(loader); model = build_model(task, args, device); opt = make_optimizer(model, args)
    model.train(); recent = []
    for _ in tqdm(range(args.steps), desc=f"{mode}/rho={rho:.2f}/seed={seed}"):
        try: x, y, mask = next(it)
        except StopIteration: it = iter(loader); x, y, mask = next(it)
        x = x.to(device, dtype=torch.long, non_blocking=True); y = y.to(device, dtype=torch.long, non_blocking=True)
        mask = mask.to(device, dtype=torch.float32, non_blocking=True)
        opt.zero_grad(set_to_none=True); _, loss = model(x, targets=y, mask=mask)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        recent.append(float(loss.detach()))
        if len(recent) > 100: recent.pop(0)

    metrics = evaluate_free(model, task, args.val_instances, args.eval_batch_size, device)
    row = {"task": task.name, "mode": mode, "rho": rho,
        "realized_local_reliability": dataset.realized_local_reliability,
        "seed": seed, "steps": args.steps, "train_size": len(dataset), "val_size": len(args.val_instances),
        "train_loss": float(np.mean(recent)), **metrics}
    del model, opt, loader
    if device.type == "cuda": torch.cuda.empty_cache()
    return row


def summarize(path):
    import pandas as pd
    df = pd.read_csv(path)
    g = df.groupby(["mode", "rho"], as_index=False).agg(
        mean_answer=("answer_accuracy", "mean"), sd_answer=("answer_accuracy", "std"),
        mean_exact=("exact_trace_accuracy", "mean"), sd_exact=("exact_trace_accuracy", "std"),
        mean_step=("trace_step_accuracy", "mean"), mean_realized_rho=("realized_local_reliability", "mean"), n=("seed", "count"))
    print("\nCANONICAL TRACE-vs-STEP SUMMARY")
    print(g.to_string(index=False))
    out = path.with_name(path.stem + "_aggregate.csv"); g.to_csv(out, index=False); print(f"saved aggregate: {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    p.add_argument("--train-size", type=int, default=100000); p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--steps", type=int, default=8000); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--train-seed", type=int, default=501); p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--trace-assignment-seed", type=int, default=1701)
    p.add_argument("--step-assignment-seed", type=int, default=1702)
    p.add_argument("--corruption-seed", type=int, default=1703); p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=Path("results/paper/trace_vs_step/canonical_trace_vs_step.csv"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if any(r < 0 or r > 1 for r in args.rhos): p.error("every rho must lie in [0,1]")
    if args.task not in TASKS or not args.task.startswith("boolean_circuit_"):
        raise ValueError("This canonical implementation currently supports boolean_circuit_* tasks")
    if args.output.exists():
        if not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
        args.output.unlink()

    task = TASKS[args.task]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = generate_unique(task, args.train_size, args.train_seed); prompts = {x.prompt for x in train}
    val = generate_unique(task, args.val_size, args.val_seed, prompts); args.val_instances = val
    depth = len(train[0].correct_trace.split())
    trace_scores, step_scores = build_reliability_scores(len(train), depth, args.trace_assignment_seed, args.step_assignment_seed)
    wrong_states = build_wrong_states(train, args.corruption_seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "val_instances"}
    manifest.update({"device": str(device), "depth": depth,
        "canonical_context": True,
        "changed_targets": "successor-state labels only; x remains the fully clean sequence",
        "trace_mode": "one reliability draw shared across all transitions of an example",
        "step_mode": "independent reliability draws per transition"})
    (args.output.parent / "config.json").write_text(json.dumps(manifest, indent=2))

    for rho in sorted(set(args.rhos)):
        for mode in ("trace", "step"):
            ds = CanonicalContextDataset(train, task, mode, rho, trace_scores, step_scores, wrong_states)
            print(f"\n{mode} rho={rho:.2f}: realized local reliability={ds.realized_local_reliability:.6f}")
            for seed in args.seeds:
                row = train_one(task, ds, mode, rho, seed, args, device); append_csv(args.output, row)
                print(f"seed={seed}: answer={100*row['answer_accuracy']:.2f}% exact={100*row['exact_trace_accuracy']:.2f}%")
            del ds
    summarize(args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__": main()
