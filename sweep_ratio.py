import argparse
import csv
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.model import GPTModel
from src.registry import TASKS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_unique(task, size: int, seed: int, excluded=None):
    excluded = set() if excluded is None else set(excluded)
    state = random.getstate()
    random.seed(seed)
    instances, prompts = [], set()
    while len(instances) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        instances.append(inst)
        prompts.add(inst.prompt)
    random.setstate(state)
    return instances


class RatioDataset(Dataset):
    def __init__(self, instances, task, condition: str, rho=None, ratio_scores=None):
        if condition not in {"outcome", "mixed_process"}:
            raise ValueError(f"unknown condition: {condition}")
        if condition == "mixed_process" and (rho is None or ratio_scores is None):
            raise ValueError("mixed_process requires rho and ratio_scores")

        tokenizer = task.tokenizer
        width = task.block_size - 1
        if tokenizer.vocab_size > 256:
            raise ValueError("uint8 storage requires tokenizer.vocab_size <= 256")

        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)

        for row, inst in enumerate(instances):
            if condition == "outcome":
                target = f" : {inst.gold}\n"
            else:
                trace = inst.correct_trace if ratio_scores[row] < rho else inst.wrong_trace
                target = f" {trace} : {inst.gold}\n"

            prompt_ids = tokenizer.encode(inst.prompt)
            full = prompt_ids + tokenizer.encode(target)
            if len(full) > task.block_size:
                raise ValueError(
                    f"{task.name}/{condition}: {len(full)} tokens exceeds block_size={task.block_size}"
                )

            sequence = torch.tensor(full, dtype=torch.uint8)
            length = len(full) - 1
            self.x[row, :length] = sequence[:-1]
            self.y[row, :length] = sequence[1:]
            self.mask[row, len(prompt_ids) - 1:length] = True

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]


def build_model(task, args, device):
    return GPTModel(
        vocab_size=task.tokenizer.vocab_size,
        block_size=task.block_size,
        pad_id=task.tokenizer.pad_id,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        dropout=args.dropout,
    ).to(device)


def extract_generation(text: str):
    line = text.split("\n", 1)[0]
    if ":" not in line:
        return None, None, False
    trace_text, answer_text = line.rsplit(":", 1)
    match = re.search(r"\d+", answer_text)
    return trace_text.strip(), match.group(0) if match else None, True


@torch.inference_mode()
def evaluate(model, task, instances, condition, args, device):
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst))

    answer_correct = trace_correct = trace_step_correct = trace_step_total = colon_count = total = 0
    max_new_tokens = 8 if condition == "outcome" else task.max_new_tokens

    for rows in buckets.values():
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]
            context = torch.tensor([ids for ids, _ in batch], dtype=torch.long, device=device)
            prompt_length = context.shape[1]
            output = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                stop_id=task.tokenizer.newline_id,
                greedy=True,
            )
            for generated_ids, (_, inst) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_length:])
                trace, answer, emitted_colon = extract_generation(tail)
                colon_count += int(emitted_colon)
                answer_correct += int(answer == inst.gold)
                if condition != "outcome":
                    trace_correct += int(trace == inst.correct_trace)
                    predicted_steps = [] if trace is None else trace.split()
                    gold_steps = inst.correct_trace.split()
                    trace_step_correct += sum(
                        predicted == gold
                        for predicted, gold in zip(predicted_steps, gold_steps)
                    )
                    trace_step_total += len(gold_steps)
                total += 1

    return {
        "answer_accuracy": answer_correct / total,
        "exact_trace_accuracy": None if condition == "outcome" else trace_correct / total,
        "trace_step_accuracy": None if condition == "outcome" else trace_step_correct / trace_step_total,
        "colon_rate": colon_count / total,
    }


def make_optimizer(model, args, device):
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_run(task, dataset, val_instances, condition, rho, seed, checkpoints, args, device, output):
    set_seed(seed)
    generator = torch.Generator().manual_seed(args.batch_seed)
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
    model = build_model(task, args, device)
    optimizer = make_optimizer(model, args, device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    label = "outcome" if condition == "outcome" else f"rho={rho:.2f}"

    model.train()
    for step in tqdm(range(1, max(checkpoints) + 1), desc=f"{task.name}/{label}/seed={seed}"):
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)

        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        mask = mask.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y, mask=mask)
        else:
            _, loss = model(x, targets=y, mask=mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step in checkpoints:
            metrics = evaluate(model, task, val_instances, condition, args, device)
            row = {
                "task": task.name,
                "condition": label,
                "rho": "" if rho is None else rho,
                "seed": seed,
                "step": step,
                "loss": float(loss.detach()),
                "answer_accuracy": metrics["answer_accuracy"],
                "exact_trace_accuracy": "" if metrics["exact_trace_accuracy"] is None else metrics["exact_trace_accuracy"],
                "trace_step_accuracy": "" if metrics["trace_step_accuracy"] is None else metrics["trace_step_accuracy"],
                "colon_rate": metrics["colon_rate"],
            }
            append_csv(output, row)
            trace_text = "n/a" if metrics["exact_trace_accuracy"] is None else f"{100 * metrics['exact_trace_accuracy']:.2f}%"
            step_text = "n/a" if metrics["trace_step_accuracy"] is None else f"{100 * metrics['trace_step_accuracy']:.2f}%"
            print(
                f"{task.name} {label} seed={seed} step={step} "
                f"answer={100 * metrics['answer_accuracy']:.2f}% "
                f"trace={trace_text} trace_step={step_text} "
                f"colon={100 * metrics['colon_rate']:.2f}%"
            )
            model.train()

    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="word_index_len32")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1000, 2000, 4000, 6000, 8000])
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--ratio-seed", type=int, default=777)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-outcome", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        parser.error("every rho must lie in [0, 1]")
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] < 1:
        parser.error("checkpoints must be positive")

    task = TASKS[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("results") / f"{task.name}_phase_{timestamp}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to append to existing output: {output}")

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {inst.prompt for inst in train_instances}
    val_instances = generate_unique(task, args.val_size, args.val_seed, train_prompts)
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))
    print(
        f"task={task.name} chance={100 * task.chance_acc:.2f}% device={device} "
        f"train={len(train_instances)} val={len(val_instances)} overlap=0 output={output}"
    )

    conditions = []
    if args.include_outcome:
        conditions.append(("outcome", None))
    conditions.extend(("mixed_process", rho) for rho in sorted(set(args.rhos)))

    for condition, rho in conditions:
        dataset = RatioDataset(
            train_instances,
            task,
            condition,
            rho=rho,
            ratio_scores=ratio_scores,
        )
        for seed in args.seeds:
            train_run(
                task,
                dataset,
                val_instances,
                condition,
                rho,
                seed,
                checkpoints,
                args,
                device,
                output,
            )
        del dataset

    print(f"saved: {output}")


if __name__ == "__main__":
    main()