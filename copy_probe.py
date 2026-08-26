import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.model import GPTModel
from src.registry import TASKS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_unique(task, size, seed, excluded=None):
    excluded = set() if excluded is None else set(excluded)
    old_state = random.getstate()
    random.seed(seed)
    instances, prompts = [], set()

    while len(instances) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        instances.append(inst)
        prompts.add(inst.prompt)

    random.setstate(old_state)
    return instances


def replace_terminal_state(trace, new_state):
    steps = trace.split()
    prefix, _ = steps[-1].rsplit(">", 1)
    steps[-1] = f"{prefix}>{new_state}"
    return " ".join(steps)


def remove_terminal_state(trace):
    """Keep the final operation but remove its answer-bearing successor."""
    steps = trace.split()
    prefix, _ = steps[-1].rsplit(">", 1)
    steps[-1] = f"{prefix}>"
    return " ".join(steps)


def terminal_state(trace):
    return trace.split()[-1].rsplit(">", 1)[1]


def sample_wrong_boolean_state(gold, rng):
    states = [f"{i:04b}" for i in range(16)]
    return rng.choice([x for x in states if x != gold])


class CopyDataset(Dataset):
    def __init__(self, instances, task, rho, ratio_scores, mode):
        if mode not in {"full", "no_terminal"}:
            raise ValueError(mode)

        tokenizer = task.tokenizer
        width = task.block_size - 1

        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)

        for row, inst in enumerate(instances):
            clean = ratio_scores[row] < rho
            trace = inst.correct_trace if clean else inst.wrong_trace

            if mode == "no_terminal":
                trace = remove_terminal_state(trace)

            target = f" {trace} : {inst.gold}\n"
            prompt_ids = tokenizer.encode(inst.prompt)
            full = prompt_ids + tokenizer.encode(target)

            if len(full) > task.block_size:
                raise ValueError(
                    f"{task.name}/{mode}: {len(full)} > block_size={task.block_size}"
                )

            seq = torch.tensor(full, dtype=torch.uint8)
            length = len(full) - 1

            self.x[row, :length] = seq[:-1]
            self.y[row, :length] = seq[1:]
            self.mask[row, len(prompt_ids) - 1:length] = True

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.mask[i]


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


def train_model(task, dataset, seed, args, device):
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

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model.train()

    for _ in tqdm(range(args.steps), leave=False):
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

    return model, float(loss.detach())


def parse_free_generation(task, text):
    line = text.split("\n", 1)[0]

    if ":" not in line:
        return None, None

    trace, answer = line.rsplit(":", 1)
    pred = task.extract_answer(answer, first=True)

    return trace.strip(), pred


@torch.inference_mode()
def evaluate_free(model, task, instances, mode, args, device):
    model.eval()

    buckets = defaultdict(list)

    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst))

    answer_correct = 0
    trace_correct = 0
    total = 0

    for rows in buckets.values():
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]

            context = torch.tensor(
                [ids for ids, _ in batch],
                dtype=torch.long,
                device=device,
            )

            prompt_len = context.shape[1]

            output = model.generate(
                context,
                max_new_tokens=task.max_new_tokens,
                stop_id=task.tokenizer.newline_id,
                greedy=True,
            )

            for generated_ids, (_, inst) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_len:])
                trace, pred = parse_free_generation(task, tail)

                expected_trace = (
                    inst.correct_trace
                    if mode == "full"
                    else remove_terminal_state(inst.correct_trace)
                )

                answer_correct += int(pred == inst.gold)
                trace_correct += int(trace == expected_trace)
                total += 1

    return {
        "answer_accuracy": answer_correct / total,
        "canonical_trace_accuracy": trace_correct / total,
    }


def make_probe_records(instances, probe_seed):
    rng = random.Random(probe_seed)
    records = []

    for inst in instances:
        decoy = sample_wrong_boolean_state(inst.gold, rng)

        records.append({
            "condition": "CC",
            "inst": inst,
            "trace": inst.correct_trace,
            "displayed": inst.gold,
        })

        records.append({
            "condition": "CW",
            "inst": inst,
            "trace": replace_terminal_state(inst.correct_trace, decoy),
            "displayed": decoy,
        })

        records.append({
            "condition": "WC",
            "inst": inst,
            "trace": replace_terminal_state(inst.wrong_trace, inst.gold),
            "displayed": inst.gold,
        })

        records.append({
            "condition": "WW",
            "inst": inst,
            "trace": inst.wrong_trace,
            "displayed": terminal_state(inst.wrong_trace),
        })

    return records


@torch.inference_mode()
def evaluate_copy_probe(model, task, instances, args, device):
    model.eval()

    records = make_probe_records(instances, args.probe_seed)
    buckets = defaultdict(list)

    for record in records:
        context_text = f"{record['inst'].prompt} {record['trace']} : "
        ids = task.tokenizer.encode(context_text)
        buckets[(record["condition"], len(ids))].append((ids, record))

    stats = {
        c: {"gold": 0, "displayed": 0, "total": 0}
        for c in ("CC", "CW", "WC", "WW")
    }

    for (condition, _), rows in buckets.items():
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]

            context = torch.tensor(
                [ids for ids, _ in batch],
                dtype=torch.long,
                device=device,
            )

            context_len = context.shape[1]

            output = model.generate(
                context,
                max_new_tokens=8,
                stop_id=task.tokenizer.newline_id,
                greedy=True,
            )

            for generated_ids, (_, record) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(
                    generated_ids[context_len:]
                ).split("\n", 1)[0]

                pred = task.extract_answer(tail, first=True)

                stats[condition]["gold"] += int(
                    pred == record["inst"].gold
                )

                stats[condition]["displayed"] += int(
                    pred == record["displayed"]
                )

                stats[condition]["total"] += 1

    result = {}

    for condition, values in stats.items():
        n = values["total"]

        result[f"{condition.lower()}_gold_accuracy"] = (
            values["gold"] / n
        )

        result[f"{condition.lower()}_displayed_accuracy"] = (
            values["displayed"] / n
        )

    return result


FIELDS = [
    "task",
    "mode",
    "rho",
    "seed",
    "steps",
    "loss",
    "free_answer_accuracy",
    "free_canonical_trace_accuracy",
    "cc_gold_accuracy",
    "cc_displayed_accuracy",
    "cw_gold_accuracy",
    "cw_displayed_accuracy",
    "wc_gold_accuracy",
    "wc_displayed_accuracy",
    "ww_gold_accuracy",
    "ww_displayed_accuracy",
]


def append_csv(path, row):
    exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not exists:
            writer.writeheader()

        writer.writerow({
            field: row.get(field, "")
            for field in FIELDS
        })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="boolean_circuit_8")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.8, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002])

    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--val-size", type=int, default=500)

    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--ratio-seed", type=int, default=777)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--probe-seed", type=int, default=991)

    parser.add_argument("--steps", type=int, default=6000)
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

    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if not args.task.startswith("boolean_circuit_"):
        parser.error(
            "This probe currently assumes a 4-bit Boolean-circuit terminal state."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    task = TASKS[args.task]

    train_instances = generate_unique(
        task,
        args.train_size,
        args.train_seed,
    )

    train_prompts = {
        inst.prompt
        for inst in train_instances
    }

    val_instances = generate_unique(
        task,
        args.val_size,
        args.val_seed,
        train_prompts,
    )

    ratio_scores = np.random.default_rng(
        args.ratio_seed
    ).random(len(train_instances))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = (
        args.output
        or Path("results")
        / f"{task.name}_terminal_copy_{timestamp}.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"task={task.name} "
        f"device={device} "
        f"train={len(train_instances)} "
        f"val={len(val_instances)} "
        f"output={output}"
    )

    for rho in args.rhos:
        for mode in ("full", "no_terminal"):

            dataset = CopyDataset(
                train_instances,
                task,
                rho,
                ratio_scores,
                mode,
            )

            for seed in args.seeds:
                print(
                    f"\nmode={mode} "
                    f"rho={rho:.2f} "
                    f"seed={seed}"
                )

                model, loss = train_model(
                    task,
                    dataset,
                    seed,
                    args,
                    device,
                )

                free = evaluate_free(
                    model,
                    task,
                    val_instances,
                    mode,
                    args,
                    device,
                )

                row = {
                    "task": task.name,
                    "mode": mode,
                    "rho": rho,
                    "seed": seed,
                    "steps": args.steps,
                    "loss": loss,
                    "free_answer_accuracy":
                        free["answer_accuracy"],
                    "free_canonical_trace_accuracy":
                        free["canonical_trace_accuracy"],
                }

                if mode == "full":
                    probe = evaluate_copy_probe(
                        model,
                        task,
                        val_instances,
                        args,
                        device,
                    )
                    row.update(probe)

                append_csv(output, row)

                print(
                    f"answer="
                    f"{100 * free['answer_accuracy']:.2f}% "
                    f"trace="
                    f"{100 * free['canonical_trace_accuracy']:.2f}%"
                )

                if mode == "full":
                    print(
                        "forced probe | "
                        f"CC gold={100 * probe['cc_gold_accuracy']:.1f}% | "
                        f"CW gold={100 * probe['cw_gold_accuracy']:.1f}% "
                        f"copy-decoy={100 * probe['cw_displayed_accuracy']:.1f}% | "
                        f"WC gold={100 * probe['wc_gold_accuracy']:.1f}% | "
                        f"WW gold={100 * probe['ww_gold_accuracy']:.1f}%"
                    )

                del model

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del dataset

    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()