import argparse
import random
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.model import GPTModel
from src.registry import TASKS


FILLER_CHAR = "."

TARGET_BUILDERS = {
    "outcome": lambda inst: f" : {inst.gold}\n",
    "answer_first": lambda inst: f" : {inst.gold} ; {inst.correct_trace}\n",
        "filler": lambda inst: f" {FILLER_CHAR * len(inst.correct_trace)} : {inst.gold}\n",
    "process": lambda inst: f" {inst.correct_trace} : {inst.gold}\n",
    "corrupted": lambda inst: f" {inst.wrong_trace} : {inst.gold}\n",
}


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
    items, prompts = [], set()
    while len(items) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        items.append(inst)
        prompts.add(inst.prompt)
    random.setstate(state)
    return items


class SupervisionDataset(Dataset):
    def __init__(self, instances, task, mode: str):
        if mode not in TARGET_BUILDERS:
            raise ValueError(f"Unknown mode: {mode}")
        tokenizer, block_size = task.tokenizer, task.block_size
        rows = []
        for inst in instances:
            prompt_ids = tokenizer.encode(inst.prompt)
            target_ids = tokenizer.encode(TARGET_BUILDERS[mode](inst))
            full = prompt_ids + target_ids
            if len(full) > block_size:
                raise ValueError(f"{task.name}/{mode}: {len(full)} tokens exceeds block_size={block_size}")
            x, y = full[:-1], full[1:]
            mask = [float(i + 1 >= len(prompt_ids)) for i in range(len(x))]
            pad = block_size - 1 - len(x)
            rows.append((x + [tokenizer.pad_id] * pad,
                         y + [tokenizer.pad_id] * pad,
                         mask + [0.0] * pad))
        self.x = torch.tensor([row[0] for row in rows], dtype=torch.long)
        self.y = torch.tensor([row[1] for row in rows], dtype=torch.long)
        self.mask = torch.tensor([row[2] for row in rows], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]


def build_model(task, args, device):
    return GPTModel(vocab_size=task.tokenizer.vocab_size,
                    block_size=task.block_size,
                    pad_id=task.tokenizer.pad_id,
                    n_embd=args.embedding,
                    n_head=args.heads,
                    n_layer=args.layers,
                    dropout=args.dropout).to(device)


def train_one(task, train_instances, mode, seed, args, device):
    set_seed(seed)
    dataset = SupervisionDataset(train_instances, task, mode)
    generator = torch.Generator().manual_seed(args.batch_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers, generator=generator,
                        pin_memory=device.type == "cuda")
    iterator = iter(loader)
    model = build_model(task, args, device)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                      fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    for step in tqdm(range(args.steps), desc=f"{task.name}/{mode}/seed={seed}", leave=False):
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
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


def extract_prediction(text: str, mode: str):
    text = text.split("\n", 1)[0]
    if ":" not in text:
        return None
    answer_part = text.split(":", 1)[1] if mode in {"outcome", "answer_first"} else text.rsplit(":", 1)[1]
    match = re.search(r"\d+", answer_part)
    return match.group(0) if match else None


@torch.inference_mode()
def evaluate(model, task, instances, mode, args, device):
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst.gold))
    correct = total = 0
    max_new_tokens = 8 if mode in {"outcome", "answer_first"} else task.max_new_tokens
    for _, rows in sorted(buckets.items()):
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]
            context = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
            prompt_len = context.shape[1]
            output = model.generate(context, max_new_tokens=max_new_tokens,
                                    stop_id=task.tokenizer.newline_id, greedy=True)
            for generated_ids, (_, gold) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_len:])
                correct += int(extract_prediction(tail, mode) == gold)
                total += 1
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["state_machine_2", "state_machine_4", "state_machine_8", "state_machine_12", "state_machine_16", "state_machine_20"])
    parser.add_argument("--modes", nargs="+", choices=list(TARGET_BUILDERS), default=["outcome", "answer_first", "filler", "process", "corrupted"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    results = defaultdict(list)
    for task_name in args.tasks:
        task = TASKS[task_name]
        train_instances = generate_unique(task, args.train_size, args.train_seed)
        train_prompts = {inst.prompt for inst in train_instances}
        val_instances = generate_unique(task, args.val_size, args.val_seed, train_prompts)
        print(f"\n{task_name}: chance={100 * task.chance_acc:.2f}% train={len(train_instances)} val={len(val_instances)}")
        for mode in args.modes:
            for seed in args.seeds:
                model, loss = train_one(task, train_instances, mode, seed, args, device)
                accuracy = evaluate(model, task, val_instances, mode, args, device)
                results[(task_name, mode)].append(accuracy)
                print(f"{task_name:18s} {mode:12s} seed={seed} loss={loss:.4f} accuracy={100 * accuracy:.2f}%")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print("\nSUMMARY")
    for (task_name, mode), values in results.items():
        array = np.asarray(values) * 100
        std = array.std(ddof=1) if len(array) > 1 else 0.0
        print(f"{task_name:18s} {mode:12s} mean={array.mean():6.2f}% std={std:6.2f}% runs={array.round(2)}")


if __name__ == "__main__":
    main()