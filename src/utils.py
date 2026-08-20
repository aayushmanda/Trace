# utils.py

import math, random, re, gc
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from src.model import GPTModel


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# INSTANCE TRACE
# ============================================================

def get_correct_trace(inst):
    if hasattr(inst, "correct_trace"): return inst.correct_trace
    if hasattr(inst, "correct"): return inst.correct
    return inst[1]


# ============================================================
# GENERATE VALIDATION FILE
# ============================================================

def generate_validation_file(task, task_name, val_size, val_seed):
    train_file = f"{task_name}/{task_name}_rho_0.txt"
    val_file = f"{task_name}/{task_name}_val.txt"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]

    train_prompts = {line.split(" ", 1)[0] for line in train_lines}

    random.seed(val_seed)

    val_instances = []
    val_prompts = set()

    for _ in tqdm(range(val_size), desc="Generate validation"):
        while True:
            inst = task.sample()

            word, c = inst.prompt.split(";")

            assert int(inst.gold) == word.index(c), (
                f"Not first-occurrence: prompt={inst.prompt}, "
                f"gold={inst.gold}, expected={word.index(c)}"
            )

            if inst.prompt in train_prompts or inst.prompt in val_prompts:
                continue

            val_instances.append(inst)
            val_prompts.add(inst.prompt)
            break

    with open(val_file, "w", encoding="utf-8") as f:
        for inst in val_instances:
            f.write(f"{inst.prompt} {get_correct_trace(inst)} : {inst.gold}\n")

    print("Saved validation:", val_file)

    return val_file


# ============================================================
# TRAIN DATASET
# ============================================================

class TrainDataset(Dataset):

    def __init__(self, file_path, tokenizer, block_size, pad_id):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        examples = []
        max_len = 0

        for line in tqdm(lines, desc=f"Encoding {file_path}", leave=False):
            prompt, continuation = line.split(" ", 1)

            word, c = prompt.split(";")
            nums = re.findall(r"\d+", continuation.rsplit(":", 1)[1])

            if not nums:
                raise ValueError(f"No gold answer in: {line}")

            assert int(nums[0]) == word.index(c), (
                f"Not first-occurrence: prompt={prompt}, "
                f"gold={nums[0]}, expected={word.index(c)}"
            )

            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(" " + continuation + "\n")

            full = p_ids + t_ids

            if len(full) > block_size:
                raise ValueError(f"Sequence too long: {len(full)} > {block_size}")

            x = full[:-1]
            y = full[1:]

            mask = [1.0 if i + 1 >= len(p_ids) else 0.0 for i in range(len(x))]

            examples.append((x, y, mask))
            max_len = max(max_len, len(x))

        xs, ys, masks = [], [], []

        for x, y, mask in examples:
            pad = max_len - len(x)

            xs.append(x + [pad_id] * pad)
            ys.append(y + [pad_id] * pad)
            masks.append(mask + [0.0] * pad)

        self.xs = torch.tensor(xs, dtype=torch.long)
        self.ys = torch.tensor(ys, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.float32)

        print("Dataset shape:", tuple(self.xs.shape))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.masks[idx]


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValDataset(Dataset):

    def __init__(self, file_path, tokenizer, block_size):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.items = []

        for line in lines:
            prompt, continuation = line.split(" ", 1)

            nums = re.findall(r"\d+", continuation.rsplit(":", 1)[1])

            if not nums:
                raise ValueError(f"No gold answer in: {line}")

            gold = nums[0]

            word, c = prompt.split(";")

            assert int(gold) == word.index(c), (
                f"Validation is not first-occurrence: "
                f"prompt={prompt}, gold={gold}, expected={word.index(c)}"
            )

            prompt_ids = tokenizer.encode(prompt)

            if len(prompt_ids) > block_size:
                raise ValueError(f"Validation prompt too long: {prompt}")

            self.items.append((prompt, gold, prompt_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================
# VALIDATION BUCKET DATASET
# ============================================================

class ValBucketDataset(Dataset):

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prompt, gold, ids = self.items[idx]
        return torch.tensor(ids, dtype=torch.long), prompt, gold


# ============================================================
# VALIDATION LOADERS
# ============================================================

def build_val_loaders(val_dataset, val_batch_size, device):
    val_by_length = {}

    for i in range(len(val_dataset)):
        prompt, gold, ids = val_dataset[i]
        val_by_length.setdefault(len(ids), []).append((prompt, gold, ids))

    val_loaders = {}

    for length, items in val_by_length.items():
        val_loaders[length] = DataLoader(
            ValBucketDataset(items),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda")
        )

    print("Validation buckets:", {k: len(v) for k, v in val_by_length.items()})

    return val_loaders


# ============================================================
# TRAIN LOADER
# ============================================================

def build_train_loader(train_dataset, batch_size, batch_seed, num_workers, device):
    generator = torch.Generator()
    generator.manual_seed(batch_seed)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=generator
    )

    return loader, generator


# ============================================================
# MODEL
# ============================================================

def build_model(vocab_size, block_size, pad_id, n_embd, n_head, n_layer, dropout, device, use_compile):
    base_model = GPTModel(
        vocab_size=vocab_size,
        block_size=block_size,
        pad_id=pad_id,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    ).to(device)

    model = torch.compile(base_model, mode="reduce-overhead") if use_compile and device == "cuda" else base_model

    return base_model, model


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model, learning_rate, weight_decay, device):
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=(device == "cuda")
        )
    except Exception:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(step, learning_rate, min_learning_rate, warmup_steps, steps):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    decay = (step - warmup_steps) / (steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))

    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


# ============================================================
# TRAIN MODEL
# ============================================================

# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(model, optimizer, train_loader, steps, learning_rate, min_learning_rate,
                warmup_steps, max_grad_norm, device, use_bf16, rho, model_seed):

    model.train()
    train_iter = iter(train_loader)

    for step in tqdm(range(steps), desc=f"Train rho={rho:.2f} seed={model_seed}", leave=False):

        try:
            xb, yb, mb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb, mb = next(train_iter)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        lr = get_lr(step, learning_rate, min_learning_rate, warmup_steps, steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(xb, targets=yb, mask=mb)
        else:
            _, loss = model(xb, targets=yb, mask=mb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return loss.item()

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, val_loaders, tokenizer, max_new_tokens, newline_id,
                   device, rho, model_seed):

    model.eval()

    correct = 0
    separator_count = 0
    total = 0

    total_batches = sum(len(loader) for loader in val_loaders.values())

    with torch.inference_mode():

        with tqdm(
            total=total_batches,
            desc=f"Val rho={rho:.2f} seed={model_seed}",
            leave=False
        ) as bar:

            for prompt_len, loader in sorted(val_loaders.items()):

                for context, prompts, golds in loader:

                    context = context.to(device, non_blocking=True)

                    out = model.generate(
                        context,
                        max_new_tokens=max_new_tokens,
                        stop_id=newline_id,
                        greedy=True
                    )

                    for row, gold in zip(out.tolist(), golds):

                        generated_ids = row[prompt_len:]
                        generated = tokenizer.decode(generated_ids).split("\n", 1)[0]

                        pred = None

                        if ":" in generated:
                            separator_count += 1

                            nums = re.findall(
                                r"\d+",
                                generated.rsplit(":", 1)[1]
                            )

                            if nums:
                                pred = nums[0]

                        correct += int(pred == gold)
                        total += 1

                    bar.update(1)

    return correct / total, separator_count / total


# ============================================================
# CLEANUP
# ============================================================

def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()