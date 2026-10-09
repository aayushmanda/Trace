"""Training utilities for the character-token GPT: config, seeding, device, loader, loop, CSV output."""
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm as _tqdm

from src.gpt import GPTModel

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    """Write dict rows; the header is the union of keys in first-seen order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(prefer=("cuda:2", "cuda:3")):
    """Prefer the shared box's free A100s; cuda:0 if CUDA_VISIBLE_DEVICES is set."""
    if not torch.cuda.is_available():
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    n = torch.cuda.device_count()
    for name in prefer:
        if int(name.split(":")[1]) < n:
            return name
    return "cuda:0"


def configure_device(device):
    """TF32 and cuDNN benchmark on CUDA, as in the paper's runs (not bitwise deterministic)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return device


def autocast_context(device, enabled=True):
    """bfloat16 autocast on CUDA (the paper's runs use it)."""
    use = enabled and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use)


def build_gpt(task, args, device):
    tok = task.tokenizer
    return GPTModel(
        vocab_size=tok.vocab_size,
        block_size=task.block_size,
        pad_id=tok.pad_id,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        dropout=args.dropout,
    ).to(device)


def make_optimizer(model, args, device):
    params = list(model.parameters())
    fused = torch.device(device).type == "cuda"
    try:
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def make_loader(dataset, args, device):
    """Shuffled minibatches; the order depends only on batch_seed, not on the model seed."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.batch_seed),
        pin_memory=torch.device(device).type == "cuda",
    )


def progress(iterable=None, **kwargs):
    """tqdm, silenced by TRACE_TQDM=0 or under a test runner."""
    quiet = os.environ.get("TRACE_TQDM", "1") in {"0", "false", "False"}
    quiet = quiet or any("unittest" in arg or "pytest" in arg for arg in sys.argv)
    kwargs.setdefault("disable", quiet)
    return _tqdm(iterable, **kwargs)


def gpt_lm_loss(model, batch, device):
    x, y, mask = batch
    x = x.to(device, dtype=torch.long, non_blocking=True)
    y = y.to(device, dtype=torch.long, non_blocking=True)
    mask = mask.to(device, dtype=torch.float32, non_blocking=True)
    _, loss = model(x, targets=y, mask=mask)
    return loss


def train_steps(model, loader, optimizer, device, n_steps, grad_clip=1.0, bf16=True, desc=None):
    """Run n_steps AdamW updates, cycling the loader; returns the last loss."""
    iterator = iter(loader)
    model.train()
    loss = None
    bar = progress(range(n_steps), desc=desc or "train", leave=False)
    for _ in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bf16):
            loss = gpt_lm_loss(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return float(loss.detach())
