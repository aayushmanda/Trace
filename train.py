"""Learning rate schedule and the single-model training loop."""

import math
import random

import torch

from config import (
    batch_size,
    device,
    learning_rate,
    max_grad_norm,
    min_learning_rate,
    steps_per_run,
    warmup_steps,
    weight_decay,
)
from data import get_batch_from_fixed_dataset
from model import GPTModel


def get_lr(step, total_steps=steps_per_run):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step > total_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(train_dataset_tensors, seed, n_steps=steps_per_run, batch_size = 256):
    """Trains a fresh GPTModel on the given dataset and returns it."""
    set_seed(seed)

    model = GPTModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.train()
    for step in range(n_steps):
        lr = get_lr(step, total_steps=n_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb, mb = get_batch_from_fixed_dataset(train_dataset_tensors, batch_size)
        _, loss = model(xb, yb, mb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return model
