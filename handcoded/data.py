"""Teacher-forced next-token batches for outcome vs process continuations."""
from dataclasses import dataclass
import random

import torch
from torch.nn import functional as F

from handcoded.config import BATCH_SEED


@dataclass
class LanguageBatch:
    inputs: torch.Tensor
    targets: torch.Tensor

    def __len__(self):
        return len(self.inputs)

    def select(self, indices):
        return LanguageBatch(self.inputs[indices], self.targets[indices])

    def to(self, device):
        return LanguageBatch(self.inputs.to(device), self.targets.to(device))


def encode_dataset(circuits, tokenizer, mode):
    """Prompt tokens are ignored in the loss (`-100`); continuation tokens are trained."""
    inputs, targets = [], []
    for circuit in circuits:
        prompt = tokenizer.prompt(circuit)
        full = prompt + tokenizer.continuation(circuit, mode)
        inputs.append(full[:-1])
        target = full[1:]
        target[: len(prompt) - 1] = [-100] * (len(prompt) - 1)
        targets.append(target)
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise RuntimeError(f"encode_dataset expected (N, T) pair, got {tuple(inputs.shape)} {tuple(targets.shape)}")
    return LanguageBatch(inputs, targets)


def language_model_loss(model, batch):
    logits = model(batch.inputs)
    return F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100)


def make_batch_schedule(size, steps, batch_size, seed=BATCH_SEED):
    rng = random.Random(seed)
    return [[rng.randrange(size) for _ in range(batch_size)] for _ in range(steps)]
