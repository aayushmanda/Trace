"""
data.py
Tokenizer, corpus builders, and batch construction.
"""
import random
import string
import torch


class Tokenizer:
    def __init__(self, extra_chars: str):
        self.chars = sorted(list(set(extra_chars)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi["\n"]

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i in self.itos)


def build_rho_corpus(n: int, rho: float, seed: int = None):
    """
    Build a training corpus of (prompt, target) pairs where `rho` controls
    the fraction/strength of some structured signal vs. noise.

    NOTE: this is a stub — replace with your actual corpus-generation logic.
    The original snippet referenced this function without defining it.
    """
    if seed is not None:
        random.seed(seed)

    examples = []
    for _ in range(n):
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        if random.random() < rho:
            prompt = f"{a}+{b}="
            target = f"{a + b}\n"
        else:
            prompt = f"{a}+{b}="
            target = f"{random.randint(0, 1998)}\n"
        examples.append((prompt, target))
    return examples


def build_clean_val_set(n: int, seed: int = None):
    """Clean (noise-free) validation set: always the correct sum."""
    if seed is not None:
        random.seed(seed)

    examples = []
    for _ in range(n):
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        prompt = f"{a}+{b}="
        target = f"{a + b}\n"
        examples.append((prompt, target))
    return examples


def make_batch(examples_list, tokenizer: Tokenizer, batch_size: int, block_size: int, device: str):
    batch = random.sample(examples_list, min(batch_size, len(examples_list)))
    xs, ys, masks = [], [], []

    for prompt, target in batch:
        p_ids = tokenizer.encode(prompt)
        t_ids = tokenizer.encode(target)
        full = (p_ids + t_ids)[:block_size]
        Lp = min(len(p_ids), len(full))
        L = len(full)

        x = full[:-1]
        y = full[1:]
        mask = [1 if (i + 1) >= Lp else 0 for i in range(L - 1)]

        pad_len = (block_size - 1) - len(x)
        x = x + [tokenizer.pad_id] * pad_len
        y = y + [tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len

        xs.append(x)
        ys.append(y)
        masks.append(mask)

    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.float, device=device),
    )