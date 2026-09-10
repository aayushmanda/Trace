import torch
from torch.utils.data import Dataset

from src.data.dataclass import ANSWER_SEP, Instance


TARGET_BUILDERS = {
    "outcome": lambda inst: f"{ANSWER_SEP}{inst.gold}\n",
    "answer_first": lambda inst: f"{ANSWER_SEP}{inst.gold} ; {inst.correct_trace}\n",
    "filler": lambda inst: f" {'.' * len(inst.correct_trace)}{ANSWER_SEP}{inst.gold}\n",
    "process": lambda inst: f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n",
    "corrupted": lambda inst: f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}\n",
}


def encode_pair(tokenizer, prompt: str, target: str, block_size: int):
    prompt_ids = tokenizer.encode(prompt)
    full = prompt_ids + tokenizer.encode(target)
    if len(full) > block_size:
        raise ValueError(f"{len(full)} tokens exceeds block_size={block_size}")
    width = block_size - 1
    x = [tokenizer.pad_id] * width
    y = [tokenizer.pad_id] * width
    mask = [0.0] * width
    n = len(full) - 1
    x[:n] = full[:-1]
    y[:n] = full[1:]
    for i in range(len(prompt_ids) - 1, n):
        mask[i] = 1.0
    if not (len(x) == len(y) == len(mask) == width):
        raise RuntimeError(f"encode_pair ranks {len(x), len(y), len(mask)} != {width}")
    return x, y, mask


class ContinuationDataset(Dataset):
    """Supervised continuation; loss is on the target tokens only."""

    def __init__(self, instances, tokenizer, block_size, targets):
        if len(instances) != len(targets):
            raise ValueError("instances and targets must align")
        if tokenizer.vocab_size > 256:
            raise ValueError("uint8 storage requires tokenizer.vocab_size <= 256")
        width = block_size - 1
        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)
        for row, (inst, target) in enumerate(zip(instances, targets)):
            if isinstance(target, Instance):
                raise TypeError("pass target strings, not Instance")
            x, y, mask = encode_pair(tokenizer, inst.prompt, target, block_size)
            self.x[row] = torch.tensor(x, dtype=torch.uint8)
            self.y[row] = torch.tensor(y, dtype=torch.uint8)
            self.mask[row] = torch.tensor(mask, dtype=torch.bool)
        if self.x.ndim != 2 or self.x.shape != self.y.shape or self.mask.shape != self.x.shape:
            raise RuntimeError(f"dataset tensors must be (N, T), got x={tuple(self.x.shape)}")

    def to(self, device):
        """Optional GPU-resident copies for small GPT sets (`data_on_device: true`)."""
        self.x = self.x.to(device, non_blocking=True)
        self.y = self.y.to(device, non_blocking=True)
        self.mask = self.mask.to(device, non_blocking=True)
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]


class SupervisionDataset(ContinuationDataset):
    def __init__(self, instances, task, mode: str):
        if mode not in TARGET_BUILDERS:
            raise ValueError(f"Unknown mode: {mode}")
        targets = [TARGET_BUILDERS[mode](inst) for inst in instances]
        super().__init__(instances, task.tokenizer, task.block_size, targets)


class RatioDataset(ContinuationDataset):
    def __init__(self, instances, task, condition: str, rho=None, ratio_scores=None):
        if condition not in {"outcome", "mixed_process"}:
            raise ValueError(f"unknown condition: {condition}")
        if condition == "mixed_process" and (rho is None or ratio_scores is None):
            raise ValueError("mixed_process requires rho and ratio_scores")
        targets = []
        for row, inst in enumerate(instances):
            if condition == "outcome":
                targets.append(f"{ANSWER_SEP}{inst.gold}\n")
            else:
                trace = inst.correct_trace if ratio_scores[row] < rho else inst.wrong_trace
                targets.append(f" {trace}{ANSWER_SEP}{inst.gold}\n")
        super().__init__(instances, task.tokenizer, task.block_size, targets)
