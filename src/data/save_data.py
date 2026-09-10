import os
import random
from typing import List

from src.data.dataclass import Instance
from src.data.registry import TASKS


def save_mixed_trace_file(instances: List[Instance], correct_ratio: float, output_file: str, seed: int = 100):
    if not 0.0 <= correct_ratio <= 1.0:
        raise ValueError("correct_ratio must be between 0 and 1")
    rng = random.Random(seed)
    unique = {}
    for inst in instances:
        unique.setdefault(inst.prompt, inst)
    instances = list(unique.values())
    rng.shuffle(instances)
    n = len(instances)
    n_correct = int(round(n * correct_ratio))
    lines = [f"{inst.prompt} {inst.correct_trace} : {inst.gold}\n" for inst in instances[:n_correct]]
    lines += [f"{inst.prompt} {inst.wrong_trace} : {inst.gold}\n" for inst in instances[n_correct:]]
    rng.shuffle(lines)
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
