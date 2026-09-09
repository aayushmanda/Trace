import os
import random
from typing import List

from task import Instance
from registry import TASKS


def save_mixed_trace_file(
    instances: List[Instance],
    correct_ratio: float,
    output_file: str,
    seed: int = 100,
):
    if not 0.0 <= correct_ratio <= 1.0:
        raise ValueError("correct_ratio must be between 0 and 1")

    rng = random.Random(seed)

    unique = {}
    for inst in instances:
        if inst.prompt not in unique:
            unique[inst.prompt] = inst

    instances = list(unique.values())
    rng.shuffle(instances)

    n = len(instances)
    n_correct = int(round(n * correct_ratio))
    n_wrong = n - n_correct

    correct_instances = instances[:n_correct]
    wrong_instances = instances[n_correct:]

    lines = []

    for inst in correct_instances:
        lines.append(f"{inst.prompt} {inst.correct_trace} : {inst.gold}\n")

    for inst in wrong_instances:
        lines.append(f"{inst.prompt} {inst.wrong_trace} : {inst.gold}\n")

    rng.shuffle(lines)

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Saved: {output_file}")
    print(f"Total: {n}")
    print(f"Correct: {n_correct} ({100*n_correct/n:.2f}%)")
    print(f"Wrong: {n_wrong} ({100*n_wrong/n:.2f}%)")


if __name__ == "__main__":

    DATASET_SIZE = 10
    CORRECT_RATIO = 0.5

    instances = [
        TASKS["word_index"].sample()
        for _ in range(DATASET_SIZE)
    ]

    save_mixed_trace_file(
        instances,
        correct_ratio=CORRECT_RATIO,
        output_file="data/word_index_rho_050.txt",
        seed=100,
    )