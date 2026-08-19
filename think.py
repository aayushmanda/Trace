import os
import math
import random
import re
import string
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTModel
from tokenizer import CharTokenizer
from dataclass import Instance, Task


# ============================================================
# Global Versioning & Hyper-parameters
# ============================================================
DATASET_VERSION = "v5_matched_strict"

batch_size = 256
steps_per_run = 6000
learning_rate = 3e-4
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
dropout = 0.05
label_smoothing = 0.0
max_grad_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 128
n_head = 4
n_layer = 4

DATASET_SIZE = 50000
DATASET_SEED = 100
VAL_SEED = 99999
CACHE_DIR = "./dataset_cache"

# Global unified vocabulary across all tasks
GLOBAL_CHARS = string.ascii_lowercase + string.digits + " ;:->+=*\n"

# ============================================================
# Choose task
# ============================================================
TASK_NAME = "word_index"  # "word_index" | "sort_letters" | "multiply" | "count_char" | "graph_path"

# ============================================================
# Task family
# ============================================================
ANSWER_SEP = " : "

GLOBAL_TOKENIZER = CharTokenizer(GLOBAL_CHARS)


from task import _sample_word_index
# ============================================================
# Registry
# ============================================================
from registry import TASKS

def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}")
    return TASKS[name]


# ============================================================
# Active task
# ============================================================
task = get_task(TASK_NAME)
tokenizer = task.tokenizer
block_size = task.block_size
max_new_tokens = task.max_new_tokens
PAD_ID = tokenizer.pad_id
NEWLINE_ID = tokenizer.newline_id
vocab_size = tokenizer.vocab_size

print(f"[TASK] {task.name} | vocab={vocab_size} | block={block_size} | "
      f"chance≈{task.chance_acc:.3f} | ceiling={task.ceiling_acc:.3f}")


# ============================================================
# Data Pools
# ============================================================
def get_or_create_master_pools(n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    fp = task.fingerprint()
    correct_path = os.path.join(cache_dir, f"{task.name}_correct_{n_samples}_{fp}.pt")
    wrong_path = os.path.join(cache_dir, f"{task.name}_wrong_{n_samples}_{fp}.pt")
    prompts_path = os.path.join(cache_dir, f"{task.name}_prompts_{n_samples}_{fp}.pt")

    if os.path.exists(correct_path) and os.path.exists(wrong_path) and os.path.exists(prompts_path):
        print(f"\n[DISK] Loading cached pools for {task.name} ...")
        correct = tuple(t.to(device) for t in torch.load(correct_path, map_location=device, weights_only=True))
        wrong = tuple(t.to(device) for t in torch.load(wrong_path, map_location=device, weights_only=True))
        prompts = torch.load(prompts_path, weights_only=False)
        print("[DISK] Loaded.\n")
        return correct, wrong, prompts

    print(f"\n[GEN] Generating master pools for {task.name} ({n_samples:,} samples)...")
    rng_state = random.getstate()
    random.seed(data_seed)

    correct_xs, correct_ys, correct_masks = [], [], []
    wrong_xs, wrong_ys, wrong_masks = [], [], []
    prompts = []

    for _ in range(n_samples):
        inst = task.sample()
        prompts.append(inst.prompt)

        for mode, xs, ys, masks in [
            ("correct_think", correct_xs, correct_ys, correct_masks),
            ("wrong_think", wrong_xs, wrong_ys, wrong_masks),
        ]:
            prompt, answer = task.render(inst, mode)
            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(answer + "\n")
            full = p_ids + t_ids

            if len(full) > block_size:
                raise ValueError(
                    f"Task '{task.name}' sequence length ({len(full)}) exceeds block_size ({block_size}). "
                    f"Prompt: {prompt!r}, Answer: {answer!r}"
                )

            Lp = len(p_ids)
            x, y = full[:-1], full[1:]
            mask = [1.0 if (i + 1) >= Lp else 0.0 for i in range(len(full) - 1)]
            pad = (block_size - 1) - len(x)
            x += [PAD_ID] * pad
            y += [PAD_ID] * pad
            mask += [0.0] * pad
            xs.append(x); ys.append(y); masks.append(mask)

    random.setstate(rng_state)

    master_correct = (
        torch.tensor(correct_xs, dtype=torch.long),
        torch.tensor(correct_ys, dtype=torch.long),
        torch.tensor(correct_masks, dtype=torch.float),
    )
    master_wrong = (
        torch.tensor(wrong_xs, dtype=torch.long),
        torch.tensor(wrong_ys, dtype=torch.long),
        torch.tensor(wrong_masks, dtype=torch.float),
    )

    print("[DISK] Saving pools ...")
    torch.save(master_correct, correct_path)
    torch.save(master_wrong, wrong_path)
    torch.save(prompts, prompts_path)
    print("[DISK] Saved.\n")

    return (tuple(t.to(device) for t in master_correct),
            tuple(t.to(device) for t in master_wrong),
            prompts)


def get_mixed_dataset_for_ratio(master_correct, master_wrong, ratio, seed=None):
    n = master_correct[0].shape[0]
    n_correct = int(n * ratio)

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g, device=device)
    else:
        perm = torch.randperm(n, device=device)

    correct_idx = perm[:n_correct]
    wrong_idx = perm[n_correct:]

    c = [t[correct_idx] for t in master_correct]
    w = [t[wrong_idx] for t in master_wrong]
    return tuple(torch.cat([c[i], w[i]], dim=0) for i in range(3))


def get_batch_from_fixed_dataset(dataset_tensors, batch_size):
    xs, ys, masks = dataset_tensors
    ix = torch.randint(0, xs.shape[0], (batch_size,), device=device)
    return xs[ix], ys[ix], masks[ix]


def build_validation_dataset(n_words=1000, master_prompts=None, val_seed=VAL_SEED) -> List[Instance]:
    rng_state = random.getstate()
    random.seed(val_seed)

    master_prompts_set = set(master_prompts) if master_prompts is not None else set()
    instances = []
    seen = set()

    attempts = 0
    max_attempts = n_words * 20

    while len(instances) < n_words and attempts < max_attempts:
        attempts += 1
        inst = task.sample()
        if inst.prompt in seen or inst.prompt in master_prompts_set:
            continue
        seen.add(inst.prompt)
        instances.append(inst)

    random.setstate(rng_state)

    if master_prompts_set:
        val_prompts = {inst.prompt for inst in instances}
        assert val_prompts.isdisjoint(master_prompts_set), "Validation set overlaps with master training pool!"

    return instances



def get_lr(step, total_steps=steps_per_run):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step > total_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


# ============================================================
# Evaluation (free-generation only; forced_correct/forced_wrong removed)
# ============================================================
@torch.no_grad()
def evaluate_all_modes(model, test_instances: List[Instance]) -> Dict[str, float]:
    model.eval()
    counts = {"free": 0}

    for inst in test_instances:
        ctx_free = task.context(inst, "free")
        ids_free = torch.tensor([tokenizer.encode(ctx_free)], dtype=torch.long, device=device)
        out_free = model.generate(ids_free, max_new_tokens=max_new_tokens, stop_id=NEWLINE_ID)[0].tolist()
        pred_free = task.extract_answer(tokenizer.decode(out_free)[len(ctx_free):])
        counts["free"] += int(pred_free == inst.gold)

    model.train()
    n = len(test_instances)
    return {k: v / n for k, v in counts.items()}


def save_phase_transition_plot(results, replicates, filename=None):
    if filename is None:
        filename = f"phase_transition_{task.name}.png"
    ratios_pct = [r * 100 for r in results.keys()]
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, rep in enumerate(replicates):
        rep_accs = [results[r][i]["free"] * 100 if i < len(results[r]) else None for r in results]
        plt.scatter(ratios_pct, rep_accs, color=colors[i % len(colors)],
                    alpha=0.7, s=25, zorder=3, label=f"Replicate {rep}")

    means = [(sum(r_dict["free"] for r_dict in accs) / len(accs)) * 100 for accs in results.values()]
    stds = [((sum(((r_dict["free"] * 100) - m) ** 2 for r_dict in accs) / len(accs)) ** 0.5
             if len(accs) > 1 else 0.0) for m, accs in zip(means, results.values())]

    lower = [max(0.0, m - s) for m, s in zip(means, stds)]
    upper = [min(100.0, m + s) for m, s in zip(means, stds)]

    plt.fill_between(ratios_pct, lower, upper, color="#1a73e8", alpha=0.15, label="±1 Std Dev")
    plt.plot(ratios_pct, means, color="#1a73e8", marker="o", linewidth=2.5,
             markersize=6, zorder=4, label="Mean Free-Gen Acc")

    plt.axhline(task.chance_acc * 100, color="gray", ls="--", alpha=0.6, label="Chance")
    plt.axhline(task.ceiling_acc * 100, color="green", ls="--", alpha=0.6, label="Ceiling")

    plt.title(f"Free-Gen Accuracy vs. Correct Trace Ratio — {task.name}", fontsize=12, fontweight="bold")
    plt.xlabel("Correct Think Trace Ratio (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xticks(ratios_pct, [f"{int(r)}%" for r in ratios_pct], rotation=45)
    plt.ylim(-2, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot saved as '{filename}'")


# ============================================================
# Main experiment
# ============================================================
def run_clean_phase_experiment():
    print(f"Model Parameters: {sum(p.numel() for p in GPTModel().parameters() if p.requires_grad):,}")

    ratios_to_test = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.0]
    replicates_to_test = [2200, 1337, 2026, 2003, 10]
    results: Dict[float, List[Dict[str, float]]] = {}

    master_correct, master_wrong, master_prompts = get_or_create_master_pools()
    val_instances = build_validation_dataset(n_words=1000, master_prompts=master_prompts, val_seed=VAL_SEED)

    print("=" * 80)
    print(f"   PHASE TRANSITION EXPERIMENT — {task.name.upper()}")
    print(f"Steps: {steps_per_run} | Batch: {batch_size} | WD: {weight_decay} | Embed: {n_embd} | Layers: {n_layer}")
    print("=" * 80)

    for ratio_idx, ratio in enumerate(ratios_to_test):
        results[ratio] = []
        print(f"\n[Testing Ratio: {ratio*100:.1f}% Correct | {(1-ratio)*100:.1f}% Wrong]")

        for rep in replicates_to_test[:1]:  # all replicates, not just the first
            run_seed = rep                   # unique per (replicate, ratio) cell
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            train_tensors = get_mixed_dataset_for_ratio(
                master_correct, master_wrong, ratio, seed=run_seed
            )

            model = GPTModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            model.train()
            for step in range(steps_per_run):
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                xb, yb, mb = get_batch_from_fixed_dataset(train_tensors, batch_size)
                _, loss = model(xb, yb, mb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            eval_metrics = evaluate_all_modes(model, val_instances)
            results[ratio].append(eval_metrics)
            print(f"Replicate {rep} | Free Acc: {eval_metrics['free']*100:.2f}%")

    print("\n" + "=" * 80)
    print(" STATISTICAL SUMMARY (Free-Generation Accuracy)")
    print("=" * 80)
    print(f"{'Correct %':<10} | {'Wrong %':<10} | {'Mean Free %':<13} | {'Std Dev %':<10}")
    print("-" * 50)
    for ratio, metrics_list in results.items():
        frees = [m["free"] for m in metrics_list]
        mean_free = sum(frees) / len(frees)
        std_free = (sum((x - mean_free) ** 2 for x in frees) / len(frees)) ** 0.5 if len(frees) > 1 else 0.0
        print(f"{ratio*100:>8.1f}% | {(1-ratio)*100:>8.1f}% | {mean_free*100:>11.2f}% | ± {std_free*100:>6.2f}%")
    print("=" * 80)

    save_phase_transition_plot(results, replicates_to_test)


if __name__ == "__main__":
    run_clean_phase_experiment()