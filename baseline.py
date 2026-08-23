# baseline_no_trace.py

import os
import random
import re

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from config import *
from src.registry import TASKS
from src.utils import (
    TrainDataset,
    ValDataset,
    build_val_loaders,
    build_train_loader,
    build_model,
    build_optimizer,
    get_lr,
    set_seed,
    cleanup,
)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_training_instances(task, size, seed):
    """Generate exactly the same underlying pool as the rho experiment."""
    print(f"Generating training pool: n={size}, seed={seed}")

    old_state = random.getstate()
    random.seed(seed)

    instances = [task.sample() for _ in tqdm(range(size), desc="Generate train")]

    random.setstate(old_state)
    return instances


def save_no_trace_file(instances, output_file, seed):
    """
    Save:
        prompt : gold

    Two shuffles intentionally mirror save_mixed_trace_file(),
    keeping prompt ordering comparable to the rho experiments.
    """
    rng = random.Random(seed)

    # Same deduplication as existing save_data.py
    unique = {}
    for inst in instances:
        if inst.prompt not in unique:
            unique[inst.prompt] = inst

    instances = list(unique.values())

    # First shuffle: same as save_mixed_trace_file
    rng.shuffle(instances)

    lines = [
        f"{inst.prompt} : {inst.gold}\n"
        for inst in instances
    ]

    # Second shuffle: same as save_mixed_trace_file
    rng.shuffle(lines)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nSaved no-trace dataset: {output_file}")
    print(f"Unique training examples: {len(lines)}")


def generate_validation(task, train_prompts, output_file, size, seed):
    """
    Same validation logic as the main experiment.

    Validation still contains a correct trace in the file, but ValDataset
    discards that trace during inference and presents only the prompt.
    """
    random.seed(seed)

    val_instances = []
    val_prompts = set()

    for _ in tqdm(range(size), desc="Generate validation"):
        while True:
            inst = task.sample()

            if inst.prompt in train_prompts:
                continue

            if inst.prompt in val_prompts:
                continue

            val_instances.append(inst)
            val_prompts.add(inst.prompt)
            break

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for inst in val_instances:
            f.write(
                f"{inst.prompt} {inst.correct_trace} : {inst.gold}\n"
            )

    print(f"Saved validation: {output_file}")
    print(f"Validation examples: {len(val_instances)}")


# ============================================================
# TRAIN
# ============================================================

def train_baseline(
    model,
    optimizer,
    train_loader,
    steps,
    device,
    use_bf16,
    model_seed,
    wandb_run=None,
):
    model.train()
    train_iter = iter(train_loader)

    final_loss = None

    for step in tqdm(
        range(steps),
        desc=f"Train no-trace seed={model_seed}",
    ):
        try:
            xb, yb, mb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb, mb = next(train_iter)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        lr = get_lr(
            step,
            learning_rate,
            min_learning_rate,
            warmup_steps,
            steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if use_bf16:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                _, loss = model(
                    xb,
                    targets=yb,
                    mask=mb,
                )
        else:
            _, loss = model(
                xb,
                targets=yb,
                mask=mb,
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()

        final_loss = loss.item()

        if wandb_run is not None and (
            step % WANDB_LOG_EVERY == 0 or step == steps - 1
        ):
            wandb_run.log(
                {
                    "train/loss": final_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/examples_seen": (step + 1) * len(xb),
                },
                step=step,
            )

    return final_loss


# ============================================================
# EVALUATION
# ============================================================

def evaluate_baseline(
    model,
    val_loaders,
    tokenizer,
    max_new_tokens,
    newline_id,
    device,
    model_seed,
):
    model.eval()

    correct = 0
    separator_count = 0
    total = 0

    with torch.inference_mode():

        for prompt_len, loader in sorted(val_loaders.items()):

            for context, prompts, golds in tqdm(
                loader,
                desc=f"Val no-trace seed={model_seed}",
                leave=False,
            ):
                context = context.to(device, non_blocking=True)

                out = model.generate(
                    context,
                    max_new_tokens=max_new_tokens,
                    stop_id=newline_id,
                    greedy=True,
                )

                for row, gold in zip(out.tolist(), golds):

                    generated_ids = row[prompt_len:]
                    generated = tokenizer.decode(
                        generated_ids
                    ).split("\n", 1)[0]

                    pred = None

                    if ":" in generated:
                        separator_count += 1

                        nums = re.findall(
                            r"\d+",
                            generated.rsplit(":", 1)[1],
                        )

                        if nums:
                            pred = nums[0]

                    correct += int(pred == gold)
                    total += 1

    return (
        correct / total,
        separator_count / total,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_no_trace_baseline():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    use_bf16 = (
        device == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    task = TASKS[task_name]

    tokenizer = task.tokenizer
    block_size = task.block_size
    max_new_tokens = task.max_new_tokens

    pad_id = tokenizer.pad_id
    newline_id = tokenizer.newline_id
    vocab_size = tokenizer.vocab_size

    print("=" * 70)
    print("NO-TRACE BASELINE")
    print("=" * 70)

    print("Task:", task_name)
    print("Device:", device)
    print("BF16:", use_bf16)
    print("Train size:", train_size)
    print("Steps:", steps)
    print("Model seeds:", model_seeds)

    # --------------------------------------------------------
    # Generate SAME underlying training instances
    # --------------------------------------------------------

    train_file = (
        f"{task_name}/{task_name}_no_trace.txt"
    )

    instances = generate_training_instances(
        task,
        train_size,
        train_seed,
    )

    # Training prompts used for validation exclusion
    unique = {}
    for inst in instances:
        if inst.prompt not in unique:
            unique[inst.prompt] = inst

    train_prompts = set(unique.keys())

    if not os.path.exists(train_file):
        save_no_trace_file(
            instances=instances,
            output_file=train_file,
            seed=train_seed,
        )
    else:
        print("Using existing:", train_file)

    del instances

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_file = f"{task_name}/{task_name}_val.txt"

    if REGENERATE_VAL or not os.path.exists(val_file):
        generate_validation(
            task=task,
            train_prompts=train_prompts,
            output_file=val_file,
            size=val_size,
            seed=val_seed,
        )
    else:
        print("Using existing validation:", val_file)

    del train_prompts

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TrainDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        block_size=block_size,
        pad_id=pad_id,
    )

    val_dataset = ValDataset(
        val_file,
        tokenizer,
        block_size,
    )

    val_loaders = build_val_loaders(
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        device=device,
    )

    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    # --------------------------------------------------------
    # Run seeds
    # --------------------------------------------------------

    all_results = {}

    for batch_size in batch_sizes:

        accuracies = []

        print("\n" + "#" * 70)
        print("BATCH SIZE:", batch_size)
        print("#" * 70)

        for model_seed in model_seeds:

            print(
                f"\nNO TRACE | batch={batch_size} "
                f"| seed={model_seed}"
            )

            set_seed(model_seed)

            run = None

            if USE_WANDB:
                run = wandb.init(
                    project=WANDB_PROJECT,
                    name=(
                        f"{task_name}_no_trace_"
                        f"bs{batch_size}_seed{model_seed}"
                    ),
                    group=f"{task_name}_no_trace",
                    config={
                        "condition": "no_trace",
                        "task": task_name,
                        "batch_size": batch_size,
                        "model_seed": model_seed,
                        "train_seed": train_seed,
                        "batch_seed": batch_seed,
                        "steps": steps,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                        "learning_rate": learning_rate,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                    },
                )

            train_loader, loader_generator = build_train_loader(
                train_dataset=train_dataset,
                batch_size=batch_size,
                batch_seed=batch_seed,
                num_workers=num_workers,
                device=device,
            )

            base_model, model = build_model(
                vocab_size=vocab_size,
                block_size=block_size,
                pad_id=pad_id,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                device=device,
                use_compile=USE_COMPILE,
            )

            optimizer = build_optimizer(
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
            )

            final_loss = train_baseline(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                steps=steps,
                device=device,
                use_bf16=use_bf16,
                model_seed=model_seed,
                wandb_run=run,
            )

            accuracy, separator_rate = evaluate_baseline(
                model=base_model,
                val_loaders=val_loaders,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                newline_id=newline_id,
                device=device,
                model_seed=model_seed,
            )

            accuracies.append(accuracy)

            print(
                f"seed={model_seed} | "
                f"loss={final_loss:.4f} | "
                f"accuracy={100 * accuracy:.2f}% | "
                f"separator={100 * separator_rate:.2f}%"
            )

            if run is not None:
                run.log(
                    {
                        "val/accuracy": accuracy,
                        "val/accuracy_percent": 100 * accuracy,
                        "val/separator_rate": separator_rate,
                        "final/train_loss": final_loss,
                    },
                    step=steps,
                )

                run.finish()

            if SAVE_MODELS:
                path = (
                    f"{task_name}/"
                    f"{task_name}_no_trace_"
                    f"batch_{batch_size}_"
                    f"seed_{model_seed}.pt"
                )

                torch.save(
                    base_model.state_dict(),
                    path,
                )

                print("Saved:", path)

            del train_loader
            del loader_generator
            del optimizer
            del model
            del base_model

            cleanup()

        values = np.asarray(accuracies) * 100

        mean = values.mean()
        std = (
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )

        all_results[batch_size] = {
            "runs": values.tolist(),
            "mean": mean,
            "std": std,
        }

        print("\nNO-TRACE RESULT")
        print(
            f"batch={batch_size} | "
            f"mean={mean:.2f}% | "
            f"std={std:.2f}% | "
            f"runs={values}"
        )

    return all_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_no_trace_baseline()

    print("\n" + "=" * 70)
    print("FINAL NO-TRACE BASELINE")
    print("=" * 70)

    for batch_size, result in results.items():
        print(
            f"batch={batch_size} | "
            f"mean={result['mean']:.2f}% | "
            f"std={result['std']:.2f}% | "
            f"runs={result['runs']}"
        )