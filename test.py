# experiment.py

import os
import random

import numpy as np
import torch

from config import *
from src.registry import TASKS

from src.utils import (
    TrainDataset,
    ValDataset,
    generate_validation_file,
    build_val_loaders,
    build_train_loader,
    build_model,
    build_optimizer,
    train_model,
    evaluate_model,
    set_seed,
    cleanup,
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.set_float32_matmul_precision("high")

use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

print("Device:", device)
print("BF16:", use_bf16)


# ============================================================
# TASK
# ============================================================

task = TASKS[task_name]
tokenizer = task.tokenizer

block_size = task.block_size
max_new_tokens = task.max_new_tokens

pad_id = tokenizer.pad_id
newline_id = tokenizer.newline_id
vocab_size = tokenizer.vocab_size

print("Task:", task_name)
print("Block size:", block_size)
print("Vocab size:", vocab_size)


# ============================================================
# VALIDATION
# ============================================================

val_file = f"{task_name}/{task_name}_val.txt"

if REGENERATE_VAL or not os.path.exists(val_file):
    generate_validation_file(task, task_name, val_size, val_seed)

val_dataset = ValDataset(val_file, tokenizer, block_size)

assert len(val_dataset) == val_size

print("Validation examples:", len(val_dataset))

val_loaders = build_val_loaders(
    val_dataset=val_dataset,
    val_batch_size=val_batch_size,
    device=device
)


# ============================================================
# RESULTS
# ============================================================

all_accuracies = {rho: [] for rho in rho_values}
all_separator_rates = {rho: [] for rho in rho_values}


# ============================================================
# EXPERIMENT
# ============================================================

for rho in rho_values:

    print("\n" + "=" * 70)
    print(f"RHO = {rho:.2f}")
    print("=" * 70)

    data_file = f"{task_name}/{task_name}_rho_{int(round(rho * 100))}.txt"


    train_dataset = TrainDataset(
        file_path=data_file,
        tokenizer=tokenizer,
        block_size=block_size,
        pad_id=pad_id
    )


    for model_seed in model_seeds:

        print(f"\nrho={rho:.2f} | seed={model_seed}")

        set_seed(model_seed)


        # ====================================================
        # DATALOADER
        # ====================================================

        train_loader, loader_generator = build_train_loader(
            train_dataset=train_dataset,
            batch_size=batch_size,
            batch_seed=batch_seed,
            num_workers=num_workers,
            device=device
        )


        # ====================================================
        # MODEL
        # ====================================================

        base_model, model = build_model(
            vocab_size=vocab_size,
            block_size=block_size,
            pad_id=pad_id,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=device,
            use_compile=USE_COMPILE
        )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer = build_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device
        )


        # ====================================================
        # TRAIN
        # ====================================================

        final_loss = train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            steps=steps,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            device=device,
            use_bf16=use_bf16,
            rho=rho,
            model_seed=model_seed
        )


        # ====================================================
        # VALIDATE
        # ====================================================

        val_acc, separator_rate = evaluate_model(
            model=base_model,
            val_loaders=val_loaders,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            newline_id=newline_id,
            device=device,
            rho=rho,
            model_seed=model_seed
        )


        all_accuracies[rho].append(val_acc)
        all_separator_rates[rho].append(separator_rate)


        print(
            f"rho={rho:.2f} | "
            f"seed={model_seed} | "
            f"loss={final_loss:.4f} | "
            f"accuracy={val_acc * 100:.2f}% | "
            f"separator={separator_rate * 100:.2f}%"
        )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        if SAVE_MODELS:
            path = f"{task_name}/{task_name}_rho_{int(round(rho * 100))}_seed_{model_seed}.pt"
            torch.save(base_model.state_dict(), path)
            print("Saved:", path)


        # ====================================================
        # CLEAN
        # ====================================================

        del train_loader, loader_generator, optimizer, model, base_model
        cleanup()


    del train_dataset
    cleanup()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

for rho in rho_values:

    values = np.array(all_accuracies[rho]) * 100

    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0

    print(
        f"rho={rho:.2f} | "
        f"mean={mean:.2f}% | "
        f"std={std:.2f}% | "
        f"runs={values}"
    )