# experiment.py

import os
import random

import numpy as np
import torch
import wandb

from torch.utils.data import Subset

from config import *
from save_data import save_mixed_trace_file
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
# GENERATE TRAINING INSTANCES
# ============================================================

def generate_training_instances(task, size, seed):

    print(
        f"Generating deterministic training pool: "
        f"n={size}, seed={seed}"
    )

    old_state = random.getstate()

    random.seed(seed)

    instances = [
        task.sample()
        for _ in range(size)
    ]

    random.setstate(old_state)

    return instances


# ============================================================
# PREPARE TRAINING FILES
# ============================================================

def prepare_training_files(
    task,
    task_name,
    rho_values,
    train_size,
    train_seed,
):

    missing_files = []

    for rho in rho_values:

        rho_pct = int(round(rho * 100))

        data_file = (
            f"{task_name}/"
            f"{task_name}_rho_{rho_pct}.txt"
        )

        if not os.path.exists(data_file):
            missing_files.append(
                (rho, data_file)
            )

    if not missing_files:
        print("All required training files already exist.")
        return

    print(
        f"{len(missing_files)} training files missing."
    )

    # All rho values share the SAME underlying instances.
    instances = generate_training_instances(
        task=task,
        size=train_size,
        seed=train_seed,
    )

    for rho, data_file in missing_files:

        print(
            f"\nGenerating rho={rho:.2f} -> "
            f"{data_file}"
        )

        save_mixed_trace_file(
            instances=instances,
            correct_ratio=rho,
            output_file=data_file,
            seed=train_seed,
        )

    del instances


# ============================================================
# WANDB
# ============================================================

def create_wandb_run(
    task_name,
    batch_size,
    rho,
    model_seed,
    train_dataset_size,
    val_dataset_size,
    block_size,
    device,
    use_bf16,
    run_steps,
    test_mode,
):

    if not USE_WANDB:
        return None

    rho_pct = int(round(rho * 100))

    prefix = "test_" if test_mode else ""

    run = wandb.init(
        project=WANDB_PROJECT,

        name=(
            f"{prefix}"
            f"{task_name}"
            f"_bs{batch_size}"
            f"_rho{rho_pct}"
            f"_seed{model_seed}"
        ),

        group=(
            f"{prefix}"
            f"{task_name}_batch_{batch_size}"
        ),

        config={
            # Experiment
            "task": task_name,
            "rho": rho,
            "batch_size": batch_size,
            "model_seed": model_seed,
            "batch_seed": batch_seed,

            # Data
            "train_size": train_dataset_size,
            "val_size": val_dataset_size,
            "train_seed": train_seed,
            "val_seed": val_seed,

            # Training
            "steps": run_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,

            # Model
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "block_size": block_size,

            # Runtime
            "device": device,
            "bf16": use_bf16,
            "compile": USE_COMPILE,

            # Debug
            "test_mode": test_mode,
        },

        tags=[
            task_name,
            f"batch-{batch_size}",
            f"rho-{rho_pct}",
            f"seed-{model_seed}",
            "test" if test_mode else "experiment",
        ],

        save_code=True,
    )

    return run


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(test_mode=False):

    # ========================================================
    # DEVICE
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    use_bf16 = (
        device == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print("\n" + "=" * 80)
    print("TRACE EXPERIMENT")
    print("=" * 80)

    print("Device:", device)
    print("BF16:", use_bf16)
    print("Test mode:", test_mode)

    # ========================================================
    # TASK
    # ========================================================

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

    # ========================================================
    # TEST MODE / FULL MODE
    # ========================================================

    if test_mode:

        active_batch_sizes = [
            batch_sizes[0]
        ]

        active_rhos = [
            rho_values[0]
        ]

        active_model_seeds = [
            model_seeds[0]
        ]

        run_steps = 5
        test_val_size = 32

        print("\nSMOKE TEST")
        print("Batch sizes:", active_batch_sizes)
        print("Rhos:", active_rhos)
        print("Seeds:", active_model_seeds)
        print("Steps:", run_steps)
        print("Validation size:", test_val_size)

    else:

        active_batch_sizes = batch_sizes
        active_rhos = rho_values
        active_model_seeds = model_seeds

        run_steps = steps
        test_val_size = val_size

    # ========================================================
    # PREPARE TRAINING DATA
    # ========================================================

    # Validation generation requires rho=0 file because it uses
    # it to avoid train/validation prompt overlap.

    required_rhos = list(
        dict.fromkeys(
            [0.0] + list(active_rhos)
        )
    )

    prepare_training_files(
        task=task,
        task_name=task_name,
        rho_values=required_rhos,
        train_size=train_size,
        train_seed=train_seed,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_file = (
        f"{task_name}/"
        f"{task_name}_val.txt"
    )

    if (
        REGENERATE_VAL
        or not os.path.exists(val_file)
    ):

        print("Generating validation dataset...")

        generate_validation_file(
            task,
            task_name,
            val_size,
            val_seed,
        )

    full_val_dataset = ValDataset(
        val_file,
        tokenizer,
        block_size,
    )

    assert len(full_val_dataset) == val_size, (
        f"Expected {val_size} validation samples, "
        f"found {len(full_val_dataset)}"
    )

    if test_mode:

        subset_size = min(
            test_val_size,
            len(full_val_dataset),
        )

        val_dataset = Subset(
            full_val_dataset,
            range(subset_size),
        )

    else:

        val_dataset = full_val_dataset

    print(
        "Validation examples:",
        len(val_dataset),
    )

    val_loaders = build_val_loaders(
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        device=device,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    all_accuracies = {
        bs: {
            rho: []
            for rho in active_rhos
        }
        for bs in active_batch_sizes
    }

    all_separator_rates = {
        bs: {
            rho: []
            for rho in active_rhos
        }
        for bs in active_batch_sizes
    }

    # ========================================================
    # EXPERIMENT
    # ========================================================

    for batch_size in active_batch_sizes:

        print("\n" + "#" * 80)
        print(
            f"BATCH SIZE = {batch_size}"
        )
        print("#" * 80)

        for rho in active_rhos:

            rho_pct = int(
                round(rho * 100)
            )

            print("\n" + "=" * 70)
            print(
                f"BATCH={batch_size} | "
                f"RHO={rho:.2f}"
            )
            print("=" * 70)

            data_file = (
                f"{task_name}/"
                f"{task_name}_rho_{rho_pct}.txt"
            )

            if not os.path.exists(data_file):

                raise FileNotFoundError(
                    f"Missing training file: "
                    f"{data_file}"
                )

            # =================================================
            # DATASET
            # =================================================

            train_dataset = TrainDataset(
                file_path=data_file,
                tokenizer=tokenizer,
                block_size=block_size,
                pad_id=pad_id,
            )

            print(
                "Training examples:",
                len(train_dataset),
            )

            # =================================================
            # MODEL SEEDS
            # =================================================

            for model_seed in active_model_seeds:

                print(
                    f"\nbatch={batch_size} | "
                    f"rho={rho:.2f} | "
                    f"seed={model_seed}"
                )

                set_seed(model_seed)

                run = None
                train_loader = None
                loader_generator = None
                optimizer = None
                model = None
                base_model = None

                try:

                    # =========================================
                    # WANDB
                    # =========================================

                    run = create_wandb_run(
                        task_name=task_name,
                        batch_size=batch_size,
                        rho=rho,
                        model_seed=model_seed,
                        train_dataset_size=len(
                            train_dataset
                        ),
                        val_dataset_size=len(
                            val_dataset
                        ),
                        block_size=block_size,
                        device=device,
                        use_bf16=use_bf16,
                        run_steps=run_steps,
                        test_mode=test_mode,
                    )

                    # =========================================
                    # DATALOADER
                    # =========================================

                    (
                        train_loader,
                        loader_generator,
                    ) = build_train_loader(
                        train_dataset=train_dataset,
                        batch_size=batch_size,
                        batch_seed=batch_seed,
                        num_workers=num_workers,
                        device=device,
                    )

                    # =========================================
                    # MODEL
                    # =========================================

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

                    # =========================================
                    # OPTIMIZER
                    # =========================================

                    optimizer = build_optimizer(
                        model=model,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        device=device,
                    )

                    # =========================================
                    # WARMUP
                    # =========================================

                    if test_mode:

                        run_warmup_steps = min(
                            warmup_steps,
                            max(
                                1,
                                run_steps // 5,
                            ),
                        )

                    else:

                        run_warmup_steps = warmup_steps

                    # =========================================
                    # TRAIN
                    # =========================================

                    final_loss = train_model(
                        model=model,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        steps=run_steps,
                        learning_rate=learning_rate,
                        min_learning_rate=min_learning_rate,
                        warmup_steps=run_warmup_steps,
                        max_grad_norm=max_grad_norm,
                        device=device,
                        use_bf16=use_bf16,
                        rho=rho,
                        model_seed=model_seed,
                        wandb_run=run,
                        log_every=(
                            1
                            if test_mode
                            else WANDB_LOG_EVERY
                        ),
                    )

                    # =========================================
                    # VALIDATION
                    # =========================================

                    val_acc, separator_rate = (
                        evaluate_model(
                            model=base_model,
                            val_loaders=val_loaders,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            newline_id=newline_id,
                            device=device,
                            rho=rho,
                            model_seed=model_seed,
                        )
                    )

                    all_accuracies[
                        batch_size
                    ][rho].append(
                        val_acc
                    )

                    all_separator_rates[
                        batch_size
                    ][rho].append(
                        separator_rate
                    )

                    print(
                        f"batch={batch_size} | "
                        f"rho={rho:.2f} | "
                        f"seed={model_seed} | "
                        f"loss={final_loss:.4f} | "
                        f"accuracy={val_acc * 100:.2f}% | "
                        f"separator={separator_rate * 100:.2f}%"
                    )

                    # =========================================
                    # WANDB FINAL METRICS
                    # =========================================

                    if run is not None:

                        run.log(
                            {
                                "val/accuracy":
                                    val_acc,

                                "val/accuracy_percent":
                                    val_acc * 100,

                                "val/separator_rate":
                                    separator_rate,

                                "final/train_loss":
                                    final_loss,
                            },
                            step=run_steps,
                        )

                        run.summary[
                            "final_accuracy"
                        ] = val_acc

                        run.summary[
                            "final_accuracy_percent"
                        ] = val_acc * 100

                        run.summary[
                            "final_separator_rate"
                        ] = separator_rate

                        run.summary[
                            "final_train_loss"
                        ] = final_loss

                    # =========================================
                    # SAVE MODEL
                    # =========================================

                    if SAVE_MODELS:

                        model_path = (
                            f"{task_name}/"
                            f"{task_name}"
                            f"_batch_{batch_size}"
                            f"_rho_{rho_pct}"
                            f"_seed_{model_seed}.pt"
                        )

                        torch.save(
                            base_model.state_dict(),
                            model_path,
                        )

                        print(
                            "Saved:",
                            model_path,
                        )

                finally:

                    # =========================================
                    # WANDB FINISH
                    # =========================================

                    if run is not None:
                        run.finish()

                    # =========================================
                    # MEMORY CLEANUP
                    # =========================================

                    del train_loader
                    del loader_generator
                    del optimizer
                    del model
                    del base_model

                    cleanup()

            del train_dataset
            cleanup()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for batch_size in active_batch_sizes:

        print(
            f"\nBATCH SIZE = "
            f"{batch_size}"
        )

        for rho in active_rhos:

            values = (
                np.array(
                    all_accuracies[
                        batch_size
                    ][rho]
                )
                * 100
            )

            if len(values) == 0:
                continue

            mean = values.mean()

            std = (
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )

            print(
                f"batch={batch_size} | "
                f"rho={rho:.2f} | "
                f"mean={mean:.2f}% | "
                f"std={std:.2f}% | "
                f"runs={values}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Full experiment:
    #
    # python experiment.py
    #
    # Smoke test:
    #
    # TRACE_TEST=1 python experiment.py

    test_mode = (
        os.environ.get(
            "TRACE_TEST",
            "0",
        )
        == "1"
    )

    run_experiment(
        test_mode=test_mode,
    )