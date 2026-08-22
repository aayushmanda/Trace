# experiment.py

import os
import random
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
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


# -----------------------------------------------------------------------------
# data

def generate_training_instances(task, size, seed):
    print(f"Generating deterministic training pool: n={size}, seed={seed}")
    old_state = random.getstate()
    random.seed(seed)
    instances = [task.sample() for _ in range(size)]
    random.setstate(old_state)
    return instances


def prepare_training_files(task, task_name, rho_values, train_size, train_seed):
    missing_files = []

    for rho in rho_values:
        rho_pct = int(round(rho * 100))
        data_file = f"{task_name}/{task_name}_rho_{rho_pct}.txt"
        if not os.path.exists(data_file):
            missing_files.append((rho, data_file))

    if not missing_files:
        print("All required training files already exist.")
        return

    print(f"{len(missing_files)} training files missing.")

    # Same underlying training instances for every rho.
    instances = generate_training_instances(task, train_size, train_seed)

    for rho, data_file in missing_files:
        print(f"Generating rho={rho:.2f} -> {data_file}")
        save_mixed_trace_file(
            instances=instances,
            correct_ratio=rho,
            output_file=data_file,
            seed=train_seed,
        )

    del instances


# -----------------------------------------------------------------------------
# wandb

def create_wandb_run(task_name, batch_size, rho, model_seed, train_dataset_size, val_dataset_size, block_size, device, use_bf16, run_steps, test_mode):
    if not USE_WANDB:
        return None

    rho_pct = int(round(rho * 100))
    prefix = "test_" if test_mode else ""

    return wandb.init(
        project=WANDB_PROJECT,
        name=f"{prefix}{task_name}_bs{batch_size}_rho{rho_pct}_seed{model_seed}",
        group=f"{prefix}{task_name}_batch_{batch_size}",
        config={
            "task": task_name,
            "rho": rho,
            "batch_size": batch_size,
            "model_seed": model_seed,
            "batch_seed": batch_seed,
            "train_size": train_dataset_size,
            "val_size": val_dataset_size,
            "train_seed": train_seed,
            "val_seed": val_seed,
            "steps": run_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "block_size": block_size,
            "device": device,
            "bf16": use_bf16,
            "compile": USE_COMPILE,
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


# -----------------------------------------------------------------------------
# rho vs accuracy summary

def log_rho_accuracy_plot(all_accuracies, active_batch_sizes, active_rhos):
    if not USE_WANDB:
        return

    os.makedirs("figures", exist_ok=True)

    summary_run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{task_name}_rho_vs_accuracy",
        job_type="summary",
    )

    for batch_size in active_batch_sizes:
        xs, means, stds = [], [], []

        for rho in active_rhos:
            values = np.asarray(all_accuracies[batch_size][rho], dtype=float) * 100

            if len(values) == 0:
                continue

            xs.append(rho)
            means.append(values.mean())
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)

        xs = np.asarray(xs)
        means = np.asarray(means)
        stds = np.asarray(stds)

        order = np.argsort(xs)
        xs, means, stds = xs[order], means[order], stds[order]

        fig, ax = plt.subplots(figsize=(8, 5.5))

        # Raw experimental means ± std.
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=1.5,
            label="Mean ± std",
            zorder=3,
        )

        # Smooth shape-preserving interpolation.
        if len(xs) >= 3:
            x_dense = np.linspace(xs.min(), xs.max(), 500)

            mean_interp = PchipInterpolator(xs, means)
            std_interp = PchipInterpolator(xs, stds)

            mean_smooth = mean_interp(x_dense)
            std_smooth = np.maximum(std_interp(x_dense), 0)

            ax.plot(
                x_dense,
                mean_smooth,
                linewidth=2.5,
                label="Interpolated mean",
                zorder=2,
            )

            ax.fill_between(
                x_dense,
                np.clip(mean_smooth - std_smooth, 0, 100),
                np.clip(mean_smooth + std_smooth, 0, 100),
                alpha=0.18,
                zorder=1,
            )

            # Estimate critical rho from maximum slope.
            derivative = mean_interp.derivative()(x_dense)
            idx = np.argmax(derivative)

            rho_c = float(x_dense[idx])
            max_slope = float(derivative[idx])

            ax.axvline(
                rho_c,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

            ax.annotate(
                rf"$\rho_c \approx {rho_c:.3f}$",
                xy=(rho_c, mean_smooth[idx]),
                xytext=(10, -35),
                textcoords="offset points",
                fontsize=11,
            )

            summary_run.summary[f"rho_c_bs_{batch_size}"] = rho_c
            summary_run.summary[f"max_slope_bs_{batch_size}"] = max_slope

        ax.set_xlabel(r"Correct Trace Ratio $\rho$", fontsize=12)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
        ax.set_title(
            f"Trace Correctness Phase Transition — Batch Size {batch_size}",
            fontsize=13,
        )

        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(0, 100)

        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

        fig.tight_layout()

        path = f"figures/rho_accuracy_bs{batch_size}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")

        summary_run.log({
            f"phase_transition/bs_{batch_size}": wandb.Image(fig),
        })

        plt.close(fig)

    summary_run.finish()


# -----------------------------------------------------------------------------
# experiment

def run_experiment(test_mode=False):

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    print("\n" + "=" * 60)
    print("TRACE EXPERIMENT")
    print("=" * 80)
    print("Device:", device)
    print("BF16:", use_bf16)
    print("Test mode:", test_mode)

    # task
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

    # smoke test / full experiment
    if test_mode:
        active_batch_sizes = [batch_sizes[0]]
        active_rhos = [rho_values[0]]
        active_model_seeds = [model_seeds[0]]
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

    # training files
    required_rhos = list(dict.fromkeys([0.0] + list(active_rhos)))

    prepare_training_files(
        task=task,
        task_name=task_name,
        rho_values=required_rhos,
        train_size=train_size,
        train_seed=train_seed,
    )

    # validation
    val_file = f"{task_name}/{task_name}_val.txt"

    if REGENERATE_VAL or not os.path.exists(val_file):
        print("Generating validation dataset...")
        generate_validation_file(task, task_name, val_size, val_seed)

    full_val_dataset = ValDataset(val_file, tokenizer, block_size)

    assert len(full_val_dataset) == val_size, (
        f"Expected {val_size} validation samples, found {len(full_val_dataset)}"
    )

    if test_mode:
        subset_size = min(test_val_size, len(full_val_dataset))
        val_dataset = Subset(full_val_dataset, range(subset_size))
    else:
        val_dataset = full_val_dataset

    print("Validation examples:", len(val_dataset))

    val_loaders = build_val_loaders(
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        device=device,
    )

    # results
    all_accuracies = {
        bs: {rho: [] for rho in active_rhos}
        for bs in active_batch_sizes
    }

    all_separator_rates = {
        bs: {rho: [] for rho in active_rhos}
        for bs in active_batch_sizes
    }

    # -------------------------------------------------------------------------
    # train

    for batch_size in active_batch_sizes:

        print("\n" + "#" * 80)
        print(f"BATCH SIZE = {batch_size}")
        print("#" * 80)

        for rho in active_rhos:

            rho_pct = int(round(rho * 100))

            print("\n" + "=" * 70)
            print(f"BATCH={batch_size} | RHO={rho:.2f}")
            print("=" * 70)

            data_file = f"{task_name}/{task_name}_rho_{rho_pct}.txt"

            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Missing training file: {data_file}")

            train_dataset = TrainDataset(
                file_path=data_file,
                tokenizer=tokenizer,
                block_size=block_size,
                pad_id=pad_id,
            )

            print("Training examples:", len(train_dataset))

            for model_seed in active_model_seeds:

                print(f"\nbatch={batch_size} | rho={rho:.2f} | seed={model_seed}")

                set_seed(model_seed)

                run = None
                train_loader = None
                loader_generator = None
                optimizer = None
                model = None
                base_model = None

                try:

                    # wandb run
                    run = create_wandb_run(
                        task_name=task_name,
                        batch_size=batch_size,
                        rho=rho,
                        model_seed=model_seed,
                        train_dataset_size=len(train_dataset),
                        val_dataset_size=len(val_dataset),
                        block_size=block_size,
                        device=device,
                        use_bf16=use_bf16,
                        run_steps=run_steps,
                        test_mode=test_mode,
                    )

                    # dataloader
                    train_loader, loader_generator = build_train_loader(
                        train_dataset=train_dataset,
                        batch_size=batch_size,
                        batch_seed=batch_seed,
                        num_workers=num_workers,
                        device=device,
                    )

                    # model
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

                    # optimizer
                    optimizer = build_optimizer(
                        model=model,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        device=device,
                    )

                    run_warmup_steps = min(warmup_steps, max(1, run_steps // 5)) if test_mode else warmup_steps

                    # train
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
                        log_every=1 if test_mode else WANDB_LOG_EVERY,
                    )

                    # validation
                    val_acc, separator_rate = evaluate_model(
                        model=base_model,
                        val_loaders=val_loaders,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        newline_id=newline_id,
                        device=device,
                        rho=rho,
                        model_seed=model_seed,
                    )

                    all_accuracies[batch_size][rho].append(val_acc)
                    all_separator_rates[batch_size][rho].append(separator_rate)

                    print(
                        f"batch={batch_size} | rho={rho:.2f} | seed={model_seed} | "
                        f"loss={final_loss:.4f} | accuracy={val_acc * 100:.2f}% | "
                        f"separator={separator_rate * 100:.2f}%"
                    )

                    # final wandb metrics
                    if run is not None:
                        run.log({
                            "val/accuracy": val_acc,
                            "val/accuracy_percent": val_acc * 100,
                            "val/separator_rate": separator_rate,
                            "final/train_loss": final_loss,
                        }, step=run_steps)

                        run.summary["final_accuracy"] = val_acc
                        run.summary["final_accuracy_percent"] = val_acc * 100
                        run.summary["final_separator_rate"] = separator_rate
                        run.summary["final_train_loss"] = final_loss

                    # model checkpoint
                    if SAVE_MODELS:
                        model_path = f"{task_name}/{task_name}_batch_{batch_size}_rho_{rho_pct}_seed_{model_seed}.pt"
                        torch.save(base_model.state_dict(), model_path)
                        print("Saved:", model_path)

                finally:

                    if run is not None:
                        run.finish()

                    del train_loader
                    del loader_generator
                    del optimizer
                    del model
                    del base_model

                    cleanup()

            del train_dataset
            cleanup()

    # -------------------------------------------------------------------------
    # final results

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for batch_size in active_batch_sizes:

        print(f"\nBATCH SIZE = {batch_size}")

        for rho in active_rhos:

            values = np.array(all_accuracies[batch_size][rho]) * 100

            if len(values) == 0:
                continue

            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0

            print(
                f"batch={batch_size} | rho={rho:.2f} | "
                f"mean={mean:.2f}% | std={std:.2f}% | runs={values}"
            )

    # -------------------------------------------------------------------------
    # rho vs validation accuracy phase-transition plot

    if USE_WANDB and not test_mode:
        log_rho_accuracy_plot(
            all_accuracies=all_accuracies,
            active_batch_sizes=active_batch_sizes,
            active_rhos=active_rhos,
        )

    return all_accuracies, all_separator_rates


# -----------------------------------------------------------------------------
# main

if __name__ == "__main__":

    # Full experiment:
    #   python experiment.py
    #
    # Smoke test:
    #   TRACE_TEST=1 python experiment.py

    test_mode = os.environ.get("TRACE_TEST", "0") == "1"
    run_experiment(test_mode=test_mode)