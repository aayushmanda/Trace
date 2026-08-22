

import os
import torch
import wandb

from config import *
from src.registry import TASKS
from src.utils import (
    TrainDataset,
    ValDataset,
    build_val_loaders,
    build_train_loader,
    build_model,
    build_optimizer,
    train_model,
    evaluate_model,
    set_seed,
)


def main():
    run = wandb.init(project=WANDB_PROJECT)
    cfg = wandb.config

    rho = float(cfg.rho)
    model_seed = int(cfg.model_seed)
    batch_size = int(cfg.batch_size)
    lr = float(cfg.learning_rate)
    layers = int(cfg.n_layer)
    embd = int(cfg.n_embd)
    heads = int(cfg.n_head)

    run.name = f"rho{rho:.2f}_lr{lr:g}_bs{batch_size}_L{layers}_seed{model_seed}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    set_seed(model_seed)

    task = TASKS[task_name]
    tokenizer = task.tokenizer

    rho_pct = int(round(rho * 100))
    train_file = f"{task_name}/{task_name}_rho_{rho_pct}.txt"
    val_file = f"{task_name}/{task_name}_val.txt"

    if not os.path.exists(train_file):
        raise FileNotFoundError(train_file)

    if not os.path.exists(val_file):
        raise FileNotFoundError(val_file)

    train_dataset = TrainDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        block_size=task.block_size,
        pad_id=tokenizer.pad_id,
    )

    val_dataset = ValDataset(
        val_file,
        tokenizer,
        task.block_size,
    )

    train_loader, _ = build_train_loader(
        train_dataset=train_dataset,
        batch_size=batch_size,
        batch_seed=batch_seed,
        num_workers=num_workers,
        device=device,
    )

    val_loaders = build_val_loaders(
        val_dataset=val_dataset,
        val_batch_size=val_batch_size,
        device=device,
    )

    base_model, model = build_model(
        vocab_size=tokenizer.vocab_size,
        block_size=task.block_size,
        pad_id=tokenizer.pad_id,
        n_embd=embd,
        n_head=heads,
        n_layer=layers,
        dropout=dropout,
        device=device,
        use_compile=USE_COMPILE,
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=lr,
        weight_decay=weight_decay,
        device=device,
    )

    final_loss = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        steps=steps,
        learning_rate=lr,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        device=device,
        use_bf16=use_bf16,
        rho=rho,
        model_seed=model_seed,
        wandb_run=run,
        log_every=WANDB_LOG_EVERY,
    )

    val_acc, separator_rate = evaluate_model(
        model=base_model,
        val_loaders=val_loaders,
        tokenizer=tokenizer,
        max_new_tokens=task.max_new_tokens,
        newline_id=tokenizer.newline_id,
        device=device,
        rho=rho,
        model_seed=model_seed,
    )

    run.log({
        "val/accuracy": val_acc,
        "val/accuracy_percent": val_acc * 100,
        "val/separator_rate": separator_rate,
        "final/train_loss": final_loss,
    })

    run.summary["final_accuracy_percent"] = val_acc * 100

    print(
        f"rho={rho:.2f} "
        f"lr={lr:g} "
        f"seed={model_seed} "
        f"accuracy={val_acc * 100:.2f}%"
    )

    run.finish()


if __name__ == "__main__":
    main()