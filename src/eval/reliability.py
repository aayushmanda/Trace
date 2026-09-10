"""Trace-reliability sweeps (GPT stack)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import RatioDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.eval.generate import evaluate_with_trace
from src.training.io import append_csv
from src.training.loop import train_with_checkpoints
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import maybe_compile, maybe_high_precision, set_seed


def main(args):
    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        raise SystemExit("every rho must lie in [0, 1]")
    checkpoints = sorted(set(args.checkpoints))
    task = TASKS[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maybe_high_precision(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("results") / f"{task.name}_phase_{timestamp}.csv"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to append to existing output: {output}")
    train_instances = generate_unique(task, args.train_size, args.train_seed)
    val_instances = generate_unique(task, args.val_size, args.val_seed, {i.prompt for i in train_instances})
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))
    conditions = []
    if args.include_outcome:
        conditions.append(("outcome", None))
    conditions.extend(("mixed_process", rho) for rho in sorted(set(args.rhos)))
    for condition, rho in conditions:
        dataset = RatioDataset(train_instances, task, condition, rho=rho, ratio_scores=ratio_scores)
        for seed in args.seeds:
            set_seed(seed)
            loader = make_loader(dataset, args, device)
            model = build_gpt(task, args, device)
            train_model = maybe_compile(model, device, enabled=getattr(args, "compile", None))
            optimizer = make_optimizer(model, args, device)
            label = "outcome" if condition == "outcome" else f"rho={rho:.2f}"

            def on_checkpoint(step, model, loss, *, _c=condition, _r=rho, _s=seed, _lab=label):
                if step == 0:
                    return
                metrics = evaluate_with_trace(model, task, val_instances, _c, args, device)
                row = {
                    "task": task.name, "condition": _lab,
                    "rho": "" if _r is None else _r, "seed": _s, "step": step,
                    "loss": float(loss.detach()) if loss is not None else "",
                    "answer_accuracy": metrics["answer_accuracy"],
                    "exact_trace_accuracy": "" if metrics["exact_trace_accuracy"] is None else metrics["exact_trace_accuracy"],
                    "trace_step_accuracy": "" if metrics["trace_step_accuracy"] is None else metrics["trace_step_accuracy"],
                    "colon_rate": metrics["colon_rate"],
                }
                append_csv(output, row)
                print(f"{task.name} {_lab} seed={_s} step={step} answer={100 * metrics['answer_accuracy']:.2f}%")

            train_with_checkpoints(
                model, loader, optimizer, device, checkpoints, on_checkpoint,
                grad_clip=args.grad_clip, desc=f"{task.name}/{label}/s{seed}",
                train_model=train_model,
            )
            del model, optimizer, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del dataset
    print(f"saved: {output}")
