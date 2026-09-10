"""Train-D=8 length eval and rho=0.80 m_min histograms (GPT stack)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import RatioDataset, SupervisionDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.eval.generate import evaluate, evaluate_with_trace
from src.eval.margins import example_min_margins
from src.training.io import write_csv
from src.training.loop import train_steps
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import maybe_high_precision, set_seed


def run_length(args):
    device = torch.device(args.device)
    maybe_high_precision(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    train_task = TASKS[f"boolean_circuit_{args.train_depth}"]
    train = generate_unique(train_task, args.train_size, 501)
    train_prompts = {inst.prompt for inst in train}
    eval_sets = {
        depth: generate_unique(TASKS[f"boolean_circuit_{depth}"], args.val_size, 101 + depth, train_prompts)
        for depth in args.eval_depths
    }
    rows = []
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for seed in args.seeds:
            set_seed(seed)
            dataset = SupervisionDataset(train, train_task, mode)
            loader = make_loader(dataset, args, device)
            model = build_gpt(train_task, args, device)
            opt = make_optimizer(model, args, device)
            loss = train_steps(model, loader, opt, device, args.steps, getattr(args, "grad_clip", 1.0),
                               desc=f"length {mode} s{seed}")
            ckpt = Path(args.ckpt_dir) / f"{mode}_trainD{args.train_depth}_s{seed}.pt"
            torch.save({"model": model.state_dict(), "mode": mode, "seed": seed}, ckpt)
            for depth, instances in eval_sets.items():
                task = TASKS[f"boolean_circuit_{depth}"]
                acc = evaluate(model, task, instances, mode, args, device)
                row = dict(
                    mode=mode, seed=seed, train_depth=args.train_depth, eval_depth=depth,
                    train_loss=loss, answer_accuracy=acc, ood=int(depth != args.train_depth),
                    chance=task.chance_acc, ckpt=str(ckpt),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(args.out, rows)
    print("wrote", args.out)
    return rows


def run_margins(args):
    device = torch.device(args.device)
    maybe_high_precision(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    task = TASKS[args.task]
    train = generate_unique(task, args.train_size, 501)
    val = generate_unique(task, args.val_size, 101, {i.prompt for i in train})
    scores = np.random.default_rng(777).random(len(train))
    rows = []
    for label, condition, rho in (("outcome", "outcome", None), ("process", "mixed_process", args.rho)):
        for seed in args.seeds:
            dataset = RatioDataset(train, task, condition, rho=rho, ratio_scores=scores)
            loader = make_loader(dataset, args, device, extra_seed=seed)
            set_seed(seed)
            model = build_gpt(task, args, device)
            opt = make_optimizer(model, args, device)
            loss = train_steps(model, loader, opt, device, args.steps, args.grad_clip,
                               desc=f"mmin {label} s{seed}")
            metrics = evaluate_with_trace(
                model, task, val, "outcome" if condition == "outcome" else "process", args, device,
            )
            margins = example_min_margins(model, task, val, device, args.eval_batch_size)
            mvals = np.array([r["m_min"] for r in margins], dtype=float)
            summary = dict(
                label=label, rho="" if rho is None else rho, seed=seed, train_loss=loss,
                answer_accuracy=metrics["answer_accuracy"],
                m_min_mean=float(np.nanmean(mvals)),
                m_min_median=float(np.nanmedian(mvals)),
                frac_positive_path=float(np.mean([r["positive_path"] for r in margins])),
                frac_mmin_gt0=float(np.mean(mvals > 0)),
            )
            print(json.dumps(summary), flush=True)
            for rec in margins:
                rows.append({**summary, **rec})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(args.out, rows)
    print("wrote", args.out)
    return rows
