"""Character-token GPT runs: outcome, answer-first and step supervision on Boolean-8 and the register machine.

    python -m src supervision --config configs/experiments/boolean8_gpt.yaml
    python -m src supervision --config configs/experiments/register16_gpt.yaml
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data import TARGET_BUILDERS, TASKS, SupervisionDataset, generate_unique
from src.train import (
    build_gpt,
    configure_device,
    default_device,
    load_yaml,
    make_loader,
    make_optimizer,
    progress,
    set_seed,
    train_steps,
    write_csv,
)


def extract_prediction(text: str, mode: str):
    text = text.split("\n", 1)[0]
    if ":" not in text:
        return None
    if mode in {"outcome", "answer_first"}:
        answer_part = text.split(":", 1)[1]
    else:
        answer_part = text.rsplit(":", 1)[1]
    match = re.search(r"\d+", answer_part)
    return match.group(0) if match else None


@torch.inference_mode()
def evaluate(model, task, instances, mode, args, device):
    """Greedy answer accuracy. Outcome-like modes get a short decode budget."""
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst.gold))
    correct = total = 0
    max_new_tokens = 8 if mode in {"outcome", "answer_first"} else task.max_new_tokens
    batch_size = args.eval_batch_size
    n_batches = sum((len(rows) + batch_size - 1) // batch_size for rows in buckets.values())
    bar = progress(total=n_batches, desc="eval", leave=False)
    for rows in buckets.values():
        for start in range(0, len(rows), batch_size):
            bar.update(1)
            batch = rows[start:start + batch_size]
            context = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
            prompt_len = context.shape[1]
            output = model.generate(context, max_new_tokens=max_new_tokens, stop_id=task.tokenizer.newline_id)
            for generated_ids, (_, gold) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_len:])
                correct += int(extract_prediction(tail, mode) == gold)
                total += 1
    bar.close()
    return correct / total


def supervision_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml(pre_args.config) if pre_args.config else {}
    p = argparse.ArgumentParser(parents=[pre], prog="python -m src supervision")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(cfg.get("tasks") or ["boolean_circuit_8"]))
    p.add_argument("--modes", nargs="+", choices=list(TARGET_BUILDERS),
                   default=list(cfg.get("modes") or ["outcome", "process"]))
    p.add_argument("--seeds", nargs="+", type=int, default=list(cfg.get("seeds") or [2001]))
    p.add_argument("--train-size", type=int, default=int(cfg.get("train_size", 1000)))
    p.add_argument("--val-size", type=int, default=int(cfg.get("val_size", 100)))
    p.add_argument("--train-seed", type=int, default=int(cfg.get("train_seed", 501)))
    p.add_argument("--val-seed", type=int, default=int(cfg.get("val_seed", 101)))
    p.add_argument("--batch-seed", type=int, default=int(cfg.get("batch_seed", 12345)))
    p.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 32)))
    p.add_argument("--eval-batch-size", type=int, default=int(cfg.get("eval_batch_size", 64)))
    p.add_argument("--steps", type=int, default=int(cfg.get("steps", 100)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 3e-4)))
    p.add_argument("--weight-decay", type=float, default=float(cfg.get("weight_decay", 0.0)))
    p.add_argument("--grad-clip", type=float, default=float(cfg.get("grad_clip", 1.0)))
    p.add_argument("--embedding", type=int, default=int(cfg.get("embedding", 128)))
    p.add_argument("--heads", type=int, default=int(cfg.get("heads", 4)))
    p.add_argument("--layers", type=int, default=int(cfg.get("layers", 2)))
    p.add_argument("--dropout", type=float, default=float(cfg.get("dropout", 0.0)))
    p.add_argument("--workers", type=int, default=int(cfg.get("workers", 0)))
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=bool(cfg.get("bf16", True)),
                   help="CUDA bfloat16 autocast during training")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=Path(cfg["output"]) if cfg.get("output") else None)
    args = p.parse_args(argv)
    args.experiment = cfg.get("experiment", "supervision")
    return args


def supervision_main(argv=None):
    args = supervision_args(argv)
    device = configure_device(args.device or default_device())
    results = defaultdict(list)
    rows = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        train_instances = generate_unique(task, args.train_size, args.train_seed)
        val_instances = generate_unique(task, args.val_size, args.val_seed, {i.prompt for i in train_instances})
        print(f"\n{task_name}: chance={100 * task.chance_acc:.2f}% train={len(train_instances)} val={len(val_instances)}")
        for mode in args.modes:
            for seed in args.seeds:
                set_seed(seed)
                loader = make_loader(SupervisionDataset(train_instances, task, mode), args, device)
                model = build_gpt(task, args, device)
                opt = make_optimizer(model, args, device)
                loss = train_steps(model, loader, opt, device, args.steps, args.grad_clip, bf16=args.bf16,
                                   desc=f"{task_name}/{mode}/s{seed}")
                accuracy = evaluate(model, task, val_instances, mode, args, device)
                results[(task_name, mode)].append(accuracy)
                rows.append({
                    "task": task_name, "mode": mode, "seed": int(seed),
                    "steps": int(args.steps), "loss": float(loss), "accuracy": float(accuracy),
                })
                print(f"{task_name:20s} {mode:12s} seed={seed} loss={loss:.4f} accuracy={100 * accuracy:.2f}%")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print("\nSUMMARY")
    summary = []
    for (task_name, mode), values in results.items():
        array = np.asarray(values) * 100
        std = array.std(ddof=1) if len(array) > 1 else 0.0
        summary.append({
            "task": task_name, "mode": mode,
            "mean_pct": float(array.mean()), "std_pct": float(std),
            "runs": [float(x) for x in array.round(2)],
        })
        print(f"{task_name:20s} {mode:12s} mean={array.mean():6.2f}% std={std:6.2f}% runs={array.round(2)}")
    if args.output and rows:
        write_csv(args.output, rows)
        persist = args.output.with_name(args.output.stem + "_persist.json")
        persist.write_text(json.dumps({
            "experiment": args.experiment,
            "config": args.config,
            "tasks": list(args.tasks),
            "modes": list(args.modes),
            "seeds": list(args.seeds),
            "steps": int(args.steps),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "csv": str(args.output),
            "rows": rows,
            "summary": summary,
        }, indent=2) + "\n")
        print(f"saved: {args.output}\npersist: {persist}")
