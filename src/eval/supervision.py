"""Five-condition supervision comparison.

    python -m src supervision --config configs/experiments/e1_five_condition_rerun.yaml
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import TARGET_BUILDERS, SupervisionDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.eval.generate import evaluate
from src.training.io import write_csv
from src.training.loop import train_steps
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.config import load_yaml
from src.training.seed import add_compile_bf16_flags, default_device, maybe_high_precision, prepare_train_model, set_seed


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml(pre_args.config) if pre_args.config else {}
    p = argparse.ArgumentParser(parents=[pre], prog="python -m src supervision")
    p.add_argument("--tasks", nargs="+", default=list(cfg.get("tasks") or ["boolean_circuit_4"]))
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
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=Path(cfg["output"]) if cfg.get("output") else None)
    add_compile_bf16_flags(p, cfg, from_yaml=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if getattr(args, "device", None) else default_device())
    device = maybe_high_precision(device, compile=getattr(args, "compile", None),
                                  bf16=getattr(args, "bf16", None),
                                  distributed=getattr(args, "distributed", None))
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
                dataset = SupervisionDataset(train_instances, task, mode)
                loader = make_loader(dataset, args, device)
                model = build_gpt(task, args, device)
                train_model = prepare_train_model(
                    model, device, compile=getattr(args, "compile", None),
                    distributed=getattr(args, "distributed", None),
                )
                opt = make_optimizer(model, args, device)
                loss = train_steps(train_model, loader, opt, device, args.steps, args.grad_clip,
                                   desc=f"{task_name}/{mode}/s{seed}")
                accuracy = evaluate(model, task, val_instances, mode, args, device)
                results[(task_name, mode)].append(accuracy)
                row = {
                    "task": task_name, "mode": mode, "seed": int(seed),
                    "steps": int(args.steps), "loss": float(loss), "accuracy": float(accuracy),
                }
                rows.append(row)
                print(f"{task_name:18s} {mode:12s} seed={seed} loss={loss:.4f} accuracy={100 * accuracy:.2f}%")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print("\nSUMMARY")
    summary = []
    for (task_name, mode), values in results.items():
        array = np.asarray(values) * 100
        std = array.std(ddof=1) if len(array) > 1 else 0.0
        rec = {
            "task": task_name, "mode": mode,
            "mean_pct": float(array.mean()), "std_pct": float(std),
            "runs": [float(x) for x in array.round(2)],
        }
        summary.append(rec)
        print(f"{task_name:18s} {mode:12s} mean={array.mean():6.2f}% std={std:6.2f}% runs={array.round(2)}")
    out = getattr(args, "output", None) or getattr(args, "out", None)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            write_csv(out, rows)
        persist = out.with_name(out.stem + "_persist.json")
        persist.write_text(json.dumps({
            "experiment": "e1_five_condition",
            "config": str(getattr(args, "config", None)) if getattr(args, "config", None) else None,
            "tasks": list(args.tasks),
            "modes": list(args.modes),
            "seeds": list(args.seeds),
            "steps": int(args.steps),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "csv": str(out),
            "rows": rows,
            "summary": summary,
        }, indent=2) + "\n")
        print(f"saved: {out}")
        print(f"persist: {persist}")
