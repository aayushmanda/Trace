"""Five-condition supervision comparison (GPT stack)."""
from collections import defaultdict

import numpy as np
import torch

from src.data.datasets import SupervisionDataset, TARGET_BUILDERS
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.eval.generate import evaluate
from src.training.loop import train_steps
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import default_device, maybe_high_precision, set_seed


def main(args):
    device = torch.device(args.device or default_device() if getattr(args, "device", None) else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    maybe_high_precision(device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    results = defaultdict(list)
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
                opt = make_optimizer(model, args, device)
                loss = train_steps(model, loader, opt, device, args.steps, args.grad_clip,
                                   desc=f"{task_name}/{mode}/s{seed}")
                accuracy = evaluate(model, task, val_instances, mode, args, device)
                results[(task_name, mode)].append(accuracy)
                print(f"{task_name:18s} {mode:12s} seed={seed} loss={loss:.4f} accuracy={100 * accuracy:.2f}%")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print("\nSUMMARY")
    for (task_name, mode), values in results.items():
        array = np.asarray(values) * 100
        std = array.std(ddof=1) if len(array) > 1 else 0.0
        print(f"{task_name:18s} {mode:12s} mean={array.mean():6.2f}% std={std:6.2f}% runs={array.round(2)}")
