"""Gradient-alignment diagnostic in the actual two-block GPT (App. F).

At checkpoints along one process-mode training trajectory of the exact
architecture and budget already used for Table 1
(configs/experiments/e1_five_condition_rerun.yaml), on a fixed held-out
batch of clean instances, this computes three gradients at the same
parameters theta:

  g_local: gradient of a state-transition-only teacher-forced CE loss
           (only the D*4 state-bit character positions in the displayed
           trace; excludes gate specifiers, separators, and the answer)
  g_proc:  gradient of the paper's own process loss on the same instances
  g_out:   gradient of the paper's own outcome loss on the same instances

and reports, per checkpoint and seed, the raw inner products A_proc =
<g_local, g_proc>, A_out = <g_local, g_out>, the normalized projections
<g_local, g>/||g_local||^2, and the cosine similarities, plus the same
broken down per transition position t = 1..D.

No model checkpoints are saved to disk; everything is computed in-memory
via train_with_checkpoints' callback, at the current live model. No new
hyperparameter search: architecture, optimizer, and data sizes are copied
from configs/experiments/e1_five_condition_rerun.yaml. Compare against
Corollary 2 and Theorem 4, which make the same asymmetry claim in the
shared-kernel model; this script tests whether it also holds, at the
gradient level, in the actual unrestricted Transformer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataclass import ANSWER_SEP
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.training.io import write_csv
from src.training.loop import train_with_checkpoints
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import default_device, maybe_high_precision, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "gradient_alignment"


def _state_char_offsets(trace: str) -> list[int]:
    """Offsets, within `trace`, of the 4 state-bit characters after each '>'."""
    offsets = []
    i = 0
    while True:
        j = trace.find(">", i)
        if j == -1:
            break
        offsets.extend(range(j + 1, j + 5))
        i = j + 5
    return offsets


def _encode(tokenizer, prompt: str, target: str, block_size: int, local_offsets=None):
    """Like datasets.encode_pair, but `local_offsets` (character offsets within
    `target`) restricts the mask to exactly those positions instead of every
    target position. `None` means the standard "every target position" mask."""
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    full = prompt_ids + target_ids
    if len(full) > block_size:
        raise ValueError(f"{len(full)} tokens exceeds block_size={block_size}")
    width = block_size - 1
    x = [tokenizer.pad_id] * width
    y = [tokenizer.pad_id] * width
    mask = [0.0] * width
    n = len(full) - 1
    x[:n] = full[:-1]
    y[:n] = full[1:]
    start = len(prompt_ids) - 1
    if local_offsets is None:
        for i in range(start, n):
            mask[i] = 1.0
    else:
        offsets = set(local_offsets)
        for k in range(len(target_ids)):
            if k in offsets:
                mask[start + k] = 1.0
    return x, y, mask


def _tensors(rows, tokenizer):
    xs, ys, masks = zip(*rows)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.float32)
    return x, y, mask


def build_reference_batch(instances, task, n_gates: int):
    """Builds, from the same instances, the process/outcome/local(+per-t) encodings."""
    tokenizer, block_size = task.tokenizer, task.block_size
    proc_rows, out_rows, local_rows = [], [], []
    per_t_rows = [[] for _ in range(n_gates)]
    for inst in instances:
        proc_target = f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n"
        out_target = f"{ANSWER_SEP}{inst.gold}\n"
        proc_rows.append(_encode(tokenizer, inst.prompt, proc_target, block_size))
        out_rows.append(_encode(tokenizer, inst.prompt, out_target, block_size))
        state_offsets = _state_char_offsets(inst.correct_trace)
        # +1 for the leading space in proc_target before the trace.
        local_rows.append(_encode(tokenizer, inst.prompt, proc_target, block_size,
                                   local_offsets=[o + 1 for o in state_offsets]))
        assert len(state_offsets) == 4 * n_gates, (len(state_offsets), n_gates)
        for t in range(n_gates):
            t_offsets = state_offsets[4 * t: 4 * t + 4]
            per_t_rows[t].append(_encode(tokenizer, inst.prompt, proc_target, block_size,
                                          local_offsets=[o + 1 for o in t_offsets]))
    batch = {
        "proc": _tensors(proc_rows, tokenizer),
        "out": _tensors(out_rows, tokenizer),
        "local": _tensors(local_rows, tokenizer),
        "per_t": [_tensors(rows, tokenizer) for rows in per_t_rows],
    }
    return batch


def _masked_loss(model, batch, device):
    x, y, mask = batch
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    _, loss = model(x, targets=y, mask=mask)
    return loss


def _flat_grad(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    flat = torch.cat(grads)
    model.zero_grad(set_to_none=True)
    return flat


def _stats(g_local, g):
    dot = float(torch.dot(g_local, g))
    n_local = float(g_local.norm())
    n_g = float(g.norm())
    return {
        "dot": dot,
        "normalized_projection": dot / (n_local ** 2) if n_local > 0 else float("nan"),
        "cosine": dot / (n_local * n_g) if n_local > 0 and n_g > 0 else float("nan"),
        "grad_norm": n_g,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--ref-size", type=int, default=256)
    p.add_argument("--ref-seed", type=int, default=101)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    args = p.parse_args(argv)

    n_gates = int(args.task.rsplit("_", 1)[-1])
    device = torch.device(default_device())
    device = maybe_high_precision(device, compile=args.compile, bf16=args.bf16)
    task = TASKS[args.task]

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    ref_instances = generate_unique(task, args.ref_size, args.ref_seed,
                                     {i.prompt for i in train_instances})
    ref_batch = build_reference_batch(ref_instances, task, n_gates)
    checkpoints = sorted({0, args.steps // 4, args.steps // 2, args.steps})

    from src.data.datasets import SupervisionDataset

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        train_dataset = SupervisionDataset(train_instances, task, "process")
        loader = make_loader(train_dataset, args, device)
        model = build_gpt(task, args, device)
        optimizer = make_optimizer(model, args, device)

        def on_checkpoint(step, model, loss, _seed=seed):
            model.eval()
            g_local = _flat_grad(model, _masked_loss(model, ref_batch["local"], device))
            g_proc = _flat_grad(model, _masked_loss(model, ref_batch["proc"], device))
            g_out = _flat_grad(model, _masked_loss(model, ref_batch["out"], device))
            row = {"seed": _seed, "step": step,
                   "train_loss": float(loss) if loss is not None else None,
                   "local_grad_norm": float(g_local.norm())}
            for name, g in [("proc", g_proc), ("out", g_out)]:
                s = _stats(g_local, g)
                for k, v in s.items():
                    row[f"{name}_{k}"] = v
            for t, batch_t in enumerate(ref_batch["per_t"]):
                g_local_t = _flat_grad(model, _masked_loss(model, batch_t, device))
                for name, g in [("proc", g_proc), ("out", g_out)]:
                    s = _stats(g_local_t, g)
                    row[f"t{t}_{name}_dot"] = s["dot"]
                    row[f"t{t}_{name}_normalized_projection"] = s["normalized_projection"]
                    row[f"t{t}_{name}_cosine"] = s["cosine"]
            rows.append(row)
            print(f"seed={_seed} step={step:5d} "
                  f"A_proc={row['proc_dot']:.4e} (cos={row['proc_cosine']:+.3f}) "
                  f"A_out={row['out_dot']:.4e} (cos={row['out_cosine']:+.3f})")
            model.train()

        train_with_checkpoints(model, loader, optimizer, device, checkpoints, on_checkpoint,
                                grad_clip=args.grad_clip, desc=f"grad-align/s{seed}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "gradient_alignment.csv"
    write_csv(csv_path, rows)
    persist = {
        "experiment": "gradient_alignment",
        "task": args.task,
        "seeds": list(args.seeds),
        "steps": args.steps,
        "checkpoints": checkpoints,
        "train_size": args.train_size,
        "ref_size": args.ref_size,
        "ref_seed": args.ref_seed,
        "train_seed": args.train_seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding": args.embedding,
        "heads": args.heads,
        "layers": args.layers,
        "csv": str(csv_path),
        "rows": rows,
    }
    (OUT / "gradient_alignment_persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
