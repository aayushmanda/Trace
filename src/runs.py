"""GPT runs: five-condition table, reliability sweep, gradient alignment.

    python -m src supervision --config configs/experiments/e1_five_condition_rerun.yaml
    python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001
    python -m src align --help
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data import (
    ANSWER_SEP,
    RatioDataset,
    SupervisionDataset,
    TARGET_BUILDERS,
    TASKS,
    generate_unique,
)
from src.train import (
    add_compile_bf16_flags,
    append_csv,
    build_gpt,
    default_device,
    load_yaml,
    make_loader,
    make_optimizer,
    maybe_high_precision,
    prepare_train_model,
    progress,
    set_seed,
    train_steps,
    train_with_checkpoints,
    write_csv,
)

import re
from collections import defaultdict

import torch

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

def extract_generation(text: str):
    line = text.split("\n", 1)[0]
    if ":" not in line:
        return None, None, False
    trace_text, answer_text = line.rsplit(":", 1)
    match = re.search(r"\d+", answer_text)
    return trace_text.strip(), match.group(0) if match else None, True

@torch.inference_mode()
def evaluate(model, task, instances, mode, args, device):
    """Greedy answer accuracy. `mode` outcome-like uses a short decode budget."""
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst.gold))
    correct = total = 0
    max_new_tokens = 8 if mode in {"outcome", "answer_first"} else task.max_new_tokens
    batch_size = getattr(args, "eval_batch_size", 128)
    n_batches = sum((len(rows) + batch_size - 1) // batch_size for rows in buckets.values())
    bar = progress(total=n_batches, desc="eval", leave=False)
    for rows in buckets.values():
        for start in range(0, len(rows), batch_size):
            bar.update(1)
            batch = rows[start:start + batch_size]
            context = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
            prompt_len = context.shape[1]
            output = model.generate(
                context, max_new_tokens=max_new_tokens,
                stop_id=task.tokenizer.newline_id, greedy=True,
            )
            for generated_ids, (_, gold) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_len:])
                correct += int(extract_prediction(tail, mode) == gold)
                total += 1
    bar.close()
    return correct / total

@torch.inference_mode()
def evaluate_with_trace(model, task, instances, condition, args, device):
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst))
    answer_correct = trace_correct = trace_step_correct = trace_step_total = colon_count = total = 0
    max_new_tokens = 8 if condition == "outcome" else task.max_new_tokens
    batch_size = getattr(args, "eval_batch_size", 128)
    n_batches = sum((len(rows) + batch_size - 1) // batch_size for rows in buckets.values())
    bar = progress(total=n_batches, desc="eval", leave=False)
    for rows in buckets.values():
        for start in range(0, len(rows), batch_size):
            bar.update(1)
            batch = rows[start:start + batch_size]
            context = torch.tensor([ids for ids, _ in batch], dtype=torch.long, device=device)
            prompt_length = context.shape[1]
            output = model.generate(
                context, max_new_tokens=max_new_tokens,
                stop_id=task.tokenizer.newline_id, greedy=True,
            )
            for generated_ids, (_, inst) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_length:])
                trace, answer, emitted_colon = extract_generation(tail)
                colon_count += int(emitted_colon)
                answer_correct += int(answer == inst.gold)
                if condition != "outcome":
                    trace_correct += int(trace == inst.correct_trace)
                    predicted_steps = [] if trace is None else trace.split()
                    gold_steps = inst.correct_trace.split()
                    trace_step_correct += sum(p == g for p, g in zip(predicted_steps, gold_steps))
                    trace_step_total += len(gold_steps)
                total += 1
    bar.close()
    return {
        "answer_accuracy": answer_correct / total,
        "exact_trace_accuracy": None if condition == "outcome" else trace_correct / total,
        "trace_step_accuracy": None if condition == "outcome" else trace_step_correct / max(trace_step_total, 1),
        "colon_rate": colon_count / total,
    }

"""Five-condition supervision comparison.

    python -m src supervision --config configs/experiments/e1_five_condition_rerun.yaml
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def supervision_args(argv=None):
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

def supervision_main(argv=None):
    args = supervision_args(argv)
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

"""Trace-reliability sweeps.

    python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

def reliability_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src reliability")
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.8])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001])
    p.add_argument("--checkpoints", nargs="+", type=int, default=[1000])
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--ratio-seed", type=int, default=777)
    p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--include-outcome", action="store_true")
    p.add_argument("--answer-loss", choices=["all", "clean"], default="all")
    p.add_argument("--save-models", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None)
    add_compile_bf16_flags(p, from_yaml=True)
    return p.parse_args(argv)

def _persist_payload(args, task, output, rows):
    return {
        "experiment": "e5_reliability",
        "task": task.name,
        "rhos": list(args.rhos),
        "seeds": list(args.seeds),
        "checkpoints": list(args.checkpoints),
        "train_size": args.train_size,
        "val_size": args.val_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "ratio_seed": args.ratio_seed,
        "batch_seed": args.batch_seed,
        "include_outcome": bool(args.include_outcome),
        "answer_loss": getattr(args, "answer_loss", "all"),
        "csv": str(output),
        "n_rows": len(rows),
        "rows": rows,
        "note": "Terminal answers remain correct; rho is trace reliability. Local = trace_step_accuracy, rollout = exact_trace_accuracy / answer_accuracy.",
    }

def reliability_main(argv=None):
    args = reliability_args(argv)
    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        raise SystemExit("every rho must lie in [0, 1]")
    checkpoints = sorted(set(args.checkpoints))
    task = TASKS[args.task]

    device = torch.device(getattr(args, "device", None) or default_device())
    device = maybe_high_precision(device, compile=getattr(args, "compile", None),
                                  bf16=getattr(args, "bf16", None),
                                  distributed=getattr(args, "distributed", None))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("results") / f"{task.name}_phase_{timestamp}.csv"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to append to existing output: {output}")
    persist = output.with_name(output.stem + "_persist.json")
    rows = []
    train_instances = generate_unique(task, args.train_size, args.train_seed)
    val_instances = generate_unique(task, args.val_size, args.val_seed, {i.prompt for i in train_instances})
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))
    conditions = []
    if args.include_outcome:
        conditions.append(("outcome", None))
    conditions.extend(("mixed_process", rho) for rho in sorted(set(args.rhos)))
    for condition, rho in conditions:
        dataset = RatioDataset(train_instances, task, condition, rho=rho, ratio_scores=ratio_scores,
                               answer_loss=getattr(args, "answer_loss", "all"))
        for seed in args.seeds:
            set_seed(seed)
            loader = make_loader(dataset, args, device)
            model = build_gpt(task, args, device)
            train_model = prepare_train_model(
                model, device, compile=getattr(args, "compile", None),
                distributed=getattr(args, "distributed", None),
            )
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
                rows.append(row)
                persist.write_text(json.dumps(_persist_payload(args, task, output, rows), indent=2) + "\n")
                print(f"{task.name} {_lab} seed={_s} step={step} answer={100 * metrics['answer_accuracy']:.2f}%")

            train_with_checkpoints(
                model, loader, optimizer, device, checkpoints, on_checkpoint,
                grad_clip=args.grad_clip, desc=f"{task.name}/{label}/s{seed}",
                train_model=train_model,
            )
            if getattr(args, "save_models", None) is not None:
                args.save_models.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.save_models / f"{task.name}_{label.replace('=', '')}_s{seed}.pt")
            del model, optimizer, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del dataset
    persist.write_text(json.dumps(_persist_payload(args, task, output, rows), indent=2) + "\n")
    print(f"saved: {output}")
    print(f"persist: {persist}")

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

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
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

def align_main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src align")
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
    p.add_argument("--train-mode", choices=["process", "outcome"], default="process",
                    help="Which objective to actually train under; g_local/g_proc/g_out "
                         "are always evaluated at every checkpoint regardless, so choosing "
                         "'outcome' makes g_out the real (not counterfactual) trained-objective "
                         "gradient and g_proc the counterfactual one, the mirror image of the "
                         "default 'process' run.")
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

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        train_dataset = SupervisionDataset(train_instances, task, args.train_mode)
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
                                grad_clip=args.grad_clip, desc=f"grad-align-{args.train_mode}/s{seed}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.train_mode == "process" else f"_{args.train_mode}_trained"
    csv_path = OUT / f"gradient_alignment{suffix}.csv"
    write_csv(csv_path, rows)
    persist = {
        "experiment": f"gradient_alignment{suffix}",
        "train_mode": args.train_mode,
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
    (OUT / f"gradient_alignment{suffix}_persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")
