"""Rank-8 LoRA on a pretrained causal LM, Boolean-8 serialization: outcome, answer-first and step training.

The paper runs SmolLM2-135M and Qwen2.5-0.5B, three seeds each, 800 updates (6400 for the 8x outcome budget).

    python -m src lora --config configs/experiments/lora_transfer.yaml
    python -m src lora --model Qwen/Qwen2.5-0.5B --conditions outcome --steps 6400 \\
        --output results/paper/qwen05/outcome_6400/s2001.csv --run-dir results/paper/qwen05/artifacts_6400
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import TASKS, generate_unique
from src.train import default_device, load_yaml, set_seed

ANSWER_RE = re.compile(r"([01]{4})")
CONDITIONS = ("outcome", "answer_first", "process")


def completion_for(inst, condition):
    if condition == "outcome":
        return f" : {inst.gold}\n"
    if condition == "answer_first":
        return f" : {inst.gold} ; {inst.correct_trace}\n"
    if condition == "process":
        return f" {inst.correct_trace} : {inst.gold}\n"
    raise ValueError(condition)


class CompletionDataset(Dataset):
    def __init__(self, instances, tokenizer, condition, max_length):
        self.rows = []
        for inst in instances:
            pids = tokenizer(inst.prompt, add_special_tokens=True)["input_ids"]
            cids = tokenizer(completion_for(inst, condition), add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                cids += [tokenizer.eos_token_id]
            ids = pids + cids
            if len(ids) > max_length:
                raise ValueError(f"{len(ids)} tokens > --max-length={max_length}")
            self.rows.append((ids, [-100] * len(pids) + cids.copy()))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        n, w = len(rows), max(len(x) for x, _ in rows)
        ids = torch.full((n, w), self.pad, dtype=torch.long)
        labels = torch.full((n, w), -100, dtype=torch.long)
        mask = torch.zeros((n, w), dtype=torch.long)
        for i, (x, y) in enumerate(rows):
            k = len(x)
            ids[i, :k] = torch.tensor(x)
            labels[i, :k] = torch.tensor(y)
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tok.pad_token = tok.eos_token
    return tok


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model(args, tok, device):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=choose_dtype(device), low_cpu_mem_usage=True)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
        ),
    )
    return model.to(device)


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total


def train(model, ds, args, seed, device):
    set_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.batch_seed),
        collate_fn=Collator(args.pad_id),
    )
    it = iter(loader)
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")
    except TypeError:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    amp, amp_dtype = device.type == "cuda", choose_dtype(device)
    model.train()
    recent = []
    pbar = tqdm(range(1, args.steps + 1), desc=f"train seed={seed}")
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        recent.append(float(loss.detach()))
        if len(recent) > 30:
            recent.pop(0)
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{np.mean(recent):.4f}")
    return float(np.mean(recent))


@torch.inference_mode()
def generate_text(model, tok, prompts, args, device, max_new):
    old = tok.padding_side
    tok.padding_side = "left"
    model.eval()
    outputs = []
    for start in tqdm(range(0, len(prompts), args.eval_batch_size), desc="free generation", leave=False):
        b = tok(prompts[start : start + args.eval_batch_size], return_tensors="pt", padding=True, truncation=False)
        b = {k: v.to(device) for k, v in b.items()}
        y = model.generate(
            **b,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        outputs += tok.batch_decode(y[:, b["input_ids"].shape[1] :], skip_special_tokens=True)
    tok.padding_side = old
    return outputs


def parse_generation(text, condition):
    line = text.split("\n", 1)[0].strip()
    if ":" not in line:
        return None, None
    if condition in {"outcome", "answer_first"}:
        m = ANSWER_RE.search(line.split(":", 1)[1])
        return None, m.group(1) if m else None
    trace, ans = line.rsplit(":", 1)
    m = ANSWER_RE.search(ans)
    return trace.strip(), m.group(1) if m else None


@torch.inference_mode()
def evaluate_free(model, tok, instances, condition, args, device):
    """Greedy free-running answer accuracy (and exact-trace accuracy for step training)."""
    max_new = args.outcome_max_new_tokens if condition in {"outcome", "answer_first"} else args.process_max_new_tokens
    texts = generate_text(model, tok, [x.prompt for x in instances], args, device, max_new)
    ans_ok = exact_trace = 0
    audit = []
    for inst, text in zip(instances, texts):
        pred_trace, pred_ans = parse_generation(text, condition)
        ans_ok += int(pred_ans == inst.gold)
        exact_trace += int(pred_trace == inst.correct_trace)
        if len(audit) < args.audit_examples:
            audit.append({"prompt": inst.prompt, "gold_trace": inst.correct_trace, "gold_answer": inst.gold,
                          "generated": text, "parsed_trace": pred_trace, "parsed_answer": pred_ans})
    n = len(instances)
    return {
        "answer_accuracy": ans_ok / n,
        "exact_trace_accuracy": exact_trace / n if condition == "process" else None,
        "audit": audit,
    }


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_adapter(model, tok, root, condition, seed):
    path = root / condition / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def cleanup(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(task, tok, ds, val, condition, seed, args, device):
    print(f"\n{'=' * 96}\ncondition={condition} seed={seed}\n{'=' * 96}")
    set_seed(seed)
    model = build_model(args, tok, device)
    trainable, total, frac = trainable_summary(model)
    print(f"trainable parameters: {trainable:,}/{total:,} ({100*frac:.3f}%)")
    loss = train(model, ds, args, seed, device)
    free = evaluate_free(model, tok, val, condition, args, device)
    gen_dir = args.run_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{condition}_seed_{seed}.json").write_text(json.dumps(free["audit"], indent=2))
    adapter_path = ""
    if args.save_adapters:
        adapter_path = str(save_adapter(model, tok, args.run_dir / "adapters", condition, seed))
    append_csv(args.output, {
        "model": args.model,
        "task": task.name,
        "condition": condition,
        "seed": seed,
        "train_size": len(ds),
        "val_size": len(val),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "trainable_parameters": trainable,
        "trainable_fraction": frac,
        "loss": loss,
        "answer_accuracy": free["answer_accuracy"],
        "exact_trace_accuracy": "" if free["exact_trace_accuracy"] is None else free["exact_trace_accuracy"],
        "adapter_path": adapter_path,
    })
    msg = f"loss={loss:.4f} | answer={100*free['answer_accuracy']:.2f}%"
    if free["exact_trace_accuracy"] is not None:
        msg += f" | exact trace={100*free['exact_trace_accuracy']:.2f}%"
    print(msg)
    cleanup(model)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(
        parents=[pre],
        prog="python -m src lora",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=cfg.get("model", "HuggingFaceTB/SmolLM2-135M"))
    p.add_argument("--task", choices=list(TASKS), default=cfg.get("task", "boolean_circuit_8"))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(cfg.get("conditions", CONDITIONS)))
    p.add_argument("--seeds", nargs="+", type=int, default=list(cfg.get("seeds", [2001, 2002, 2003])))
    p.add_argument("--train-size", type=int, default=int(cfg.get("train_size", 12000)))
    p.add_argument("--val-size", type=int, default=int(cfg.get("val_size", 300)))
    p.add_argument("--train-seed", type=int, default=int(cfg.get("train_seed", 501)))
    p.add_argument("--val-seed", type=int, default=int(cfg.get("val_seed", 101)))
    p.add_argument("--batch-seed", type=int, default=int(cfg.get("batch_seed", 12345)))
    p.add_argument("--steps", type=int, default=int(cfg.get("steps", 800)))
    p.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 16)))
    p.add_argument("--eval-batch-size", type=int, default=int(cfg.get("eval_batch_size", 32)))
    p.add_argument("--workers", type=int, default=int(cfg.get("workers", 2)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 3e-4)))
    p.add_argument("--weight-decay", type=float, default=float(cfg.get("weight_decay", 0.0)))
    p.add_argument("--grad-clip", type=float, default=float(cfg.get("grad_clip", 1.0)))
    p.add_argument("--lora-rank", type=int, default=int(cfg.get("lora_rank", 8)))
    p.add_argument("--max-length", type=int, default=int(cfg.get("max_length", 384)))
    p.add_argument("--process-max-new-tokens", type=int, default=int(cfg.get("process_max_new_tokens", 128)))
    p.add_argument("--outcome-max-new-tokens", type=int, default=int(cfg.get("outcome_max_new_tokens", 20)))
    p.add_argument("--audit-examples", type=int, default=int(cfg.get("audit_examples", 8)))
    p.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=bool(cfg.get("save_adapters", True)))
    p.add_argument("--output", type=Path, default=Path(cfg.get("output", "results/paper/smollm/smollm_trace.csv")))
    p.add_argument("--run-dir", type=Path, default=Path(cfg.get("run_dir", "results/paper/smollm/artifacts")))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=cfg.get("device"))
    return p.parse_args(argv)


def run(args):
    device = torch.device(args.device or default_device())
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        print("WARNING: no CUDA GPU; this will be slow")
    task = TASKS[args.task]
    tok = load_tokenizer(args.model)
    args.pad_id = tok.pad_token_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        args.output.unlink()

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    val = generate_unique(task, args.val_size, args.val_seed, excluded={x.prompt for x in train_instances})
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg["device"] = str(device)
    cfg["minibatch_order"] = "fixed across model seeds; only model initialization seed changes"
    (args.run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"model={args.model} task={task.name} device={device} train={len(train_instances)} val={len(val)}")
    print(f"example:\n  prompt = {train_instances[0].prompt}\n  trace  = {train_instances[0].correct_trace}"
          f"\n  gold   = {train_instances[0].gold}")

    for condition in args.conditions:
        ds = CompletionDataset(train_instances, tok, condition, args.max_length)
        for seed in args.seeds:
            run_one(task, tok, ds, val, condition, seed, args, device)
        del ds
        gc.collect()
    print(f"\nsaved results: {args.output}\nsaved artifacts: {args.run_dir}")


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
