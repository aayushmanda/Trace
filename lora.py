
import argparse
import csv
import gc
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.registry import TASKS
from sweep_ratio import generate_unique


BITS_RE = re.compile(r"(?<![01])([01]{4})(?![01])")
STEP_RE = re.compile(r"Step\s+(\d+)\s*:\s*([01]{4})", re.IGNORECASE)
ANSWER_RE = re.compile(r"Final\s+answer\s*:\s*([01]{4})", re.IGNORECASE)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_gate_string(encoded: str):
    gates, i = [], 0
    while i < len(encoded):
        op = encoded[i]
        if op == "x":
            gate = encoded[i:i + 2]
            i += 2
        elif op in {"c", "s"}:
            gate = encoded[i:i + 3]
            i += 3
        elif op == "t":
            gate = encoded[i:i + 4]
            i += 4
        else:
            raise ValueError(f"Cannot parse gate string at {encoded[i:]} in {encoded!r}")
        gates.append(gate)
    return gates


def parse_boolean_prompt(compact_prompt: str):
    # Existing repository format: i0101;ux0c12s03t012...
    match = re.fullmatch(r"i([01]{4});u(.+)", compact_prompt)
    if not match:
        raise ValueError(f"Unexpected Boolean-circuit prompt: {compact_prompt!r}")
    initial, encoded_gates = match.groups()
    return initial, parse_gate_string(encoded_gates)


def gate_to_english(gate: str):
    op = gate[0]
    if op == "x":
        return f"NOT wire {gate[1]}"
    if op == "c":
        return f"CNOT with control wire {gate[1]} and target wire {gate[2]}"
    if op == "s":
        return f"SWAP wire {gate[1]} with wire {gate[2]}"
    if op == "t":
        return (
            f"TOFFOLI with control wires {gate[1]} and {gate[2]} "
            f"and target wire {gate[3]}"
        )
    raise ValueError(gate)


def trace_states(trace: str):
    states = []
    for item in trace.split():
        if ">" not in item:
            raise ValueError(f"Unexpected trace item: {item!r}")
        _, state = item.rsplit(">", 1)
        if not re.fullmatch(r"[01]{4}", state):
            raise ValueError(f"Unexpected Boolean state: {state!r}")
        states.append(state)
    return states


def natural_language_prompt(inst):
    """
    Same prompt for outcome-only and every process-rho condition.

    Do not put a requested output format in the prompt: that would make X differ
    across supervision conditions. The response format is learned only from the
    supervised continuation.
    """
    initial, gates = parse_boolean_prompt(inst.prompt)
    lines = [
        "Execute the following reversible Boolean circuit on a 4-bit state.",
        "Wires are indexed 0, 1, 2, 3 from left to right.",
        "NOT flips its target bit.",
        "CNOT flips its target bit when its control bit is 1.",
        "SWAP exchanges its two wires.",
        "TOFFOLI flips its target bit when both control bits are 1.",
        f"Initial state: {initial}",
        "Operations:",
    ]
    lines.extend(f"{i}. {gate_to_english(gate)}" for i, gate in enumerate(gates, 1))
    lines.append("Solution:")
    return "\n".join(lines)


def process_completion(inst, clean: bool):
    states = trace_states(inst.correct_trace if clean else inst.wrong_trace)
    lines = ["Reasoning trace:"]
    lines.extend(f"Step {i}: {state}" for i, state in enumerate(states, 1))
    # Terminal answer intentionally stays correct under corrupted process supervision.
    lines.append(f"Final answer: {inst.gold}")
    return "\n".join(lines) + "\n"


def outcome_completion(inst):
    return f"Final answer: {inst.gold}\n"


def local_prefix(inst, transition_index: int):
    """
    Gold-prefix query for local next-state competence.

    transition_index is zero based. The model is given all previous correct
    states and asked to continue at exactly the next state slot.
    """
    states = trace_states(inst.correct_trace)
    prefix = natural_language_prompt(inst) + "\nReasoning trace:\n"
    for i in range(transition_index):
        prefix += f"Step {i + 1}: {states[i]}\n"
    prefix += f"Step {transition_index + 1}: "
    return prefix, states[transition_index]


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class CompletionDataset(Dataset):
    """
    Causal-LM SFT with zero loss on the input prompt and loss only on completion.

    Reliability assignments are nested because each training row receives a
    fixed ratio_score and is clean iff ratio_score < rho.
    """

    def __init__(
        self,
        instances,
        tokenizer,
        max_length: int,
        condition: str,
        rho: Optional[float] = None,
        ratio_scores: Optional[np.ndarray] = None,
    ):
        if condition not in {"outcome", "process"}:
            raise ValueError(condition)
        if condition == "process" and (rho is None or ratio_scores is None):
            raise ValueError("process requires rho and ratio_scores")

        self.rows: list[EncodedExample] = []
        self.realized_rho = None

        if condition == "process":
            clean_flags = np.asarray(ratio_scores) < float(rho)
            self.realized_rho = float(clean_flags.mean())
        else:
            clean_flags = None

        eos = tokenizer.eos_token_id
        for i, inst in enumerate(tqdm(instances, desc=f"tokenize/{condition}/{rho}")):
            prompt = natural_language_prompt(inst) + "\n"
            if condition == "outcome":
                completion = outcome_completion(inst)
            else:
                completion = process_completion(inst, bool(clean_flags[i]))

            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if eos is not None:
                completion_ids = completion_ids + [eos]

            ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids.copy()

            if len(ids) > max_length:
                raise ValueError(
                    f"Example has {len(ids)} tokens > --max-length={max_length}. "
                    "Increase --max-length; do not silently truncate the trace."
                )
            self.rows.append(EncodedExample(ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row.input_ids, row.labels


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        max_len = max(len(ids) for ids, _ in rows)
        batch = len(rows)
        input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        for i, (ids, labs) in enumerate(rows):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(labs, dtype=torch.long)
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def choose_dtype(args, device):
    if device.type != "cuda":
        return torch.float32
    if args.bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    # Default to bf16 when available because these models support it well.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_lora_model(args, tokenizer, device):
    dtype = choose_dtype(args, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = (
        "all-linear"
        if args.target_modules == ["all-linear"]
        else args.target_modules
    )
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.to(device)
    return model


def trainable_parameter_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total


def make_optimizer(model, args):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_lora(model, dataset, seed, args, device):
    set_all_seeds(seed)
    generator = torch.Generator().manual_seed(args.batch_seed + seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=generator,
        pin_memory=device.type == "cuda",
        collate_fn=CausalCollator(args.pad_token_id),
    )
    iterator = iter(loader)
    optimizer = make_optimizer(model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    use_amp = device.type == "cuda" and choose_dtype(args, device) in {
        torch.float16,
        torch.bfloat16,
    }
    amp_dtype = choose_dtype(args, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and amp_dtype == torch.float16),
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = []
    pbar = tqdm(range(1, args.max_steps + 1), desc=f"train/seed={seed}")
    for optimizer_step in pbar:
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += float(loss.detach()) * args.grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running.append(accumulated_loss)
        if len(running) > 50:
            running.pop(0)
        if optimizer_step % 10 == 0:
            pbar.set_postfix(
                loss=f"{np.mean(running):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    return {
        "final_train_loss_50": float(np.mean(running)),
        "optimizer_steps": args.max_steps,
        "effective_batch_size": args.batch_size * args.grad_accum,
    }


def parse_completion(text: str, depth: int):
    answer_match = ANSWER_RE.search(text)
    answer = answer_match.group(1) if answer_match else None

    indexed = {}
    for idx, state in STEP_RE.findall(text):
        idx = int(idx)
        if 1 <= idx <= depth and idx not in indexed:
            indexed[idx] = state
    states = [indexed.get(i) for i in range(1, depth + 1)]
    return states, answer


@torch.inference_mode()
def batched_generate(model, tokenizer, prompts, args, device, max_new_tokens):
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []

    model.eval()
    for start in tqdm(
        range(0, len(prompts), args.eval_batch_size),
        desc="generate",
        leave=False,
    ):
        batch_prompts = prompts[start:start + args.eval_batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_width = encoded["input_ids"].shape[1]
        tails = generated[:, input_width:]
        outputs.extend(
            tokenizer.batch_decode(tails, skip_special_tokens=True)
        )

    tokenizer.padding_side = old_padding
    return outputs


@torch.inference_mode()
def evaluate_free(model, tokenizer, instances, condition, args, device):
    prompts = [natural_language_prompt(inst) + "\n" for inst in instances]
    max_new = args.outcome_max_new_tokens if condition == "outcome" else args.process_max_new_tokens
    texts = batched_generate(model, tokenizer, prompts, args, device, max_new)

    answer_correct = 0
    exact_trace_correct = 0
    step_correct = 0
    step_total = 0
    depth = len(trace_states(instances[0].correct_trace))

    audit = []
    for inst, text in zip(instances, texts):
        states, answer = parse_completion(text, depth)
        gold_states = trace_states(inst.correct_trace)
        answer_correct += int(answer == inst.gold)

        if condition == "process":
            exact_trace_correct += int(states == gold_states)
            for pred, gold in zip(states, gold_states):
                step_correct += int(pred == gold)
                step_total += 1

        if len(audit) < args.audit_examples:
            audit.append(
                {
                    "compact_prompt": inst.prompt,
                    "gold_answer": inst.gold,
                    "gold_states": gold_states,
                    "generated": text,
                    "parsed_answer": answer,
                    "parsed_states": states,
                }
            )

    n = len(instances)
    return {
        "answer_accuracy": answer_correct / n,
        "exact_trace_accuracy": (
            None if condition == "outcome" else exact_trace_correct / n
        ),
        "free_step_accuracy": (
            None if condition == "outcome" else step_correct / step_total
        ),
        "audit": audit,
    }


@torch.inference_mode()
def evaluate_teacher_states(model, tokenizer, instances, args, device):
    prefixes, targets = [], []
    for inst in instances:
        depth = len(trace_states(inst.correct_trace))
        for t in range(depth):
            prefix, target = local_prefix(inst, t)
            prefixes.append(prefix)
            targets.append(target)

    texts = batched_generate(
        model,
        tokenizer,
        prefixes,
        args,
        device,
        args.local_max_new_tokens,
    )

    correct = 0
    for text, target in zip(texts, targets):
        match = BITS_RE.search(text)
        pred = match.group(1) if match else None
        correct += int(pred == target)
    return correct / len(targets)


def append_csv(path: Path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


RESULT_FIELDS = [
    "model",
    "task",
    "condition",
    "rho",
    "realized_rho",
    "seed",
    "train_size",
    "val_size",
    "max_steps",
    "effective_batch_size",
    "lora_rank",
    "lora_alpha",
    "trainable_parameters",
    "trainable_fraction",
    "final_train_loss_50",
    "answer_accuracy",
    "exact_trace_accuracy",
    "free_step_accuracy",
    "teacher_state_accuracy",
    "predicted_exact_from_teacher",
]


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(
    task,
    tokenizer,
    dataset,
    val_instances,
    condition,
    rho,
    seed,
    args,
    device,
    result_csv,
    audit_dir,
    adapter_dir,
):
    print("\n" + "=" * 88)
    print(f"condition={condition} rho={rho} seed={seed}")
    print("=" * 88)

    set_all_seeds(seed)
    model = build_lora_model(args, tokenizer, device)
    trainable, total, fraction = trainable_parameter_summary(model)
    print(
        f"LoRA trainable parameters: {trainable:,} / {total:,} "
        f"({100*fraction:.4f}%)"
    )

    train_metrics = train_lora(model, dataset, seed, args, device)

    free = evaluate_free(
        model, tokenizer, val_instances, condition, args, device
    )

    if condition == "process":
        local_instances = val_instances[: min(args.local_eval_size, len(val_instances))]
        teacher = evaluate_teacher_states(
            model, tokenizer, local_instances, args, device
        )
        depth = len(trace_states(val_instances[0].correct_trace))
        predicted_exact = teacher ** depth
    else:
        teacher = None
        predicted_exact = None

    row = {
        "model": args.model,
        "task": task.name,
        "condition": condition,
        "rho": "" if rho is None else rho,
        "realized_rho": (
            "" if dataset.realized_rho is None else dataset.realized_rho
        ),
        "seed": seed,
        "train_size": len(dataset),
        "val_size": len(val_instances),
        "max_steps": args.max_steps,
        "effective_batch_size": train_metrics["effective_batch_size"],
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "trainable_parameters": trainable,
        "trainable_fraction": fraction,
        "final_train_loss_50": train_metrics["final_train_loss_50"],
        "answer_accuracy": free["answer_accuracy"],
        "exact_trace_accuracy": (
            "" if free["exact_trace_accuracy"] is None
            else free["exact_trace_accuracy"]
        ),
        "free_step_accuracy": (
            "" if free["free_step_accuracy"] is None
            else free["free_step_accuracy"]
        ),
        "teacher_state_accuracy": "" if teacher is None else teacher,
        "predicted_exact_from_teacher": (
            "" if predicted_exact is None else predicted_exact
        ),
    }
    append_csv(result_csv, RESULT_FIELDS, row)

    audit_dir.mkdir(parents=True, exist_ok=True)
    label = "outcome" if condition == "outcome" else f"rho_{rho:.2f}"
    (audit_dir / f"{label}_seed_{seed}.json").write_text(
        json.dumps(free["audit"], indent=2)
    )

    if args.save_adapters:
        path = adapter_dir / label / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    print(
        f"answer={100*free['answer_accuracy']:.2f}% "
        + (
            ""
            if condition == "outcome"
            else (
                f"exact={100*free['exact_trace_accuracy']:.2f}% "
                f"free-step={100*free['free_step_accuracy']:.2f}% "
                f"teacher-state={100*teacher:.2f}% "
                f"teacher^D={100*predicted_exact:.2f}%"
            )
        )
    )
    cleanup_model(model)


def summarize_results(result_csv: Path, output_dir: Path):
    import pandas as pd

    df = pd.read_csv(result_csv)
    process = df[df["condition"] == "process"].copy()
    outcome = df[df["condition"] == "outcome"].copy()

    summary = {
        "num_runs": int(len(df)),
        "conditions": {},
    }

    for condition, group in df.groupby("condition"):
        if condition == "process":
            for rho, rg in group.groupby("rho"):
                key = f"process_rho_{rho:g}"
                summary["conditions"][key] = {
                    "n": int(len(rg)),
                    "answer_mean": float(rg["answer_accuracy"].mean()),
                    "answer_std": float(rg["answer_accuracy"].std(ddof=1)),
                    "exact_mean": float(rg["exact_trace_accuracy"].mean()),
                    "exact_std": float(rg["exact_trace_accuracy"].std(ddof=1)),
                    "teacher_mean": float(rg["teacher_state_accuracy"].mean()),
                    "teacher_std": float(rg["teacher_state_accuracy"].std(ddof=1)),
                }
        else:
            summary["conditions"]["outcome"] = {
                "n": int(len(group)),
                "answer_mean": float(group["answer_accuracy"].mean()),
                "answer_std": float(group["answer_accuracy"].std(ddof=1)),
            }

    if len(process):
        pred = process["predicted_exact_from_teacher"].to_numpy(float)
        obs = process["exact_trace_accuracy"].to_numpy(float)
        summary["local_to_global"] = {
            "mae": float(np.mean(np.abs(pred - obs))),
            "correlation": (
                float(np.corrcoef(pred, obs)[0, 1])
                if np.std(pred) > 0 and np.std(obs) > 0
                else None
            ),
        }

        aggregate = (
            process.groupby("rho", as_index=False)
            .agg(
                answer=("answer_accuracy", "mean"),
                exact=("exact_trace_accuracy", "mean"),
                free_step=("free_step_accuracy", "mean"),
                teacher=("teacher_state_accuracy", "mean"),
                teacher_pred=("predicted_exact_from_teacher", "mean"),
            )
            .sort_values("rho")
        )
        exact_values = aggregate["exact"].to_numpy(float)
        summary["monotone_exact_over_tested_rhos"] = bool(
            np.all(np.diff(exact_values) >= -0.03)
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(aggregate["rho"], aggregate["answer"], marker="o", label="answer")
        ax.plot(aggregate["rho"], aggregate["exact"], marker="o", label="exact rollout")
        ax.plot(aggregate["rho"], aggregate["teacher"], marker="o", label="teacher-state")
        ax.plot(
            aggregate["rho"],
            aggregate["teacher_pred"],
            marker="x",
            linestyle="--",
            label="teacher-state^D",
        )
        if len(outcome):
            ax.axhline(
                outcome["answer_accuracy"].mean(),
                linestyle=":",
                label="outcome-only answer",
            )
        ax.set_xlabel("trace reliability rho")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Pretrained LoRA replication: local acquisition and rollout")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "pretrained_reliability.png", dpi=180)
        plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def validate_task(task):
    if not task.name.startswith("boolean_circuit_"):
        raise ValueError(
            "This script intentionally implements one existing task family only: "
            "boolean_circuit_*. Use e.g. --task boolean_circuit_8."
        )
    probe = task.sample()
    initial, gates = parse_boolean_prompt(probe.prompt)
    correct = trace_states(probe.correct_trace)
    wrong = trace_states(probe.wrong_trace)
    if len(gates) != len(correct) or len(correct) != len(wrong):
        raise AssertionError("Task parse sanity check failed")
    if probe.gold != correct[-1]:
        raise AssertionError("Gold answer does not equal final correct state")
    if any(c == w for c, w in zip(correct, wrong)):
        # The repository's construction normally makes every displayed corrupted
        # successor invalid; equality with the canonical clean state can in rare
        # trajectories occur only if paths reconverge. Do not assert on it.
        pass


def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.8, 1.0])
    p.add_argument("--include-outcome", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])

    p.add_argument("--train-size", type=int, default=30_000)
    p.add_argument("--val-size", type=int, default=1_000)
    p.add_argument("--local-eval-size", type=int, default=250)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--ratio-seed", type=int, default=777)
    p.add_argument("--batch-seed", type=int, default=12345)

    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["all-linear"],
        help='Use "all-linear" or explicit modules such as q_proj k_proj v_proj o_proj.',
    )

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--save-adapters", action="store_true")

    p.add_argument("--process-max-new-tokens", type=int, default=160)
    p.add_argument("--outcome-max-new-tokens", type=int, default=24)
    p.add_argument("--local-max-new-tokens", type=int, default=12)
    p.add_argument("--audit-examples", type=int, default=10)

    p.add_argument("--output-dir", type=Path, default=Path("results/pretrained_lora"))
    p.add_argument("--run-name")
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        raise ValueError("Every rho must lie in [0,1]")
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if not 20_000 <= args.train_size <= 50_000:
        print(
            f"WARNING: requested train-size={args.train_size}; the proposed paper "
            "replication was 20k-50k examples."
        )

    task = TASKS[args.task]
    validate_task(task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "WARNING: no CUDA GPU detected. The script will run, but pretrained "
            "LoRA fine-tuning will be very slow on CPU."
        )
    else:
        torch.set_float32_matmul_precision("high")

    tokenizer = load_tokenizer(args.model)
    args.pad_token_id = tokenizer.pad_token_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{task.name}_{Path(args.model).name}_{timestamp}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = output_dir / "results.csv"
    if result_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{result_csv} exists; pass --overwrite to replace")
    if result_csv.exists():
        result_csv.unlink()

    # Fixed underlying X,Y pool for every condition.
    train_instances = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {inst.prompt for inst in train_instances}
    val_instances = generate_unique(
        task, args.val_size, args.val_seed, excluded=train_prompts
    )
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["device"] = str(device)
    config["actual_run_directory"] = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    print(
        f"model={args.model} task={task.name} device={device} "
        f"train={len(train_instances)} val={len(val_instances)} "
        f"train/val prompt overlap=0"
    )

    conditions = []
    if args.include_outcome:
        conditions.append(("outcome", None))
    conditions.extend(("process", rho) for rho in sorted(set(args.rhos)))

    for condition, rho in conditions:
        dataset = CompletionDataset(
            train_instances,
            tokenizer,
            max_length=args.max_length,
            condition=condition,
            rho=rho,
            ratio_scores=ratio_scores,
        )
        if condition == "process":
            print(
                f"rho={rho:.3f}: realized clean fraction="
                f"{dataset.realized_rho:.5f}"
            )

        for seed in args.seeds:
            run_one(
                task=task,
                tokenizer=tokenizer,
                dataset=dataset,
                val_instances=val_instances,
                condition=condition,
                rho=rho,
                seed=seed,
                args=args,
                device=device,
                result_csv=result_csv,
                audit_dir=output_dir / "audit",
                adapter_dir=output_dir / "adapters",
            )
        del dataset
        gc.collect()

    summary = summarize_results(result_csv, output_dir)
    print("\nFINAL SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, indent=2))
    print(f"\nsaved: {output_dir}")


if __name__ == "__main__":
    main()
