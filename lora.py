#!/usr/bin/env python3
"""
Fast SmolLM replication of Trace reliability experiments.

Run this file FROM THE ROOT of https://github.com/aayushmanda/Trace

It deliberately reuses Trace's own:
  - TASKS registry
  - boolean_circuit_* task sampler
  - generate_unique(...)
  - exact compact prompt / trace / answer serialization

Only the model is changed:
  Trace GPT-from-scratch  ->  pretrained HuggingFaceTB/SmolLM2-135M + LoRA.

Default pilot:
  outcome-only
  rho = 0.0, 0.5, 0.8, 1.0
  one model seed
  6k training examples
  400 optimizer updates

The central test is:
  1) valid process supervision >> outcome / corrupted supervision;
  2) teacher-forced local state accuracy improves before exact free rollout;
  3) reliability rho orders acquisition.

Install once:
  uv add transformers peft accelerate

Example:
  uv run smollm_trace_experiment.py

Paper-scale follow-up:
  uv run smollm_trace_experiment.py \
    --rhos 0.0 0.3 0.5 0.6 0.7 0.8 1.0 \
    --seeds 2001 2002 2003 \
    --train-size 12000 --steps 800
"""

import argparse
import csv
import gc
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.registry import TASKS
from sweep_ratio import generate_unique


# Trace's Boolean trace items look like:
#   x0>0101
#   c12>0111
#   s03>1110
#   t012>1010
STEP_RE = re.compile(r"([xcst]\d{1,3}>([01]{4}))")
ANSWER_RE = re.compile(r"([01]{4})")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trace_steps(trace: str):
    return trace.strip().split()


def completion_for(inst, condition: str, clean: bool | None = None):
    """
    Preserve the serialization already used by Trace.

    sweep_ratio.py:
        outcome:       " : {gold}\\n"
        mixed process: " {trace} : {gold}\\n"
    compare_supervision.py:
        answer_first:  " : {gold} ; {correct_trace}\\n"
    """
    if condition == "outcome":
        return f" : {inst.gold}\n"
    if condition == "answer_first":
        return f" : {inst.gold} ; {inst.correct_trace}\n"
    if condition == "process":
        trace = inst.correct_trace if clean else inst.wrong_trace
        return f" {trace} : {inst.gold}\n"
    raise ValueError(condition)


class CompletionDataset(Dataset):
    """
    HuggingFace causal-LM SFT with zero loss on the Trace prompt.

    For process supervision the underlying training examples are identical for all
    rho. A fixed ratio_score is assigned to each example, so reliability assignments
    are nested exactly as in Trace's sweep_ratio.py.
    """

    def __init__(self, instances, tokenizer, condition, rho, ratio_scores, max_length):
        self.rows = []
        self.realized_rho = None

        if condition == "process":
            if rho is None:
                raise ValueError("process condition needs rho")
            clean_flags = np.asarray(ratio_scores) < float(rho)
            self.realized_rho = float(clean_flags.mean())
        else:
            clean_flags = None

        eos = tokenizer.eos_token_id

        for i, inst in enumerate(instances):
            prompt = inst.prompt
            if condition == "process":
                completion = completion_for(inst, condition, bool(clean_flags[i]))
            else:
                completion = completion_for(inst, condition)

            # Deliberately tokenize prompt and supervised continuation separately.
            # This gives an exact completion-loss mask.
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if eos is not None:
                target_ids = target_ids + [eos]

            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids.copy()

            if len(ids) > max_length:
                raise ValueError(
                    f"{len(ids)} tokens > --max-length={max_length}. "
                    "Increase max-length rather than truncating a reasoning trace."
                )

            self.rows.append((ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        width = max(len(ids) for ids, _ in rows)
        batch = len(rows)

        input_ids = torch.full((batch, width), self.pad_id, dtype=torch.long)
        labels = torch.full((batch, width), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, width), dtype=torch.long)

        for i, (ids, labs) in enumerate(rows):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            labels[i, :n] = torch.tensor(labs)
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(args, tokenizer, device):
    dtype = choose_dtype(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.full_finetune:
        model.to(device)
        return model

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    model.to(device)
    return model


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total


def train(model, dataset, args, seed, device):
    set_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.batch_seed + seed),
        collate_fn=Collator(args.pad_id),
    )
    iterator = iter(loader)

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    amp = device.type == "cuda"
    amp_dtype = choose_dtype(device)

    model.train()
    losses = []
    pbar = tqdm(range(1, args.steps + 1), desc=f"train seed={seed}")

    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp,
        ):
            loss = model(**batch).loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach()))
        if len(losses) > 30:
            losses.pop(0)

        if step % 10 == 0:
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return float(np.mean(losses))


@torch.inference_mode()
def generate_text(model, tokenizer, prompts, args, device, max_new_tokens):
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    outputs = []

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
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        input_width = encoded["input_ids"].shape[1]
        tails = generated[:, input_width:]
        outputs.extend(tokenizer.batch_decode(tails, skip_special_tokens=True))

    tokenizer.padding_side = old_padding
    return outputs


def parse_free_generation(text: str, condition: str):
    # Trace's targets terminate at newline. Ignore anything SmolLM emits afterward.
    line = text.split("\n", 1)[0].strip()

    if ":" not in line:
        return None, None

    if condition in {"outcome", "answer_first"}:
        right = line.split(":", 1)[1]
        match = ANSWER_RE.search(right)
        return None, match.group(1) if match else None

    trace_text, answer_text = line.rsplit(":", 1)
    answer_match = ANSWER_RE.search(answer_text)
    answer = answer_match.group(1) if answer_match else None
    return trace_text.strip(), answer


@torch.inference_mode()
def evaluate_free(model, tokenizer, instances, condition, args, device):
    prompts = [inst.prompt for inst in instances]
    max_new = args.outcome_max_new_tokens if condition in {"outcome", "answer_first"} else args.process_max_new_tokens
    texts = generate_text(model, tokenizer, prompts, args, device, max_new)

    answer_correct = 0
    exact_correct = 0
    step_correct = 0
    step_total = 0

    audits = []

    for inst, text in zip(instances, texts):
        pred_trace, pred_answer = parse_free_generation(text, condition)
        answer_correct += int(pred_answer == inst.gold)

        if condition == "process":
            gold_steps = trace_steps(inst.correct_trace)
            pred_steps = [] if pred_trace is None else pred_trace.split()

            exact_correct += int(pred_trace == inst.correct_trace)
            step_correct += sum(
                p == g for p, g in zip(pred_steps, gold_steps)
            )
            step_total += len(gold_steps)

        if len(audits) < args.audit_examples:
            audits.append({
                "prompt": inst.prompt,
                "gold_trace": inst.correct_trace,
                "wrong_trace": inst.wrong_trace,
                "gold_answer": inst.gold,
                "generated": text,
                "parsed_trace": pred_trace,
                "parsed_answer": pred_answer,
            })

    n = len(instances)
    return {
        "answer_accuracy": answer_correct / n,
        "exact_trace_accuracy": None if condition != "process" else exact_correct / n,
        "free_step_accuracy": None if condition != "process" else step_correct / step_total,
        "audit": audits,
    }


def local_prefix(inst, transition_index):
    """
    Teacher-force all previous CLEAN Trace steps, then ask SmolLM for the next step.

    This keeps the exact Trace serialization:
       prompt step_1 step_2 ... step_t
    """
    gold_steps = trace_steps(inst.correct_trace)
    previous = gold_steps[:transition_index]

    prefix = inst.prompt + " "
    if previous:
        prefix += " ".join(previous) + " "

    return prefix, gold_steps[transition_index]


@torch.inference_mode()
def evaluate_teacher_forced_local(model, tokenizer, instances, args, device):
    prefixes = []
    gold_steps = []

    depth = len(trace_steps(instances[0].correct_trace))

    for inst in instances:
        for t in range(depth):
            prefix, gold = local_prefix(inst, t)
            prefixes.append(prefix)
            gold_steps.append(gold)

    texts = generate_text(
        model,
        tokenizer,
        prefixes,
        args,
        device,
        args.local_max_new_tokens,
    )

    step_correct = 0
    state_correct = 0

    for text, gold in zip(texts, gold_steps):
        pred_match = STEP_RE.search(text)
        gold_match = STEP_RE.fullmatch(gold)

        pred_step = pred_match.group(1) if pred_match else None
        pred_state = pred_match.group(2) if pred_match else None
        gold_state = gold_match.group(2)

        step_correct += int(pred_step == gold)
        state_correct += int(pred_state == gold_state)

    total = len(gold_steps)
    return {
        "teacher_step_accuracy": step_correct / total,
        "teacher_state_accuracy": state_correct / total,
    }


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def cleanup(model):
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
    output_csv,
    audit_dir,
):
    print("\n" + "=" * 92)
    print(f"condition={condition} rho={rho} seed={seed}")
    print("=" * 92)

    set_seed(seed)
    model = build_model(args, tokenizer, device)
    trainable, total, fraction = trainable_summary(model)
    print(
        f"trainable parameters: {trainable:,}/{total:,} "
        f"({100*fraction:.3f}%)"
    )

    final_loss = train(model, dataset, args, seed, device)
    free = evaluate_free(
        model, tokenizer, val_instances, condition, args, device
    )

    if condition == "process":
        local_instances = val_instances[: min(args.local_eval_size, len(val_instances))]
        local = evaluate_teacher_forced_local(
            model, tokenizer, local_instances, args, device
        )
        depth = len(trace_steps(val_instances[0].correct_trace))
        predicted_from_state = local["teacher_state_accuracy"] ** depth
        predicted_from_step = local["teacher_step_accuracy"] ** depth
    else:
        local = {
            "teacher_step_accuracy": None,
            "teacher_state_accuracy": None,
        }
        predicted_from_state = None
        predicted_from_step = None

    label = condition if condition != "process" else f"rho={rho:.2f}"

    row = {
        "model": args.model,
        "task": task.name,
        "condition": label,
        "rho": "" if rho is None else rho,
        "realized_rho": "" if dataset.realized_rho is None else dataset.realized_rho,
        "seed": seed,
        "train_size": len(dataset),
        "val_size": len(val_instances),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "full_finetune": args.full_finetune,
        "trainable_parameters": trainable,
        "trainable_fraction": fraction,
        "loss": final_loss,
        "answer_accuracy": free["answer_accuracy"],
        "exact_trace_accuracy": "" if free["exact_trace_accuracy"] is None else free["exact_trace_accuracy"],
        "free_step_accuracy": "" if free["free_step_accuracy"] is None else free["free_step_accuracy"],
        "teacher_step_accuracy": "" if local["teacher_step_accuracy"] is None else local["teacher_step_accuracy"],
        "teacher_state_accuracy": "" if local["teacher_state_accuracy"] is None else local["teacher_state_accuracy"],
        "teacher_step_pow_D": "" if predicted_from_step is None else predicted_from_step,
        "teacher_state_pow_D": "" if predicted_from_state is None else predicted_from_state,
    }
    append_csv(output_csv, row)

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_name = condition if condition != "process" else f"rho_{rho:.2f}"
    (audit_dir / f"{audit_name}_seed_{seed}.json").write_text(
        json.dumps(free["audit"], indent=2)
    )

    print(
        f"loss={final_loss:.4f} "
        f"answer={100*free['answer_accuracy']:.2f}%"
    )
    if condition == "process":
        print(
            f"exact rollout={100*free['exact_trace_accuracy']:.2f}% | "
            f"free step={100*free['free_step_accuracy']:.2f}% | "
            f"teacher step={100*local['teacher_step_accuracy']:.2f}% | "
            f"teacher state={100*local['teacher_state_accuracy']:.2f}% | "
            f"teacher-state^D={100*predicted_from_state:.2f}%"
        )

    cleanup(model)


def summarize(output_csv: Path):
    import pandas as pd

    df = pd.read_csv(output_csv)
    print("\nFINAL SUMMARY")
    print("=" * 92)

    columns = [
        "condition",
        "seed",
        "answer_accuracy",
        "exact_trace_accuracy",
        "free_step_accuracy",
        "teacher_state_accuracy",
        "teacher_state_pow_D",
    ]
    print(df[columns].to_string(index=False))

    process = df[df["rho"].notna()].copy()
    if len(process) > 1:
        pred = process["teacher_state_pow_D"].astype(float).to_numpy()
        obs = process["exact_trace_accuracy"].astype(float).to_numpy()
        mae = float(np.mean(np.abs(pred - obs)))
        corr = (
            float(np.corrcoef(pred, obs)[0, 1])
            if np.std(pred) > 0 and np.std(obs) > 0
            else float("nan")
        )
        print(f"\nlocal->global diagnostic: MAE={100*mae:.2f} pp, corr={corr:.4f}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Small pretrained BASE LM, not instruction-tuned.
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
    )
    parser.add_argument("--task", default="boolean_circuit_8")

    # Cheap but informative reliability points.
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 0.8, 1.0],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2001],
    )
    parser.add_argument(
        "--include-answer-first",
        action="store_true",
        help="Adds the Trace answer-first control; costs one extra training per seed.",
    )

    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--local-eval-size", type=int, default=60)

    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--ratio-seed", type=int, default=777)
    parser.add_argument("--batch-seed", type=int, default=12345)

    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train all 135M parameters instead of LoRA. Stronger but slower/more memory.",
    )

    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--process-max-new-tokens", type=int, default=128)
    parser.add_argument("--outcome-max-new-tokens", type=int, default=20)
    parser.add_argument("--local-max-new-tokens", type=int, default=16)
    parser.add_argument("--audit-examples", type=int, default=8)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/smollm_trace_pilot.csv"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if any(rho < 0 or rho > 1 for rho in args.rhos):
        parser.error("every rho must be in [0,1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        print("WARNING: no CUDA GPU; this experiment will be slow.")

    if args.task not in TASKS:
        raise KeyError(f"Unknown Trace task: {args.task}")
    if not args.task.startswith("boolean_circuit_"):
        raise ValueError(
            "This compact pilot intentionally targets Trace's boolean_circuit_* "
            "family because its intermediate states have a clean 4-bit local metric."
        )

    task = TASKS[args.task]
    tokenizer = load_tokenizer(args.model)
    args.pad_id = tokenizer.pad_token_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} exists. Pass --overwrite or choose another --output."
            )
        args.output.unlink()

    # Exactly the same underlying X,Y pool for every supervision condition.
    train_instances = generate_unique(
        task,
        args.train_size,
        args.train_seed,
    )
    train_prompts = {inst.prompt for inst in train_instances}
    val_instances = generate_unique(
        task,
        args.val_size,
        args.val_seed,
        excluded=train_prompts,
    )

    # Nested reliability assignment, just like sweep_ratio.py.
    ratio_scores = np.random.default_rng(args.ratio_seed).random(
        len(train_instances)
    )

    print(
        f"model={args.model} task={task.name} device={device} "
        f"train={len(train_instances)} val={len(val_instances)} "
        f"prompt_overlap=0"
    )
    print(
        "Trace example:\n"
        f"  prompt        = {train_instances[0].prompt}\n"
        f"  correct trace = {train_instances[0].correct_trace}\n"
        f"  wrong trace   = {train_instances[0].wrong_trace}\n"
        f"  gold          = {train_instances[0].gold}"
    )

    conditions = [("outcome", None)]
    if args.include_answer_first:
        conditions.append(("answer_first", None))
    conditions.extend(
        ("process", rho)
        for rho in sorted(set(args.rhos))
    )

    audit_dir = args.output.parent / (args.output.stem + "_audit")

    for condition, rho in conditions:
        dataset = CompletionDataset(
            train_instances,
            tokenizer,
            condition,
            rho,
            ratio_scores,
            args.max_length,
        )

        if condition == "process":
            print(
                f"\nrho={rho:.3f}: realized clean fraction="
                f"{dataset.realized_rho:.4f}"
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
                output_csv=args.output,
                audit_dir=audit_dir,
            )

        del dataset
        gc.collect()

    summarize(args.output)
    print(f"\nsaved results: {args.output}")
    print(f"saved generations: {audit_dir}")


if __name__ == "__main__":
    main()
