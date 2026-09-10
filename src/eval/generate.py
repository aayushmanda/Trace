import re
from collections import defaultdict

import torch

from src.training.progress import progress


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
