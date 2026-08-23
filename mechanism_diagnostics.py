import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.model import GPTModel
from src.registry import TASKS


SUMMARY_FIELDS = [
    "task", "condition", "rho", "seed", "step", "train_loss",
    "answer_accuracy", "exact_trace_accuracy", "free_step_accuracy",
    "colon_rate", "mean_first_error", "teacher_full_step_accuracy",
    "teacher_state_accuracy", "teacher_state_token_accuracy",
    "teacher_state_nll", "predicted_exact_trace_probability",
    "gradient_alignment", "gradient_cosine", "clean_corrupt_cosine",
    "clean_state_gradient_norm", "training_gradient_norm",
    "between_component_variance", "projected_gradient_mean",
    "projected_gradient_variance", "clean_state_loss",
    "diagnostic_training_loss",
]

STEP_FIELDS = [
    "task", "condition", "rho", "seed", "step", "transition_index",
    "free_step_accuracy", "first_error_rate", "teacher_full_step_accuracy",
    "teacher_state_accuracy", "teacher_state_token_accuracy",
    "teacher_state_nll",
]


def configure_runtime(deterministic: bool):
    torch.set_float32_matmul_precision("highest")
    if not deterministic:
        return
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_unique(task, size: int, seed: int, excluded=None):
    excluded = set() if excluded is None else set(excluded)
    state = random.getstate()
    random.seed(seed)
    instances, prompts = [], set()
    while len(instances) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        instances.append(inst)
        prompts.add(inst.prompt)
    random.setstate(state)
    return instances


def split_step(task_name: str, step: str):
    """Return the serialized operation prefix and next-state target."""
    if task_name.startswith("state_machine_"):
        return step[:-2], step[-2:]
    if task_name.startswith("register_machine_"):
        return step[:-4], step[-4:]
    if task_name.startswith("boolean_circuit_"):
        prefix, state = step.rsplit(">", 1)
        return prefix + ">", state
    raise ValueError(
        f"Mechanism diagnostics support state_machine, register_machine, and "
        f"boolean_circuit tasks, not {task_name!r}."
    )


def nested_clean_flags(size: int, rho: float, assignment_order: np.ndarray):
    flags = np.zeros(size, dtype=bool)
    flags[assignment_order[: round(rho * size)]] = True
    return flags


class SequenceDataset(Dataset):
    """Compact continuation dataset with optional next-state-only masking."""

    def __init__(self, instances, task, mode: str, clean_flags=None, state_only=False):
        if mode not in {"outcome", "process", "corrupted", "mixed"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "mixed" and clean_flags is None:
            raise ValueError("mixed mode requires clean_flags")
        if state_only and mode != "process":
            raise ValueError("state_only is defined only for clean process traces")

        tokenizer = task.tokenizer
        width = task.block_size - 1
        if tokenizer.vocab_size > 256:
            raise ValueError("uint8 storage requires tokenizer.vocab_size <= 256")
        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)

        for row, inst in enumerate(instances):
            if mode == "outcome":
                continuation = f" : {inst.gold}\n"
            else:
                clean = mode == "process" or (mode == "mixed" and clean_flags[row])
                trace = inst.correct_trace if clean else inst.wrong_trace
                continuation = f" {trace} : {inst.gold}\n"

            prompt_ids = tokenizer.encode(inst.prompt)
            full = prompt_ids + tokenizer.encode(continuation)
            if len(full) > task.block_size:
                raise ValueError(
                    f"{task.name}/{mode}: {len(full)} tokens exceeds block_size={task.block_size}"
                )
            sequence = torch.tensor(full, dtype=torch.uint8)
            length = len(full) - 1
            self.x[row, :length] = sequence[:-1]
            self.y[row, :length] = sequence[1:]

            if not state_only:
                self.mask[row, len(prompt_ids) - 1:length] = True
                continue

            cursor = 1  # continuation begins with one space
            for step in inst.correct_trace.split():
                step_start = continuation.find(step, cursor)
                if step_start < 0:
                    raise RuntimeError(f"Could not locate step {step!r} in {continuation!r}")
                prefix, state = split_step(task.name, step)
                state_start = step_start + len(prefix)
                for offset in range(state_start, state_start + len(state)):
                    self.mask[row, len(prompt_ids) + offset - 1] = True
                cursor = step_start + len(step)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]


def build_model(task, args, device):
    return GPTModel(
        vocab_size=task.tokenizer.vocab_size,
        block_size=task.block_size,
        pad_id=task.tokenizer.pad_id,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        dropout=args.dropout,
    ).to(device)


def make_optimizer(model, args, device):
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device.type == "cuda" and not args.deterministic:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def parse_generation(task, tail: str, outcome: bool):
    line = tail.split("\n", 1)[0]
    emitted_colon = ":" in line
    if not emitted_colon:
        return None, None, False
    if outcome:
        trace = None
    else:
        trace = line.rsplit(":", 1)[0].strip()
    return trace, task.extract_answer(line), True


@torch.inference_mode()
def evaluate_free(model, task, instances, outcome: bool, eval_batch_size: int, device):
    model.eval()
    buckets = defaultdict(list)
    for inst in instances:
        ids = task.tokenizer.encode(inst.prompt)
        buckets[len(ids)].append((ids, inst))

    depth = len(instances[0].correct_trace.split())
    step_correct = np.zeros(depth, dtype=np.int64)
    first_errors = np.zeros(depth, dtype=np.int64)
    answer_correct = exact_trace = colon_count = total = 0
    first_error_sum = 0
    max_new_tokens = 8 if outcome else task.max_new_tokens

    for rows in buckets.values():
        for start in range(0, len(rows), eval_batch_size):
            batch = rows[start:start + eval_batch_size]
            context = torch.tensor([ids for ids, _ in batch], dtype=torch.long, device=device)
            prompt_length = context.shape[1]
            output = model.generate(
                context, max_new_tokens=max_new_tokens,
                stop_id=task.tokenizer.newline_id, greedy=True,
            )
            for generated_ids, (_, inst) in zip(output.tolist(), batch):
                tail = task.tokenizer.decode(generated_ids[prompt_length:])
                trace, answer, emitted_colon = parse_generation(task, tail, outcome)
                colon_count += int(emitted_colon)
                answer_correct += int(answer == inst.gold)
                if not outcome:
                    gold_steps = inst.correct_trace.split()
                    predicted_steps = [] if trace is None else trace.split()
                    first_error = depth + 1
                    for index, gold_step in enumerate(gold_steps):
                        correct = index < len(predicted_steps) and predicted_steps[index] == gold_step
                        step_correct[index] += int(correct)
                        if not correct and first_error == depth + 1:
                            first_error = index + 1
                            first_errors[index] += 1
                    exact_trace += int(trace == inst.correct_trace)
                    first_error_sum += first_error
                total += 1

    return {
        "answer_accuracy": answer_correct / total,
        "exact_trace_accuracy": None if outcome else exact_trace / total,
        "free_step_accuracy": None if outcome else float(step_correct.sum() / (total * depth)),
        "colon_rate": colon_count / total,
        "mean_first_error": None if outcome else first_error_sum / total,
        "per_step_accuracy": None if outcome else step_correct / total,
        "first_error_rate": None if outcome else first_errors / total,
    }


def build_teacher_queries(task, instances):
    full_step_queries, state_queries = [], []
    tokenizer = task.tokenizer
    for inst in instances:
        steps = inst.correct_trace.split()
        previous = []
        for index, step in enumerate(steps):
            before_step = f"{inst.prompt} " + (" ".join(previous) + " " if previous else "")
            prefix, state = split_step(task.name, step)
            full_step_queries.append((index, tokenizer.encode(before_step), tokenizer.encode(step)))
            state_queries.append((index, tokenizer.encode(before_step + prefix), tokenizer.encode(state)))
            previous.append(step)
    return full_step_queries, state_queries


@torch.inference_mode()
def score_teacher_queries(model, queries, depth: int, batch_size: int, device):
    model.eval()
    groups = defaultdict(list)
    for transition_index, context, target in queries:
        groups[(len(context), len(target))].append((transition_index, context, target))

    exact = np.zeros(depth, dtype=np.int64)
    count = np.zeros(depth, dtype=np.int64)
    token_correct = np.zeros(depth, dtype=np.int64)
    token_count = np.zeros(depth, dtype=np.int64)
    nll_sum = np.zeros(depth, dtype=np.float64)

    for (context_length, target_length), rows in groups.items():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            context = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
            target = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)

            generated = model.generate(context, max_new_tokens=target_length, greedy=True)
            generated_target = generated[:, context_length:context_length + target_length]
            generated_exact = generated_target.eq(target).all(dim=1)

            sequence = torch.cat((context, target), dim=1)
            logits, _ = model(sequence[:, :-1])
            target_logits = logits[:, context_length - 1:context_length - 1 + target_length]
            predicted_tokens = target_logits.argmax(dim=-1)
            losses = F.cross_entropy(
                target_logits.reshape(-1, target_logits.shape[-1]),
                target.reshape(-1), reduction="none",
            ).view(len(batch), target_length)

            for row_index, (transition_index, _, _) in enumerate(batch):
                exact[transition_index] += int(generated_exact[row_index])
                count[transition_index] += 1
                token_correct[transition_index] += int(predicted_tokens[row_index].eq(target[row_index]).sum())
                token_count[transition_index] += target_length
                nll_sum[transition_index] += float(losses[row_index].sum())

    return {
        "exact": exact / np.maximum(count, 1),
        "token_accuracy": token_correct / np.maximum(token_count, 1),
        "nll": nll_sum / np.maximum(token_count, 1),
    }


def evaluate_teacher_forced(model, task, instances, batch_size: int, device):
    depth = len(instances[0].correct_trace.split())
    full_queries, state_queries = build_teacher_queries(task, instances)
    full = score_teacher_queries(model, full_queries, depth, batch_size, device)
    state = score_teacher_queries(model, state_queries, depth, batch_size, device)
    return {
        "teacher_full_step_accuracy": float(full["exact"].mean()),
        "teacher_state_accuracy": float(state["exact"].mean()),
        "teacher_state_token_accuracy": float(state["token_accuracy"].mean()),
        "teacher_state_nll": float(state["nll"].mean()),
        "predicted_exact_trace_probability": float(np.prod(full["exact"])),
        "per_step_full": full["exact"],
        "per_step_state": state["exact"],
        "per_step_state_token": state["token_accuracy"],
        "per_step_state_nll": state["nll"],
    }


def gradient_vector(model, batch, device):
    x, y, mask = batch
    x = x.to(device, dtype=torch.long, non_blocking=True)
    y = y.to(device, dtype=torch.long, non_blocking=True)
    mask = mask.to(device, dtype=torch.float32, non_blocking=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y, mask=mask)
    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False)
    vector = torch.cat([gradient.detach().float().reshape(-1) for gradient in gradients]).cpu()
    return vector, float(loss.detach()), float(mask.sum())


def dataset_gradients(model, dataset, microbatch_size: int, device):
    vectors, losses, weights = [], [], []
    for start in range(0, len(dataset), microbatch_size):
        stop = min(start + microbatch_size, len(dataset))
        vector, loss, weight = gradient_vector(
            model, (dataset.x[start:stop], dataset.y[start:stop], dataset.mask[start:stop]), device
        )
        vectors.append(vector)
        losses.append(loss)
        weights.append(weight)
    total_weight = max(sum(weights), 1.0)
    mean = sum(vector * weight for vector, weight in zip(vectors, weights)) / total_weight
    mean_loss = sum(loss * weight for loss, weight in zip(losses, weights)) / total_weight
    return mean, vectors, mean_loss


def safe_cosine(first, second):
    denominator = float(first.norm() * second.norm())
    return float(torch.dot(first, second) / denominator) if denominator > 0 else math.nan


def gradient_diagnostics(model, diagnostic_sets, condition: str, rho, microbatch_size: int, device):
    model.eval()
    clean_state, _, clean_state_loss = dataset_gradients(
        model, diagnostic_sets["clean_state"], microbatch_size, device
    )
    reference_norm = float(clean_state.norm())
    reference_unit = clean_state / max(reference_norm, 1e-12)

    if condition == "outcome":
        actual, actual_vectors, actual_loss = dataset_gradients(
            model, diagnostic_sets["outcome"], microbatch_size, device
        )
        projections = np.asarray([float(torch.dot(vector, reference_unit)) for vector in actual_vectors])
        clean_corrupt_cosine = between_variance = None
        projected_mean = float(projections.mean())
        projected_variance = float(projections.var())
    else:
        clean, clean_vectors, clean_loss = dataset_gradients(
            model, diagnostic_sets["process"], microbatch_size, device
        )
        corrupt, corrupt_vectors, corrupt_loss = dataset_gradients(
            model, diagnostic_sets["corrupted"], microbatch_size, device
        )
        actual = rho * clean + (1.0 - rho) * corrupt
        actual_loss = rho * clean_loss + (1.0 - rho) * corrupt_loss
        clean_projection = np.asarray([float(torch.dot(vector, reference_unit)) for vector in clean_vectors])
        corrupt_projection = np.asarray([float(torch.dot(vector, reference_unit)) for vector in corrupt_vectors])
        projected_mean = float(rho * clean_projection.mean() + (1.0 - rho) * corrupt_projection.mean())
        projected_variance = float(
            rho * clean_projection.var()
            + (1.0 - rho) * corrupt_projection.var()
            + rho * (1.0 - rho) * (clean_projection.mean() - corrupt_projection.mean()) ** 2
        )
        clean_corrupt_cosine = safe_cosine(clean, corrupt)
        between_variance = float(rho * (1.0 - rho) * (clean - corrupt).square().sum())

    denominator = float(clean_state.square().sum())
    alignment = float(torch.dot(clean_state, actual) / denominator) if denominator > 0 else math.nan
    return {
        "gradient_alignment": alignment,
        "gradient_cosine": safe_cosine(clean_state, actual),
        "clean_corrupt_cosine": clean_corrupt_cosine,
        "clean_state_gradient_norm": reference_norm,
        "training_gradient_norm": float(actual.norm()),
        "between_component_variance": between_variance,
        "projected_gradient_mean": projected_mean,
        "projected_gradient_variance": projected_variance,
        "clean_state_loss": clean_state_loss,
        "diagnostic_training_loss": actual_loss,
    }


def append_csv(path: Path, row: dict, fields):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def train_run(
    task, train_dataset, val_instances, teacher_instances, diagnostic_sets,
    condition, rho, seed, checkpoints, gradient_checkpoints, args, device,
    summary_output, step_output,
):
    set_seed(seed)
    generator = torch.Generator().manual_seed(args.batch_seed)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, generator=generator, pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)
    model = build_model(task, args, device)
    optimizer = make_optimizer(model, args, device)
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    label = "outcome" if condition == "outcome" else f"rho={rho:.4f}"

    for step in tqdm(range(1, max(checkpoints) + 1), desc=f"{task.name}/{label}/seed={seed}"):
        model.train()
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        mask = mask.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y, mask=mask)
        else:
            _, loss = model(x, targets=y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step not in checkpoints:
            continue

        free = evaluate_free(
            model, task, val_instances, condition == "outcome", args.eval_batch_size, device
        )
        teacher = evaluate_teacher_forced(
            model, task, teacher_instances, args.teacher_batch_size, device
        )
        gradients = {}
        if step in gradient_checkpoints:
            gradients = gradient_diagnostics(
                model, diagnostic_sets, condition, rho,
                args.gradient_microbatch_size, device,
            )

        summary = {
            "task": task.name, "condition": label, "rho": rho, "seed": seed,
            "step": step, "train_loss": float(loss.detach()), **free, **teacher, **gradients,
        }
        append_csv(summary_output, summary, SUMMARY_FIELDS)

        depth = len(teacher["per_step_full"])
        for transition_index in range(depth):
            row = {
                "task": task.name, "condition": label, "rho": rho, "seed": seed,
                "step": step, "transition_index": transition_index + 1,
                "free_step_accuracy": None if free["per_step_accuracy"] is None else free["per_step_accuracy"][transition_index],
                "first_error_rate": None if free["first_error_rate"] is None else free["first_error_rate"][transition_index],
                "teacher_full_step_accuracy": teacher["per_step_full"][transition_index],
                "teacher_state_accuracy": teacher["per_step_state"][transition_index],
                "teacher_state_token_accuracy": teacher["per_step_state_token"][transition_index],
                "teacher_state_nll": teacher["per_step_state_nll"][transition_index],
            }
            append_csv(step_output, row, STEP_FIELDS)

        gradient_text = ""
        if gradients:
            gradient_text = (
                f" align={gradients['gradient_alignment']:.3f}"
                f" cosine={gradients['gradient_cosine']:.3f}"
            )
        exact_text = "n/a" if free["exact_trace_accuracy"] is None else f"{100 * free['exact_trace_accuracy']:.2f}%"
        print(
            f"{task.name} {label} seed={seed} step={step} "
            f"answer={100 * free['answer_accuracy']:.2f}% exact={exact_text} "
            f"teacher_state={100 * teacher['teacher_state_accuracy']:.2f}% "
            f"predicted_exact={100 * teacher['predicted_exact_trace_probability']:.2f}%"
            f"{gradient_text}"
        )

    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+",
        default=["state_machine_16", "register_machine_16", "boolean_circuit_8"],
    )
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.8, 0.85, 0.9, 0.95, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1000, 2000, 4000, 6000, 8000])
    parser.add_argument("--gradient-checkpoints", nargs="+", type=int)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--teacher-size", type=int, default=256)
    parser.add_argument("--gradient-size", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--diagnostic-seed", type=int, default=303)
    parser.add_argument("--assignment-seed", type=int, default=777)
    parser.add_argument("--batch-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--gradient-microbatch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-outcome", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if any(rho < 0.0 or rho > 1.0 for rho in args.rhos):
        parser.error("Every rho must lie in [0, 1].")
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] < 1:
        parser.error("Checkpoints must be positive.")
    gradient_checkpoints = (
        sorted(set(args.gradient_checkpoints))
        if args.gradient_checkpoints
        else sorted(set([checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]))
    )
    if not set(gradient_checkpoints).issubset(checkpoints):
        parser.error("Every gradient checkpoint must also be an evaluation checkpoint.")

    configure_runtime(args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = args.output_dir / f"mechanism_summary_{timestamp}.csv"
    step_output = args.output_dir / f"mechanism_steps_{timestamp}.csv"

    print(
        f"device={device} deterministic={args.deterministic} bf16={args.bf16} "
        f"summary={summary_output} steps={step_output}"
    )

    for task_name in args.tasks:
        if task_name not in TASKS:
            parser.error(f"Unknown task {task_name!r}.")
        task = TASKS[task_name]
        split_step(task.name, task.sample().correct_trace.split()[0])  # fail early if unsupported

        train_instances = generate_unique(task, args.train_size, args.train_seed)
        train_prompts = {inst.prompt for inst in train_instances}
        val_instances = generate_unique(task, args.val_size, args.val_seed, train_prompts)
        excluded = train_prompts | {inst.prompt for inst in val_instances}
        diagnostic_instances = generate_unique(task, args.gradient_size, args.diagnostic_seed, excluded)
        teacher_instances = val_instances[: min(args.teacher_size, len(val_instances))]

        assignment_order = np.random.default_rng(args.assignment_seed).permutation(len(train_instances))
        diagnostic_sets = {
            "clean_state": SequenceDataset(diagnostic_instances, task, "process", state_only=True),
            "process": SequenceDataset(diagnostic_instances, task, "process"),
            "corrupted": SequenceDataset(diagnostic_instances, task, "corrupted"),
            "outcome": SequenceDataset(diagnostic_instances, task, "outcome"),
        }
        print(
            f"\n{task.name}: chance={100 * task.chance_acc:.2f}% "
            f"train={len(train_instances)} val={len(val_instances)} "
            f"teacher={len(teacher_instances)} gradient={len(diagnostic_instances)} overlap=0"
        )

        conditions = [("outcome", None)] if args.include_outcome else []
        conditions += [("mixed", rho) for rho in sorted(set(args.rhos))]
        for condition, rho in conditions:
            if condition == "outcome":
                train_dataset = SequenceDataset(train_instances, task, "outcome")
            else:
                flags = nested_clean_flags(len(train_instances), rho, assignment_order)
                train_dataset = SequenceDataset(train_instances, task, "mixed", clean_flags=flags)
                assert int(flags.sum()) == round(rho * len(train_instances))
            for seed in args.seeds:
                train_run(
                    task, train_dataset, val_instances, teacher_instances, diagnostic_sets,
                    condition, rho, seed, checkpoints, gradient_checkpoints, args, device,
                    summary_output, step_output,
                )
            del train_dataset

    print(f"\nsaved summary: {summary_output}")
    print(f"saved per-step diagnostics: {step_output}")


if __name__ == "__main__":
    main()