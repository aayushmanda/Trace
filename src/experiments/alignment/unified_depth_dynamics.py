"""Paired UnifiedExecutor depth sweep with exact directional token-gradient probes.

Run from the repository root, for example:
  uv run --offline --no-project --with 'torch==2.7.1' --with numpy \
    python -m src.experiments.alignment.unified_depth_dynamics \
    --depths 2 4 6 8 --seeds 0 1 2 --steps 10000

Each (depth, seed) pair starts process and outcome from byte-identical
parameters and uses identical circuit prompts and minibatch indices. Depths
use different parameter counts, so comparisons across depths are descriptive. For each
trained model, task directions come from a separately sampled clean process
probe: negative gradients of local intermediate-state cross-entropy on the
full probe and its shards. The token directional contribution is -D_v ell_e,
computed by JVP.
This measures the paper's A, S, and C on the actual masked token losses.
Directions are recomputed at each checkpoint, so the statistic diagnoses the
current neural tangent geometry; it is not a fixed-coordinate escape theorem.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, jvp

from handcoded.data import encode_dataset, language_model_loss, make_batch_schedule
from handcoded.eval import free_run_metrics, make_generation_evaluation
from handcoded.gates import make_circuits
from handcoded.models import build_random_trainable_unified
from handcoded.tokenizer import make_tokenizer


def local_state_loss(model, batch, colon_id, n_states):
    logits = model(batch.inputs)
    losses = F.cross_entropy(logits.transpose(1, 2), batch.targets, ignore_index=-100, reduction="none")
    positions = torch.arange(batch.targets.shape[1], device=batch.targets.device)[None, :]
    colon_positions = (batch.targets == colon_id).int().argmax(dim=1)[:, None]
    mask = (batch.targets >= 0) & (batch.targets < n_states) & (positions < colon_positions)
    if not mask.any():
        raise ValueError("No intermediate-state tokens in task-direction probe")
    return losses[mask].mean()


def task_directions(model, process_probe, tokenizer, count):
    """Orthonormal negative gradients of clean local-state loss on probe shards."""
    if len(process_probe) < count:
        raise ValueError("Task probe must contain at least one example per direction")
    parameters = tuple(model.parameters())
    vectors = []
    for shard in range(count):
        if shard == 0:
            subset = process_probe
        else:
            indices = torch.arange(shard - 1, len(process_probe), count - 1, device=process_probe.inputs.device)
            subset = process_probe.select(indices)
        loss = local_state_loss(model, subset, tokenizer.colon, tokenizer.n_states)
        gradients = torch.autograd.grad(loss, parameters)
        candidate = [-g.detach() for g in gradients]
        for basis in vectors:
            coefficient = sum((a * b).sum() for a, b in zip(candidate, basis))
            candidate = [a - coefficient * b for a, b in zip(candidate, basis)]
        norm = torch.sqrt(sum((g * g).sum() for g in candidate))
        if not torch.isfinite(norm) or norm.item() <= 1e-12:
            continue
        vectors.append([g / norm for g in candidate])
    if not vectors:
        raise ValueError("All local-state task gradients vanished")
    return vectors


def token_directional_derivatives(model, batch, vector):
    """Return -D_v ell for every supervised token, with no loss normalization."""
    params = dict(model.named_parameters())
    tangent = {name: direction for name, direction in zip(params, vector)}

    def per_token_loss(current):
        logits = functional_call(model, current, (batch.inputs,))
        return F.cross_entropy(logits.transpose(1, 2), batch.targets, ignore_index=-100, reduction="none")

    _, derivative = jvp(per_token_loss, (params,), (tangent,))
    return -derivative[batch.targets != -100].detach()


def own_direction(model, own_probe):
    """Unit-norm negative gradient of the format's own token loss on its probe set."""
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(language_model_loss(model, own_probe), parameters)
    norm = torch.sqrt(sum((g * g).sum() for g in gradients))
    return [-g.detach() / norm for g in gradients]


def random_directions(model, count, seed):
    """Fixed Gaussian unit directions in parameter space (null for A and C)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = []
    for _ in range(count):
        draw = [torch.randn(p.shape, generator=generator).to(p.device) for p in model.parameters()]
        norm = torch.sqrt(sum((g * g).sum() for g in draw))
        vectors.append([g / norm for g in draw])
    return vectors


def a_and_c(contributions):
    a = contributions.abs().mean().item()
    return a, (abs(contributions.mean().item()) / a if a > 0 else None)


def diagnostics(model, mode_batch, process_probe, tokenizer, directions, own_probe=None, n_random=0, random_seed=0):
    model.eval()
    basis = task_directions(model, process_probe, tokenizer, directions)
    with torch.no_grad():
        contributions = torch.stack(
            [token_directional_derivatives(model, mode_batch, vector) for vector in basis], dim=1
        )
    one = contributions[:, 0]
    a = one.abs().mean().item()
    s_signed = one.mean().item()
    row = {
        "tokens": len(one), "task_directions": len(basis),
        "A": a, "S": abs(s_signed), "signed_speed": s_signed,
        "C": abs(s_signed) / a if a > 0 else None,
    }
    if own_probe is not None:
        own = own_direction(model, own_probe)
        with torch.no_grad():
            row["A_own"], row["C_own"] = a_and_c(token_directional_derivatives(model, mode_batch, own))
    if n_random:
        with torch.no_grad():
            pairs = [a_and_c(token_directional_derivatives(model, mode_batch, v))
                     for v in random_directions(model, n_random, random_seed)]
        row["A_rand"] = sum(p[0] for p in pairs) / n_random
        row["C_rand"] = sum(p[1] for p in pairs) / n_random
    if len(basis) > 1:
        magnitudes = contributions.norm(dim=1)
        population = contributions.mean(dim=0).norm()
        row.update(
            task_subspace_A=magnitudes.mean().item(),
            task_subspace_S=population.item(),
            task_subspace_C=(population / magnitudes.mean()).item() if magnitudes.mean() > 0 else None,
        )
    return row


def check_jvp(model, batch, vector, rel_tol=0.06):
    """One finite-difference validation of the exact JVP implementation."""
    params = dict(model.named_parameters())
    tangent = {name: direction for name, direction in zip(params, vector)}
    eps = 1e-3
    with torch.no_grad():
        plus = {name: value + eps * tangent[name] for name, value in params.items()}
        minus = {name: value - eps * tangent[name] for name, value in params.items()}
        upper = F.cross_entropy(
            functional_call(model, plus, (batch.inputs,)).transpose(1, 2),
            batch.targets, ignore_index=-100, reduction="none",
        )
        lower = F.cross_entropy(
            functional_call(model, minus, (batch.inputs,)).transpose(1, 2),
            batch.targets, ignore_index=-100, reduction="none",
        )
        estimate = -((upper - lower) / (2 * eps))[batch.targets != -100]
    exact = token_directional_derivatives(model, batch, vector)
    error = (exact - estimate).abs().max().item()
    scale = max(1e-4, exact.abs().max().item())
    if error > rel_tol * scale:
        raise AssertionError(f"JVP check failed: max error {error:.4g}, scale {scale:.4g}")
    return error


def checkpoint_row(model, mode, step, depth, seed, train_batch, test_batch, probe_batch, evaluation, tokenizer, directions,
                   own_probe=None, n_random=0):
    with torch.no_grad():
        diagnostic_loss = language_model_loss(model, test_batch).item()
        accuracy = free_run_metrics(model, evaluation, tokenizer, mode)
    measured = diagnostics(model, test_batch, probe_batch, tokenizer, directions, own_probe, n_random, random_seed=seed)
    return {
        "depth": depth, "seed": seed, "mode": mode, "step": step,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(train_batch), "diagnostic_loss": diagnostic_loss,
        "answer_accuracy": accuracy["final_answer"],
        "exact_continuation": accuracy["exact_continuation"],
        **measured,
    }


def run_one(depth, seed, args, device, writer, output_file):
    tokenizer = make_tokenizer(args.n_bits, args.n_gates)
    train = make_circuits(args.train_size, args.data_seed + seed, depth, n_bits=args.n_bits, n_gates=args.n_gates)
    test = make_circuits(args.test_size, args.test_seed + seed, depth, n_bits=args.n_bits, n_gates=args.n_gates)
    # An independently sampled stream supplies task directions; it is never used for fitting or token diagnostics.
    task_probe = make_circuits(args.task_probe, args.probe_seed + seed, depth, n_bits=args.n_bits, n_gates=args.n_gates)
    datasets = {mode: encode_dataset(train, tokenizer, mode).to(device) for mode in ("process", "outcome")}
    test_data = {mode: encode_dataset(test, tokenizer, mode).to(device) for mode in ("process", "outcome")}
    probe_data = encode_dataset(task_probe, tokenizer, "process").to(device)
    own_probes = {mode: encode_dataset(task_probe, tokenizer, mode).to(device) for mode in ("process", "outcome")}
    evaluation = make_generation_evaluation(test[:args.eval_examples], tokenizer, device)
    diagnostic_data = {mode: test_data[mode].select(slice(0, min(args.diagnostic_examples, len(test))))
                       for mode in ("process", "outcome")}
    schedule = make_batch_schedule(len(train), args.steps, args.batch_size, args.batch_seed + seed)
    base = build_random_trainable_unified(tokenizer, depth, seed=args.model_seed + seed, device=device)
    models = {mode: copy.deepcopy(base) for mode in ("process", "outcome")}
    assert all(torch.equal(a, b) for a, b in zip(models["process"].parameters(), models["outcome"].parameters()))
    checkpoints = set(args.checkpoints) | {0, args.steps}
    for mode, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0, fused=device.type == "cuda")
        for step in range(args.steps + 1):
            if step in checkpoints:
                row = checkpoint_row(model, mode, step, depth, seed, datasets[mode],
                                     diagnostic_data[mode], probe_data, evaluation, tokenizer, args.directions,
                                     own_probes[mode], args.random_directions)
                writer.writerow(row)
                output_file.flush()
                print(json.dumps(row), flush=True)
                if args.check_jvp and step == 0:
                    vector = task_directions(model, probe_data, tokenizer, 1)[0]
                    print("JVP max finite-difference error", check_jvp(model, diagnostic_data[mode], vector), flush=True)
            if step == args.steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = datasets[mode].select(schedule[step])
            loss = language_model_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if args.save_models is not None:
            args.save_models.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_models / f"D{depth}_s{seed}_{mode}.pt")
    del models, base, datasets, test_data, probe_data, own_probes
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-bits", type=int, default=2)
    parser.add_argument("--n-gates", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--task-probe", type=int, default=64)
    parser.add_argument("--diagnostic-examples", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--random-directions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 100, 500, 2000, 5000, 10000])
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--test-seed", type=int, default=9000)
    parser.add_argument("--probe-seed", type=int, default=19000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--batch-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/paper/unified_depth_dynamics.csv"))
    parser.add_argument("--check-jvp", action="store_true")
    parser.add_argument("--save-models", type=Path, default=None, help="Directory for final state dicts.")
    args = parser.parse_args()
    if min(args.depths) < 1 or args.steps < 1 or min(args.seeds) < 0:
        parser.error("depths and steps must be positive, seeds nonnegative")
    if args.task_probe < args.directions or args.test_size < max(args.diagnostic_examples, args.eval_examples):
        parser.error("probe or test set is too small for the requested diagnostics")
    return args


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    fields = ["depth", "seed", "mode", "step", "parameters", "train_examples", "diagnostic_loss", "answer_accuracy",
              "exact_continuation", "tokens", "task_directions", "A", "S", "signed_speed", "C",
              "task_subspace_A", "task_subspace_S", "task_subspace_C", "A_own", "C_own", "A_rand", "C_rand"]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for depth in args.depths:
            for seed in args.seeds:
                run_one(depth, seed, args, device, writer, output_file)


if __name__ == "__main__":
    main()
