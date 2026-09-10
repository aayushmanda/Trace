"""Mechanism test: local credit on a deep outcome architecture (answer-only).

Three conditions share the outcome residual layout except ordinary process, which
is the known-good comparison (process tokens). L_out+local never puts trace tokens
in the LM target: H_t is a linear readout of block-t hidden states.

If outcome ≈ chance while outcome+local ≈ process ≈ 100%, that isolates credit
placement. Mixed-format P̂_g is a negative transfer result, not this test.

Do not claim an ε^{D-1} mixing-ball rate from C_t; report C_t vs D only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.plot_style import apply_style
from src.training.config import load_yaml
from src.training.io import write_csv
from src.training.optim import make_adamw
from src.training.progress import progress
from src.training.seed import add_compile_bf16_flags, autocast_context, configure_device, default_device, set_seed

ROOT = Path(__file__).resolve().parents[2]

import handcoded as h
from handcoded.local_credit import (
    chance_accuracy,
    collect_hiddens,
    counterfactual_accuracy,
    credit_signals,
    gold_states_tensor,
    has_gate_tokens,
    outcome_plus_local_loss,
    trained_target_ids,
    train_linear_probes,
)
from handcoded.models import attach_local_heads
from src.eval import executor_comparison as c


CONDITIONS = ("outcome", "outcome_local", "process")
PATCH_METHODS = ("oracle_slot", "random_subspace", "unstructured", "wrong_layer")


def _mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_gold(circuits, tokenizer, mode, device):
    data = h.encode_dataset(circuits, tokenizer, mode).to(device)
    gold = gold_states_tensor(circuits, device)
    return data, gold


def answer_accuracy(model, circuits, tokenizer, device, mode):
    return c.generation_metrics(model, circuits, tokenizer, device, mode)["free_answer_accuracy"]


def evaluate_probes(model, probe_train, probe_eval, tokenizer, device, n_states, probe_steps, probe_lr):
    was_training = model.training
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    train_h, train_y = collect_hiddens(model, probe_train, tokenizer, device)
    eval_h, eval_y = collect_hiddens(model, probe_eval, tokenizer, device)
    probes, train_acc, eval_acc = train_linear_probes(
        train_h, train_y, eval_h, eval_y, n_states, steps=probe_steps, lr=probe_lr,
    )
    for param in model.parameters():
        param.requires_grad_(True)
    model.train(was_training)
    return probes, train_acc, eval_acc


def train_loss(condition, model, batch, gold, lambda_local):
    if condition == "outcome_local":
        loss, _, _ = outcome_plus_local_loss(model, batch, gold, lambda_local)
        return loss
    return h.language_model_loss(model, batch)


def run_patches(oracle, circuits, tokenizer, device, methods=PATCH_METHODS):
    rows = []
    generator = torch.Generator().manual_seed(0)
    depth = oracle.depth
    for layer in range(depth):
        for method in methods:
            kwargs = {"device": device, "generator": generator}
            if method == "wrong_layer":
                kwargs["method"] = "oracle_slot"
                kwargs["wrong_layer"] = (layer + 1) % depth if depth > 1 else layer
            else:
                kwargs["method"] = method
            acc = counterfactual_accuracy(oracle, circuits, tokenizer, layer, **kwargs)
            rows.append({
                "model": "oracle_outcome",
                "layer": layer,
                "method": method,
                "counterfactual_accuracy": acc,
            })
    return rows


def plot_results(output, metrics, probes, credit, patches, depth):
    apply_style()
    output = Path(output)
    colors = {"outcome": "#dd8452", "outcome_local": "#247ba0", "process": "#55a868"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for condition in CONDITIONS:
        rows = [r for r in metrics if r["condition"] == condition]
        if not rows:
            continue
        ax.plot(
            [r["step"] for r in rows], [r["answer_accuracy"] for r in rows],
            marker="o", color=colors[condition], label=condition,
        )
    ax.axhline(chance_accuracy(), color="#888", ls="--", lw=1, label="chance")
    ax.set_xlabel("step")
    ax.set_ylabel("answer accuracy")
    ax.set_title("Outcome-local credit (answer only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "answer_accuracy.pdf")
    fig.savefig(output / "answer_accuracy.png", dpi=140)
    plt.close(fig)

    latest = {}
    for row in probes:
        latest[(row["condition"], row["layer"])] = row
    if latest:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for condition in CONDITIONS:
            xs = [i + 1 for i in range(depth)]
            ys = [latest.get((condition, i), {}).get("probe_eval_accuracy") for i in range(depth)]
            if any(y is not None for y in ys):
                ax.plot(xs, ys, marker="o", color=colors[condition], label=condition)
        ax.set_xlabel("block t")
        ax.set_ylabel(r"probe acc $H_t(h_t)\to s_t$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Linear state probes (held-out circuits)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "probe_accuracy.pdf")
        plt.close(fig)

    if credit:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for condition in CONDITIONS:
            rows = [r for r in credit if r["condition"] == condition]
            if not rows:
                continue
            by_layer = {}
            for row in rows:
                by_layer.setdefault(row["layer"], []).append(row["C_t"])
            xs = sorted(by_layer)
            ys = [sum(by_layer[i]) / len(by_layer[i]) for i in xs]
            ax.plot([i + 1 for i in xs], ys, marker="o", color=colors[condition], label=condition)
        ax.set_xlabel("block t")
        ax.set_ylabel(r"$C_t=\langle g^{\mathrm{term}},g^{\mathrm{local}}\rangle/\|g^{\mathrm{local}}\|^2$")
        ax.set_title(r"$C_t$ vs layer (not an $\varepsilon^{D-1}$ claim)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "credit_C.pdf")
        plt.close(fig)

    if patches:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        methods = []
        for row in patches:
            if row["method"] not in methods:
                methods.append(row["method"])
        means = []
        for method in methods:
            vals = [r["counterfactual_accuracy"] for r in patches if r["method"] == method]
            means.append(sum(vals) / max(len(vals), 1))
        ax.bar(methods, means, color="#4c72b0")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("counterfactual execution accuracy")
        ax.set_title("Oracle outcome patch (positive control vs noise)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output / "patch_accuracy.pdf")
        plt.close(fig)


def run_experiment(args):
    output = _mkdir(args.output)
    set_seed(args.seed)
    device = configure_device(args.device, compile=False, bf16=getattr(args, "bf16", False))
    tokenizer = h.make_tokenizer()
    depth = args.depth
    train = c.unique_circuits(args.train_size, args.train_seed, depth)
    val = c.unique_circuits(args.val_size, args.val_seed, depth, c.circuit_keys(train))
    probe_eval = c.unique_circuits(
        args.probe_size, args.probe_seed, depth, c.circuit_keys(train) | c.circuit_keys(val),
    )
    probe_train = train[: min(len(train), max(args.probe_size, 8))]
    patch_circuits = val[: min(len(val), args.patch_circuits)]

    outcome_data, outcome_gold = encode_gold(train, tokenizer, "outcome", device)
    process_data, _ = encode_gold(train, tokenizer, "process", device)
    if has_gate_tokens(trained_target_ids(outcome_data), tokenizer):
        raise RuntimeError("outcome targets contain process/gate tokens")

    base = h.build_random_trainable_outcome_architecture(tokenizer, depth, seed=args.seed, device=device)
    attach_local_heads(base, seed=args.seed + 17, device=device)
    models = {
        "outcome": copy.deepcopy(base),
        "outcome_local": copy.deepcopy(base),
        "process": h.build_random_trainable_process_architecture(
            tokenizer, depth, seed=args.seed, device=device,
        ),
    }
    data = {"outcome": outcome_data, "outcome_local": outcome_data, "process": process_data}
    gold = {"outcome": outcome_gold, "outcome_local": outcome_gold, "process": None}
    mode = {"outcome": "outcome", "outcome_local": "outcome", "process": "process"}
    optimizers = {
        name: make_adamw(model.parameters(), args.lr, weight_decay=0.0, device=device)
        for name, model in models.items()
    }
    schedule = h.make_batch_schedule(len(train), args.steps, args.batch_size, args.batch_seed)
    checkpoints = sorted(set(args.checkpoints) | {0, args.steps})
    checkpoints = [k for k in checkpoints if 0 <= k <= args.steps]

    metrics, probe_rows, credit_rows = [], [], []

    def checkpoint(step):
        for name, model in models.items():
            model.eval()
            acc = answer_accuracy(model, val, tokenizer, device, mode[name])
            metrics.append({
                "condition": name, "step": step, "depth": depth, "seed": args.seed,
                "answer_accuracy": acc, "lambda_local": args.lambda_local,
                "format": mode[name],
            })
            if name == "process":
                continue
            probes, train_acc, eval_acc = evaluate_probes(
                model, probe_train, probe_eval, tokenizer, device, tokenizer.n_states,
                args.probe_steps, args.probe_lr,
            )
            for layer, (tr, ev) in enumerate(zip(train_acc, eval_acc)):
                probe_rows.append({
                    "condition": name, "step": step, "depth": depth, "seed": args.seed,
                    "layer": layer, "probe_train_accuracy": tr, "probe_eval_accuracy": ev,
                    "answer_accuracy": acc,
                })
            index = torch.tensor(schedule[max(step - 1, 0)], device=device)
            batch = outcome_data.select(index)
            gold_batch = outcome_gold[index]
            model.train()
            for row in credit_signals(model, batch, gold_batch, probes):
                credit_rows.append({
                    "condition": name, "step": step, "depth": depth, "seed": args.seed,
                    **row, "note": "C_t vs D only; not an eps^{D-1} claim",
                })
            model.eval()
        print(json.dumps({
            "step": step,
            "answer": {r["condition"]: round(r["answer_accuracy"], 4) for r in metrics if r["step"] == step},
        }), flush=True)

    oracle = h.HandcodedOutcomeTransformer(tokenizer, depth).to(device).eval()
    patch_rows = run_patches(oracle, patch_circuits, tokenizer, device)
    write_csv(output / "patch.csv", patch_rows)
    oracle_mean = sum(r["counterfactual_accuracy"] for r in patch_rows if r["method"] == "oracle_slot") / depth
    random_mean = sum(r["counterfactual_accuracy"] for r in patch_rows if r["method"] == "random_subspace") / depth
    print(json.dumps({
        "oracle_patch_mean": round(oracle_mean, 4),
        "random_patch_mean": round(random_mean, 4),
    }), flush=True)

    checkpoint(0)
    for step in progress(range(1, args.steps + 1), desc=f"outcome-local D={depth}"):
        index = torch.tensor(schedule[step - 1], device=device)
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            batch = data[name].select(index)
            gold_batch = None if gold[name] is None else gold[name][index]
            with autocast_context(device):
                loss = train_loss(name, model, batch, gold_batch, args.lambda_local)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizers[name].step()
        if step in checkpoints:
            checkpoint(step)

    write_csv(output / "metrics.csv", metrics)
    if probe_rows:
        write_csv(output / "probes.csv", probe_rows)
    if credit_rows:
        write_csv(output / "credit.csv", credit_rows)
    table = []
    for name in CONDITIONS:
        final = next(r for r in metrics if r["condition"] == name and r["step"] == args.steps)
        row = {"condition": name, "answer_accuracy": final["answer_accuracy"], "depth": depth, "seed": args.seed}
        for layer in range(depth):
            match = [
                r for r in probe_rows
                if r["condition"] == name and r["step"] == args.steps and r["layer"] == layer
            ]
            row[f"s{layer + 1}_probe_acc"] = match[0]["probe_eval_accuracy"] if match else ""
        table.append(row)
    write_csv(output / "table.csv", table)
    plot_results(output, metrics, probe_rows, credit_rows, patch_rows, depth)
    persist = {
        "experiment": "outcome_local",
        "mechanism_test": True,
        "mixed_format_is_negative_transfer": True,
        "no_eps_D_minus_1_claim": True,
        "depth": depth,
        "seed": args.seed,
        "steps": args.steps,
        "lambda_local": args.lambda_local,
        "oracle_patch_mean": oracle_mean,
        "random_patch_mean": random_mean,
        "final_answer": {r["condition"]: r["answer_accuracy"] for r in metrics if r["step"] == args.steps},
        "csv": str(output / "metrics.csv"),
    }
    (output / "persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print("wrote", output)
    return persist


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/experiments/outcome_local.yaml")
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml(pre_args.config) if Path(pre_args.config).exists() or (ROOT / pre_args.config).exists() else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / str(cfg.get("output", "results/paper_revision_v2/outcome_local")))
    parser.add_argument("--depth", type=int, default=int(cfg.get("depth", 2)))
    parser.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    parser.add_argument("--steps", type=int, default=int(cfg.get("steps", 40)))
    parser.add_argument("--train-size", type=int, default=int(cfg.get("train_size", 64)))
    parser.add_argument("--val-size", type=int, default=int(cfg.get("val_size", 16)))
    parser.add_argument("--probe-size", type=int, default=int(cfg.get("probe_size", 16)))
    parser.add_argument("--patch-circuits", type=int, default=int(cfg.get("patch_circuits", 8)))
    parser.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 16)))
    parser.add_argument("--lr", type=float, default=float(cfg.get("lr", 0.002)))
    parser.add_argument("--lambda-local", type=float, default=float(cfg.get("lambda_local", cfg.get("lambda", 1.0))))
    parser.add_argument("--grad-clip", type=float, default=float(cfg.get("grad_clip", 1.0)))
    parser.add_argument("--probe-steps", type=int, default=int(cfg.get("probe_steps", 40)))
    parser.add_argument("--probe-lr", type=float, default=float(cfg.get("probe_lr", 0.05)))
    parser.add_argument("--train-seed", type=int, default=int(cfg.get("train_seed", 123)))
    parser.add_argument("--val-seed", type=int, default=int(cfg.get("val_seed", 8000)))
    parser.add_argument("--probe-seed", type=int, default=int(cfg.get("probe_seed", 9000)))
    parser.add_argument("--batch-seed", type=int, default=int(cfg.get("batch_seed", 2026)))
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(cfg.get("checkpoints") or [0, 20, 40]))
    parser.add_argument("--device", default=cfg.get("device"))
    add_compile_bf16_flags(parser, cfg, from_yaml=True)
    args = parser.parse_args(argv)
    if args.smoke:
        args.depth = 2
        args.steps = 30
        args.train_size = 32
        args.val_size = 8
        args.probe_size = 8
        args.patch_circuits = 4
        args.batch_size = 8
        args.checkpoints = [0, 15, 30]
        args.probe_steps = 20
        args.output = ROOT / "results/paper_revision_v2/outcome_local"
    if args.device is None:
        args.device = "cpu" if args.smoke else default_device()
    args.compile = False
    return args


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
