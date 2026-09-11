"""Mechanism test: local credit on a deep outcome architecture (answer-only).

Three conditions share the outcome residual layout. Default process still uses the
one-block process architecture (historical). Pass `--matched-architecture` to train
process on the same D-block outcome net (`build_random_trainable_outcome_architecture`)
so the depth table is not mixing architectures. L_out+local never puts trace tokens
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

import nnsight  # experiment path: probes and causal patches require nnsight

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
import numpy as np

from src.eval import executor_comparison as c
from src.eval.rule_credit import credit_summary


CONDITIONS = ("outcome", "outcome_local", "process")
# Construction-slot interventions plus the layerwise trained-probe path.
# `probe_subspace` is the identification object on trained nets; mixed-format
# P̂_g is not.
PATCH_METHODS = ("oracle_slot", "random_subspace", "unstructured", "wrong_layer", "probe_subspace")
SLOT_PATCH_METHODS = ("oracle_slot", "random_subspace", "unstructured", "wrong_layer")
REQUIRED_PERSIST_KEYS = (
    "experiment",
    "mechanism_test",
    "mixed_format_is_negative_transfer",
    "no_eps_D_minus_1_claim",
    "depth",
    "seed",
    "steps",
    "lambda_local",
    "oracle_patch_mean",
    "random_patch_mean",
    "probe_subspace_oracle_mean",
    "final_answer",
    "final_train_answer",
    "final_probe_eval",
    "csv",
    "report",
    "interventions",
)


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


def run_patches(
    model, circuits, tokenizer, device, methods=None, model_name="oracle_outcome",
    depth=None, probes=None, donor_ids=None,
):
    """Activation-patch `model` at every block with each method in `methods`.

    Works identically on the hand-built oracle and on any trained model that shares
    `HandcodedOutcomeTransformer`'s residual layout (`state_slots`, `_answer_index`,
    `depth`) — `outcome`, `outcome_local`, and process *when* it is the matched
    D-block copy. The one-block process architecture has no per-step state slots.

    `probe_subspace` needs a trained linear probe per block (the Appendix-M gap).
    """
    rows = []
    generator = torch.Generator().manual_seed(0)
    depth = model.depth if depth is None else depth
    if methods is None:
        methods = (
            (*SLOT_PATCH_METHODS, "probe_subspace", "probe_wrong_layer")
            if probes is not None else SLOT_PATCH_METHODS
        )
    was_training = model.training
    model.eval()
    for layer in range(depth):
        for method in methods:
            kwargs = {"device": device, "generator": generator, "donor_ids": donor_ids}
            if method == "wrong_layer":
                kwargs["method"] = "oracle_slot"
                kwargs["wrong_layer"] = (layer + 1) % depth if depth > 1 else layer
            elif method == "probe_wrong_layer":
                if probes is None:
                    raise ValueError("probe_wrong_layer needs trained probes")
                wrong = (layer + 1) % depth if depth > 1 else layer
                kwargs["method"] = "probe_subspace"
                kwargs["probe"] = probes[wrong]
                kwargs["wrong_layer"] = wrong
            elif method == "probe_subspace":
                if probes is None:
                    raise ValueError("probe_subspace patch needs trained probes")
                kwargs["method"] = "probe_subspace"
                kwargs["probe"] = probes[layer]
            else:
                kwargs["method"] = method
            acc = counterfactual_accuracy(model, circuits, tokenizer, layer, **kwargs)
            rows.append({
                "model": model_name,
                "layer": layer,
                "method": method,
                "counterfactual_accuracy": acc,
            })
    model.train(was_training)
    return rows


@torch.no_grad()
def measure_induced_credit(model, tokenizer, circuits, device, depth, backgrounds=2, seed=8127):
    """Gold-prefix Def 20 readout on a D-block (or process-length) net.

    Reports mixing radius ||P̂_g − U||_2, outcome Rule credit, and the process
    cell floor from the same tables. Not an ε^{D−1} claim unless mixing is small.
    """
    tables, masses = c.read_local_rules(
        model, tokenizer, depth, device, backgrounds=backgrounds, seed=seed,
    )
    p_hat = tables.mean(0).detach().cpu().numpy()
    k = tokenizer.n_states
    uniform = np.ones((k, k), dtype=np.float64) / k
    radii = [
        float(np.linalg.norm(p_hat[g] - uniform, ord=2))
        for g in range(len(tokenizer.gates))
    ]
    gate_index = {name: i for i, name in enumerate(tokenizer.gates)}
    credit_mean, credit_max, n_terms = credit_summary(p_hat, gate_index, circuits, k)
    floors = []
    for circuit in circuits:
        source = circuit.start
        for t, gate in enumerate(circuit.gates):
            gold = circuit.states[t]
            cell = float(p_hat[gate_index[gate], source, gold])
            floors.append((1.0 - 1.0 / k) / max(cell, 1e-30))
            source = gold
    return {
        "mixing_radius_max": float(max(radii) if radii else float("nan")),
        "mixing_radius_mean": float(np.mean(radii) if radii else float("nan")),
        "outcome_rule_credit_mean": float(credit_mean),
        "outcome_rule_credit_max": float(credit_max),
        "n_credit_terms": int(n_terms),
        "process_floor_mean": float(np.mean(floors) if floors else float("nan")),
        "state_mass_mean": float(masses.mean().cpu()),
        "readout": "gold_prefix_def20",
        "n_tables": int(tables.shape[0]),
    }


def _mean_patch(rows, method, depth):
    vals = [r["counterfactual_accuracy"] for r in rows if r["method"] == method]
    if not vals:
        return float("nan")
    return sum(vals) / max(len(vals), depth)


def write_report(output, persist, metrics, probes, credit, patches, trained_patches):
    """Short markdown: how to read success/failure. Not a theorem."""
    output = Path(output)
    depth = persist.get("depth")
    steps = persist.get("steps")
    answers = persist.get("final_answer") or {}
    train_answers = persist.get("final_train_answer") or {}
    probe_eval = persist.get("final_probe_eval") or {}
    chance = chance_accuracy()
    undertrained = bool(persist.get("smoke")) or (isinstance(steps, int) and steps < 200)

    def fmt(value):
        if value is None or value == "":
            return "n/a"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Causal identification (deep outcome architecture)",
        "",
        "This is the **mechanism** test: do trained D-block outcome Transformers *use* a",
        "layerwise state $s_t$ in the remaining computation $\\Phi_{g_{t+1:D}}(s_t)$?",
        "It is **not** mixed-format $\\widehat P_g$ (that readout is a **limit of transfer**:",
        "D=4 local ~99.9%, composition TV 0.833, gradient cosine 0.028).",
        "Probes and causal patches use **nnsight** traces (`NNsight` on the `nn.Module`);",
        "$C_t$ stays in autograd.",
        "",
        "## What would count as success (confirmation budget, not this smoke)",
        "",
        "- **Credit placement:** held-out answer accuracy: outcome ≈ chance "
        f"({chance:.3f}) while outcome+local ≈ process ≈ 1. Same architecture/init/format",
        "  for outcome vs outcome+local; process is the known-good token-trace comparison.",
        "- **Decodable state:** linear probes $h_t \\to s_t$ on held-out circuits are high",
        "  for outcome+local (and, if the net actually stores $s_t$, possibly outcome too).",
        "- **Causal use:** `probe_subspace` patch at block $t$ to donor $s_t'$ makes the",
        "  remaining net output $\\Phi_{g_{t+1:D}}(s_t')$. Oracle-slot patch on the",
        "  **hand-built** executor stays ~1 vs random-subspace ~chance (positive control).",
        "- **Controls:** random subspace and wrong-layer stay near chance.",
        "- **Credit vs depth:** report $C_t = \\langle g_t^{\\mathrm{term}}, g_t^{\\mathrm{local}}\\rangle / \\|g_t^{\\mathrm{local}}\\|^2$",
        "  vs $D$. **Do not** claim an $\\varepsilon^{D-1}$ mixing-ball rate unless mixing is checked.",
        "",
        "## What failure means",
        "",
        "- **All conditions ~0% held-out answer:** undertrained. Not a negative mechanism result.",
        "  Read `train_loss` / `train_subset_answer_accuracy` for plumbing.",
        "- **Probes high, `probe_subspace` ~chance:** $s_t$ is linearly readable but not",
        "  causally used by later blocks (the hole this experiment is for).",
        "- **Oracle-slot high on a *trained* net, probes low:** the construction coordinates",
        "  moved, but that is not evidence the net computed through $\\Phi$.",
        "- Mixed $\\widehat P_g$ agreement is **not** confirmation of this test.",
        "",
        "## This run",
        "",
        f"- depth $D$ = {depth}, seed = {persist.get('seed')}, steps = {steps}, "
        f"$\\lambda$ = {persist.get('lambda_local')}",
        f"- device = `{persist.get('device')}`, smoke = {persist.get('smoke')}",
        f"- matched D-block process = {persist.get('matched_architecture')}",
        f"- oracle_slot mean (hand-built) = {fmt(persist.get('oracle_patch_mean'))}",
        f"- random_subspace mean (hand-built) = {fmt(persist.get('random_patch_mean'))}",
        f"- probe_subspace mean (hand-built) = {fmt(persist.get('probe_subspace_oracle_mean'))}",
        "",
        "| condition | val answer | train-subset answer |",
        "|---|---:|---:|",
    ]
    for name in CONDITIONS:
        lines.append(
            f"| {name} | {fmt(answers.get(name))} | {fmt(train_answers.get(name))} |"
        )
    lines.extend(["", "Held-out probe eval accuracy (final checkpoint):", ""])
    if probe_eval:
        for name, layers in probe_eval.items():
            lines.append(f"- `{name}`: " + ", ".join(
                f"t={i + 1} → {fmt(acc)}" for i, acc in enumerate(layers)
            ))
    else:
        lines.append("- (no outcome-model probes; process is not patched)")
    lines.extend([
        "",
        "Trained-net patch means (final checkpoint, average over layers):",
        "",
        f"- outcome oracle_slot = {fmt((persist.get('final_trained_patch_oracle_slot_mean') or {}).get('outcome'))}",
        f"- outcome probe_subspace = {fmt((persist.get('final_trained_patch_probe_subspace_mean') or {}).get('outcome'))}",
        f"- outcome_local oracle_slot = {fmt((persist.get('final_trained_patch_oracle_slot_mean') or {}).get('outcome_local'))}",
        f"- outcome_local probe_subspace = {fmt((persist.get('final_trained_patch_probe_subspace_mean') or {}).get('outcome_local'))}",
        f"- process oracle_slot = {fmt((persist.get('final_trained_patch_oracle_slot_mean') or {}).get('process'))}",
        f"- process probe_subspace = {fmt((persist.get('final_trained_patch_probe_subspace_mean') or {}).get('process'))}",
        "",
    ])
    if undertrained:
        lines.extend([
            "**This dump is plumbing / smoke.** Held-out accuracies near 0 are expected at",
            "this budget. Do not paste them into the paper. Confirmation YAML:",
            "`configs/experiments/causal_identification.yaml` (one `--depth` at a time;",
            "requires `--confirm`; do not launch a D-grid from `RUN.md`).",
            "",
        ])
    lines.extend([
        "CSV: `metrics.csv`, `probes.csv`, `credit.csv`, `patch.csv`, `trained_patch.csv`,",
        "`probe_vs_patch.csv`, `table.csv`. JSON: `persist.json`.",
        "",
        "No GSM8K / LLM transfer is claimed.",
        "",
    ])
    text = "\n".join(lines)
    (output / "report.md").write_text(text)
    return text


def missing_persist_keys(persist):
    return [key for key in REQUIRED_PERSIST_KEYS if key not in persist]


def plot_results(output, metrics, probes, credit, patches, depth, trained_patches=None):
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

    if trained_patches:
        final_step = max(r["step"] for r in trained_patches)
        methods = []
        for row in trained_patches:
            if row["method"] not in methods:
                methods.append(row["method"])
        models_present = [m for m in ("outcome", "outcome_local") if any(r["condition"] == m for r in trained_patches)]
        oracle_means = {}
        for row in patches or []:
            oracle_means.setdefault(row["method"], []).append(row["counterfactual_accuracy"])
        oracle_means = {k: sum(v) / len(v) for k, v in oracle_means.items()}

        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        width = 0.8 / (len(models_present) + 1)
        xs = list(range(len(methods)))
        for i, name in enumerate(["oracle_outcome", *models_present]):
            vals = []
            for method in methods:
                if name == "oracle_outcome":
                    vals.append(oracle_means.get(method, float("nan")))
                else:
                    rows = [
                        r for r in trained_patches
                        if r["condition"] == name and r["step"] == final_step and r["method"] == method
                    ]
                    vals.append(sum(r["counterfactual_accuracy"] for r in rows) / len(rows) if rows else float("nan"))
            offs = [x + (i - (len(models_present)) / 2) * width for x in xs]
            color = colors.get(name, "#4c72b0") if name != "oracle_outcome" else "#333333"
            ax.bar(offs, vals, width=width, label=name, color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.axhline(chance_accuracy(), color="#888", ls="--", lw=1)
        ax.set_ylabel("counterfactual answer accuracy")
        ax.set_title(f"Patch accuracy at final checkpoint (step {final_step}), oracle vs trained")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "patch_accuracy_trained_vs_oracle.pdf")
        fig.savefig(output / "patch_accuracy_trained_vs_oracle.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name in models_present:
            rows = [r for r in trained_patches if r["condition"] == name and r["method"] == "oracle_slot"]
            by_step = {}
            for row in rows:
                by_step.setdefault(row["step"], []).append(row["counterfactual_accuracy"])
            xs_step = sorted(by_step)
            ys = [sum(by_step[s]) / len(by_step[s]) for s in xs_step]
            ax.plot(xs_step, ys, marker="o", color=colors[name], label=f"{name} (oracle_slot)")
            rows_r = [r for r in trained_patches if r["condition"] == name and r["method"] == "random_subspace"]
            by_step_r = {}
            for row in rows_r:
                by_step_r.setdefault(row["step"], []).append(row["counterfactual_accuracy"])
            xs_r = sorted(by_step_r)
            ys_r = [sum(by_step_r[s]) / len(by_step_r[s]) for s in xs_r]
            ax.plot(xs_r, ys_r, marker="x", ls="--", color=colors[name], alpha=0.6, label=f"{name} (random_subspace)")
            rows_p = [r for r in trained_patches if r["condition"] == name and r["method"] == "probe_subspace"]
            by_step_p = {}
            for row in rows_p:
                by_step_p.setdefault(row["step"], []).append(row["counterfactual_accuracy"])
            if by_step_p:
                xs_p = sorted(by_step_p)
                ys_p = [sum(by_step_p[s]) / len(by_step_p[s]) for s in xs_p]
                ax.plot(xs_p, ys_p, marker="s", color=colors[name], alpha=0.85, label=f"{name} (probe_subspace)")
        if oracle_means:
            ax.axhline(oracle_means.get("oracle_slot", float("nan")), color="#333", ls=":", lw=1, label="oracle oracle_slot")
            if "probe_subspace" in oracle_means:
                ax.axhline(oracle_means["probe_subspace"], color="#333", ls="-.", lw=1, label="oracle probe_subspace")
        ax.axhline(chance_accuracy(), color="#888", ls="--", lw=1, label="chance")
        ax.set_xlabel("step")
        ax.set_ylabel("counterfactual answer accuracy (mean over layers)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Trained-model patch accuracy over training")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "trained_patch_vs_step.pdf")
        fig.savefig(output / "trained_patch_vs_step.png", dpi=140)
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
    matched = bool(getattr(args, "matched_architecture", False))
    if matched:
        process_model = copy.deepcopy(base)
    else:
        process_model = h.build_random_trainable_process_architecture(
            tokenizer, depth, seed=args.seed, device=device,
        )
    models = {
        "outcome": copy.deepcopy(base),
        "outcome_local": copy.deepcopy(base),
        "process": process_model,
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

    metrics, probe_rows, credit_rows, trained_patch_rows, induced_rows = [], [], [], [], []
    run_trained_patches = not getattr(args, "skip_trained_patches", False)
    last_loss = {name: float("nan") for name in CONDITIONS}
    last_parts = {name: {} for name in CONDITIONS}
    train_eval = train[: min(len(train), int(getattr(args, "train_eval_size", 32)))]
    log_n = min(args.batch_size, len(train))

    def logged_losses(name, model):
        batch = data[name].select(torch.arange(log_n, device=device))
        gold_batch = None if gold[name] is None else gold[name][:log_n]
        if name == "outcome_local":
            loss, terminal, local = outcome_plus_local_loss(model, batch, gold_batch, args.lambda_local)
            return {
                "train_loss": float(loss.detach()),
                "terminal_loss": float(terminal.detach()),
                "local_loss": float(local.detach()),
            }
        loss = h.language_model_loss(model, batch)
        return {"train_loss": float(loss.detach())}

    def patch_now(step):
        if not run_trained_patches:
            return False
        if getattr(args, "smoke", False):
            return step == args.steps
        return True

    def checkpoint(step):
        for name, model in models.items():
            model.eval()
            acc = answer_accuracy(model, val, tokenizer, device, mode[name])
            train_acc = answer_accuracy(model, train_eval, tokenizer, device, mode[name])
            with torch.no_grad():
                parts = logged_losses(name, model)
            last_loss[name] = parts["train_loss"]
            last_parts[name] = parts
            row = {
                "condition": name, "step": step, "depth": depth, "seed": args.seed,
                "answer_accuracy": acc,
                "train_subset_answer_accuracy": train_acc,
                "lambda_local": args.lambda_local,
                "format": mode[name],
                **parts,
            }
            metrics.append(row)
            if getattr(args, "induced_credit", False):
                try:
                    induced = measure_induced_credit(
                        model, tokenizer, val, device, depth,
                        backgrounds=int(getattr(args, "induced_backgrounds", 2)),
                    )
                    induced_rows.append({
                        "condition": name, "step": step, "depth": depth, "seed": args.seed,
                        "theorem": "cor:near-mixing / Def 20 gold-prefix",
                        **induced,
                    })
                except Exception as exc:
                    print(json.dumps({
                        "induced_credit_failed": True, "condition": name, "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                    }), flush=True)
            if name == "process" and not matched:
                # One-block process has no per-step state_slots. Matched D-block process
                # shares the outcome residual layout and is patched below.
                continue
            probes, tr_acc, ev_acc = evaluate_probes(
                model, probe_train, probe_eval, tokenizer, device, tokenizer.n_states,
                args.probe_steps, args.probe_lr,
            )
            for layer, (tr, ev) in enumerate(zip(tr_acc, ev_acc)):
                probe_rows.append({
                    "condition": name, "step": step, "depth": depth, "seed": args.seed,
                    "layer": layer, "probe_train_accuracy": tr, "probe_eval_accuracy": ev,
                    "answer_accuracy": acc,
                })
            if patch_now(step):
                for prow in run_patches(
                    model, patch_circuits, tokenizer, device, model_name=name, probes=probes,
                    donor_ids=getattr(args, "patch_donors", None),
                ):
                    trained_patch_rows.append({
                        "condition": name, "step": step, "depth": depth, "seed": args.seed,
                        **prow,
                    })
            index = torch.tensor(schedule[max(step - 1, 0)], device=device)
            batch = outcome_data.select(index)
            gold_batch = outcome_gold[index]
            model.train()
            for crow in credit_signals(model, batch, gold_batch, probes):
                credit_rows.append({
                    "condition": name, "step": step, "depth": depth, "seed": args.seed,
                    **crow, "note": "C_t vs D only; not an eps^{D-1} claim",
                })
            model.eval()
        summary = {
            "step": step,
            "answer": {r["condition"]: round(r["answer_accuracy"], 4) for r in metrics if r["step"] == step},
            "train_subset_answer": {
                r["condition"]: round(r["train_subset_answer_accuracy"], 4)
                for r in metrics if r["step"] == step
            },
            "train_loss": {r["condition"]: round(r["train_loss"], 4) for r in metrics if r["step"] == step},
        }
        step_patches = [r for r in trained_patch_rows if r["step"] == step]
        if step_patches:
            summary["trained_probe_subspace_mean"] = {
                name: round(
                    sum(r["counterfactual_accuracy"] for r in step_patches
                        if r["condition"] == name and r["method"] == "probe_subspace") / depth, 4,
                )
                for name in (("outcome", "outcome_local", "process") if matched else ("outcome", "outcome_local"))
                if any(r["condition"] == name for r in step_patches)
            }
        print(json.dumps(summary), flush=True)

    oracle = h.HandcodedOutcomeTransformer(tokenizer, depth).to(device).eval()
    oracle_probes, _, _ = evaluate_probes(
        oracle, probe_train, probe_eval, tokenizer, device, tokenizer.n_states,
        args.probe_steps, args.probe_lr,
    )
    patch_rows = run_patches(
        oracle, patch_circuits, tokenizer, device, probes=oracle_probes, model_name="oracle_outcome",
        donor_ids=getattr(args, "patch_donors", None),
    )
    write_csv(output / "patch.csv", patch_rows)
    oracle_mean = _mean_patch(patch_rows, "oracle_slot", depth)
    random_mean = _mean_patch(patch_rows, "random_subspace", depth)
    probe_patch_mean = _mean_patch(patch_rows, "probe_subspace", depth)
    print(json.dumps({
        "oracle_patch_mean": round(oracle_mean, 4),
        "random_patch_mean": round(random_mean, 4),
        "probe_subspace_oracle_mean": round(probe_patch_mean, 4),
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
            last_loss[name] = float(loss.detach())
        if step in checkpoints:
            checkpoint(step)

    write_csv(output / "metrics.csv", metrics)
    if probe_rows:
        write_csv(output / "probes.csv", probe_rows)
    if credit_rows:
        write_csv(output / "credit.csv", credit_rows)
    if induced_rows:
        write_csv(output / "induced_credit.csv", induced_rows)
    if trained_patch_rows:
        write_csv(output / "trained_patch.csv", trained_patch_rows)
    table = []
    patch_methods_present = []
    for row in trained_patch_rows:
        if row["method"] not in patch_methods_present:
            patch_methods_present.append(row["method"])
    for name in CONDITIONS:
        final = next(r for r in metrics if r["condition"] == name and r["step"] == args.steps)
        row = {
            "condition": name,
            "answer_accuracy": final["answer_accuracy"],
            "train_subset_answer_accuracy": final.get("train_subset_answer_accuracy", ""),
            "train_loss": final.get("train_loss", ""),
            "depth": depth,
            "seed": args.seed,
        }
        for layer in range(depth):
            match = [
                r for r in probe_rows
                if r["condition"] == name and r["step"] == args.steps and r["layer"] == layer
            ]
            row[f"s{layer + 1}_probe_acc"] = match[0]["probe_eval_accuracy"] if match else ""
            for method in ("oracle_slot", "probe_subspace", "random_subspace", "wrong_layer"):
                patch_match = [
                    r for r in trained_patch_rows
                    if r["condition"] == name and r["step"] == args.steps
                    and r["layer"] == layer and r["method"] == method
                ]
                row[f"s{layer + 1}_{method}_patch_acc"] = (
                    patch_match[0]["counterfactual_accuracy"] if patch_match else ""
                )
        table.append(row)
    write_csv(output / "table.csv", table)

    # Decodable-but-not-used pattern: high probe_eval_accuracy, probe_subspace ~ chance.
    probe_vs_patch = []
    methods_for_cmp = patch_methods_present or list(PATCH_METHODS)
    patch_names = ("outcome", "outcome_local", "process") if matched else ("outcome", "outcome_local")
    for name in patch_names:
        for layer in range(depth):
            probe_match = [
                r for r in probe_rows
                if r["condition"] == name and r["step"] == args.steps and r["layer"] == layer
            ]
            patches_here = {
                method: next(
                    (r["counterfactual_accuracy"] for r in trained_patch_rows
                     if r["condition"] == name and r["step"] == args.steps
                     and r["layer"] == layer and r["method"] == method),
                    "",
                )
                for method in methods_for_cmp
            }
            probe_vs_patch.append({
                "condition": name, "depth": depth, "seed": args.seed, "layer": layer,
                "probe_eval_accuracy": probe_match[0]["probe_eval_accuracy"] if probe_match else "",
                **{f"patch_{method}": value for method, value in patches_here.items()},
            })
    if probe_vs_patch and trained_patch_rows:
        write_csv(output / "probe_vs_patch.csv", probe_vs_patch)

    def _final_patch_mean(method):
        names = ("outcome", "outcome_local", "process") if matched else ("outcome", "outcome_local")
        return {
            name: sum(
                r["counterfactual_accuracy"] for r in trained_patch_rows
                if r["condition"] == name and r["step"] == args.steps and r["method"] == method
            ) / depth
            for name in names
            if any(r["condition"] == name and r["step"] == args.steps for r in trained_patch_rows)
        }

    final_probe_eval = {}
    for name in (("outcome", "outcome_local", "process") if matched else ("outcome", "outcome_local")):
        accs = []
        for layer in range(depth):
            match = [
                r for r in probe_rows
                if r["condition"] == name and r["step"] == args.steps and r["layer"] == layer
            ]
            if match:
                accs.append(match[0]["probe_eval_accuracy"])
        if accs:
            final_probe_eval[name] = accs

    plot_results(output, metrics, probe_rows, credit_rows, patch_rows, depth, trained_patches=trained_patch_rows)
    persist = {
        "experiment": "causal_identification",
        "mechanism_test": True,
        "mixed_format_is_negative_transfer": True,
        "no_eps_D_minus_1_claim": True,
        "smoke": bool(getattr(args, "smoke", False)),
        "device": str(device),
        "matched_architecture": matched,
        "induced_credit": bool(getattr(args, "induced_credit", False)),
        "process_patching_not_applicable": (
            None if matched else
            "process is a one-block, reused architecture with no per-execution-step "
            "residual state slot; activation patching as implemented here (patch a "
            "fixed layer's fixed state_slots range) does not apply to it and was not run"
        ),
        "depth": depth,
        "seed": args.seed,
        "steps": args.steps,
        "lambda_local": args.lambda_local,
        "lr": args.lr,
        "oracle_patch_mean": oracle_mean,
        "random_patch_mean": random_mean,
        "probe_subspace_oracle_mean": probe_patch_mean,
        "final_answer": {r["condition"]: r["answer_accuracy"] for r in metrics if r["step"] == args.steps},
        "final_train_answer": {
            r["condition"]: r.get("train_subset_answer_accuracy")
            for r in metrics if r["step"] == args.steps
        },
        "final_train_loss": {r["condition"]: r.get("train_loss") for r in metrics if r["step"] == args.steps},
        "final_probe_eval": final_probe_eval,
        "final_trained_patch_oracle_slot_mean": _final_patch_mean("oracle_slot"),
        "final_trained_patch_random_subspace_mean": _final_patch_mean("random_subspace"),
        "final_trained_patch_probe_subspace_mean": _final_patch_mean("probe_subspace"),
        "csv": str(output / "metrics.csv"),
        "report": str(output / "report.md"),
        "interventions": "nnsight",
        "nnsight_version": getattr(nnsight, "__version__", None),
    }
    missing = missing_persist_keys(persist)
    if missing:
        raise RuntimeError(f"persist.json missing required keys: {missing}")
    write_report(output, persist, metrics, probe_rows, credit_rows, patch_rows, trained_patch_rows)
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
    parser.add_argument(
        "--confirm", action="store_true",
        help="Allow a confirmation YAML (one depth; hours). Do not pass unless you intend to run it.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / str(cfg.get("output", "results/paper_revision_v2/causal_identification")),
    )
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
    parser.add_argument("--train-eval-size", type=int, default=int(cfg.get("train_eval_size", 32)))
    parser.add_argument("--train-seed", type=int, default=int(cfg.get("train_seed", 123)))
    parser.add_argument("--val-seed", type=int, default=int(cfg.get("val_seed", 8000)))
    parser.add_argument("--probe-seed", type=int, default=int(cfg.get("probe_seed", 9000)))
    parser.add_argument("--batch-seed", type=int, default=int(cfg.get("batch_seed", 2026)))
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(cfg.get("checkpoints") or [0, 20, 40]))
    parser.add_argument("--skip-trained-patches", action="store_true",
                         default=bool(cfg.get("skip_trained_patches", False)),
                         help="Skip activation-patching the trained outcome/outcome_local models at each "
                              "checkpoint (patch.csv for the hand-coded oracle is always produced).")
    parser.add_argument(
        "--matched-architecture", action="store_true",
        default=bool(cfg.get("matched_architecture", False)),
        help="Train process on the same D-block outcome architecture (not the one-block process net).",
    )
    parser.add_argument(
        "--induced-credit", action="store_true",
        default=bool(cfg.get("induced_credit", False)),
        help="At each checkpoint, gold-prefix Def 20 readout of P̂_g and Rule credit vs process floor.",
    )
    parser.add_argument(
        "--induced-backgrounds", type=int,
        default=int(cfg.get("induced_backgrounds", 2)),
    )
    parser.add_argument(
        "--patch-donors", nargs="+", type=int, default=cfg.get("patch_donors"),
        help="Donor states for nnsight patches (default: all 16). Smoke uses a 4-state subset.",
    )
    parser.add_argument("--device", default=cfg.get("device"))
    add_compile_bf16_flags(parser, cfg, from_yaml=True)
    args = parser.parse_args(argv)
    args.smoke = bool(args.smoke)
    # Only top-level `do_not_launch` blocks (confirmation YAML). Nested
    # confirmation.do_not_launch on the smoke YAML is documentation.
    blocked = bool(cfg.get("do_not_launch"))
    if blocked and not args.smoke and not args.confirm:
        raise SystemExit(
            "This YAML is the confirmation protocol. Pass --confirm for one --depth "
            "(hours; prefer cuda:2/3), or --smoke for plumbing. A D-grid is not launched."
        )
    if args.smoke:
        # Enough updates to move train_loss / check keys. Held-out answer may stay 0.
        args.depth = 2
        args.steps = 80
        args.train_size = 16
        args.val_size = 8
        args.probe_size = 8
        args.patch_circuits = 4
        args.batch_size = 8
        args.lr = 0.01
        args.checkpoints = [0, 40, 80]
        args.probe_steps = 40
        args.train_eval_size = 16
        args.patch_donors = [0, 5, 10, 15]
        args.output = ROOT / "results/paper_revision_v2/causal_identification/smoke"
    if args.device is None:
        args.device = "cpu" if args.smoke else default_device()
    args.compile = False
    return args


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
