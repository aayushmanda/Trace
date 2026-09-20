"""Empirical escape times under process vs outcome supervision.

Executor-level test of the NeurIPS draft claim: from small init scale ε,
outcome escape τ_a = inf{t : max_i γ_i(t) = a} is Θ(log 1/ε) at D=2 and
Θ(ε^{-(D-2)}) at D>2, while process escape stays Θ(1). Architecture is held
fixed (process-length residual/positions); only supervision placement varies.
This does not recover QK/OV circuits and does not prove the γ-flow theorem.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from handcoded.data import encode_dataset, language_model_loss, make_batch_schedule
from handcoded.gates import make_circuits
from handcoded.models import build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from handcoded.train import train_step
from src.experiments.handcoded_trace_gradients import (
    ProcessSlices,
    assert_d4_continuation_slices,
    continuation_slices,
)
from src.plot_style import apply_style
from src.training.io import write_csv
from src.training.optim import make_adamw
from src.training.seed import configure_device, default_device, prepare_train_model, set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "handcoded_escape_times"
SMOKE_OUT = ROOT / "results" / "handcoded_escape_times_smoke"

PROCESS_A_DEFAULT = 0.95
ANSWER_A_DEFAULT = 0.95
ESCAPE_A_DEFAULT = 0.5
DEFAULT_INIT_STDS = (0.005, 0.01, 0.02, 0.04)
DEFAULT_DEPTHS = (2, 3, 4)
SMOKE_INIT_STDS = (0.01, 0.04)
SMOKE_DEPTHS = (2, 3)
MODE_COLORS = {"process": "#087eaa", "outcome": "#c56b08"}
EPS_STYLES = ("-", "--", ":", "-.")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None, help="Gradient steps (default 2000; 40 with --smoke).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Overrides --seed if given.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--eval-size", type=int, default=None, help="Fixed gold eval circuits (process-encoded diagnostic).")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--depths", type=int, nargs="+", default=None, help="Overrides --depth if given.")
    p.add_argument("--init-std", type=float, default=None)
    p.add_argument("--init-stds", type=float, nargs="+", default=None, help="Overrides --init-std if given.")
    p.add_argument("--mode", choices=("process", "outcome"), default=None)
    p.add_argument("--modes", nargs="+", choices=("process", "outcome"), default=None, help="Overrides --mode if given.")
    p.add_argument("--escape-a", type=float, default=ESCAPE_A_DEFAULT, help="Local competence threshold a for τ_a.")
    p.add_argument("--answer-a", type=float, default=ANSWER_A_DEFAULT, help="Task-solve threshold on answer_acc.")
    p.add_argument("--process-a", type=float, default=PROCESS_A_DEFAULT, help="Exact process-trace accuracy threshold.")
    p.add_argument(
        "--lstate-thr", type=float, default=None,
        help="Optional L_state hitting threshold (default -log(escape-a)).",
    )
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="Tiny run: depths 2 3, two ε, 1 seed, 40 steps.")
    p.add_argument(
        "--circuit-match", action="store_true",
        help="Optional Hungarian QK/OV cosine to the construction (off: wrong scientific object).",
    )
    p.add_argument("--data-seed", type=int, default=123)
    p.add_argument("--eval-seed", type=int, default=9000)
    p.add_argument("--batch-seed", type=int, default=2026)
    return p.parse_args(argv)


def _unique(values) -> list:
    out, seen = [], set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve(args: argparse.Namespace) -> argparse.Namespace:
    smoke = bool(args.smoke)
    args.steps = 40 if args.steps is None and smoke else (2000 if args.steps is None else args.steps)
    args.batch_size = 32 if args.batch_size is None and smoke else (64 if args.batch_size is None else args.batch_size)
    args.train_size = 128 if args.train_size is None and smoke else (4096 if args.train_size is None else args.train_size)
    args.log_every = 5 if args.log_every is None and smoke else (50 if args.log_every is None else args.log_every)
    args.eval_size = 16 if args.eval_size is None and smoke else (64 if args.eval_size is None else args.eval_size)
    if args.modes is not None:
        args.modes = _unique(args.modes)
    elif args.mode is not None:
        args.modes = [args.mode]
    else:
        args.modes = ["process", "outcome"]
    if args.depths is not None:
        args.depths = _unique(int(d) for d in args.depths)
    elif args.depth is not None:
        args.depths = [int(args.depth)]
    else:
        args.depths = list(SMOKE_DEPTHS if smoke else DEFAULT_DEPTHS)
    if args.init_stds is not None:
        args.init_stds = _unique(float(x) for x in args.init_stds)
    elif args.init_std is not None:
        args.init_stds = [float(args.init_std)]
    else:
        args.init_stds = list(SMOKE_INIT_STDS if smoke else DEFAULT_INIT_STDS)
    args.seeds = list(args.seeds) if args.seeds is not None else [args.seed]
    if args.lstate_thr is None:
        args.lstate_thr = -math.log(args.escape_a)
    if args.output is None:
        args.output = SMOKE_OUT if smoke else DEFAULT_OUT
    if not (0.0 < args.escape_a < 1.0):
        raise ValueError(f"--escape-a must lie in (0, 1), got {args.escape_a}")
    if not (0.0 < args.answer_a <= 1.0):
        raise ValueError(f"--answer-a must lie in (0, 1], got {args.answer_a}")
    if not (0.0 < args.process_a <= 1.0):
        raise ValueError(f"--process-a must lie in (0, 1], got {args.process_a}")
    for depth in args.depths:
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
    for std in args.init_stds:
        if std <= 0:
            raise ValueError(f"init-std must be positive, got {std}")
    return args


def _checkpoints(steps: int, log_every: int) -> list[int]:
    marks = {0, steps}
    if steps >= 1:
        marks.add(1)
    marks.update(range(log_every, steps + 1, log_every))
    return sorted(m for m in marks if 0 <= m <= steps)


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        return torch.device(default_device())
    device = torch.device(name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        print(f"{device} requested but CUDA is unavailable; falling back to cpu", flush=True)
        return torch.device("cpu")
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        print(f"{device} missing (count={torch.cuda.device_count()}); falling back to cpu", flush=True)
        return torch.device("cpu")
    return torch.device("cuda", index)


def _nanmean(values: torch.Tensor) -> float:
    if values.numel() == 0 or not torch.isfinite(values).any():
        return float("nan")
    return float(values[torch.isfinite(values)].mean())


def _ce_and_p_correct(logits: torch.Tensor, targets: torch.Tensor, indices: tuple[int, ...] | list[int]) -> tuple[float, float]:
    """Mean CE and mean gold-token softmax probability at teacher-forced indices."""
    if not indices:
        return float("nan"), float("nan")
    pos = list(indices)
    pos_logits = logits[:, pos, :].float()
    pos_targets = targets[:, pos]
    valid = pos_targets != -100
    if not bool(valid.any()):
        return float("nan"), float("nan")
    log_probs = F.log_softmax(pos_logits, dim=-1)
    safe_targets = pos_targets.clamp(min=0)
    gold_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    gold_logp = gold_logp.masked_fill(~valid, float("nan"))
    gold_p = gold_logp.exp()
    return _nanmean(-gold_logp), _nanmean(gold_p)


@torch.no_grad()
def teacher_forced_from_logits(logits: torch.Tensor, batch, tokenizer) -> dict[str, float]:
    """Same token / process / answer accuracies as `handcoded_circuit_match`."""
    pred = logits.argmax(dim=-1)
    targets = batch.targets
    valid = targets != -100
    token_acc = float((pred == targets)[valid].float().mean()) if valid.any() else float("nan")
    process_acc = float(((pred == targets) | ~valid).all(dim=1).float().mean())
    colon = batch.inputs == tokenizer.colon
    answer_acc = float((pred[colon] == targets[colon]).float().mean()) if colon.any() else float("nan")
    loss = float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100))
    return {
        "loss": loss,
        "token_acc": token_acc,
        "process_acc": process_acc,
        "answer_acc": answer_acc,
    }


@torch.no_grad()
def teacher_forced_metrics(model, batch, tokenizer) -> dict[str, float]:
    model.eval()
    return teacher_forced_from_logits(model(batch.inputs).float(), batch, tokenizer)


@torch.no_grad()
def colon_answer_stats(logits: torch.Tensor, batch, tokenizer) -> tuple[float, float, float]:
    """CE / P(gold s_D) / argmax accuracy on the token after COLON."""
    colon = batch.inputs == tokenizer.colon
    if not bool(colon.any()):
        return float("nan"), float("nan"), float("nan")
    pos_logits = logits[colon].float()
    pos_targets = batch.targets[colon]
    valid = pos_targets != -100
    if not bool(valid.any()):
        return float("nan"), float("nan"), float("nan")
    log_probs = F.log_softmax(pos_logits, dim=-1)
    gold_logp = log_probs.gather(-1, pos_targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    gold_logp = gold_logp.masked_fill(~valid, float("nan"))
    pred = pos_logits.argmax(dim=-1)
    acc = float((pred[valid] == pos_targets[valid]).float().mean())
    return _nanmean(-gold_logp), _nanmean(gold_logp.exp()), acc


@torch.no_grad()
def evaluate_escape(model, eval_process, eval_outcome, train_eval, tokenizer, slices: ProcessSlices, mode: str) -> dict[str, float]:
    """Diagnostic local competence on gold process encodings; answer stats mode-matched."""
    model.eval()
    process_logits = model(eval_process.inputs).float()
    process_tf = teacher_forced_from_logits(process_logits, eval_process, tokenizer)
    L_state, p_correct = _ce_and_p_correct(process_logits, eval_process.targets, slices.state)
    L_plus, _ = _ce_and_p_correct(process_logits, eval_process.targets, slices.continuation)
    L_answer_proc, p_answer_proc, answer_acc_proc = colon_answer_stats(process_logits, eval_process, tokenizer)

    outcome_logits = model(eval_outcome.inputs).float()
    outcome_tf = teacher_forced_from_logits(outcome_logits, eval_outcome, tokenizer)
    L_answer_out, p_answer_out, answer_acc_out = colon_answer_stats(outcome_logits, eval_outcome, tokenizer)

    train_loss = float(language_model_loss(model, train_eval))

    if mode == "process":
        L_answer, p_answer, answer_acc = L_answer_proc, p_answer_proc, process_tf["answer_acc"]
    else:
        L_answer, p_answer, answer_acc = L_answer_out, p_answer_out, outcome_tf["answer_acc"]
        # Training tokens are COLON, s_D, EOS: there is no gold-trace CE in the objective.
        # L_state below is still the process-encoded diagnostic (not the train loss).

    return {
        "train_loss": train_loss,
        "L_state": L_state if mode == "process" else float("nan"),
        "L_state_diag": L_state,
        "p_correct": p_correct,
        "L_answer": L_answer,
        "p_answer": p_answer,
        "L_plus": L_plus,
        "process_acc": process_tf["process_acc"],
        "answer_acc": answer_acc,
        "token_acc": process_tf["token_acc"] if mode == "process" else outcome_tf["token_acc"],
        "L_answer_process": L_answer_proc,
        "p_answer_process": p_answer_proc,
        "answer_acc_process": process_tf["answer_acc"],
        "L_answer_outcome": L_answer_out,
        "p_answer_outcome": p_answer_out,
        "answer_acc_outcome": outcome_tf["answer_acc"],
        "eval_process_ce": process_tf["loss"],
        "eval_outcome_ce": outcome_tf["loss"],
    }


def _finite_ge(value, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= threshold


def _finite_le(value, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) <= threshold


def hitting_times(rows: list[dict], args: argparse.Namespace) -> dict:
    sentinel = int(args.steps) + 1
    t_escape_p = t_escape_Lstate = t_answer = t_process = t_answer_acc = t_p_answer = sentinel
    escaped_p = escaped_Lstate = escaped_answer = escaped_process = False
    for row in rows:
        step = int(row["step"])
        if not escaped_p and _finite_ge(row.get("p_correct"), args.escape_a):
            t_escape_p, escaped_p = step, True
        lstate = row.get("L_state_diag", row.get("L_state"))
        if not escaped_Lstate and _finite_le(lstate, args.lstate_thr):
            t_escape_Lstate, escaped_Lstate = step, True
        hit_acc = _finite_ge(row.get("answer_acc"), args.answer_a)
        hit_p = _finite_ge(row.get("p_answer"), args.escape_a)
        if not escaped_answer and (hit_acc or hit_p):
            t_answer, escaped_answer = step, True
        if t_answer_acc == sentinel and hit_acc:
            t_answer_acc = step
        if t_p_answer == sentinel and hit_p:
            t_p_answer = step
        if not escaped_process and _finite_ge(row.get("process_acc"), args.process_a):
            t_process, escaped_process = step, True
    final = rows[-1]
    return {
        "seed": int(final["seed"]),
        "depth": int(final["depth"]),
        "init_std": float(final["init_std"]),
        "mode": final["mode"],
        "t_escape_p": t_escape_p,
        "t_escape_Lstate": t_escape_Lstate,
        "t_answer": t_answer,
        "t_answer_acc": t_answer_acc,
        "t_p_answer": t_p_answer,
        "t_process": t_process,
        "escaped_p": escaped_p,
        "escaped_Lstate": escaped_Lstate,
        "escaped_answer": escaped_answer,
        "escaped_process": escaped_process,
        "final_p_correct": float(final["p_correct"]),
        "final_L_state": float(final["L_state"]) if math.isfinite(float(final["L_state"])) else float("nan"),
        "final_L_state_diag": float(final["L_state_diag"]),
        "final_answer_acc": float(final["answer_acc"]),
        "final_p_answer": float(final["p_answer"]),
        "final_process_acc": float(final["process_acc"]),
        "final_L_answer": float(final["L_answer"]),
        "final_train_loss": float(final["train_loss"]),
        "steps": int(args.steps),
        "escape_a": float(args.escape_a),
        "answer_a": float(args.answer_a),
        "process_a": float(args.process_a),
        "lstate_thr": float(args.lstate_thr),
        "sentinel": sentinel,
    }


def _log_line(row: dict) -> str:
    lstate = row["L_state_diag"]
    lstate_s = f"{lstate:.4f}" if math.isfinite(float(lstate)) else "nan"
    return (
        f"mode={row['mode']:<8} D={row['depth']} ε={row['init_std']:.4g} "
        f"step={row['step']:<5d} train={row['train_loss']:.4f} "
        f"pγ={row['p_correct']:.3f} Lstate={lstate_s} "
        f"pA={row['p_answer']:.3f} ans={row['answer_acc']:.3f} proc={row['process_acc']:.3f}"
    )


def run_one(
    *, seed, depth, init_std, mode, args, tokenizer, train_data, eval_process, eval_outcome,
    train_eval, device, checkpoints, slices, star=None,
) -> tuple[list[dict], dict]:
    set_seed(seed)
    # Process-length residual/positions for both modes: outcome supervision is still the
    # short continuation COLON, s_D, EOS on the same backbone. The 2×2 in handcoded/train.py
    # stretches outcome *architecture* to process length; here architecture is held fixed
    # and only supervision placement varies.
    model = build_random_trainable_process_architecture(
        tokenizer, depth, seed=seed, init_std=init_std, device=device,
    )
    configure_device(device)
    train_model = prepare_train_model(model, device)
    optimizer = make_adamw([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=0, device=device)
    schedule = make_batch_schedule(len(train_data), args.steps, args.batch_size, seed=args.batch_seed + seed)
    checkpoint_set = set(checkpoints)
    rows: list[dict] = []

    def checkpoint_row(step: int) -> dict:
        row = evaluate_escape(model, eval_process, eval_outcome, train_eval, tokenizer, slices, mode)
        if star is not None:
            from src.experiments.handcoded_circuit_targets import circuit_match_metrics
            row.update(circuit_match_metrics(model, star))
        row.update({
            "step": step, "seed": seed, "depth": depth, "init_std": init_std, "mode": mode,
        })
        return row

    row = checkpoint_row(0)
    rows.append(row)
    print(_log_line(row), flush=True)
    label = f"escape/{mode}/D{depth}/ε{init_std:g}/s{seed}"
    progress = tqdm(enumerate(schedule, 1), total=len(schedule), desc=label)
    for step, indices in progress:
        train_step(train_model, train_data.select(indices), optimizer, grad_clip_norm=args.grad_clip)
        if step not in checkpoint_set:
            continue
        row = checkpoint_row(step)
        rows.append(row)
        progress.set_postfix(
            pγ=f"{row['p_correct']:.2f}",
            ans=f"{row['answer_acc']:.2f}",
            loss=f"{row['train_loss']:.3f}",
        )
        print(_log_line(row), flush=True)
    return rows, hitting_times(rows, args)


def draw_escape_times(metrics: pd.DataFrame, hitting: pd.DataFrame, path: Path, *, escape_a: float, sentinel: int) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    grouped = metrics.groupby(["step", "mode", "depth", "init_std"], as_index=False).mean(numeric_only=True)
    depths = sorted(grouped["depth"].unique())
    inits = sorted(grouped["init_std"].unique())
    style_for = {eps: EPS_STYLES[i % len(EPS_STYLES)] for i, eps in enumerate(inits)}
    min_depth = min(depths) if depths else 2

    ax = axes[0]
    ax.axhline(1.0 / 16.0, color="#444444", linewidth=0.9, alpha=0.7, label=r"uniform $1/K$")
    ax.axhline(escape_a, color="#6a4c93", linewidth=0.9, alpha=0.7, linestyle=":", label=rf"$a={escape_a:g}$")
    for (mode, depth, init_std), sub in grouped.groupby(["mode", "depth", "init_std"]):
        sub = sub.sort_values("step")
        ax.plot(
            sub["step"], sub["p_correct"], style_for[float(init_std)],
            color=MODE_COLORS.get(mode, "#333333"),
            linewidth=1.4 + 0.35 * (int(depth) - min_depth),
            label=rf"{mode} $D$={int(depth)} $\varepsilon$={float(init_std):g}",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(r"$p_{\mathrm{correct}}$")
    ax.set_title(r"(a) local competence vs step")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=True, loc="best", fontsize=7, ncol=1)

    hit = hitting.groupby(["mode", "depth", "init_std"], as_index=False).mean(numeric_only=True)
    ymax = max(float(sentinel), float(hit[["t_escape_p", "t_answer"]].max().max()) if len(hit) else sentinel)

    def _hit_panel(ax, column, title, ylabel):
        for (mode, depth), sub in hit.groupby(["mode", "depth"]):
            sub = sub.sort_values("init_std")
            ax.plot(
                sub["init_std"], sub[column], "-o",
                color=MODE_COLORS.get(mode, "#333333"),
                linewidth=2.0,
                markersize=6.0,
                label=rf"{mode} $D$={int(depth)}",
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"init scale $\varepsilon$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.05 * ymax, 1.08 * ymax)
        ax.axhline(sentinel, color="#444444", linewidth=0.8, alpha=0.55, linestyle="--")
        ax.legend(frameon=True, loc="best", fontsize=8)

    _hit_panel(axes[1], "t_escape_p", r"(b) $t_{\mathrm{escape}}$ vs $\varepsilon$", r"$t_{\mathrm{escape},p}$")
    _hit_panel(axes[2], "t_answer", r"(c) $t_{\mathrm{answer}}$ vs $\varepsilon$", r"$t_{\mathrm{answer}}$")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, args: argparse.Namespace, tokenizer, device, slices_by_depth: dict[int, ProcessSlices]) -> None:
    k = int(tokenizer.n_states)
    c = k - 1
    uniform = 1.0 / k
    slice_lines = []
    for depth in sorted(slices_by_depth):
        sl = slices_by_depth[depth]
        slice_lines.append(
            f"- D={depth}: first={depth + 1}; gate `{list(sl.gate)}`; state `{list(sl.state)}`; "
            f"syntax `{list(sl.syntax)}`; answer `{list(sl.answer)}`"
        )
    lines = [
        "# Hand-coded escape times (supervision placement)",
        "",
        "Executor-level diagnostic for the draft *Supervision Placement and Escape from Flat Compositional Losses*.",
        "This run does **not** prove the γ-flow theorem, does **not** recover QK/OV circuits, and is not an L(τ+)/L(τ-) probe.",
        "",
        "## Paper objects vs columns",
        "",
        r"- Process loss \(L_{\mathrm{proc}}=\log K-\frac1D\sum_i\log(1+c\gamma_i)\) is local successor NLL on gold \(s_k\).",
        r"- Column `L_state` is the neural analogue on a **gold process-encoded eval batch** (mean CE at state target indices).",
        r"  Under outcome supervision those tokens are not in the training objective, so `L_state` is logged as NaN for outcome rows.",
        r"  `L_state_diag` is the same CE for both modes (eval-only diagnostic).",
        r"- Outcome loss \(L_{\mathrm{out}}=\log K-\log(1+c\prod\gamma_i)\) is terminal answer NLL. Column `L_answer` is CE on gold \(s_D\) after COLON,",
        "  using the mode-matched encoding (process: COLON after the trace; outcome: COLON after SEP).",
        r"- Local competence proxy: paper \(p_{\mathrm{correct}}=(1+c\gamma)/K\). Column `p_correct` is the mean softmax probability of the gold successor \(s_k\)",
        "  at process state positions on the gold process eval batch (even when training outcome).",
        r"- `p_answer` is \(P(\text{gold }s_D\mid\text{COLON context})\) on the mode-matched batch.",
        r"- Escape \(\tau_a=\inf\{t:\max_i\gamma_i(t)=a\}\), **not** time to zero loss. We log `t_escape_p` = first checkpoint with `p_correct >= a`",
        rf"  (`--escape-a`, this run a={args.escape_a:g}). Mean \(p_{{\mathrm{{correct}}}}\) is a lower bound on \(\max_i p_i\); a miss means the mean has not left the box.",
        r"- `t_answer` = first checkpoint with `answer_acc >= --answer-a` **or** `p_answer >= a` (task solve, distinct from local escape).",
        r"- `t_process` = first checkpoint with `process_acc >= --process-a` (exact gold process continuation). Typically never under outcome training.",
        r"- If a threshold is never hit, the time is the sentinel `steps+1` and `escaped_*=False`.",
        "",
        "## Constants",
        "",
        f"- K={k} states, c=K-1={c}, uniform p=1/K={uniform:.6f}.",
        f"- Vocab V={len(tokenizer.tokens)} (K + M gates + SEP/COLON/EOS/PAD).",
        f"- Architecture: `build_random_trainable_process_architecture` for **both** modes (process-length residual/positions).",
        "  Outcome supervision is the short continuation `COLON, s_D, EOS` on that backbone. Supervision varies; architecture does not.",
        f"- Init: N(0, ε) with ε in {args.init_stds}. Same seed ⇒ same Gaussian directions, scaled by ε.",
        f"- Depths D={list(args.depths)}. Prediction: outcome escape Θ(log 1/ε) at D=2 and Θ(ε^{{-(D-2)}}) at D>2; process escape Θ(1).",
        f"- Optimizer AdamW lr={args.lr}, weight_decay=0. Steps={args.steps}, batch={args.batch_size}, train_size={args.train_size}.",
        f"- Device={device}. Seeds={args.seeds}. L_state threshold for `t_escape_Lstate` is -log(a)={args.lstate_thr:.4f} unless overridden.",
        "",
        "## Token slices (process teacher-forced targets)",
        "",
        "Sequence: 0 s0; 1..D prompt gates; D+1 SEP; then pairs (g_k, s_k); then COLON, answer s_D; EOS is the last next-token target.",
        *slice_lines,
        "Outcome continuation is `[COLON, s_D, EOS]`; prompt is still start+gates+SEP. Answer NLL is the token after COLON.",
        "",
        "## Caveats",
        "",
        "- Empirical hitting times on a discrete log grid are not a proof of the scaling. Fit log vs power in ε only after a paper-scale grid.",
        "- Outcome-trained models never see trace tokens, so `p_correct` at process state positions is a **diagnostic readout**, not a train metric.",
        "- Circuit cosine / Hungarian QK match is off unless `--circuit-match` (wrong object for this figure).",
        "",
        "## Smoke / paper-scale",
        "",
        "```",
        "python -m src.experiments.handcoded_escape_times --smoke --device cuda:2 --output results/handcoded_escape_times_smoke",
        "python experiments/run.py handcoded-escape-times --smoke --device cuda:2 --output results/handcoded_escape_times_smoke",
        "```",
        "",
        "Paper-scale grid suggestion (not run by --smoke):",
        "",
        "```",
        "python -m src.experiments.handcoded_escape_times \\",
        "  --modes process outcome --depths 2 3 4 \\",
        "  --init-stds 0.005 0.01 0.02 0.04 \\",
        "  --steps 2000 --train-size 20000 --batch-size 64 --device cuda:2 \\",
        "  --output results/handcoded_escape_times",
        "```",
        "",
        f"This run: smoke={bool(args.smoke)} modes={args.modes} depths={args.depths} init_stds={args.init_stds} "
        f"steps={args.steps} seeds={args.seeds} output={args.output}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _maybe_star(tokenizer, depth: int, enabled: bool):
    if not enabled:
        return None
    from handcoded.models import HandcodedProcessTransformer
    from src.experiments.handcoded_circuit_targets import extract_star_circuits, verify_star_targets
    star_model = HandcodedProcessTransformer(tokenizer, depth).eval()
    star = extract_star_circuits(star_model, tokenizer=tokenizer, depth=depth)
    verify_star_targets(star)
    return star


def main(argv=None) -> int:
    args = _resolve(parse_args(argv))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = make_tokenizer()
    k = int(tokenizer.n_states)
    if 4 in args.depths:
        assert_d4_continuation_slices()
    slices_by_depth = {depth: continuation_slices(depth) for depth in args.depths}
    for depth, sl in slices_by_depth.items():
        print(
            f"continuation slices D={depth}: gate={list(sl.gate)} state={list(sl.state)} "
            f"syntax={list(sl.syntax)} answer={list(sl.answer)}",
            flush=True,
        )

    device = _resolve_device(args.device)
    configure_device(device)
    print(
        f"device={device} steps={args.steps} seeds={args.seeds} modes={args.modes} "
        f"depths={args.depths} init_stds={args.init_stds} batch={args.batch_size} "
        f"K={k} a={args.escape_a} uniform={1.0 / k:.4f}",
        flush=True,
    )
    write_notes(out / "NOTES.md", args, tokenizer, device, slices_by_depth)

    checkpoints = _checkpoints(args.steps, args.log_every)
    metric_rows: list[dict] = []
    hit_rows: list[dict] = []
    n_jobs = len(args.depths) * len(args.init_stds) * len(args.modes) * len(args.seeds)
    job = 0
    for depth in args.depths:
        slices = slices_by_depth[depth]
        train_circuits = make_circuits(args.train_size, args.data_seed, depth=depth)
        eval_circuits = make_circuits(args.eval_size, args.eval_seed, depth=depth)
        process_train = encode_dataset(train_circuits, tokenizer, "process").to(device)
        outcome_train = encode_dataset(train_circuits, tokenizer, "outcome").to(device)
        eval_process = encode_dataset(eval_circuits, tokenizer, "process").to(device)
        eval_outcome = encode_dataset(eval_circuits, tokenizer, "outcome").to(device)
        n_train_eval = min(args.eval_size, len(process_train), len(outcome_train))
        train_eval_process = process_train.select(slice(0, n_train_eval))
        train_eval_outcome = outcome_train.select(slice(0, n_train_eval))
        star = _maybe_star(tokenizer, depth, args.circuit_match)
        for init_std in args.init_stds:
            for mode in args.modes:
                train_data = process_train if mode == "process" else outcome_train
                train_eval = train_eval_process if mode == "process" else train_eval_outcome
                for seed in args.seeds:
                    job += 1
                    print(
                        f"[{job}/{n_jobs}] mode={mode} D={depth} ε={init_std:g} seed={seed}",
                        flush=True,
                    )
                    rows, hit = run_one(
                        seed=seed, depth=depth, init_std=init_std, mode=mode, args=args,
                        tokenizer=tokenizer, train_data=train_data, eval_process=eval_process,
                        eval_outcome=eval_outcome, train_eval=train_eval, device=device,
                        checkpoints=checkpoints, slices=slices, star=star,
                    )
                    metric_rows.extend(rows)
                    hit_rows.append(hit)

    csv_path = write_csv(out / "metrics.csv", metric_rows)
    hit_path = write_csv(out / "hitting_times.csv", hit_rows)
    frame = pd.DataFrame(metric_rows)
    hitting = pd.DataFrame(hit_rows)
    draw_escape_times(
        frame, hitting, out / "escape_times",
        escape_a=args.escape_a, sentinel=args.steps + 1,
    )
    init_rows = frame[frame["step"] == 0]
    summary = {
        "experiment": "handcoded_escape_times",
        "K": k,
        "c": k - 1,
        "uniform_p": 1.0 / k,
        "escape_a": args.escape_a,
        "answer_a": args.answer_a,
        "process_a": args.process_a,
        "lstate_thr": args.lstate_thr,
        "depths": args.depths,
        "init_stds": args.init_stds,
        "modes": args.modes,
        "steps": args.steps,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "lr": args.lr,
        "device": str(device),
        "smoke": bool(args.smoke),
        "circuit_match": bool(args.circuit_match),
        "architecture": "build_random_trainable_process_architecture",
        "token_slices": {str(d): sl.as_dict() for d, sl in slices_by_depth.items()},
        "csv": str(csv_path),
        "hitting_csv": str(hit_path),
        "n_metric_rows": len(metric_rows),
        "n_runs": len(hit_rows),
        "p_correct_step0_mean": float(init_rows["p_correct"].mean()) if len(init_rows) else None,
        "hitting_times": hit_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {hit_path}")
    print(f"wrote {out / 'escape_times.pdf'}")
    print(f"wrote {out / 'NOTES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
