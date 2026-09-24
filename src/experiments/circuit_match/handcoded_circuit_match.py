"""Does GD rediscover the hand-coded process-executor factorization?

Trains a random-init copy of `HandcodedProcessTransformer` (same residual
partition, 4 heads, MLP width KM) on Boolean process supervision at D=4,
and logs cosine / pointer / OV-block / MLP-transition similarity to the
constructed maps extracted by `handcoded_circuit_targets`.
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
from tqdm.auto import tqdm

from handcoded.data import encode_dataset, make_batch_schedule
from handcoded.eval import free_run_metrics, make_generation_evaluation
from handcoded.gates import make_circuits
from handcoded.models import HandcodedProcessTransformer, build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from handcoded.train import train_step
from src.experiments.circuit_match.handcoded_circuit_targets import (
    STAR_HEAD_NAMES,
    algebra_audit,
    circuit_match_metrics,
    extract_star_circuits,
    star_qk_route_entries,
    verify_star_targets,
)
from src.experiments.circuit_match.handcoded_trace_gradients import (
    CORRUPTION_SEED,
    assert_d4_continuation_slices,
    check_star_dirs,
    continuation_slices,
    draw_trace_gradients,
    extract_grad_match_rows,
    make_matched_eval_batches,
    matched_loss_and_grads,
    notes_section,
    star_direction_vectors,
)
from src.plot_style import apply_style
from src.training.io import write_csv
from src.training.optim import make_adamw
from src.training.seed import configure_device, default_device, prepare_train_model, set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "handcoded_circuit_match"

CIRCUIT_KEYS = (
    "sim_qk_S", "sim_ov_S", "sim_qk_G", "sim_ov_G", "sim_qk_F", "sim_ov_F", "mlp_sim",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=None, help="Gradient steps (default 2000; 40 with --smoke).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Overrides --seed if given.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-size", type=int, default=None, help="Teacher-forced / free-run eval prefix.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="Tiny CPU/GPU run: 40 steps, small data.")
    p.add_argument("--verify-only", action="store_true", help="Extract and check star circuits, then exit.")
    p.add_argument("--no-free-run", action="store_true")
    p.add_argument("--data-seed", type=int, default=123)
    p.add_argument("--test-seed", type=int, default=9000)
    p.add_argument("--batch-seed", type=int, default=2026)
    p.add_argument(
        "--corruption", choices=("symmetric", "coherent"), default="symmetric",
        help="Matched wrong-trace law for the eval-only L+/L- probe (default: symmetric).",
    )
    p.add_argument(
        "--probe-batch-size", type=int, default=None,
        help="Matched (τ+, τ-) pairs for the loss/grad probe (default 16 smoke / 32 full).",
    )
    return p.parse_args(argv)


def _resolve(args: argparse.Namespace) -> argparse.Namespace:
    smoke = bool(args.smoke)
    args.steps = 40 if args.steps is None and smoke else (2000 if args.steps is None else args.steps)
    args.batch_size = 32 if args.batch_size is None and smoke else (64 if args.batch_size is None else args.batch_size)
    args.train_size = 128 if args.train_size is None and smoke else (4096 if args.train_size is None else args.train_size)
    args.test_size = 32 if args.test_size is None and smoke else (256 if args.test_size is None else args.test_size)
    args.log_every = 5 if args.log_every is None and smoke else (50 if args.log_every is None else args.log_every)
    args.eval_size = 16 if args.eval_size is None and smoke else (64 if args.eval_size is None else args.eval_size)
    args.probe_batch_size = (
        16 if args.probe_batch_size is None and smoke
        else (32 if args.probe_batch_size is None else args.probe_batch_size)
    )
    args.seeds = list(args.seeds) if args.seeds is not None else [args.seed]
    return args


def _checkpoints(steps: int, log_every: int) -> list[int]:
    marks = {0, steps}
    if steps >= 1:
        marks.add(1)
    marks.update(range(log_every, steps + 1, log_every))
    return sorted(m for m in marks if 0 <= m <= steps)


@torch.no_grad()
def teacher_forced_metrics(model, batch, tokenizer) -> dict[str, float]:
    logits = model(batch.inputs)
    pred = logits.argmax(dim=-1)
    targets = batch.targets
    valid = targets != -100
    token_acc = float((pred == targets)[valid].float().mean()) if valid.any() else float("nan")
    process_acc = float(((pred == targets) | ~valid).all(dim=1).float().mean())
    colon = batch.inputs == tokenizer.colon
    answer_acc = float((pred[colon] == targets[colon]).float().mean()) if colon.any() else float("nan")
    loss = float(torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=-100,
    ))
    return {
        "loss": loss,
        "token_acc": token_acc,
        "process_acc": process_acc,
        "answer_acc": answer_acc,
    }


@torch.no_grad()
def evaluate_model(model, star, tokenizer, train_batch, test_batch, train_gen, test_gen, *, free_run: bool) -> dict:
    model.eval()
    row = dict(circuit_match_metrics(model, star))
    train_m = teacher_forced_metrics(model, train_batch, tokenizer)
    test_m = teacher_forced_metrics(model, test_batch, tokenizer)
    row.update({
        "train_loss": train_m["loss"],
        "train_token_acc": train_m["token_acc"],
        "train_process_acc": train_m["process_acc"],
        "train_answer_acc": train_m["answer_acc"],
        "test_loss": test_m["loss"],
        "test_token_acc": test_m["token_acc"],
        "test_process_acc": test_m["process_acc"],
        "test_answer_acc": test_m["answer_acc"],
    })
    if free_run:
        train_fr = free_run_metrics(model, train_gen, tokenizer, "process")
        test_fr = free_run_metrics(model, test_gen, tokenizer, "process")
        row["train_free_run_process"] = train_fr["exact_continuation"]
        row["train_free_run_answer"] = train_fr["final_answer"]
        row["test_free_run_process"] = test_fr["exact_continuation"]
        row["test_free_run_answer"] = test_fr["final_answer"]
    else:
        for key in ("train_free_run_process", "train_free_run_answer", "test_free_run_process", "test_free_run_answer"):
            row[key] = float("nan")
    return row


def _save_star_npz(star, path: Path) -> None:
    payload = {f"qk_{n}": star.c_qk[n].numpy() for n in STAR_HEAD_NAMES}
    payload.update({f"ov_{n}": star.c_ov[n].numpy() for n in STAR_HEAD_NAMES})
    payload.update({"mlp_in": star.w_in.numpy(), "mlp_bias": star.bias.numpy(), "mlp_out": star.w_out.numpy()})
    import numpy as np
    np.savez_compressed(path, **payload)


def draw_circuit_match(frame: pd.DataFrame, path: Path) -> None:
    apply_style()
    grouped = frame.groupby("step", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    colors = {
        "sim_qk_S": "#087eaa", "sim_ov_S": "#087eaa",
        "sim_qk_G": "#c56b08", "sim_ov_G": "#c56b08",
        "sim_qk_F": "#6a4c93", "sim_ov_F": "#6a4c93",
        "mlp_sim": "#2a9d8f",
    }
    styles = {
        "sim_qk_S": "-", "sim_ov_S": "--",
        "sim_qk_G": "-", "sim_ov_G": "--",
        "sim_qk_F": "-", "sim_ov_F": "--",
        "mlp_sim": "-",
    }
    labels = {
        "sim_qk_S": r"$C_{QK}^{S}$", "sim_ov_S": r"$C_{OV}^{S}$",
        "sim_qk_G": r"$C_{QK}^{G}$", "sim_ov_G": r"$C_{OV}^{G}$",
        "sim_qk_F": r"$C_{QK}^{F}$", "sim_ov_F": r"$C_{OV}^{F}$",
        "mlp_sim": r"MLP $W^*$",
    }
    ax = axes[0]
    for key in CIRCUIT_KEYS:
        ax.plot(grouped["step"], grouped[key], styles[key], color=colors[key], label=labels[key], linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("cosine to star circuit")
    ax.set_title("(a) Learned maps vs construction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="lower right", fontsize=9, ncol=2)

    ax = axes[1]
    ax.plot(grouped["step"], grouped["train_process_acc"], "-", color="#087eaa", label="train process", linewidth=2.0)
    ax.plot(grouped["step"], grouped["train_answer_acc"], "--", color="#087eaa", label="train answer", linewidth=2.0)
    ax.plot(grouped["step"], grouped["test_process_acc"], "-", color="#c56b08", label="test process", linewidth=2.0)
    ax.plot(grouped["step"], grouped["test_answer_acc"], "--", color="#c56b08", label="test answer", linewidth=2.0)
    if "test_free_run_answer" in grouped and grouped["test_free_run_answer"].notna().any():
        ax.plot(grouped["step"], grouped["test_free_run_answer"], ":", color="#6a4c93", label="test free-run answer", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("(b) Process supervision competence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, star, audit, args, slices=None):
    L = star.layout
    routes = {name: {str(d): o for d, o in star.routes[name].items()} for name in STAR_HEAD_NAMES}
    mismatches = [row for row in audit if not row["ok"]]
    lines = [
        "# Hand-coded circuit match (Paper 2 start)",
        "",
        "Learned model: random-init `HandcodedProcessTransformer` (same residual partition and 4 heads).",
        "Task: Boolean process executor, process supervision, D=4, same semantic tokenizer.",
        "",
        "## Constants (from code)",
        "",
        f"- K={L.K}, M={L.M}, D={L.D}, P=3D+4={L.P}, d=3K+2M+P={L.d}, KM={L.KM}",
        f"- vocab V={L.vocab} (K+M+4 specials: SEP, COLON, EOS, PAD)",
        f"- heads={L.n_heads} (S,G,F programmed; head 3 unused in the construction), 1 layer",
        f"- α={80.0}, MLP bias={-1.5}, unembed scale={20.0}",
        f"- residual slices: S={L.S.start}:{L.S.stop}, G={L.G.start}:{L.G.stop}, "
        f"Pos={L.Pos.start}:{L.Pos.stop}, R={L.R.start}:{L.R.stop}, "
        f"G_out={L.G_out.start}:{L.G_out.stop}, S_out={L.S_out.start}:{L.S_out.stop}",
        "",
        "## Similarity",
        "",
        r"- `sim(C, C*) = <vec(C), vec(C*)> / (||C||_F ||C*||_F)` (cosine = unit-Frobenius inner product).",
        "- Centered cosine subtracts the mean of vec(C) first; logged as `sim_*_centered`.",
        "- Learned heads are unlabeled: Hungarian assignment maximizes cosine(QK)+cosine(OV) vs {S,G,F}*.",
        "- Pointer accuracy: argmax over causal sources of the Pos×Pos block of C_QK equals the programmed origin.",
        "- OV block: cosine of C_OV[S,R] / C_OV[G,G_out] / C_OV[S,S_out] vs I, plus Frobenius mass in that block.",
        "- MLP: Hungarian-align hidden units by W_in rows; `mlp_sim` is the mean of aligned W_in and W_out cosines.",
        "  `mlp_readout_acc` is argmax S_out of MLP(e_s^R + e_g^G) vs φ(s,g); `mlp_hidden_peak` is whether that unit fires.",
        "  Random-init `mlp_sim` is a small positive (~0.1) because Hungarian can match 832 rows to sparse targets; QK/OV start near 0.",
        "",
        "## Programmed D=4 routes (dest←origin)",
        "",
        f"- Head S: {routes['S']}",
        f"- Head G: {routes['G']}",
        f"- Head F: {routes['F']}",
        "",
        "Sequence (0-index, length-16 input): 0 s0; 1–4 prompt gates; 5 SEP; 6 g1; 7 s1; …; 12 g4; 13 s4; 14 COLON; 15 s4. EOS is the next-token target.",
        "",
        "## Checklist vs code",
        "",
    ]
    for row in audit:
        mark = "match" if row["ok"] else "MISMATCH (trust code)"
        extra = f" — {row['note']}" if row.get("note") else ""
        lines.append(f"- {row['item']}: user `{row['user']}` / code `{row['code']}` → {mark}{extra}")
    lines += [
        "",
        *(notes_section(slices or continuation_slices(args.depth), args.corruption)),
        "## Smoke / run",
        "",
        "```",
        "python -m src.experiments.circuit_match.handcoded_circuit_match --smoke --output results/handcoded_circuit_match",
        "python -m src.experiments.circuit_match.handcoded_circuit_match --smoke --corruption coherent --output results/handcoded_circuit_match_probe_smoke",
        "```",
        "",
        f"Default non-smoke: steps={2000}, batch={64}, train_size={4096}. Seeds used in this run: {args.seeds}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    return mismatches


def _log_line(row: dict) -> str:
    line = (
        f"step={row['step']:<5d} loss={row['train_loss']:.4f} "
        f"proc={row['train_process_acc']:.3f} ans={row['train_answer_acc']:.3f} "
        f"QK_S={row['sim_qk_S']:.3f} OV_S={row['sim_ov_S']:.3f} "
        f"QK_G={row['sim_qk_G']:.3f} OV_G={row['sim_ov_G']:.3f} "
        f"QK_F={row['sim_qk_F']:.3f} OV_F={row['sim_ov_F']:.3f} "
        f"MLP={row['mlp_sim']:.3f} φ={row['mlp_readout_acc']:.3f}"
    )
    if "L_plus" in row:
        line += (
            f" L+={row['L_plus']:.4f} L-={row['L_minus']:.4f} "
            f"ΔL={row['delta_L_trace']:+.3f}"
        )
    return line


def _checkpoint_row(
    model, star, tokenizer, train_eval_batch, test_eval_batch, train_gen, test_gen,
    *, free_run, star_dirs, probe_plus, probe_minus, slices, step, seed, corruption,
):
    row = evaluate_model(
        model, star, tokenizer, train_eval_batch, test_eval_batch, train_gen, test_gen,
        free_run=free_run,
    )
    row.update(matched_loss_and_grads(model, star_dirs, probe_plus, probe_minus, slices))
    row.update({"step": step, "seed": seed, "corruption": corruption})
    return row


def run_seed(
    seed, args, tokenizer, star, train_data, test_data, train_eval_batch, test_eval_batch,
    train_gen, test_gen, device, checkpoints, probe_plus, probe_minus, star_dirs, slices,
):
    set_seed(seed)
    model = build_random_trainable_process_architecture(
        tokenizer, args.depth, seed=seed, init_std=args.init_std, device=device,
    )
    configure_device(device)
    train_model = prepare_train_model(model, device)
    optimizer = make_adamw([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=0, device=device)
    schedule = make_batch_schedule(len(train_data), args.steps, args.batch_size, seed=args.batch_seed + seed)
    free_run = not args.no_free_run
    rows = []
    row = _checkpoint_row(
        model, star, tokenizer, train_eval_batch, test_eval_batch, train_gen, test_gen,
        free_run=free_run, star_dirs=star_dirs, probe_plus=probe_plus, probe_minus=probe_minus,
        slices=slices, step=0, seed=seed, corruption=args.corruption,
    )
    rows.append(row)
    print(_log_line(row), flush=True)
    checkpoint_set = set(checkpoints)
    progress = tqdm(enumerate(schedule, 1), total=len(schedule), desc=f"circuit-match/s{seed}")
    for step, indices in progress:
        train_step(train_model, train_data.select(indices), optimizer, grad_clip_norm=args.grad_clip)
        if step not in checkpoint_set:
            continue
        row = _checkpoint_row(
            model, star, tokenizer, train_eval_batch, test_eval_batch, train_gen, test_gen,
            free_run=free_run, star_dirs=star_dirs, probe_plus=probe_plus, probe_minus=probe_minus,
            slices=slices, step=step, seed=seed, corruption=args.corruption,
        )
        rows.append(row)
        progress.set_postfix(
            loss=f"{row['train_loss']:.3f}",
            ans=f"{row['train_answer_acc']:.2f}",
            Lp=f"{row['L_plus']:.3f}",
            Lm=f"{row['L_minus']:.3f}",
            dL=f"{row['delta_L_trace']:+.2f}",
        )
        print(_log_line(row), flush=True)
    return rows


def main(argv=None) -> int:
    args = _resolve(parse_args(argv))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = make_tokenizer()
    assert_d4_continuation_slices()
    slices = continuation_slices(args.depth)
    print(f"continuation slices D={slices.depth}: gate={list(slices.gate)} state={list(slices.state)} "
          f"syntax={list(slices.syntax)} answer={list(slices.answer)}", flush=True)
    star_model = HandcodedProcessTransformer(tokenizer, args.depth).eval()
    star = extract_star_circuits(star_model, tokenizer=tokenizer, depth=args.depth)
    verify_star_targets(star)
    star_dirs = star_direction_vectors(star_model)
    star_d_norms = check_star_dirs(star_dirs)
    audit = algebra_audit(star)
    entries = star_qk_route_entries(star)
    print("star C_QK Pos(d),Pos(o) entries (α/sqrt(P) = "
          f"{80.0 / math.sqrt(star.layout.P):.4f}):", flush=True)
    for name, routes in entries.items():
        print(f"  {name}: {routes}", flush=True)
    _save_star_npz(star, out / "star_targets.npz")
    mismatches = write_notes(out / "NOTES.md", star, audit, args, slices=slices)
    (out / "algebra_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"verified star circuits d={star.layout.d} P={star.layout.P} heads={star.layout.n_heads}", flush=True)
    if args.verify_only:
        print(f"wrote {out / 'NOTES.md'} (verify-only)")
        return 0

    device = torch.device(args.device or default_device())
    print(
        f"device={device} steps={args.steps} seeds={args.seeds} batch={args.batch_size} "
        f"corruption={args.corruption} probe_batch={args.probe_batch_size}",
        flush=True,
    )

    train_circuits = make_circuits(args.train_size, args.data_seed, depth=args.depth)
    test_circuits = make_circuits(args.test_size, args.test_seed, depth=args.depth)
    train_data = encode_dataset(train_circuits, tokenizer, "process").to(device)
    test_data = encode_dataset(test_circuits, tokenizer, "process").to(device)
    n_eval = min(args.eval_size, len(train_data), len(test_data))
    train_eval_batch = train_data.select(slice(0, n_eval))
    test_eval_batch = test_data.select(slice(0, n_eval))
    train_gen = make_generation_evaluation(train_circuits[:n_eval], tokenizer, device)
    test_gen = make_generation_evaluation(test_circuits[:n_eval], tokenizer, device)
    n_probe = min(args.probe_batch_size, len(test_circuits))
    probe_plus, probe_minus = make_matched_eval_batches(
        test_circuits[:n_probe], tokenizer, corruption=args.corruption,
        seed=CORRUPTION_SEED, device=device,
    )
    checkpoints = _checkpoints(args.steps, args.log_every)

    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(
            seed, args, tokenizer, star, train_data, test_data,
            train_eval_batch, test_eval_batch, train_gen, test_gen,
            device, checkpoints, probe_plus, probe_minus, star_dirs, slices,
        ))

    csv_path = write_csv(out / "metrics.csv", rows)
    grad_csv = write_csv(out / "grad_match.csv", extract_grad_match_rows(rows))
    frame = pd.DataFrame(rows)
    draw_circuit_match(frame, out / "circuit_match")
    draw_trace_gradients(frame, out / "trace_gradients")
    summary = {
        "experiment": "handcoded_circuit_match",
        "depth": args.depth,
        "layout": star.layout.as_dict(),
        "routes": {n: {str(d): o for d, o in star.routes[n].items()} for n in STAR_HEAD_NAMES},
        "qk_route_entries": entries,
        "mismatches": mismatches,
        "steps": args.steps,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "lr": args.lr,
        "device": str(device),
        "smoke": bool(args.smoke),
        "corruption": args.corruption,
        "probe_batch_size": n_probe,
        "token_slices": slices.as_dict(),
        "star_d_norms": star_d_norms,
        "csv": str(csv_path),
        "grad_csv": str(grad_csv),
        "final": rows[-1] if rows else {},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {grad_csv}")
    print(f"wrote {out / 'circuit_match.pdf'}")
    print(f"wrote {out / 'trace_gradients.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
