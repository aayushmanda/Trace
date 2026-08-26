"""
Early-training prediction of later executor acquisition.

This experiment trains each (rho, seed) ONCE to a long horizon. During only
an early window, it measures the change in clean teacher-forced successor-
state NLL caused by the actual AdamW update.

For a fixed held-out diagnostic set,
    H_n     = mean clean successor-state NLL
    Delta_n = H_n - H_{n+1}

Using only early updates, it estimates
    a_hat = mean(Delta_n)
    v_hat = var(Delta_n)

For target exact-trace accuracy A and depth D,
    q = 1 - A^(1/D).

If a whole successor state is greedily wrong, at least one state token has
p(correct) <= 1/2, hence the state's summed NLL is >= log 2. Therefore a
conservative local-risk target is
    H_target = q * log 2.

The script then predicts late acquisition without fitting to late outcomes:
  drift-only:
    H_early - k*a_hat <= H_target

  noise-adjusted:
    H_early - k*a_hat + sqrt(2 log(1/beta))*sqrt(k*v_hat) <= H_target

It compares these predictions with the first free-run checkpoint whose
exact-trace accuracy reaches --target-exact.

Run from the repository root.
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.registry import TASKS
from sweep_ratio import RatioDataset, build_model, evaluate, generate_unique, make_optimizer, set_seed


def split_step(task_name: str, step: str):
    if task_name.startswith("state_machine_"):
        return step[:-2], step[-2:]
    if task_name.startswith("register_machine_"):
        return step[:-4], step[-4:]
    if task_name.startswith("boolean_circuit_"):
        prefix, state = step.rsplit(">", 1)
        return prefix + ">", state
    raise ValueError("Supported tasks: state_machine_*, register_machine_*, boolean_circuit_*.")


def append_csv(path: Path, fields, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def finite_or_blank(x):
    if x is None:
        return ""
    x = float(x)
    return x if math.isfinite(x) else ""


def build_state_queries(task, instances):
    tokenizer = task.tokenizer
    queries = []
    for example_index, inst in enumerate(instances):
        previous = []
        for transition_index, step in enumerate(inst.correct_trace.split()):
            prefix, state = split_step(task.name, step)
            before_step = f"{inst.prompt} " + (" ".join(previous) + " " if previous else "")
            queries.append((example_index, transition_index, tokenizer.encode(before_step + prefix), tokenizer.encode(state)))
            previous.append(step)
    return queries


@torch.inference_mode()
def score_clean_states(model, queries, batch_size, device):
    """Return summed state NLL and minimum token margin per transition."""
    was_training = model.training
    model.eval()
    groups = defaultdict(list)
    for row in queries:
        groups[(len(row[2]), len(row[3]))].append(row)

    nlls, margins, transition_ids = [], [], []
    for (context_length, target_length), rows in groups.items():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            contexts = torch.tensor([r[2] for r in batch], dtype=torch.long, device=device)
            targets = torch.tensor([r[3] for r in batch], dtype=torch.long, device=device)
            sequence = torch.cat((contexts, targets), dim=1)
            logits, _ = model(sequence[:, :-1])
            target_logits = logits[:, context_length - 1:context_length - 1 + target_length, :].float()

            token_nll = F.cross_entropy(
                target_logits.reshape(-1, target_logits.shape[-1]),
                targets.reshape(-1), reduction="none",
            ).view(len(batch), target_length)

            correct_logits = target_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            wrong_logits = target_logits.clone()
            wrong_logits.scatter_(-1, targets.unsqueeze(-1), float("-inf"))
            token_margin = correct_logits - wrong_logits.max(dim=-1).values

            state_nll = token_nll.sum(dim=1)
            state_margin = token_margin.min(dim=1).values
            nlls.extend(state_nll.cpu().numpy().tolist())
            margins.extend(state_margin.cpu().numpy().tolist())
            transition_ids.extend([r[1] for r in batch])

    if was_training:
        model.train()
    return {
        "state_nll": np.asarray(nlls, dtype=np.float64),
        "state_margin": np.asarray(margins, dtype=np.float64),
        "transition_index": np.asarray(transition_ids, dtype=np.int64),
    }


def summarize_scores(scores, tail_fraction):
    nll = scores["state_nll"]
    margin = scores["state_margin"]
    k = max(1, int(math.ceil(tail_fraction * len(margin))))
    return {
        "clean_state_nll": float(nll.mean()),
        "clean_state_nll_std": float(nll.std(ddof=1)) if len(nll) > 1 else 0.0,
        "mean_state_margin": float(margin.mean()),
        "median_state_margin": float(np.median(margin)),
        "target_quantile_margin": float(np.quantile(margin, tail_fraction)),
        "tail_mean_margin": float(np.sort(margin)[:k].mean()),
        "positive_state_margin_rate": float((margin > 0).mean()),
    }


def drift_crossing_step(early_step, current_risk, target_risk, drift):
    gap = current_risk - target_risk
    if gap <= 0:
        return float(early_step)
    if drift <= 0:
        return math.inf
    return float(early_step + gap / drift)


def noise_crossing_step(early_step, current_risk, target_risk, drift, variance, beta):
    gap = current_risk - target_risk
    if gap <= 0:
        return float(early_step)
    if drift <= 0:
        return math.inf
    variance = max(float(variance), 0.0)
    z = math.sqrt(2.0 * math.log(1.0 / beta))
    b = z * math.sqrt(variance)
    x = (b + math.sqrt(b * b + 4.0 * drift * gap)) / (2.0 * drift)
    return float(early_step + x * x)


CHECKPOINT_FIELDS = [
    "task", "rho", "seed", "step", "train_loss", "answer_accuracy",
    "exact_trace_accuracy", "trace_step_accuracy", "colon_rate",
    "clean_state_nll", "clean_state_nll_std", "mean_state_margin",
    "median_state_margin", "target_quantile_margin", "tail_mean_margin",
    "positive_state_margin_rate",
]

PROGRESS_FIELDS = [
    "task", "rho", "seed", "step", "clean_state_nll_before",
    "clean_state_nll_after", "risk_progress", "mean_state_margin_before",
    "mean_state_margin_after", "mean_margin_progress",
    "target_quantile_margin_before", "target_quantile_margin_after",
    "target_quantile_margin_progress", "positive_state_margin_rate_before",
    "positive_state_margin_rate_after",
]

SUMMARY_FIELDS = [
    "task", "rho", "seed", "depth", "target_exact",
    "required_local_accuracy", "tail_fraction", "target_clean_state_nll",
    "early_end", "num_early_progress_samples", "early_clean_state_nll",
    "early_mean_state_margin", "early_target_quantile_margin",
    "early_positive_state_margin_rate", "a_hat_risk_progress",
    "v_hat_risk_progress", "risk_progress_snr", "mean_margin_progress",
    "v_margin_progress", "predicted_acquisition_step_drift",
    "predicted_acquisition_step_noise_adjusted",
    "predicted_acquire_by_horizon_drift",
    "predicted_acquire_by_horizon_noise_adjusted",
    "observed_acquisition_step", "observed_acquire_by_horizon",
]


def train_one(task, dataset, val_instances, diagnostic_queries, rho, seed, args, device, checkpoint_csv, progress_csv):
    set_seed(seed)
    generator = torch.Generator().manual_seed(args.batch_seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, generator=generator,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)
    model = build_model(task, args, device)
    optimizer = make_optimizer(model, args, device)
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    eval_steps = sorted(set(args.checkpoints + [args.early_end]))
    max_step = max(eval_steps)
    checkpoint_records, progress_records = [], []
    model.train()

    pbar = tqdm(range(1, max_step + 1), desc=f"{task.name}/rho={rho:.3f}/seed={seed}")
    for step in pbar:
        try:
            x, y, mask = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, mask = next(iterator)

        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        mask = mask.to(device, dtype=torch.float32, non_blocking=True)

        measure_progress = args.progress_start <= step <= args.early_end and step % args.progress_every == 0
        if measure_progress:
            before = summarize_scores(
                score_clean_states(model, diagnostic_queries, args.diagnostic_batch_size, device),
                args.tail_fraction,
            )

        optimizer.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y, mask=mask)
        else:
            _, loss = model(x, targets=y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if measure_progress:
            after = summarize_scores(
                score_clean_states(model, diagnostic_queries, args.diagnostic_batch_size, device),
                args.tail_fraction,
            )
            row = {
                "task": task.name, "rho": rho, "seed": seed, "step": step,
                "clean_state_nll_before": before["clean_state_nll"],
                "clean_state_nll_after": after["clean_state_nll"],
                "risk_progress": before["clean_state_nll"] - after["clean_state_nll"],
                "mean_state_margin_before": before["mean_state_margin"],
                "mean_state_margin_after": after["mean_state_margin"],
                "mean_margin_progress": after["mean_state_margin"] - before["mean_state_margin"],
                "target_quantile_margin_before": before["target_quantile_margin"],
                "target_quantile_margin_after": after["target_quantile_margin"],
                "target_quantile_margin_progress": after["target_quantile_margin"] - before["target_quantile_margin"],
                "positive_state_margin_rate_before": before["positive_state_margin_rate"],
                "positive_state_margin_rate_after": after["positive_state_margin_rate"],
            }
            append_csv(progress_csv, PROGRESS_FIELDS, row)
            progress_records.append(row)

        if step in eval_steps:
            metrics = evaluate(model, task, val_instances, "mixed_process", args, device)
            clean = summarize_scores(
                score_clean_states(model, diagnostic_queries, args.diagnostic_batch_size, device),
                args.tail_fraction,
            )
            row = {
                "task": task.name, "rho": rho, "seed": seed, "step": step,
                "train_loss": float(loss.detach()),
                "answer_accuracy": metrics["answer_accuracy"],
                "exact_trace_accuracy": metrics["exact_trace_accuracy"],
                "trace_step_accuracy": metrics["trace_step_accuracy"],
                "colon_rate": metrics["colon_rate"],
                **clean,
            }
            append_csv(checkpoint_csv, CHECKPOINT_FIELDS, row)
            checkpoint_records.append(row)
            pbar.write(
                f"rho={rho:.3f} seed={seed} step={step} "
                f"ans={100*metrics['answer_accuracy']:.1f}% "
                f"exact={100*metrics['exact_trace_accuracy']:.1f}% "
                f"H={clean['clean_state_nll']:.4f} margin={clean['mean_state_margin']:.4f}"
            )
            model.train()

    early_checkpoint = next(r for r in checkpoint_records if r["step"] == args.early_end)
    early_progress = [r for r in progress_records if r["step"] <= args.early_end]
    risk_progress = np.asarray([r["risk_progress"] for r in early_progress], dtype=np.float64)
    margin_progress = np.asarray([r["mean_margin_progress"] for r in early_progress], dtype=np.float64)

    a_hat = float(risk_progress.mean()) if len(risk_progress) else math.nan
    v_hat = float(risk_progress.var(ddof=1)) if len(risk_progress) > 1 else 0.0
    margin_a = float(margin_progress.mean()) if len(margin_progress) else math.nan
    margin_v = float(margin_progress.var(ddof=1)) if len(margin_progress) > 1 else 0.0
    snr = a_hat / math.sqrt(v_hat) if v_hat > 0 else (math.inf if a_hat > 0 else 0.0)

    pred_drift = drift_crossing_step(
        args.early_end, early_checkpoint["clean_state_nll"],
        args.target_clean_state_nll, a_hat,
    )
    pred_noise = noise_crossing_step(
        args.early_end, early_checkpoint["clean_state_nll"],
        args.target_clean_state_nll, a_hat, v_hat, args.beta,
    )

    observed = math.inf
    for r in sorted(checkpoint_records, key=lambda z: z["step"]):
        if r["step"] >= args.early_end and r["exact_trace_accuracy"] >= args.target_exact:
            observed = float(r["step"])
            break

    horizon = float(max(eval_steps))
    summary = {
        "task": task.name, "rho": rho, "seed": seed, "depth": args.depth,
        "target_exact": args.target_exact,
        "required_local_accuracy": args.required_local_accuracy,
        "tail_fraction": args.tail_fraction,
        "target_clean_state_nll": args.target_clean_state_nll,
        "early_end": args.early_end,
        "num_early_progress_samples": len(early_progress),
        "early_clean_state_nll": early_checkpoint["clean_state_nll"],
        "early_mean_state_margin": early_checkpoint["mean_state_margin"],
        "early_target_quantile_margin": early_checkpoint["target_quantile_margin"],
        "early_positive_state_margin_rate": early_checkpoint["positive_state_margin_rate"],
        "a_hat_risk_progress": a_hat,
        "v_hat_risk_progress": v_hat,
        "risk_progress_snr": snr,
        "mean_margin_progress": margin_a,
        "v_margin_progress": margin_v,
        "predicted_acquisition_step_drift": finite_or_blank(pred_drift),
        "predicted_acquisition_step_noise_adjusted": finite_or_blank(pred_noise),
        "predicted_acquire_by_horizon_drift": int(pred_drift <= horizon),
        "predicted_acquire_by_horizon_noise_adjusted": int(pred_noise <= horizon),
        "observed_acquisition_step": finite_or_blank(observed),
        "observed_acquire_by_horizon": int(observed <= horizon),
    }

    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def report_prediction_quality(summaries):
    print("\n" + "=" * 72)
    print("EARLY -> LATE PREDICTION SUMMARY")
    print("=" * 72)
    obs = np.asarray([r["observed_acquire_by_horizon"] for r in summaries], dtype=np.int64)

    for field, name in [
        ("predicted_acquire_by_horizon_drift", "drift-only"),
        ("predicted_acquire_by_horizon_noise_adjusted", "noise-adjusted"),
    ]:
        pred = np.asarray([r[field] for r in summaries], dtype=np.int64)
        tp = int(((pred == 1) & (obs == 1)).sum())
        tn = int(((pred == 0) & (obs == 0)).sum())
        fp = int(((pred == 1) & (obs == 0)).sum())
        fn = int(((pred == 0) & (obs == 1)).sum())
        acc = (tp + tn) / max(len(pred), 1)
        print(f"{name:<18} acc={acc:.3f} TP={tp} TN={tn} FP={fp} FN={fn}")

    for field, name in [
        ("predicted_acquisition_step_drift", "drift-only"),
        ("predicted_acquisition_step_noise_adjusted", "noise-adjusted"),
    ]:
        pairs = [(float(r[field]), float(r["observed_acquisition_step"])) for r in summaries if r[field] != "" and r["observed_acquisition_step"] != ""]
        if len(pairs) >= 2:
            p = np.asarray([x[0] for x in pairs])
            o = np.asarray([x[1] for x in pairs])
            corr = float(np.corrcoef(p, o)[0, 1])
            mae = float(np.abs(p - o).mean())
            print(f"{name:<18} finite_pairs={len(pairs)} Pearson_r={corr:.3f} MAE={mae:.1f} updates")

    print("\nDo not tune these predictor thresholds using the same late outcomes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="boolean_circuit_8")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000])

    parser.add_argument("--early-end", type=int, default=2000)
    parser.add_argument("--progress-start", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--target-exact", type=float, default=0.80)
    parser.add_argument("--beta", type=float, default=0.10)

    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--diagnostic-size", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--diagnostic-seed", type=int, default=303)
    parser.add_argument("--ratio-seed", type=int, default=777)
    parser.add_argument("--batch-seed", type=int, default=12345)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        parser.error("Every rho must lie in [0,1].")
    if not 0.0 < args.target_exact < 1.0:
        parser.error("--target-exact must lie in (0,1).")
    if not 0.0 < args.beta < 1.0:
        parser.error("--beta must lie in (0,1).")
    if args.progress_start > args.early_end:
        parser.error("--progress-start cannot exceed --early-end.")

    task = TASKS[args.task]
    if not (task.name.startswith("state_machine_") or task.name.startswith("register_machine_") or task.name.startswith("boolean_circuit_")):
        parser.error("Supported tasks: state_machine_*, register_machine_*, boolean_circuit_*.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {inst.prompt for inst in train_instances}
    val_instances = generate_unique(task, args.val_size, args.val_seed, train_prompts)
    excluded = train_prompts | {inst.prompt for inst in val_instances}
    diagnostic_instances = generate_unique(task, args.diagnostic_size, args.diagnostic_seed, excluded)

    depths = {len(inst.correct_trace.split()) for inst in train_instances}
    if len(depths) != 1:
        raise ValueError("Expected fixed trace depth.")
    args.depth = depths.pop()
    args.required_local_accuracy = args.target_exact ** (1.0 / args.depth)
    args.tail_fraction = 1.0 - args.required_local_accuracy
    args.target_clean_state_nll = args.tail_fraction * math.log(2.0)

    diagnostic_queries = build_state_queries(task, diagnostic_instances)
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("results") / f"{task.name}_early_prediction_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_csv = output_dir / "checkpoints.csv"
    progress_csv = output_dir / "early_progress.csv"
    summary_csv = output_dir / "prediction_summary.csv"

    print("=" * 72)
    print("EARLY ACQUISITION PREDICTION")
    print("=" * 72)
    print(f"task={task.name} depth={args.depth} device={device}")
    print(f"train={len(train_instances)} val={len(val_instances)} diagnostic={len(diagnostic_instances)}")
    print(f"early_end={args.early_end} progress_every={args.progress_every}")
    print(f"target_exact={args.target_exact:.3f} required_local={args.required_local_accuracy:.5f}")
    print(f"tail_fraction={args.tail_fraction:.5f} H_target={args.target_clean_state_nll:.6f}")
    print(f"output={output_dir}")

    summaries = []
    for rho in sorted(set(args.rhos)):
        dataset = RatioDataset(train_instances, task, "mixed_process", rho=rho, ratio_scores=ratio_scores)
        for seed in args.seeds:
            summary = train_one(
                task, dataset, val_instances, diagnostic_queries, rho, seed,
                args, device, checkpoint_csv, progress_csv,
            )
            append_csv(summary_csv, SUMMARY_FIELDS, summary)
            summaries.append(summary)
            print(
                f"SUMMARY rho={rho:.3f} seed={seed} "
                f"a_hat={summary['a_hat_risk_progress']:.3e} "
                f"v_hat={summary['v_hat_risk_progress']:.3e} "
                f"pred_noise={summary['predicted_acquisition_step_noise_adjusted']} "
                f"observed={summary['observed_acquisition_step']}"
            )
        del dataset

    report_prediction_quality(summaries)
    print("\nSaved:")
    print(f"  {checkpoint_csv}")
    print(f"  {progress_csv}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()