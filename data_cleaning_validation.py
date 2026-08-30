
import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.registry import TASKS
from sweep_ratio import (
    RatioDataset,
    build_model,
    evaluate,
    generate_unique,
    make_optimizer,
    set_seed,
)
from early_acquisition_prediction import build_state_queries, score_clean_states




def atomic_json_dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def pava_increasing(y, w=None):
    """Weighted pool-adjacent-violators algorithm for isotonic regression."""
    y = np.asarray(y, dtype=np.float64)
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, dtype=np.float64)
    if len(y) != len(w):
        raise ValueError("y and w must have the same length")

    blocks = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, wi, yi])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            a, b = blocks[-2], blocks[-1]
            weight = a[2] + b[2]
            mean = (a[2] * a[3] + b[2] * b[3]) / weight
            blocks[-2:] = [[a[0], b[1], weight, mean]]

    fitted = np.empty_like(y)
    for start, stop, _, mean in blocks:
        fitted[start:stop + 1] = mean
    return fitted


def inverse_monotone(x, y, target):
    """
    Invert a nondecreasing curve using linear interpolation.

    This is a continuous interpolation of the generalized inverse. It is more
    useful than a coarse grid crossing when predicting a filtering fraction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if np.any(np.diff(y) < -1e-12):
        raise ValueError("y must be nondecreasing")
    if target <= y[0]:
        return float(x[0])
    if target > y[-1]:
        return math.inf

    j = int(np.searchsorted(y, target, side="left"))
    if j == 0:
        return float(x[0])
    if y[j] == y[j - 1]:
        return float(x[j])
    alpha = (target - y[j - 1]) / (y[j] - y[j - 1])
    return float(x[j - 1] + alpha * (x[j] - x[j - 1]))


def effective_rho(rho0, f):
    denom = rho0 + (1.0 - f) * (1.0 - rho0)
    return rho0 / denom if denom > 0 else 1.0


def required_filter_fraction(rho0, rho_target):
    if rho_target <= rho0:
        return 0.0
    if not math.isfinite(rho_target) or rho_target >= 1.0:
        return 1.0 if rho_target <= 1.0 + 1e-12 else math.inf
    value = 1.0 - rho0 * (1.0 - rho_target) / (rho_target * (1.0 - rho0))
    return float(np.clip(value, 0.0, 1.0))


def local_barrier(rho, k):
    """
    Exact barrier for matched contexts, Q(y*)=0, and K equally likely
    corrupted competitors. Here q_max=1/K:
        a = rho
        b = (1-rho)/K.
    """
    a = float(rho)
    b = float((1.0 - rho) / k)
    if a <= b:
        return 0.0
    terms = 0.0
    if a > 0:
        terms += a * math.log(2.0 * a / (a + b))
    if b > 0:
        terms += b * math.log(2.0 * b / (a + b))
    return terms


def analytic_theory_checks(target_exact, depth, rho0=None, rho_target=None):
    checks = {}

    # Population competitor-concentration boundaries.
    boundaries = {str(k): 1.0 / (k + 1.0) for k in (1, 3, 15)}
    checks["competitor_boundaries"] = boundaries

    # Local error needed for target exact rollout under homogeneous composition.
    eps = 1.0 - target_exact ** (1.0 / depth)
    checks["target_exact"] = target_exact
    checks["depth"] = depth
    checks["required_local_accuracy"] = target_exact ** (1.0 / depth)
    checks["required_local_error"] = eps
    checks["composition_reconstruction"] = (1.0 - eps) ** depth

    # Numerically verify the quadratic barrier exponent near each population tie.
    slopes = {}
    for k in (1, 3, 15):
        rho_star = 1.0 / (k + 1.0)
        max_eps = min(1e-2, (1.0 - rho_star) / 10.0)
        offsets = np.geomspace(max_eps / 100.0, max_eps, 8)
        values = np.asarray([local_barrier(rho_star + d, k) for d in offsets])
        valid = values > 0
        slope = np.polyfit(np.log(offsets[valid]), np.log(values[valid]), 1)[0]
        slopes[str(k)] = float(slope)
        if not (1.85 <= slope <= 2.15):
            raise AssertionError(f"Barrier did not look quadratic for K={k}: slope={slope:.4f}")
    checks["quadratic_barrier_loglog_slopes"] = slopes

    if rho0 is not None and rho_target is not None and math.isfinite(rho_target):
        fstar = required_filter_fraction(rho0, rho_target)
        if math.isfinite(fstar):
            reconstructed = effective_rho(rho0, fstar)
            checks["filter_formula"] = {
                "rho0": rho0,
                "rho_target": rho_target,
                "f_star": fstar,
                "rho_reconstructed": reconstructed,
                "absolute_error": abs(reconstructed - rho_target),
            }
            if abs(reconstructed - rho_target) > 1e-9 and rho_target < 1.0:
                raise AssertionError("Filtering formula consistency check failed")
    return checks


def score_local_margins(model, queries, batch_size, device):
    scores = score_clean_states(model, queries, batch_size, device)
    margins = scores["state_margin"]
    return {
        "positive_local_margin_rate": float(np.mean(margins > 0.0)),
        "mean_local_margin": float(np.mean(margins)),
        "median_local_margin": float(np.median(margins)),
        "p10_local_margin": float(np.quantile(margins, 0.10)),
        "p05_local_margin": float(np.quantile(margins, 0.05)),
    }


def train_final_model(task, dataset, val_instances, diagnostic_queries, seed, args, device, desc):
    set_seed(seed)
    generator = torch.Generator().manual_seed(args.batch_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)
    model = build_model(task, args, device)
    optimizer = make_optimizer(model, args, device)
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    model.train()
    final_loss = math.nan
    pbar = tqdm(range(1, args.steps + 1), desc=desc)
    for step in pbar:
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
        final_loss = float(loss.detach())

        if step % max(args.steps // 10, 1) == 0:
            pbar.set_postfix(loss=f"{final_loss:.4f}")

    free = evaluate(model, task, val_instances, "mixed_process", args, device)
    local = score_local_margins(model, diagnostic_queries, args.diagnostic_batch_size, device)
    depth = len(val_instances[0].correct_trace.split())
    local_predicted_exact = local["positive_local_margin_rate"] ** depth

    result = {
        "seed": seed,
        "step": args.steps,
        "train_loss": final_loss,
        "answer_accuracy": float(free["answer_accuracy"]),
        "exact_trace_accuracy": float(free["exact_trace_accuracy"]),
        "trace_step_accuracy": float(free["trace_step_accuracy"]),
        "colon_rate": float(free["colon_rate"]),
        **local,
        "local_predicted_exact": float(local_predicted_exact),
    }

    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_fixed_data(task, args):
    train_instances = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {inst.prompt for inst in train_instances}
    val_instances = generate_unique(task, args.val_size, args.val_seed, train_prompts)
    diagnostic_instances = val_instances[: min(args.diagnostic_size, len(val_instances))]
    diagnostic_queries = build_state_queries(task, diagnostic_instances)
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))
    return train_instances, val_instances, diagnostic_queries, ratio_scores


def calibration_paths(run_dir):
    return (
        run_dir / "calibration.csv",
        run_dir / "prediction.json",
        run_dir / "calibration.png",
    )


def run_calibration(task, args, device, run_dir):
    calibration_csv, prediction_json, calibration_png = calibration_paths(run_dir)
    if calibration_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{calibration_csv} exists; pass --overwrite to replace")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_instances, val_instances, diagnostic_queries, ratio_scores = make_fixed_data(task, args)
    depth = len(val_instances[0].correct_trace.split())
    rows = []

    for rho in sorted(set(args.calibration_rhos)):
        dataset = RatioDataset(
            train_instances,
            task,
            "mixed_process",
            rho=rho,
            ratio_scores=ratio_scores,
        )
        for seed in args.seeds:
            result = train_final_model(
                task, dataset, val_instances, diagnostic_queries, seed, args, device,
                desc=f"CAL rho={rho:.3f} seed={seed}",
            )
            row = {
                "task": task.name,
                "rho": rho,
                "depth": depth,
                "train_size": len(train_instances),
                "val_size": len(val_instances),
                "diagnostic_transitions": len(diagnostic_queries),
                **result,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(calibration_csv, index=False)
        del dataset

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("rho", as_index=False)
        .agg(
            mean_local=("positive_local_margin_rate", "mean"),
            sd_local=("positive_local_margin_rate", "std"),
            mean_exact=("exact_trace_accuracy", "mean"),
            sd_exact=("exact_trace_accuracy", "std"),
            n=("seed", "count"),
        )
        .sort_values("rho")
    )

    x = grouped["rho"].to_numpy(float)
    y = grouped["mean_local"].to_numpy(float)
    weights = grouped["n"].to_numpy(float)
    y_iso = pava_increasing(y, weights)
    grouped["isotonic_local"] = y_iso

    required_local = args.target_exact ** (1.0 / depth)
    rho_target = inverse_monotone(x, y_iso, required_local)
    if not math.isfinite(rho_target):
        raise RuntimeError(
            f"Calibration never reaches required local rate {required_local:.5f}. "
            "Add higher calibration rhos or increase training budget."
        )

    if args.base_rho >= rho_target:
        print(
            f"WARNING: base_rho={args.base_rho:.4f} is already at/above predicted "
            f"target reliability {rho_target:.4f}; f*=0. Choose a lower base rho "
            "for a meaningful rescue experiment."
        )

    f_star = required_filter_fraction(args.base_rho, rho_target)
    analytic = analytic_theory_checks(
        args.target_exact, depth, rho0=args.base_rho, rho_target=rho_target
    )

    predicted = df["local_predicted_exact"].to_numpy(float)
    observed = df["exact_trace_accuracy"].to_numpy(float)
    calibration_mae = float(np.mean(np.abs(predicted - observed)))
    calibration_corr = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if np.std(predicted) > 0 and np.std(observed) > 0 else math.nan
    )

    prediction = {
        "created_at": datetime.now().isoformat(),
        "task": task.name,
        "depth": depth,
        "target_exact": args.target_exact,
        "required_local_accuracy": required_local,
        "predicted_required_rho": rho_target,
        "base_rho": args.base_rho,
        "predicted_filter_fraction": f_star,
        "formula": (
            "rho_f=rho0/[rho0+(1-f)(1-rho0)]; "
            "f*=1-rho0(1-rho_q)/[rho_q(1-rho0)]"
        ),
        "local_to_global_calibration_mae": calibration_mae,
        "local_to_global_calibration_correlation": calibration_corr,
        "calibration_rhos": [float(v) for v in x],
        "calibration_mean_local": [float(v) for v in y],
        "calibration_isotonic_local": [float(v) for v in y_iso],
        "calibration_mean_exact": [float(v) for v in grouped["mean_exact"]],
        "config": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "diagnostic_size": args.diagnostic_size,
            "train_seed": args.train_seed,
            "val_seed": args.val_seed,
            "ratio_seed": args.ratio_seed,
            "batch_seed": args.batch_seed,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "embedding": args.embedding,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "seeds": list(args.seeds),
        },
        "analytic_theory_checks": analytic,
    }
    atomic_json_dump(prediction_json, prediction)

    grouped.to_csv(run_dir / "calibration_aggregate.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, y, "o-", label="measured positive-margin rate")
    ax.plot(x, y_iso, "--", label="isotonic fit")
    ax.axhline(required_local, linestyle=":", label=f"required local={required_local:.3f}")
    ax.axvline(rho_target, linestyle=":", label=f"predicted rho*={rho_target:.3f}")
    ax.set_xlabel("trace reliability rho")
    ax.set_ylabel("positive local-margin rate")
    ax.set_title(f"{task.name}: frozen acquisition-frontier calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(calibration_png, dpi=180)
    plt.close(fig)

    print("\nFROZEN THEORY PREDICTION")
    print("=" * 72)
    print(f"task                    : {task.name}")
    print(f"depth D                 : {depth}")
    print(f"target exact rollout q  : {args.target_exact:.3f}")
    print(f"required local q^(1/D)  : {required_local:.5f}")
    print(f"predicted required rho  : {rho_target:.5f}")
    print(f"base rho0               : {args.base_rho:.5f}")
    print(f"predicted filtering f*  : {f_star:.5f}")
    print(f"calibration MAE         : {100*calibration_mae:.2f} percentage points")
    print(f"calibration correlation : {calibration_corr:.5f}")
    print(f"prediction frozen at    : {prediction_json}")
    print("=" * 72)
    return prediction_json


def load_prediction_and_lock_args(path: Path, args):
    prediction = json.loads(path.read_text())
    cfg = prediction["config"]

    # Ensure intervention reconstructs exactly the calibration data/model protocol.
    args.task = prediction["task"]
    args.target_exact = float(prediction["target_exact"])
    args.base_rho = float(prediction["base_rho"])
    for key in (
        "train_size", "val_size", "diagnostic_size", "train_seed", "val_seed",
        "ratio_seed", "batch_seed", "batch_size", "steps", "lr",
        "weight_decay", "grad_clip", "embedding", "heads", "layers", "dropout",
    ):
        setattr(args, key, cfg[key])
    if not args.intervention_seeds:
        args.intervention_seeds = list(cfg["seeds"])
    return prediction


def nested_filter_orders(clean_flags, seed):
    rng = np.random.default_rng(seed)
    corrupt = np.flatnonzero(~clean_flags)
    all_indices = np.arange(len(clean_flags))
    return rng.permutation(corrupt), rng.permutation(all_indices)


def build_filtered_subset(train_instances, ratio_scores, rho0, mode, f, corrupt_order, random_order):
    clean_flags = ratio_scores < rho0
    n_corrupt = int((~clean_flags).sum())
    n_remove = int(round(f * n_corrupt))

    if mode == "targeted":
        removed = corrupt_order[:n_remove]
    elif mode == "random":
        removed = random_order[:n_remove]
    else:
        raise ValueError(mode)

    keep_mask = np.ones(len(train_instances), dtype=bool)
    keep_mask[removed] = False
    keep = np.flatnonzero(keep_mask)

    subset_instances = [train_instances[i] for i in keep]
    subset_scores = ratio_scores[keep]
    realized_rho = float(np.mean(subset_scores < rho0))
    return subset_instances, subset_scores, realized_rho, n_remove


def auto_filter_fractions(f_star):
    anchors = [0.0, 0.25, 0.50, 0.75]
    if math.isfinite(f_star):
        anchors += [
            max(0.0, f_star - 0.15),
            max(0.0, f_star - 0.075),
            f_star,
            min(1.0, f_star + 0.075),
            min(1.0, f_star + 0.15),
        ]
    return sorted({round(float(np.clip(v, 0.0, 1.0)), 6) for v in anchors})


def observed_crossing(df, mode, target):
    g = (
        df[df["filter_mode"] == mode]
        .groupby("filter_fraction", as_index=False)["exact_trace_accuracy"]
        .mean()
        .sort_values("filter_fraction")
    )
    passed = g[g["exact_trace_accuracy"] >= target]
    return math.inf if passed.empty else float(passed.iloc[0]["filter_fraction"])


def run_intervention(prediction_path: Path, args, device, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction = load_prediction_and_lock_args(prediction_path, args)
    task = TASKS[args.task]

    f_star = float(prediction["predicted_filter_fraction"])
    rho_target = float(prediction["predicted_required_rho"])
    depth = int(prediction["depth"])

    train_instances, val_instances, diagnostic_queries, ratio_scores = make_fixed_data(task, args)
    clean_flags = ratio_scores < args.base_rho
    corrupt_order, random_order = nested_filter_orders(clean_flags, args.filter_seed)

    fractions = (
        sorted(set(args.filter_fractions))
        if args.filter_fractions
        else auto_filter_fractions(f_star)
    )
    seeds = args.intervention_seeds
    if not seeds:
        raise ValueError("No intervention seeds specified")

    intervention_csv = run_dir / "intervention.csv"
    if intervention_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{intervention_csv} exists; pass --overwrite to replace")

    rows = []
    for f in fractions:
        if not 0.0 <= f <= 1.0:
            raise ValueError("Every filter fraction must lie in [0,1]")

        for mode in ("targeted", "random"):
            subset_instances, subset_scores, realized_rho, n_remove = build_filtered_subset(
                train_instances, ratio_scores, args.base_rho, mode, f,
                corrupt_order, random_order,
            )
            dataset = RatioDataset(
                subset_instances,
                task,
                "mixed_process",
                rho=args.base_rho,
                ratio_scores=subset_scores,
            )
            theory_rho = (
                effective_rho(args.base_rho, f) if mode == "targeted"
                else args.base_rho
            )

            for seed in seeds:
                result = train_final_model(
                    task, dataset, val_instances, diagnostic_queries, seed, args, device,
                    desc=f"INT {mode} f={f:.3f} seed={seed}",
                )
                row = {
                    "task": task.name,
                    "filter_mode": mode,
                    "filter_fraction": f,
                    "predicted_filter_fraction": f_star,
                    "target_exact": args.target_exact,
                    "base_rho": args.base_rho,
                    "theory_effective_rho": theory_rho,
                    "realized_rho": realized_rho,
                    "num_original": len(train_instances),
                    "num_removed": n_remove,
                    "num_retained": len(subset_instances),
                    "depth": depth,
                    **result,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(intervention_csv, index=False)
            del dataset

    df = pd.DataFrame(rows)
    predicted = df["local_predicted_exact"].to_numpy(float)
    observed = df["exact_trace_accuracy"].to_numpy(float)
    intervention_mae = float(np.mean(np.abs(predicted - observed)))
    intervention_corr = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if np.std(predicted) > 0 and np.std(observed) > 0 else math.nan
    )

    targeted_cross = observed_crossing(df, "targeted", args.target_exact)
    random_cross = observed_crossing(df, "random", args.target_exact)

    targeted = df[df["filter_mode"] == "targeted"]
    eff_rho_mae = float(
        np.mean(np.abs(targeted["realized_rho"] - targeted["theory_effective_rho"]))
    )

    summary = {
        "prediction_file": str(prediction_path),
        "task": task.name,
        "target_exact": args.target_exact,
        "base_rho": args.base_rho,
        "predicted_required_rho": rho_target,
        "predicted_filter_fraction": f_star,
        "tested_filter_fractions": fractions,
        "observed_targeted_crossing": None if not math.isfinite(targeted_cross) else targeted_cross,
        "observed_random_crossing": None if not math.isfinite(random_cross) else random_cross,
        "targeted_crossing_absolute_error": (
            None if not math.isfinite(targeted_cross)
            else abs(targeted_cross - f_star)
        ),
        "intervention_local_to_global_mae": intervention_mae,
        "intervention_local_to_global_correlation": intervention_corr,
        "targeted_effective_rho_formula_mae": eff_rho_mae,
        "analytic_theory_checks": analytic_theory_checks(
            args.target_exact, depth, args.base_rho, rho_target
        ),
    }
    atomic_json_dump(run_dir / "intervention_summary.json", summary)

    aggregate = (
        df.groupby(["filter_mode", "filter_fraction"], as_index=False)
        .agg(
            mean_exact=("exact_trace_accuracy", "mean"),
            sd_exact=("exact_trace_accuracy", "std"),
            mean_local=("positive_local_margin_rate", "mean"),
            mean_local_predicted_exact=("local_predicted_exact", "mean"),
            mean_realized_rho=("realized_rho", "mean"),
            n=("seed", "count"),
        )
    )
    aggregate.to_csv(run_dir / "intervention_aggregate.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mode in ("targeted", "random"):
        g = aggregate[aggregate["filter_mode"] == mode].sort_values("filter_fraction")
        ax.errorbar(
            g["filter_fraction"], g["mean_exact"], yerr=g["sd_exact"],
            marker="o", capsize=3, label=f"{mode} filtering",
        )
    ax.axhline(args.target_exact, linestyle=":", label=f"target={args.target_exact:.2f}")
    ax.axvline(f_star, linestyle="--", label=f"pre-registered f*={f_star:.3f}")
    ax.set_xlabel("fraction of corrupted examples removed (same count for random control)")
    ax.set_ylabel("exact rollout accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{task.name}: theory-predicted data-cleaning rescue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "filtering_rescue.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mode in ("targeted", "random"):
        g = aggregate[aggregate["filter_mode"] == mode].sort_values("filter_fraction")
        ax.plot(g["filter_fraction"], g["mean_local"], marker="o", label=f"{mode}: local margin +")
        ax.plot(
            g["filter_fraction"], g["mean_exact"], marker="x", linestyle="--",
            label=f"{mode}: exact rollout",
        )
    ax.axvline(f_star, linestyle=":", label=f"predicted f*={f_star:.3f}")
    ax.set_xlabel("filter fraction")
    ax.set_ylabel("accuracy / positive-margin rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Local acquisition versus global execution")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "local_vs_global_filtering.png", dpi=180)
    plt.close(fig)

    print("\nINTERVENTION VALIDATION")
    print("=" * 72)
    print(f"frozen predicted f*           : {f_star:.5f}")
    print(f"targeted observed crossing    : {targeted_cross if math.isfinite(targeted_cross) else 'not reached'}")
    print(f"random observed crossing      : {random_cross if math.isfinite(random_cross) else 'not reached'}")
    print(f"local^D -> rollout MAE        : {100*intervention_mae:.2f} percentage points")
    print(f"local^D -> rollout correlation: {intervention_corr:.5f}")
    print(f"effective-rho formula MAE     : {eff_rho_mae:.6f}")
    print(f"saved                         : {run_dir}")
    print("=" * 72)


def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--stage", choices=["calibrate", "intervene", "all"], default="calibrate")
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument(
        "--calibration-rhos", nargs="+", type=float,
        default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00],
    )
    p.add_argument("--base-rho", type=float, default=0.80)
    p.add_argument("--target-exact", type=float, default=0.70)
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003, 2004, 2005])
    p.add_argument(
        "--intervention-seeds", nargs="*", type=int, default=None,
        help="Defaults to calibration seeds recorded in the frozen prediction.",
    )
    p.add_argument(
        "--filter-fractions", nargs="*", type=float, default=None,
        help="If omitted, points are placed automatically around frozen f*.",
    )
    p.add_argument("--filter-seed", type=int, default=9191)

    p.add_argument("--train-size", type=int, default=100_000)
    p.add_argument("--val-size", type=int, default=2_000)
    p.add_argument("--diagnostic-size", type=int, default=1_000)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--ratio-seed", type=int, default=777)
    p.add_argument("--batch-seed", type=int, default=12345)

    p.add_argument("--steps", type=int, default=8_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--diagnostic-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true")

    p.add_argument("--prediction", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/data_cleaning"))
    p.add_argument("--run-name", type=str)
    p.add_argument("--overwrite", action="store_true")
    return p


def validate_args(args):
    if not 0.0 < args.target_exact < 1.0:
        raise ValueError("--target-exact must lie in (0,1)")
    if not 0.0 < args.base_rho < 1.0:
        raise ValueError("--base-rho must lie in (0,1)")
    if any(not 0.0 <= r <= 1.0 for r in args.calibration_rhos):
        raise ValueError("All calibration rhos must lie in [0,1]")
    if args.filter_fractions and any(not 0.0 <= f <= 1.0 for f in args.filter_fractions):
        raise ValueError("All filter fractions must lie in [0,1]")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.task}_{timestamp}"
    run_dir = args.output_dir / run_name

    if args.stage in {"calibrate", "all"}:
        task = TASKS[args.task]
        prediction_path = run_calibration(task, args, device, run_dir)
        if args.stage == "calibrate":
            return
    else:
        if args.prediction is None:
            parser.error("--stage intervene requires --prediction PATH")
        prediction_path = args.prediction
        if args.run_name is None:
            run_dir = prediction_path.parent / "intervention"

    if args.stage in {"intervene", "all"}:
        run_intervention(prediction_path, args, device, run_dir)


if __name__ == "__main__":
    main()