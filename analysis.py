"""Everything that happens after the models are trained.

Analysis reads the results store, never the GPU: fits, tables and figures are
reproduced from `results/*.jsonl` alone, so a plot can be redrawn or a threshold
refitted months later without touching a training run.

The object of interest is the critical ratio rho_c: the location of the
transition in

    acc(rho) = a_lo + (a_hi - a_lo) / (1 + exp(-(rho - rho_c)/tau))

with a_lo and a_hi *pinned* to the task's chance floor and Bayes ceiling, so the
only free parameters are where the transition sits and how wide it is. A fit is
reported as unreliable -- and callers refuse to quote it -- when the curve never
left the floor or the fitted threshold ran off the ratio axis.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tasks import get_task  # noqa: E402

# A fit is only trusted if the curve actually contains a transition: it must
# rise by at least this fraction of the task's chance->ceiling span, and the
# fitted threshold must land near the [0, 1] the ratio axis lives on.
MIN_RISE_FRACTION = 0.15
RHO_C_BOUNDS = (-0.25, 1.25)

# |rho_c(B) - rho_c(A)| below this counts as "the transition persists".
SHIFT_TOLERANCE = 0.10

ACCURACY_METRICS = ("free_acc", "forced_acc", "forced_wrong_acc", "answer_tf_acc")

METRIC_LABEL = {
    "free_acc": "end-to-end (free generation)",
    "forced_acc": "answer forcing (clean trace)",
    "forced_wrong_acc": "answer forcing (corrupted trace)",
    "answer_tf_acc": "teacher-forced answer match",
    "answer_nll": "answer NLL",
}

TASK_COLORS = ["#1a73e8", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
TASK_MARKERS = ["o", "s", "^", "D", "v", "P"]
METRIC_STYLE = {
    "free_acc": ("#1a73e8", "o"),
    "forced_acc": ("#2ca02c", "s"),
    "forced_wrong_acc": ("#d62728", "^"),
    "answer_tf_acc": ("#9467bd", "D"),
}
COND_STYLE = {
    "A": ("#1a73e8", "o", "L_full (full supervision)"),
    "B": ("#d62728", "s", "L_ans (answer-only loss)"),
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def group_by(records: Iterable[dict], keys: Sequence[str]) -> Dict[Tuple, List[dict]]:
    out: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(out)


def curve(records: Sequence[dict], metric: str) -> Dict[str, List[float]]:
    """Mean +/- std of `metric` against rho, over seeds. Sorted by rho."""
    by_ratio: Dict[float, List[float]] = defaultdict(list)
    for r in records:
        value = r.get(metric)
        if r.get("ratio") is not None and value is not None:
            by_ratio[float(r["ratio"])].append(float(value))

    ratios = sorted(by_ratio)
    means = [float(np.mean(by_ratio[r])) for r in ratios]
    stds = [float(np.std(by_ratio[r])) for r in ratios]
    return {"ratios": ratios, "means": means, "stds": stds,
            "n_seeds": [len(by_ratio[r]) for r in ratios]}


def baseline(records: Sequence[dict], metric: str = "free_acc") -> Optional[Tuple[float, float]]:
    """(mean, std) of the ratio-free runs -- condition C -- or None."""
    values = [
        float(r[metric]) for r in records if r.get("ratio") is None and r.get(metric) is not None
    ]
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values))


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


def _logistic(rho, rho_c, tau, a_lo, a_hi):
    return a_lo + (a_hi - a_lo) / (1.0 + np.exp(-(rho - rho_c) / np.abs(tau)))


def fit_threshold(ratios, means, a_lo: float, a_hi: float) -> Tuple[float, float]:
    """Least-squares (rho_c, tau) with the two plateaus pinned to the task."""
    x, y = np.asarray(ratios, float), np.asarray(means, float)
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            lambda r, rc, t: _logistic(r, rc, t, a_lo, a_hi), x, y, p0=[0.15, 0.05], maxfev=20000
        )
        return float(popt[0]), float(abs(popt[1]))
    except Exception as exc:  # noqa: BLE001 -- a failed fit is informational
        print(f"[warn] logistic fit failed ({exc}); falling back to interpolation")
        return interpolated_threshold(ratios, means, a_lo, a_hi), float("nan")


def interpolated_threshold(ratios, means, a_lo: float, a_hi: float) -> float:
    """Fit-free fallback: the first rho where the curve crosses the midpoint."""
    midpoint = 0.5 * (a_lo + a_hi)
    pairs = sorted(zip(ratios, means))
    for (r0, a0), (r1, a1) in zip(pairs, pairs[1:]):
        if a0 < midpoint <= a1:
            return r1 if a1 == a0 else r0 + (midpoint - a0) * (r1 - r0) / (a1 - a0)
    return float("nan")


def summarize_curve(task, records: Sequence[dict], metric: str = "free_acc") -> dict:
    """Fitted statistics for one ratio sweep, with an explicit reliability flag."""
    task = get_task(task)
    c = curve(records, metric)
    if not c["ratios"]:
        return {**c, "task": task.name, "metric": metric, "rho_c": float("nan"),
                "tau": float("nan"), "chance": task.chance_acc, "ceiling": task.ceiling_acc,
                "rise": 0.0, "reliable": False, "unreliable_reason": "no records"}

    rho_c, tau = fit_threshold(c["ratios"], c["means"], task.chance_acc, task.ceiling_acc)
    rise = max(c["means"]) - min(c["means"])
    span = task.ceiling_acc - task.chance_acc
    rose_enough = rise >= MIN_RISE_FRACTION * span
    in_range = not math.isnan(rho_c) and RHO_C_BOUNDS[0] <= rho_c <= RHO_C_BOUNDS[1]

    if not rose_enough:
        reason = f"curve is flat (rose {rise * 100:.1f}%, needs {MIN_RISE_FRACTION * span * 100:.1f}%)"
    elif not in_range:
        reason = f"fitted rho_c={rho_c:.3f} is off the ratio axis"
    else:
        reason = ""

    return {
        **c,
        "task": task.name,
        "metric": metric,
        "rho_c": rho_c,
        "tau": tau,
        "chance": task.chance_acc,
        "ceiling": task.ceiling_acc,
        "rise": rise,
        "reliable": rose_enough and in_range,
        "unreliable_reason": reason,
    }


def fit_scaling_exponent(batch_sizes, rho_cs) -> Tuple[float, float, float]:
    """log rho_c = alpha * log B + c. Returns (alpha, intercept, r^2).

    The SNR account predicts alpha = -1/2.
    """
    pairs = [(b, r) for b, r in zip(batch_sizes, rho_cs) if r == r and r > 0]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log(np.array([p[0] for p in pairs], float))
    y = np.log(np.array([p[1] for p in pairs], float))
    alpha, intercept = np.polyfit(x, y, 1)
    resid = y - (alpha * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(intercept), r2


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def print_curve_table(summaries: Sequence[dict], title: str = "PHASE TRANSITION") -> None:
    print("\n" + "=" * 86)
    print(f" {title}")
    print("=" * 86)
    print(f"{'task':<14} | {'metric':<18} | {'rho_c':>8} | {'tau':>7} | "
          f"{'chance':>7} | {'ceiling':>7} | fit")
    print("-" * 86)
    for s in summaries:
        note = "ok" if s["reliable"] else f"UNRELIABLE: {s['unreliable_reason']}"
        print(
            f"{s['task']:<14} | {s['metric']:<18} | {s['rho_c']:>8.4f} | {s['tau']:>7.4f} | "
            f"{s['chance']:>7.4f} | {s['ceiling']:>7.4f} | {note}"
        )
    print("=" * 86)


def print_ratio_table(summary: dict, title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"{'correct %':>10} | {'wrong %':>9} | {'mean':>10} | {'std':>9} | {'seeds':>5}")
    print("-" * 70)
    for rho, mean, std, n in zip(
        summary["ratios"], summary["means"], summary["stds"], summary["n_seeds"]
    ):
        print(
            f"{rho * 100:>9.1f}% | {(1 - rho) * 100:>8.1f}% | {mean * 100:>9.2f}% | "
            f"± {std * 100:>6.2f}% | {n:>5}"
        )
    print("=" * 70)


def print_forcing_table(task, records: Sequence[dict]) -> None:
    """What answer forcing adds: the same runs read four different ways.

    A gap between the end-to-end column and the forced column is generation
    overhead, not missing computation. A small gap between the two forced
    columns means the model is not conditioning on the trace it was handed.
    """
    task = get_task(task)
    curves = {m: curve(records, m) for m in ACCURACY_METRICS}
    nll = curve(records, "answer_nll")
    trunc = curve(records, "truncation_rate")
    ratios = curves["free_acc"]["ratios"]
    if not ratios:
        return

    print("\n" + "=" * 96)
    print(f" ANSWER FORCING -- {task.name}")
    print("=" * 96)
    print(
        f"{'rho':>7} | {'free':>9} | {'forced':>9} | {'forced(wrong)':>14} | "
        f"{'tf match':>9} | {'answer NLL':>10} | {'trunc':>7}"
    )
    print("-" * 96)
    for i, rho in enumerate(ratios):
        def at(c, j=i):
            return c["means"][j] if j < len(c["means"]) else float("nan")

        print(
            f"{rho * 100:>6.1f}% | {at(curves['free_acc']) * 100:>8.2f}% | "
            f"{at(curves['forced_acc']) * 100:>8.2f}% | {at(curves['forced_wrong_acc']) * 100:>13.2f}% | "
            f"{at(curves['answer_tf_acc']) * 100:>8.2f}% | {at(nll):>10.4f} | "
            f"{at(trunc) * 100:>6.2f}%"
        )
    print("=" * 96)


def print_ablation_verdicts(per_task: Dict[str, dict]) -> None:
    """Does the transition survive when the trace tokens carry no gradient?"""
    print("\n" + "=" * 88)
    print(" SUPERVISION ABLATION: FITTED THRESHOLDS AND VERDICT")
    print("=" * 88)
    print(f"{'task':<14} | {'rho_c(A)':>9} | {'rho_c(B)':>9} | {'shift':>8} | "
          f"{'L_direct':>9} | verdict")
    print("-" * 88)

    notes: List[str] = []
    for name, entry in per_task.items():
        a, b, direct = entry.get("A"), entry.get("B"), entry.get("direct")
        rho_a = a["rho_c"] if a else float("nan")
        rho_b = b["rho_c"] if b else float("nan")
        shift = rho_b - rho_a
        direct_str = f"{direct[0] * 100:8.2f}%" if direct else "        -"

        bad = [c for c, s in (("A", a), ("B", b)) if s and not s["reliable"]]
        if a is None or b is None:
            verdict = "incomplete (need both A and B)"
        elif bad:
            verdict = f"NO VERDICT -- unreliable fit in {', '.join(bad)}"
            notes += [f"   {name} condition {c}: {entry[c]['unreliable_reason']}" for c in bad]
        elif abs(shift) < SHIFT_TOLERANCE:
            verdict = "PERSISTS under L_ans -> trace-copy drift"
        else:
            verdict = "SHIFTS under L_ans -> supervision density causal"

        print(f"{name:<14} | {rho_a:>9.4f} | {rho_b:>9.4f} | {shift:>+8.4f} | "
              f"{direct_str} | {verdict}")

    print("-" * 88)
    for n in notes:
        print(n)
    print(f" shift tolerance = {SHIFT_TOLERANCE:.2f}; L_direct is the no-trace baseline (no rho axis)")
    print("=" * 88)


def print_scaling_table(rows: Sequence[Tuple[int, int, dict]]) -> None:
    print("\n" + "=" * 74)
    print(" CRITICAL RATIO vs. BATCH SIZE")
    print("=" * 74)
    print(f"{'batch':>7} | {'steps':>7} | {'rho_c':>8} | {'tau':>8} | fit")
    print("-" * 74)
    for b, steps, s in rows:
        note = "ok" if s["reliable"] else f"UNRELIABLE: {s['unreliable_reason']}"
        print(f"{b:>7} | {steps:>7} | {s['rho_c']:>8.4f} | {s['tau']:>8.4f} | {note}")
    alpha, _, r2 = fit_scaling_exponent([b for b, _, _ in rows], [s["rho_c"] for _, _, s in rows])
    print("-" * 74)
    print(f" rho_c ~ B^alpha with alpha = {alpha:+.3f} (r^2 = {r2:.3f});  SNR theory predicts -0.500")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _finish(ax, title: str, xlabel: str, ylabel: str, ratios_pct=None) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=8)
    if ratios_pct:
        ax.set_xticks(ratios_pct)
        ax.set_xticklabels([f"{int(r)}%" for r in ratios_pct], rotation=45, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)


def _save(fig, figures_dir: str, filename: str) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def plot_metrics(task, records: Sequence[dict], figures_dir: str, filename=None) -> str:
    """One task, every metric: what the end-to-end curve hides.

    The end-to-end and answer-forced curves answer different questions, and the
    gap between them is exactly the generation overhead the metric confounds
    into accuracy.
    """
    task = get_task(task)
    fig, (ax, ax_nll) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    ratios_pct = None
    for metric in ACCURACY_METRICS:
        c = curve(records, metric)
        if not c["ratios"]:
            continue
        color, marker = METRIC_STYLE[metric]
        ratios_pct = [r * 100 for r in c["ratios"]]
        means = [m * 100 for m in c["means"]]
        stds = [s * 100 for s in c["stds"]]
        ax.plot(ratios_pct, means, color=color, marker=marker, linewidth=2.2,
                markersize=5, zorder=3, label=METRIC_LABEL[metric])
        ax.fill_between(ratios_pct, [max(0.0, m - s) for m, s in zip(means, stds)],
                        [min(100.0, m + s) for m, s in zip(means, stds)],
                        color=color, alpha=0.12, zorder=2)

    ax.axhline(task.chance_acc * 100, color="gray", linestyle=":", linewidth=1.5,
               label=f"chance floor ({task.chance_acc:.3f})")
    ax.axhline(task.ceiling_acc * 100, color="black", linestyle="--", linewidth=1.5,
               label=f"Bayes ceiling ({task.ceiling_acc:.3f})")
    ax.set_ylim(-2, 105)
    _finish(ax, f"{task.name}: end-to-end vs. answer forcing",
            "Correct trace ratio ρ (%)", "Accuracy (%)", ratios_pct)

    nll = curve(records, "answer_nll")
    trunc = curve(records, "truncation_rate")
    if nll["ratios"]:
        x = [r * 100 for r in nll["ratios"]]
        ax_nll.plot(x, nll["means"], color="#1a73e8", marker="o", linewidth=2.2,
                    markersize=5, label="answer NLL (teacher-forced)")
        ax_nll.fill_between(x, [m - s for m, s in zip(nll["means"], nll["stds"])],
                            [m + s for m, s in zip(nll["means"], nll["stds"])],
                            color="#1a73e8", alpha=0.12)
    if trunc["ratios"]:
        ax2 = ax_nll.twinx()
        ax2.plot([r * 100 for r in trunc["ratios"]], [m * 100 for m in trunc["means"]],
                 color="#d62728", marker="^", linestyle="--", linewidth=1.6,
                 markersize=4, label="truncated generations")
        ax2.set_ylabel("Truncated (%)", fontsize=10, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
    _finish(ax_nll, f"{task.name}: answer likelihood and generation overhead",
            "Correct trace ratio ρ (%)", "NLL of gold answer (nats/token)",
            [r * 100 for r in nll["ratios"]] if nll["ratios"] else None)

    return _save(fig, figures_dir, filename or f"metrics_{task.name}.png")


def plot_seed_scatter(task, records: Sequence[dict], figures_dir: str,
                      metric: str = "free_acc", filename=None) -> str:
    """The classic figure: per-seed points, mean, and a +/-1 std band."""
    task = get_task(task)
    by_ratio: Dict[float, List[Tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r.get("ratio") is not None and r.get(metric) is not None:
            by_ratio[float(r["ratio"])].append((int(r["seed"]), float(r[metric])))

    ratios = sorted(by_ratio)
    ratios_pct = [r * 100 for r in ratios]
    seeds = sorted({s for v in by_ratio.values() for s, _ in v})
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for i, seed in enumerate(seeds):
        xs = [r * 100 for r in ratios if any(s == seed for s, _ in by_ratio[r])]
        ys = [a * 100 for r in ratios for s, a in by_ratio[r] if s == seed]
        ax.scatter(xs, ys, color=colors[i % len(colors)], alpha=0.7, s=25, zorder=3,
                   label=f"seed {seed}")

    means = [float(np.mean([a for _, a in by_ratio[r]])) * 100 for r in ratios]
    stds = [float(np.std([a for _, a in by_ratio[r]])) * 100 for r in ratios]
    ax.fill_between(ratios_pct, [max(0.0, m - s) for m, s in zip(means, stds)],
                    [min(100.0, m + s) for m, s in zip(means, stds)],
                    color="#1a73e8", alpha=0.15, label="±1 std")
    ax.plot(ratios_pct, means, color="#1a73e8", marker="o", linewidth=2.5,
            markersize=6, zorder=4, label="mean")
    ax.axhline(task.chance_acc * 100, color="gray", linestyle=":", linewidth=1.5,
               label=f"chance ({task.chance_acc:.3f})")
    ax.axhline(task.ceiling_acc * 100, color="black", linestyle="--", linewidth=1.5,
               label=f"Bayes ceiling ({task.ceiling_acc:.3f})")
    ax.set_ylim(-2, 105)
    _finish(ax, f"{task.name}: {METRIC_LABEL[metric]}", "Correct trace ratio ρ (%)",
            "Accuracy (%)", ratios_pct)
    return _save(fig, figures_dir, filename or f"phase_{task.name}_{metric}.png")


def plot_family(summaries: Sequence[dict], figures_dir: str,
                filename: str = "task_family.png") -> str:
    """Every task overlaid, raw and normalized to its own chance/Bayes span."""
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    for i, s in enumerate(summaries):
        color, marker = TASK_COLORS[i % len(TASK_COLORS)], TASK_MARKERS[i % len(TASK_MARKERS)]
        ratios_pct = [r * 100 for r in s["ratios"]]
        means = [a * 100 for a in s["means"]]
        stds = [d * 100 for d in s["stds"]]
        label = f"{s['task']}  (ρ_c={s['rho_c']:.3f})"

        ax_raw.plot(ratios_pct, means, color=color, marker=marker, linewidth=2.2,
                    markersize=5, label=label, zorder=3)
        ax_raw.fill_between(ratios_pct, [max(0.0, m - d) for m, d in zip(means, stds)],
                            [min(100.0, m + d) for m, d in zip(means, stds)],
                            color=color, alpha=0.12, zorder=2)
        ax_raw.axhline(s["chance"] * 100, color=color, linestyle=":", linewidth=1.0)
        ax_raw.axhline(s["ceiling"] * 100, color=color, linestyle="--", linewidth=0.8)

        span = s["ceiling"] - s["chance"]
        ax_norm.plot(ratios_pct, [(a - s["chance"]) / span * 100 for a in s["means"]],
                     color=color, marker=marker, linewidth=2.2, markersize=5,
                     label=label, zorder=3)
        if not math.isnan(s["rho_c"]):
            ax_norm.axvline(s["rho_c"] * 100, color=color, linestyle=":", linewidth=1.2, alpha=0.8)

    ax_raw.set_ylim(-2, 105)
    ax_norm.set_ylim(-5, 110)
    ax_norm.axhline(50, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    _finish(ax_raw, "Accuracy vs. correct-trace ratio", "Correct trace ratio ρ (%)", "Accuracy (%)")
    _finish(ax_norm, "Normalized to each task's chance/Bayes span",
            "Correct trace ratio ρ (%)", "(acc − chance) / (ceiling − chance)  (%)")
    return _save(fig, figures_dir, filename)


def plot_ablation(task, curves: Dict[str, dict], direct: Optional[Tuple[float, float]],
                  figures_dir: str, filename=None) -> str:
    """L_full vs. L_ans over rho, with L_direct as a horizontal baseline."""
    task = get_task(task)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ratios_pct = None
    for cond, s in sorted(curves.items()):
        color, marker, label = COND_STYLE[cond]
        ratios_pct = [r * 100 for r in s["ratios"]]
        means = [a * 100 for a in s["means"]]
        stds = [d * 100 for d in s["stds"]]
        ax.plot(ratios_pct, means, color=color, marker=marker, linewidth=2.5,
                markersize=6, zorder=4, label=f"{label}   ρ_c={s['rho_c']:.3f}")
        ax.fill_between(ratios_pct, [max(0.0, m - d) for m, d in zip(means, stds)],
                        [min(100.0, m + d) for m, d in zip(means, stds)],
                        color=color, alpha=0.15, zorder=2)
        if not math.isnan(s["rho_c"]):
            ax.axvline(s["rho_c"] * 100, color=color, linestyle=":", linewidth=1.4,
                       alpha=0.9, zorder=3)

    if direct:
        mean, std = direct[0] * 100, direct[1] * 100
        ax.axhline(mean, color="#2ca02c", linestyle="-.", linewidth=2.0, zorder=3,
                   label=f"L_direct (no trace) = {mean:.1f}%")
        if std > 0 and ratios_pct:
            ax.fill_between([min(ratios_pct), max(ratios_pct)], mean - std, mean + std,
                            color="#2ca02c", alpha=0.12, zorder=1)

    ax.axhline(task.chance_acc * 100, color="gray", linestyle=":", linewidth=1.5,
               label=f"chance floor ({task.chance_acc:.3f})")
    ax.axhline(task.ceiling_acc * 100, color="black", linestyle="--", linewidth=1.5,
               label=f"Bayes ceiling ({task.ceiling_acc:.3f})")
    ax.set_ylim(-2, 105)
    _finish(ax, f"Supervision ablation on {task.name}: L_full vs. L_ans vs. L_direct",
            "Correct trace ratio ρ (%)", "Accuracy (%)", ratios_pct)
    return _save(fig, figures_dir, filename or f"ablation_{task.name}.png")


def plot_scaling(rows: Sequence[Tuple[int, int, dict]], figures_dir: str,
                 filename: str = "rho_c_vs_batch.png", title: str = "") -> str:
    """rho_c against batch size on log-log axes, against the B^-1/2 prediction."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    batches = [b for b, _, s in rows if s["reliable"]]
    rhos = [s["rho_c"] for _, _, s in rows if s["reliable"]]
    unreliable = [(b, s["rho_c"]) for b, _, s in rows if not s["reliable"]]

    if batches:
        ax.plot(batches, rhos, color="#1a73e8", marker="o", linewidth=2.2,
                markersize=7, label="fitted ρ_c")
        alpha, intercept, r2 = fit_scaling_exponent(batches, rhos)
        grid = np.array(sorted(batches), float)
        ax.plot(grid, np.exp(intercept) * grid**alpha, color="#1a73e8", linestyle="--",
                linewidth=1.4, label=f"fit: ρ_c ∝ B^{alpha:+.3f}  (r²={r2:.3f})")
        anchor = rhos[0] * batches[0] ** 0.5
        ax.plot(grid, anchor * grid**-0.5, color="black", linestyle=":", linewidth=1.6,
                label="SNR prediction: B^−1/2")
    if unreliable:
        ax.scatter([b for b, _ in unreliable], [r for _, r in unreliable],
                   color="#d62728", marker="x", s=60, label="unreliable fit")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, title or "Critical ratio vs. batch size", "Batch size B", "ρ_c")
    return _save(fig, figures_dir, filename)


# ---------------------------------------------------------------------------
# Report: one entry point per experiment
# ---------------------------------------------------------------------------


def report(experiment: str, records: Sequence[dict], figures_dir: str = "figures") -> None:
    """Tables + figures for a finished (or partially finished) experiment."""
    if not records:
        print(f"[warn] no records for experiment {experiment!r}")
        return

    if experiment in ("phase", "family"):
        _report_ratio(records, figures_dir)
    elif experiment == "ablation":
        _report_ablation(records, figures_dir)
    elif experiment in ("batch", "tokens"):
        _report_scaling(experiment, records, figures_dir)
    else:
        _report_ratio(records, figures_dir)


def _report_ratio(records: Sequence[dict], figures_dir: str) -> None:
    summaries = []
    for (task_name,), rows in sorted(group_by(records, ["task"]).items()):
        summary = summarize_curve(task_name, rows, "free_acc")
        summaries.append(summary)
        print_ratio_table(summary, f"{task_name} -- end-to-end accuracy vs. ρ")
        print_forcing_table(task_name, rows)
        plot_seed_scatter(task_name, rows, figures_dir)
        plot_metrics(task_name, rows, figures_dir)

    all_summaries = summaries + [
        summarize_curve(s["task"], [r for r in records if r["task"] == s["task"]], m)
        for s in summaries
        for m in ("forced_acc",)
    ]
    print_curve_table(all_summaries, "PHASE TRANSITION ACROSS TASKS AND METRICS")
    if len(summaries) > 1:
        plot_family(summaries, figures_dir)


def _report_ablation(records: Sequence[dict], figures_dir: str) -> None:
    per_task: Dict[str, dict] = {}
    for (task_name,), rows in sorted(group_by(records, ["task"]).items()):
        curves, entry = {}, {}
        for cond in ("A", "B"):
            cond_rows = [r for r in rows if r.get("condition") == cond]
            if not cond_rows:
                continue
            s = summarize_curve(task_name, cond_rows, "free_acc")
            curves[cond] = s
            entry[cond] = s
            print_ratio_table(s, f"{task_name} -- condition {cond}")
            print_forcing_table(task_name, cond_rows)

        direct_rows = [r for r in rows if r.get("condition") == "C"]
        entry["direct"] = baseline(direct_rows) if direct_rows else None
        per_task[task_name] = entry
        if curves:
            plot_ablation(task_name, curves, entry["direct"], figures_dir)

    print_ablation_verdicts(per_task)


def _report_scaling(experiment: str, records: Sequence[dict], figures_dir: str) -> None:
    for (task_name,), rows in sorted(group_by(records, ["task"]).items()):
        table = []
        for (batch,), brows in sorted(group_by(rows, ["batch_size"]).items()):
            steps = sorted({r["steps"] for r in brows})[0]
            s = summarize_curve(task_name, brows, "free_acc")
            table.append((int(batch), int(steps), s))
            print_ratio_table(s, f"{task_name} -- B={batch}, steps={steps}")
        print_scaling_table(table)
        held = "steps fixed" if experiment == "batch" else "sample budget fixed"
        plot_scaling(table, figures_dir, filename=f"rho_c_vs_batch_{experiment}_{task_name}.png",
                     title=f"{task_name}: ρ_c vs. batch size ({held})")
