"""Publication figures from existing CSVs only. No training, no invented numbers.

Writes PDF+PNG (300 dpi) under Paper/figures/. TeX names are reused when the
manuscript already includes them; new layouts use revision_*.pdf.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plot_style import apply_style
from src.experiments import reject_extra_flags

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
FIG = PAPER / "figures"
DATA = PAPER / "data"

PROCESS = "#087eaa"
OUTCOME = "#c56b08"
MIXED = "#6b4c9a"
EXACT = "#1b7f5a"
STEP = "#b42318"
CHANCE = "#4a4a4a"
BASELINE = "#5c5c5c"

LABELS = {"process": "Process", "outcome": "Outcome", "both": "Mixed format"}
MODE_COLOR = {"process": PROCESS, "outcome": OUTCOME, "both": MIXED}

# apply_style() sets DejaVu / #EAEAF2. Extra sizes are NeurIPS-readable (not 21pt,
# which collides on two-column panels) while staying well above tiny 7–8pt captions.
PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "lines.linewidth": 2.0,
}


def style(**extra):
    params = dict(PAPER_RC)
    params.update(extra)
    apply_style(extra=params)


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / stem
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(".pdf"))
    return path.with_suffix(".pdf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_stats(frame, x, y):
    g = frame.groupby(x)[y].agg(["mean", "std", "count"])
    g["std"] = g["std"].fillna(0.0)
    return g.reset_index()


def errorbar(ax, stats, x, color, label, marker="o", ls="-"):
    yerr = np.where(stats["count"] > 1, stats["std"], 0.0)
    ax.errorbar(
        stats[x], stats["mean"], yerr=yerr,
        color=color, label=label, marker=marker, ls=ls,
        ms=7, lw=2.0, capsize=3.2, capthick=1.3, elinewidth=1.4,
        zorder=3,
    )


def scatter_seeds(ax, frame, x, y, color):
    ax.scatter(frame[x], frame[y], s=22, color=color, alpha=0.32, linewidths=0, zorder=2)


def chance_line(ax, p, label=None):
    ax.axhline(p, color=CHANCE, ls="--", lw=1.4, label=label or f"Chance ({100 * p:.2f}%)")


def outcome_line(ax, value, label="Outcome-only"):
    ax.axhline(value, color=OUTCOME, ls=":", lw=2.2, label=label)


# ---------------------------------------------------------------------------
# Fig 1: archived reliability (TeX: figures/legacy_reliability.pdf)
# ---------------------------------------------------------------------------
def plot_legacy_reliability():
    style()
    fsm = pd.read_csv(ROOT / "results/reliability_sweeps/state_machine_16_phase_20260824_045723.csv")
    bc = pd.read_csv(ROOT / "results/reliability_sweeps/boolean_circuit_8_phase_20260823_151153.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True, layout="constrained")

    proc = fsm[fsm.condition != "outcome"].copy()
    proc["rho"] = proc["rho"].astype(float)
    ax = axes[0]
    for col, color, label, marker, ls in [
        ("answer_accuracy", OUTCOME, "Answer accuracy", "o", "-"),
        ("exact_trace_accuracy", EXACT, "Exact trace", "s", "--"),
        ("trace_step_accuracy", PROCESS, "Trace step (free-run)", "^", ":"),
    ]:
        ax.plot(proc.rho, proc[col], color=color, marker=marker, ls=ls, ms=6, label=label)
    outcome_line(ax, float(fsm[fsm.condition == "outcome"].answer_accuracy.mean()))
    chance_line(ax, 1 / 16)
    ax.annotate(
        r"$7.15\%\to 99.75\%$" + "\n" + r"$\rho=0.86\to 0.87$",
        xy=(0.87, 0.9975), xytext=(0.12, 0.48),
        arrowprops={"arrowstyle": "->", "color": ".25", "lw": 1.1},
        fontsize=10, color=".2",
    )
    ax.set(title="(a) FSM, depth 16 (1 seed)", xlabel=r"Trace reliability $\rho$",
           ylabel="Accuracy", xlim=(-0.02, 1.04), ylim=(-0.04, 1.06))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    proc = bc[bc.condition != "outcome"].copy()
    proc["rho"] = proc["rho"].astype(float)
    ax = axes[1]
    for col, color, label, marker, ls in [
        ("answer_accuracy", OUTCOME, "Answer accuracy", "o", "-"),
        ("exact_trace_accuracy", EXACT, "Exact trace", "s", "--"),
        ("trace_step_accuracy", PROCESS, "Trace step (free-run)", "^", ":"),
    ]:
        stats = seed_stats(proc, "rho", col)
        errorbar(ax, stats, "rho", color, label, marker=marker, ls=ls)
        scatter_seeds(ax, proc, "rho", col, color)
    outcome_line(ax, float(bc[bc.condition == "outcome"].answer_accuracy.mean()))
    chance_line(ax, 1 / 16)
    ax.set(title="(b) Boolean, depth 8 (5 seeds)",
           xlabel=r"Trace reliability $\rho$", xlim=(-0.02, 1.04), ylim=(-0.04, 1.06))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return save(fig, "legacy_reliability")


# ---------------------------------------------------------------------------
# Empirical centerpiece from the newer multi-seed CSVs
# ---------------------------------------------------------------------------
def plot_revision_reliability():
    style()
    rm = pd.read_csv(ROOT / "results/reliability_sweeps/register_machine_16_seeds_2001-2003.csv")
    bc = pd.read_csv(ROOT / "results/reliability_sweeps/boolean_circuit_8_seeds_1000-1800.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True, layout="constrained")
    panels = [
        (axes[0], rm, "Register machine, $D{=}16$ (3 seeds)", 0.0588),
        (axes[1], bc, "Boolean circuit, $D{=}8$ (5 seeds)", 1 / 16),
    ]
    for i, (ax, df, title, chance) in enumerate(panels):
        proc = df[df.condition != "outcome"].copy()
        proc["rho"] = proc["rho"].astype(float)
        for col, color, label, marker, ls in [
            ("answer_accuracy", OUTCOME, "Answer accuracy", "o", "-"),
            ("exact_trace_accuracy", EXACT, "Exact trace", "s", "--"),
            ("trace_step_accuracy", PROCESS, "Trace step (free-run)", "^", ":"),
        ]:
            stats = seed_stats(proc, "rho", col)
            errorbar(ax, stats, "rho", color, label, marker=marker, ls=ls)
            scatter_seeds(ax, proc, "rho", col, color)
        outcome_line(ax, float(df[df.condition == "outcome"].answer_accuracy.mean()))
        chance_line(ax, chance)
        ax.set(title=title, xlabel=r"Trace reliability $\rho$",
               xlim=(-0.02, 1.04), ylim=(-0.04, 1.06))
        if i == 0:
            ax.set_ylabel("Accuracy")
        ax.legend(loc="upper left" if i == 0 else "lower left", fontsize=8, framealpha=0.9)
    return save(fig, "revision_reliability")


# ---------------------------------------------------------------------------
# Fig 2: local vs rollout (TeX: figures/local_rollout_gap.pdf)
# ---------------------------------------------------------------------------
def plot_local_rollout():
    style()
    mech = pd.read_csv(ROOT / "results/mechanism/mechanism_summary_20260824_012625.csv")
    rows = mech[(mech.rho.notna()) & (mech.step == 8000)].copy()
    rows["rho"] = rows["rho"].astype(float)
    rows = rows.sort_values("rho")
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    series = [
        ("teacher_state_accuracy", PROCESS, "Local state (gold prefix)", "o", "-"),
        ("free_step_accuracy", MIXED, "Free-running step", "^", ":"),
        ("exact_trace_accuracy", EXACT, "Exact free-running trace", "s", "--"),
        ("answer_accuracy", OUTCOME, "Final answer", "D", "-."),
    ]
    for col, color, label, marker, ls in series:
        ax.plot(rows.rho, rows[col], color=color, marker=marker, ls=ls, ms=7, label=label)
    chance_line(ax, 1 / 16)
    ax.annotate(
        "Local 67.97%\nExact trace 7.50%",
        xy=(0.5, 0.6796875), xytext=(0.04, 0.84),
        arrowprops={"arrowstyle": "->", "color": ".25", "lw": 1.1},
        fontsize=11, color=".2",
    )
    ax.set(
        xlabel=r"Trace reliability $\rho$", ylabel="Accuracy",
        title="Boolean circuit, depth 8, seed 2005",
        ylim=(-0.04, 1.06), xlim=(-0.02, 1.04),
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return save(fig, "local_rollout_gap")


# ---------------------------------------------------------------------------
# Fig 4: trained-executor bridge (TeX name, same four panels)
# ---------------------------------------------------------------------------
def _curve(ax, frame, mode, field, label=None, color=None, style_ls="-"):
    f = frame[frame["mode"] == mode].dropna(subset=[field])
    if f.empty:
        return
    g = f.groupby("step")[field].agg(["mean", "std", "count"])
    x = g.index.to_numpy()
    y = g["mean"].to_numpy()
    sd = g["std"].fillna(0).to_numpy()
    color = color or MODE_COLOR[mode]
    ax.plot(x, y, style_ls, color=color, label=label or LABELS[mode], lw=2.0, marker="o", ms=5)
    if (g["count"] > 1).any():
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16)


def plot_trained_bridge(frame):
    style()
    d4 = frame[frame.depth == 4]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2), layout="constrained")
    for mode in ["process", "both"]:
        _curve(axs[0, 0], d4, mode, "epsilon_rule")
        _curve(axs[0, 1], d4, mode, "composition_tv_with_invalid")
    axs[0, 0].set(title=r"(A) Induced rule strength", ylabel=r"$\widehat{\varepsilon}_{\mathrm{rule}}$")
    axs[0, 0].legend(loc="lower right")
    axs[0, 1].set(title="(B) Composition error", ylabel="Total variation", ylim=(-0.04, 1.06))
    axs[0, 1].legend()
    _curve(axs[1, 0], d4, "both", "gradient_cosine", "Composed-table gradient", MIXED)
    _curve(axs[1, 0], d4, "both", "descent_oracle_rule_cosine", "True-rule direction", PROCESS, "--")
    _curve(axs[1, 0], d4, "both", "descent_random_rule_cosine_mean", "Random-rule mean", BASELINE, ":")
    axs[1, 0].axhline(0, color=".55", lw=1.0)
    axs[1, 0].set(title="(C) Gradient cosine (mixed)", ylabel="Cosine", ylim=(-0.38, 1.06))
    axs[1, 0].legend(fontsize=8)
    for mode in ["process", "outcome", "both"]:
        _curve(axs[1, 1], d4, mode, "free_answer_accuracy")
    chance_line(axs[1, 1], 1 / 16, "Chance (6.25%)")
    axs[1, 1].set(title="(D) Answer accuracy", ylabel="Accuracy", ylim=(-0.04, 1.06))
    axs[1, 1].legend(fontsize=8)
    for ax in axs.flat:
        ax.set_xlabel("Training update")
    return save(fig, "trained_executor_bridge")


def plot_trained_depths(frame, protocol):
    style()
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 7.8), layout="constrained")
    for row, depth in enumerate(protocol["depths"]):
        d = frame[frame.depth == depth]
        for mode in ["process", "both"]:
            _curve(axes[row, 0], d, mode, "epsilon_rule")
            _curve(axes[row, 1], d, mode, "composition_tv_with_invalid")
        _curve(axes[row, 2], d, "both", "gradient_actual_norm", "Actual answer gradient", MIXED)
        _curve(axes[row, 2], d, "both", "gradient_rule_pullback_norm", "Rule pullback", PROCESS, "--")
        for mode in MODE_COLOR:
            _curve(axes[row, 3], d, mode, "free_answer_accuracy")
        chance_line(axes[row, 3], 1 / 16, "Chance" if row == 0 else None)
        axes[row, 0].set_ylabel(f"$D={depth}$")
        axes[row, 2].set_yscale("log")
        for ax in axes[row]:
            ax.set_xlabel("Training update")
        axes[row, 1].set_ylim(-0.04, 1.06)
        axes[row, 3].set_ylim(-0.04, 1.06)
    for ax, title in zip(axes[0], [
        r"Rule strength",
        "Composition TV",
        "Gradient norms",
        "Answer accuracy",
    ]):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    axes[0, 3].legend(fontsize=7)
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    return save(fig, "trained_executor_depths")


# ---------------------------------------------------------------------------
# Honest transfer-limit figure (revision_transfer_limit.pdf)
# ---------------------------------------------------------------------------
def plot_transfer_limit(frame, protocol):
    style()
    final = frame[frame.step == protocol["steps"]].copy()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.45), layout="constrained")

    ax = axs[0]
    depths = np.array(protocol["depths"], dtype=float)
    width = 0.22
    offsets = {"process": -width, "both": 0.0, "outcome": width}
    for mode, color in MODE_COLOR.items():
        sub = final[final["mode"] == mode]
        stats = sub.groupby("depth")["free_answer_accuracy"].agg(["mean", "std", "count"])
        x = stats.index.to_numpy() + offsets[mode]
        yerr = np.where(stats["count"] > 1, stats["std"].fillna(0), 0)
        ax.bar(x, stats["mean"], width=width, color=color, label=LABELS[mode],
               yerr=yerr, capsize=3, error_kw={"elinewidth": 1.2})
    chance_line(ax, 1 / 16)
    ax.set(
        xlabel="Circuit depth $D$", ylabel="Answer accuracy",
        title="(A) Answer accuracy",
        xticks=protocol["depths"], ylim=(-0.04, 1.12),
    )
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1]
    last = final.dropna(subset=["local_rule_accuracy", "composition_tv_with_invalid"])
    for mode, marker in [("process", "o"), ("both", "s")]:
        sub = last[last["mode"] == mode]
        ax.scatter(
            sub.local_rule_accuracy, sub.composition_tv_with_invalid,
            c=MODE_COLOR[mode], marker=marker, s=55, alpha=0.8,
            label=LABELS[mode], zorder=3,
        )
    proc = last[last["mode"] == "process"]
    ax.annotate("Process\n$D=2,4,6$", (proc.local_rule_accuracy.mean(), 0.04),
                ha="right", va="bottom", fontsize=9, color=PROCESS)
    for depth, xytext in [(2, (-18, 10)), (4, (-36, -14)), (6, (8, 6))]:
        grp = last[(last["mode"] == "both") & (last.depth == depth)]
        ax.annotate(
            f"$D={depth}$",
            (grp.local_rule_accuracy.mean(), grp.composition_tv_with_invalid.mean()),
            textcoords="offset points", xytext=xytext, fontsize=9, color=MIXED,
        )
    ax.axhline(0.833, color=OUTCOME, ls=":", lw=1.8, label=r"Mixed $D{=}4$ TV $0.833$")
    ax.set(
        xlabel="Local rule accuracy", ylabel="Composition TV",
        title="(B) Transfer fails at TV $0.833$",
        xlim=(0.992, 1.003), ylim=(-0.06, 1.08),
    )
    ax.legend(loc="lower left", fontsize=8)
    return save(fig, "revision_transfer_limit")


# ---------------------------------------------------------------------------
# Induced-rule / split-verdict from results/induced_rule/
# ---------------------------------------------------------------------------
def plot_induced_and_split():
    from analyze_induced import refit_in_ball

    style()
    depth_df = pd.read_csv(ROOT / "results/induced_rule/depth.csv")
    last = depth_df[depth_df.step == depth_df.step.max()].copy()
    fitted = last[last.exponent_fit_conditional.notna()].copy()
    rows = []
    for _, r in fitted.iterrows():
        slope, r2, n_pts = refit_in_ball(r, mode="conditional", eps_max=0.5)
        rows.append(dict(
            depth=int(r.depth), seed=int(r.seed),
            predicted=int(r.depth) - 1,
            full=float(r.exponent_fit_conditional),
            in_ball=slope, r2=r2, n_pts=n_pts,
            delta=float(r.delta_comp_tv),
            acc=float(r.free_answer_acc),
            recovery=float(r.rule_recovery),
        ))
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.2), layout="constrained")
    cmap = plt.get_cmap("viridis")
    depths = sorted(fitted.depth.unique())
    for i, D in enumerate(depths):
        grp = fitted[fitted.depth == D]
        xs = np.array([json.loads(v) for v in grp.eps_by_lambda_conditional])
        ys = np.array([json.loads(v) for v in grp.credit_by_lambda_conditional])
        keys = list(xs[0].keys())
        X = np.array([[d[k] for k in keys] for d in xs]).mean(axis=0)
        Y = np.array([[d[k] for k in keys] for d in ys]).mean(axis=0)
        c = cmap(i / max(len(depths) - 1, 1))
        ax[0].loglog(X, Y, "o-", color=c, ms=6, label=f"$D={int(D)}$")
        ax[0].loglog(X, Y[0] * (X / X[0]) ** (int(D) - 1), "--", color=c, lw=1.3, alpha=0.7)
    ax[0].set(xlabel=r"$\widehat{\varepsilon}_{\mathrm{rule}}$ (rescaled)",
              ylabel=r"Conditional credit", title=r"(a) Credit vs $\varepsilon^{D-1}$")
    ax[0].legend(fontsize=10)

    g = tbl.groupby("depth")
    d = np.array(sorted(g.groups), dtype=float)
    ax[1].plot(d, d - 1, "k--", lw=1.6, label="$D-1$ (kernel theory)")
    m = g["in_ball"].agg(["mean", "std", "count"])
    yerr = np.where(m["count"] > 1, m["std"].fillna(0), 0)
    ax[1].errorbar(d, m["mean"], yerr=yerr, fmt="o-", color=PROCESS, ms=7,
                   capsize=3, label=r"Fit in $\varepsilon\leq 1/2$")
    ax[1].set(xlabel="Depth $D$", ylabel="Exponent", title="(b) Split-verdict exponent",
              xticks=d)
    ax[1].legend(fontsize=9)

    m_tv = g["delta"].agg(["mean", "std", "count"])
    m_acc = g["acc"].agg(["mean", "std", "count"])
    ax[2].errorbar(d, m_tv["mean"], yerr=np.where(m_tv["count"] > 1, m_tv["std"].fillna(0), 0),
                   fmt="s-", color=OUTCOME, ms=7, capsize=3, label=r"$\delta_{\mathrm{comp}}$")
    ax[2].errorbar(d, m_acc["mean"], yerr=np.where(m_acc["count"] > 1, m_acc["std"].fillna(0), 0),
                   fmt="o-", color=PROCESS, ms=7, capsize=3, label="Free-run answer")
    ax[2].set(xlabel="Depth $D$", ylabel="Value", title="(c) Composition vs answer",
              xticks=d, ylim=(-0.05, 1.05))
    ax[2].legend(fontsize=9)
    save(fig, "induced_rule")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.25), layout="constrained")
    ax[0].plot(d, d - 1, "k--", lw=1.8, label="$D-1$ (kernel theory)")
    ax[0].errorbar(d, m["mean"], yerr=yerr, fmt="o-", color=PROCESS, ms=8,
                   capsize=3.5, lw=2, label=r"GPT readout, $\varepsilon\leq 1/2$")
    for _, r in tbl.iterrows():
        ax[0].scatter(r.depth, r.in_ball, color=PROCESS, s=28, alpha=0.4, zorder=2)
    ax[0].set(xlabel="Depth $D$", ylabel="Exponent", title="(a) Credit exponent",
              xticks=d)
    ax[0].legend(fontsize=10)
    ax[1].errorbar(d, m_tv["mean"],
                   yerr=np.where(m_tv["count"] > 1, m_tv["std"].fillna(0), 0),
                   fmt="s-", color=OUTCOME, ms=8, capsize=3.5, lw=2,
                   label=r"$\delta_{\mathrm{comp}}$")
    ax[1].set(xlabel="Depth $D$", ylabel="Composition TV", title="(b) Composition error",
              xticks=d, ylim=(-0.05, 1.05))
    ax[1].legend(fontsize=10)
    save(fig, "revision_split_verdict")
    print("split-verdict table:\n", tbl.to_string(index=False))
    return tbl


def plot_architectures():
    style()
    fig, ax = plt.subplots(figsize=(7.0, 3.05), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, text, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, 0.53, boxstyle="round,pad=0.06",
            edgecolor=color, facecolor="white", lw=1.6,
        ))
        ax.text(x + w / 2, y + 0.265, text, ha="center", va="center", fontsize=11)

    def arrow(x, y, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x, y),
                    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": ".3"})

    ax.text(0.15, 2.68, "Process architecture: one block reused across output positions",
            fontsize=13, weight="bold", color=PROCESS)
    box(0.2, 1.92, 1.3, "Prompt", PROCESS)
    box(2.15, 1.92, 2.1, "One causal block", PROCESS)
    box(5, 1.92, 3.9, r"Generated trace: $s_1,s_2,\ldots,s_D$", PROCESS)
    arrow(1.6, 2.18, 2.07, 2.18)
    arrow(4.32, 2.18, 4.92, 2.18)
    ax.plot([6.9, 6.9, 3.2], [1.84, 1.63, 1.63], lw=1.2, color=".35")
    arrow(3.2, 1.63, 3.2, 1.84)
    ax.text(5, 1.38, "Feed generated states back", ha="center", fontsize=11)

    ax.text(0.15, 1.02, "Outcome architecture: one internal block per transition",
            fontsize=13, weight="bold", color=OUTCOME)
    for x, w, t in [
        (0.2, 1.3, r"$h_0$"), (2.15, 1.4, r"$F_1\to h_1$"),
        (4.2, 1.4, r"$F_2\to h_2$"), (6.3, 1.4, r"$F_D\to h_D$"),
        (8.4, 1.3, r"Answer $s_D$"),
    ]:
        box(x, 0.2, w, t, OUTCOME)
    for x, x2 in [(1.6, 2.07), (3.62, 4.12), (5.67, 6.22), (7.77, 8.32)]:
        arrow(x, 0.465, x2, 0.465)
    ax.text(5.94, 0.64, r"$\cdots$", ha="center", fontsize=13)
    return save(fig, "architectures")


def write_provenance(paths):
    DATA.mkdir(parents=True, exist_ok=True)
    sources = []
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            sources.append({"file": rel, "sha256": sha256(p), "bytes": p.stat().st_size})
    payload = {"builder": "python experiments/run.py plot-paper-figures", "sources": sources}
    out = DATA / "revision_figure_sources.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)


def main(argv=None):
    reject_extra_flags(argv, __doc__)
    protocol = json.loads((ROOT / "results/executor_comparison/depth_replication/protocol.json").read_text())
    frame = pd.read_csv(ROOT / "results/executor_comparison/depth_replication/metrics.csv")
    plot_legacy_reliability()
    plot_revision_reliability()
    plot_local_rollout()
    plot_trained_bridge(frame)
    plot_trained_depths(frame, protocol)
    plot_transfer_limit(frame, protocol)
    plot_induced_and_split()
    plot_architectures()
    write_provenance([
        "results/reliability_sweeps/state_machine_16_phase_20260824_045723.csv",
        "results/reliability_sweeps/boolean_circuit_8_phase_20260823_151153.csv",
        "results/reliability_sweeps/register_machine_16_seeds_2001-2003.csv",
        "results/reliability_sweeps/boolean_circuit_8_seeds_1000-1800.csv",
        "results/mechanism/mechanism_summary_20260824_012625.csv",
        "results/executor_comparison/depth_replication/metrics.csv",
        "results/executor_comparison/depth_replication/protocol.json",
        "results/induced_rule/depth.csv",
        "results/induced_rule/fraction.csv",
        "results/induced_rule/trajectory.csv",
    ])


if __name__ == "__main__":
    main()
