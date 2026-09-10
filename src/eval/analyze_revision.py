"""Figures for the theory–Transformer bridge."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import apply_style

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "revision"
FIG = ROOT / "Paper" / "figures"
ONSET_MIN = 0.5


def _load(name):
    path = RES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def induced_figure(df, path):
    df = df.copy()
    if "state_on_set_mass" in df:
        df["readout_ok"] = df["state_on_set_mass"] >= ONSET_MIN
    else:
        df["readout_ok"] = True
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))
    for cond, color in (("process", "tab:blue"), ("outcome", "tab:orange"), ("both", "tab:green")):
        sub = df[(df.condition == cond) & df.readout_ok] if cond != "outcome" else df[df.condition == cond]
        if cond == "outcome":
            sub = df[(df.condition == cond) & df.readout_ok]
        if sub.empty:
            continue
        for D, g in sub.groupby("depth"):
            early = g[g.probe_step == 1] if "probe_step" in g else g
            s = early.groupby("step")["eps_rule_hat"].mean()
            ax[0].plot(s.index, s.values, label=f"{cond} D={int(D)}", color=color,
                       alpha=0.4 + 0.15 * min(int(D), 8) / 8)
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel(r"$\widehat{\varepsilon}_{\mathrm{rule}}$ at $t=1$")
    ax[0].set_title("(a) induced-rule strength")
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    last = df[df.step == df.step.max()]
    last_ok = last[last.readout_ok] if "readout_ok" in last else last
    if "credit_by_lambda_conditional" in last_ok and last_ok.credit_by_lambda_conditional.notna().any():
        cmap = plt.get_cmap("viridis")
        depths = sorted(last_ok.depth.unique())
        for i, D in enumerate(depths):
            grp = last_ok[(last_ok.depth == D) & last_ok.credit_by_lambda_conditional.astype(str).str.startswith("{")]
            if grp.empty:
                continue
            xs = [json.loads(r) if isinstance(r, str) else {} for r in grp.eps_by_lambda_conditional]
            ys = [json.loads(r) if isinstance(r, str) else {} for r in grp.credit_by_lambda_conditional]
            if not xs or not xs[0]:
                continue
            keys = list(xs[0])
            X = np.array([[d[k] for k in keys] for d in xs]).mean(0)
            Y = np.array([[d[k] for k in keys] for d in ys]).mean(0)
            c = cmap(i / max(len(depths) - 1, 1))
            ax[1].loglog(X, Y, "o-", color=c, ms=4, label=f"$D={int(D)}$")
            ax[1].loglog(X, Y[0] * (X / X[0]) ** (D - 1), "--", color=c, lw=1, alpha=.6)
    ax[1].set_xlabel(r"rescaled $\widehat{\varepsilon}_{\mathrm{rule}}$")
    ax[1].set_ylabel("conditional credit")
    ax[1].set_title(r"(b) credit $\propto \varepsilon^{D-1}$")
    ax[1].legend(fontsize=8, frameon=False)
    if {"cos_true_outcome", "cos_true_process"} <= set(df.columns):
        lastp = last.dropna(subset=["cos_true_process"], how="all")
        if not lastp.empty:
            g = lastp.groupby("condition")[["cos_true_outcome", "cos_true_process",
                                            "cos_random_outcome", "cos_random_process"]].mean()
            g.T.plot(kind="bar", ax=ax[2], rot=20)
            ax[2].set_ylabel("parameter-space cosine")
            ax[2].set_title("(c) pullback vs outcome/process grad")
            ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def length_figure(df, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for mode, color in (("process", "tab:blue"), ("outcome", "tab:orange")):
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        g = sub.groupby("eval_depth")["answer_accuracy"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0), fmt="o-", color=color, label=mode)
    if not df.empty:
        ax.axhline(df.chance.iloc[0], color="k", ls=":", lw=1, label="chance")
        ax.axvline(df.train_depth.iloc[0], color="gray", ls="--", lw=1)
    ax.set_xlabel("evaluation depth")
    ax.set_ylabel("greedy answer accuracy")
    ax.set_title("Train $D=8$, evaluate longer circuits")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def margin_figure(df, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for label, color in (("process", "tab:blue"), ("outcome", "tab:orange")):
        sub = df[df.label == label]
        if sub.empty:
            continue
        ax.hist(sub.m_min.dropna(), bins=40, density=True, alpha=0.45, color=color, label=label)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$m_{\min}$ on the gold path")
    ax.set_ylabel("density")
    ax.set_title(r"Greedy flips when $m_{\min}$ crosses 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def main():
    apply_style()
    induced = _load("induced_rule.csv")
    if not induced.empty:
        induced_figure(induced, FIG / "revision_induced_rule.pdf")
    length = _load("length_generalization.csv")
    if not length.empty:
        length_figure(length, FIG / "revision_length_generalization.pdf")
    margins = _load("mmin_histograms.csv")
    if not margins.empty:
        margin_figure(margins, FIG / "revision_mmin.pdf")
    frac = ROOT / "results/architecture_controls_n10/success_fraction.csv"
    if frac.exists():
        print(pd.read_csv(frac).to_string(index=False))


if __name__ == "__main__":
    main()
