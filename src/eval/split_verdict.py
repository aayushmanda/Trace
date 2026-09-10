"""Split-verdict depth table: predicted D−1 vs fitted exponent, δ_comp, LM pullback cosine."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.plot_style import apply_style
from src.training.config import load_yaml
from src.training.io import write_csv

ROOT = Path(__file__).resolve().parents[2]


def _import_refit():
    import sys
    exp = ROOT / "experiments"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    from analyze_induced import refit_in_ball  # noqa: PLC0415
    return refit_in_ball


def load_frame(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_table(induced, pullback=None, eps_max=0.5):
    refit_in_ball = _import_refit()
    if induced is None or induced.empty:
        induced = pullback if pullback is not None else pd.DataFrame()
        pullback = None
    if induced.empty:
        return pd.DataFrame()
    last = induced[induced.step == induced.step.max()].copy()
    if "state_on_set_mass" in last:
        last = last[last.state_on_set_mass >= 0.5]
    if "probe_step" in last:
        last = last[last.probe_step == 1]
    rows = []
    for _, r in last.iterrows():
        slope, r2, n_pts = np.nan, np.nan, 0
        if str(r.get("eps_by_lambda_conditional", "")).startswith("{"):
            slope, r2, n_pts = refit_in_ball(r, mode="conditional", eps_max=eps_max)
        cos = r.get("cos_true_outcome", np.nan)
        rel = r.get("rel_grad_err_outcome", np.nan)
        rows.append(dict(
            depth=int(r.depth), seed=int(r.seed),
            condition=r.get("condition", ""),
            predicted_exponent=int(r.depth) - 1,
            fitted_exponent_in_ball=slope, r2_in_ball=r2, n_ball=n_pts,
            delta_comp=r.get("delta_comp_tv", np.nan),
            cos_lm_outcome=cos, rel_grad_err_outcome=rel,
            eps_rule_hat=r.get("eps_rule_hat", np.nan),
            eps_step_std=r.get("eps_step_std", np.nan),
            table_step_tv=r.get("table_step_tv", np.nan),
            background_eps_std=r.get("background_eps_std", np.nan),
        ))
    out = pd.DataFrame(rows)
    if pullback is not None and not pullback.empty:
        pb = pullback[pullback.step == pullback.step.max()]
        key = ["depth", "seed"]
        cols = [c for c in ("cos_true_outcome", "rel_grad_err_outcome") if c in pb.columns]
        if cols:
            merged = pb[key + cols].drop_duplicates(key)
            out = out.merge(merged, on=key, how="left", suffixes=("", "_pb"))
            if "cos_true_outcome" in out and out["cos_lm_outcome"].isna().any():
                out["cos_lm_outcome"] = out["cos_lm_outcome"].fillna(out["cos_true_outcome"])
    return out


def figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))
    g = df.groupby("depth")
    d = np.array(sorted(g.groups), dtype=float)
    pred = d - 1
    fit = g["fitted_exponent_in_ball"].mean().reindex(d)
    fit_sd = g["fitted_exponent_in_ball"].std().reindex(d).fillna(0)
    ax[0].plot(d, pred, "k--", lw=1.2, label=r"$D-1$ (predicted)")
    ax[0].errorbar(d, fit.values, yerr=fit_sd.values, fmt="o-", ms=6, label=r"fit in $\varepsilon\le 1/2$")
    ax[0].set_xlabel("depth $D$")
    ax[0].set_ylabel("exponent")
    ax[0].set_title("(a) split verdict: credit exponent")
    ax[0].set_xticks(d)
    ax[0].legend(frameon=False, fontsize=8)

    dc = g["delta_comp"].mean().reindex(d)
    cs = g["cos_lm_outcome"].mean().reindex(d)
    ax[1].plot(d, dc.values, "s-", color="tab:red", label=r"$\delta_{\mathrm{comp}}$")
    ax[1].plot(d, cs.values, "o-", color="tab:blue", label=r"$\cos(-\nabla L_{\mathrm{out}}^{\mathrm{LM}}, J^\top W)$")
    ax[1].set_xlabel("depth $D$")
    ax[1].set_ylabel("value")
    ax[1].set_title("(b) composition TV and exact-LM cosine")
    ax[1].set_xticks(d)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def run(args):
    cfg = load_yaml(args.config) if getattr(args, "config", None) else {}
    induced_path = Path(getattr(args, "input", None) or cfg.get("induced_csv") or ROOT / "results/revision/induced_rule.csv")
    pull_path = Path(getattr(args, "pullback", None) or cfg.get("pullback_csv") or ROOT / "results/revision/pullback.csv")
    out = Path(getattr(args, "out", None) or cfg.get("out") or ROOT / "results/revision/split_verdict.csv")
    fig_path = Path(getattr(args, "figure", None) or cfg.get("figure") or ROOT / "results/revision/split_verdict.pdf")
    induced = load_frame(induced_path)
    pullback = load_frame(pull_path)
    table = build_table(induced, pullback)
    if table.empty:
        print("no induced-rule rows yet; wrote nothing. Run induced/pullback first.")
        return table
    write_csv(out, table.to_dict(orient="records"))
    print(table.to_string(index=False))
    print("wrote", out)
    figure(table, fig_path)
    return table
