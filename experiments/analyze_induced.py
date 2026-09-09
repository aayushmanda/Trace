"""Summarize the induced-rule measurements and build the paper figure."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "Paper" / "figures"


def agg(series):
    v = np.asarray(series, dtype=float)
    return f"{v.mean():.3f}$\\pm${v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def depth_table(df):
    last = df[df.step == df.step.max()]
    rows = []
    for D, grp in last.groupby("depth"):
        rows.append(dict(
            depth=int(D), target=int(D) - 1, n=len(grp),
            slope_cond=agg(grp.exponent_fit_conditional),
            r2_cond=f"{grp.exponent_r2_conditional.astype(float).mean():.4f}",
            slope_prop=agg(grp.exponent_fit_proportional),
            eps_hat=agg(grp.eps_rule_hat),
            gamma_hat=agg(grp.gamma_hat),
            delta_tv=agg(grp.delta_comp_tv),
            recovery=agg(grp.rule_recovery),
            acc=agg(grp.free_answer_acc),
        ))
    return pd.DataFrame(rows)


def traj_table(df):
    out = []
    for (cond, step), grp in df.groupby(["condition", "step"]):
        out.append(dict(condition=cond, step=int(step), n=len(grp),
                        acc=agg(grp.free_answer_acc),
                        eps_hat=agg(grp.eps_rule_hat),
                        gamma_hat=agg(grp.gamma_hat),
                        delta_tv=agg(grp.delta_comp_tv),
                        recovery=agg(grp.rule_recovery),
                        credit=agg(grp.credit_mean),
                        on_set=agg(grp.state_on_set_mass)))
    return pd.DataFrame(out).sort_values(["condition", "step"])


def figure(depth_df, traj_df, path):
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.5))

    # (a) credit against the model's own eps_rule, per depth
    last = depth_df[depth_df.step == depth_df.step.max()]
    cmap = plt.get_cmap("viridis")
    depths = sorted(last.depth.unique())
    for i, D in enumerate(depths):
        grp = last[last.depth == D]
        xs = np.array([json.loads(r) for r in grp.eps_by_lambda_conditional])
        ys = np.array([json.loads(r) for r in grp.credit_by_lambda_conditional])
        keys = list(xs[0].keys())
        X = np.array([[d[k] for k in keys] for d in xs]).mean(axis=0)
        Y = np.array([[d[k] for k in keys] for d in ys]).mean(axis=0)
        c = cmap(i / max(len(depths) - 1, 1))
        ax[0].loglog(X, Y, "o-", color=c, ms=4, label=f"$D={D}$")
        ax[0].loglog(X, Y[0] * (X / X[0]) ** (D - 1), "--", color=c, lw=1, alpha=.6)
    ax[0].set_xlabel(r"$\widehat{\varepsilon}_{\rm rule}$ (rescaled)")
    ax[0].set_ylabel(r"conditional credit $\|\mathrm{Rule}(-\nabla_{P_t}\ell_{\rm out})\|_F$")
    ax[0].set_title("(a) credit follows $\\varepsilon^{D-1}$")
    ax[0].legend(fontsize=8, frameon=False)

    # (b) fitted exponent against D-1
    g = last.groupby("depth")
    m = g.exponent_fit_conditional.astype(float).agg(["mean", "std"])
    p = g.exponent_fit_proportional.astype(float).agg(["mean", "std"])
    d = np.array(m.index, dtype=float)
    ax[1].plot(d, d - 1, "k--", lw=1, label="$D-1$ (predicted)")
    ax[1].errorbar(d, m["mean"], yerr=m["std"].fillna(0), fmt="o-", ms=5,
                   label="measured, marginal-controlled")
    ax[1].errorbar(d, p["mean"], yerr=p["std"].fillna(0), fmt="s-", ms=5,
                   color="tab:orange", label="measured, marginal free")
    ax[1].set_xlabel("composition depth $D$"); ax[1].set_ylabel("fitted exponent")
    ax[1].set_title("(b) the exponent is set by depth")
    ax[1].set_xticks(d); ax[1].legend(fontsize=8, frameon=False)

    # (c) trajectory of the measured scales
    for cond, style in (("process", "-"), ("both", "--"), ("outcome", ":")):
        grp = traj_df[traj_df.condition == cond]
        if grp.empty:
            continue
        s = grp.groupby("step")
        ax[2].plot(s.eps_rule_hat.mean().index, s.eps_rule_hat.mean().values,
                   style, color="tab:blue", label=f"$\\widehat{{\\varepsilon}}$ {cond}")
        ax[2].plot(s.delta_comp_tv.mean().index, s.delta_comp_tv.mean().values,
                   style, color="tab:red", label=f"$\\delta_{{\\rm comp}}$ {cond}")
    ax[2].set_xscale("symlog", linthresh=250)
    ax[2].set_xlabel("training step"); ax[2].set_ylabel("value")
    ax[2].set_title("(c) does the model enter the regime?")
    ax[2].legend(fontsize=7, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def refit_in_ball(row, mode="conditional", eps_max=0.5):
    """Refit the credit exponent using only the rescalings that satisfy the
    theorem's hypothesis eps_rule <= 1/2.

    The stored full-range fit includes lambda = 1, where the measured
    eps_rule of a trained model is around 1.0-1.5 and therefore outside the
    ball the depth bound is stated on.  Restricting to the hypothesis region
    is not selection: it is evaluating the claim where the claim applies.
    """
    xs = json.loads(row[f"eps_by_lambda_{mode}"])
    ys = json.loads(row[f"credit_by_lambda_{mode}"])
    keys = [k for k in xs if xs[k] <= eps_max]
    if len(keys) < 3:
        return np.nan, np.nan, len(keys)
    lx = np.log([xs[k] for k in keys])
    ly = np.log([ys[k] for k in keys])
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    resid = ly - (slope * lx + b)
    r2 = 1.0 - float((resid ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30))
    return float(slope), r2, len(keys)


def exponent_table(*paths, eps_max=0.5):
    """Per-depth exponent, full range against the hypothesis region."""
    frames = []
    for f in paths:
        f = Path(f)
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.step == d.step.max()]
        d = d[d.exponent_fit_conditional.notna()]
        if len(d):
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    fit = d.apply(lambda r: pd.Series(refit_in_ball(r, eps_max=eps_max),
                                      index=["slope_in_ball", "r2_in_ball", "n_points"]), axis=1)
    d = pd.concat([d, fit], axis=1)
    g = d.groupby("depth")
    return g.agg(runs=("seed", "count"), target=("exponent_target", "first"),
                 full_range=("exponent_fit_conditional", "mean"),
                 in_ball=("slope_in_ball", "mean"), in_ball_sd=("slope_in_ball", "std"),
                 r2=("r2_in_ball", "mean"),
                 eps_at_lambda1=("eps_rule_hat", "mean")).round(3)


def readout_validity(df):
    """The readout is only well posed where predictive mass lands on the
    sixteen valid state strings.  Anything with a low on-set mass is an
    out-of-distribution query and its scales must not be reported."""
    if "condition" not in df:
        return pd.DataFrame()
    return df.groupby("condition").agg(
        on_set_mass=("state_on_set_mass", "mean"),
        min_on_set=("state_on_set_mass", "min")).round(4)


def main():
    dpath, tpath = RES / "induced_rule_depth.csv", RES / "induced_rule_traj.csv"
    depth_df = pd.read_csv(dpath) if dpath.exists() else pd.DataFrame()
    traj_df = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
    tbl = exponent_table(RES / "induced_rule_depth.csv",
                         RES / "induced_rule_fraction.csv",
                         RES / "induced_rule_fast.csv")
    if not tbl.empty:
        print("\n=== credit exponent: full range against the hypothesis region eps <= 1/2 ===")
        print(tbl.to_string())
    if not depth_df.empty:
        print("\n=== depth sweep (final checkpoint) ===")
        print(depth_table(depth_df).to_string(index=False))
    if not traj_df.empty:
        print("\n=== readout validity (discard conditions with low on-set mass) ===")
        print(readout_validity(traj_df).to_string())
        print("\n=== trajectory (depth 4) ===")
        print(traj_table(traj_df).to_string(index=False))
    if not depth_df.empty and not traj_df.empty:
        FIG.mkdir(parents=True, exist_ok=True)
        figure(depth_df, traj_df, FIG / "induced_rule.pdf")


if __name__ == "__main__":
    main()
