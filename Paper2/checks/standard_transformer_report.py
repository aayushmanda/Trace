"""Summarize standard_transformer.py runs and draw Paper2/figures/standard_transformer.pdf.

    uv run python Paper2/checks/standard_transformer_report.py
"""
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "checks/results/standard_transformer"
LAWS = ["D1", "D0", "D-1", "Dlam0.1"]
LABEL = {"D1": r"$\mathcal{D}_{1}$ (independent)", "D0": r"$\mathcal{D}_{0}$ (one anti-copy)",
         "D-1": r"$\mathcal{D}_{-1}$ (two anti-copies)", "Dlam0.1": r"$\mathcal{D}_{\lambda=0.1}$ (mixture)"}
COLOR = {"D1": "#2a78d6", "D0": "#eb6834", "D-1": "#1baf7a", "Dlam0.1": "#eda100"}
VARIANTS = [("adamw", "0.02"), ("adamw", "0.002"), ("sgd", "0.02"), ("sgd", "0.002")]


def load(opt, std):
    runs, cfg = [], None
    files = [RES / f"{opt}_std{std}.json"] + sorted((RES / "parts").glob(f"{opt}_std{std}_*.json"))
    for f in files:
        if f.exists():
            d = json.loads(f.read_text())
            cfg = d["config"]
            runs += d["runs"]
    return cfg, runs


def route_step(run, thr=0.5):
    for c in run["curve"]:
        if max(max(h) for h in c["src_att"]) >= thr:
            return c["step"]
    return None


def summarize():
    out = {}
    for opt, std in VARIANTS:
        cfg, runs = load(opt, std)
        if not runs:
            continue
        key = f"{opt}_std{std}"
        out[key] = dict(config=cfg, laws={})
        for law in LAWS:
            rs = [r for r in runs if r["law"] == law]
            if not rs:
                continue
            hits = [r["hit_step"] for r in rs if r["hit_step"] is not None]
            routes = [route_step(r) for r in rs]
            out[key]["laws"][law] = dict(
                seeds=len(rs), solved=len(hits),
                hit_median=statistics.median(hits) if hits else None,
                hit_min=min(hits) if hits else None, hit_max=max(hits) if hits else None,
                hit_steps=[r["hit_step"] for r in rs],
                final_clean_acc=[round(r["final"]["clean_acc"], 4) for r in rs],
                route_steps=routes,
                final_src_att_max=[round(max(max(h) for h in r["final"]["src_att"]), 3) for r in rs],
            )
        # Mann-Whitney U on time-to-99% (unsolved runs share the censoring budget, ranked last).
        h = {law: [x if x is not None else math.inf for x in v["hit_steps"]] for law, v in out[key]["laws"].items()}
        if len(h) == 4:
            pairs = [("D1", "D0"), ("D1", "Dlam0.1"), ("D1", "D-1"), ("D-1", "D0")]
            tests = {f"{x}_vs_{y}": mannwhitneyu(h[x], h[y]).pvalue for x, y in pairs}
            tests["D1+D-1_vs_D0+Dlam0.1"] = mannwhitneyu(h["D1"] + h["D-1"], h["D0"] + h["Dlam0.1"]).pvalue
            out[key]["mannwhitney_p"] = {k: float(v) for k, v in tests.items()}
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    for key, v in out.items():
        c = v["config"]
        print(f"\n{key}: lr={c['lr']} steps={c['steps']} eval_every={c['eval_every']}")
        for law, s in v["laws"].items():
            print(f"  {law:8s} solved {s['solved']}/{s['seeds']}  t99 median={s['hit_median']} "
                  f"range=[{s['hit_min']},{s['hit_max']}]  hits={s['hit_steps']}  final_src_att={s['final_src_att_max']}")
        print("  Mann-Whitney p:", {k: round(p, 4) for k, p in v.get("mannwhitney_p", {}).items()})
    return out


def plot():
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "xtick.color": "#555555",
                         "ytick.color": "#555555", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharey=True)
    for ax, (opt, std) in zip(axes.flat, VARIANTS):
        cfg, runs = load(opt, std)
        for law in LAWS:
            rs = [r for r in runs if r["law"] == law]
            if not rs:
                continue
            steps = np.array([c["step"] for c in rs[0]["curve"]])
            acc = np.array([[c["clean_acc"] for c in r["curve"]] for r in rs])
            ax.plot(steps[1:], acc.mean(0)[1:], color=COLOR[law], lw=1.6, label=LABEL[law])
        nseeds = sorted({sum(r["law"] == law for r in runs) for law in LAWS})
        ax.set_title(f"{'AdamW' if opt == 'adamw' else 'SGD'}, init std {std} "
                     f"({'/'.join(map(str, nseeds))} seeds per law)", fontsize=8.5)
        ax.set_ylim(0.45, 1.02)
        ax.set_xscale("log")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        if opt == "sgd":
            ax.set_xlabel("training step")
        if std == "0.02":
            ax.set_ylabel("clean-rule accuracy\n(mean over seeds)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = ROOT / "figures/standard_transformer.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    summarize()
    plot()
