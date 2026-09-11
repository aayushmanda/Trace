"""Handcoded oracles + same-architecture init/early measurements.

Stack lock: constructions and D-block outcome architecture only. Not §7 mixed
one-block transfer, not GPT character-token.

  6. Handcoded oracles. TRUE tables (Thm 1 executor): ||q̄||, ||b̄||, Rule credit,
     process floor vs D, and nnsight oracle_slot vs random_subspace patches.

  4. Init / early training. At step 0 (and optional short training) on
     build_random_trainable_outcome_architecture, gold-prefix Def 20 P̂_g,
     mixing radius, outcome Rule credit vs process floor as a function of D.

Writes results/closure_handcoded/{oracle_*.csv, init_*.csv, figures}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments import closure_kernel as ck
import handcoded as h
from src.eval import executor_comparison as c
from src.eval.local_credit import (
    evaluate_probes,
    measure_induced_credit,
    run_patches,
)
from src.plot_style import apply_style
from src.training.seed import configure_device, default_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "closure_handcoded"
PROCESS = "#087eaa"
OUTCOME = "#c56b08"
CHANCE = "#4a4a4a"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def oracle_true_tables(rng, depths, n=200):
    """Theorem 1 executor: P_g = T_g, not a mixing ball. Credit stays Θ(1)."""
    rows = []
    tables = ck.TRUE
    for depth in depths:
        fwds, bwds, credits, floors = [], [], [], []
        for _ in range(n):
            s0, gates, y = ck.sample_circuit(rng, depth)
            q, b, p_y = ck.forward_backward(tables, gates, s0, y)
            s = s0
            for t, g in enumerate(gates):
                gold = int(ck.PERM[g][s])
                fwd = float(np.linalg.norm(ck.PI @ q[t]))
                bwd = float(np.linalg.norm(ck.PI @ b[t + 1]))
                fwds.append(fwd)
                bwds.append(bwd)
                credits.append(fwd * bwd / max(p_y, 1e-30))
                floors.append((1 - 1 / ck.K) / max(float(tables[g, s, gold]), 1e-30))
                s = gold
        rows.append({
            "depth": int(depth),
            "fwd_mean": float(np.mean(fwds)),
            "bwd_mean": float(np.mean(bwds)),
            "credit_mean": float(np.mean(credits)),
            "process_floor_mean": float(np.mean(floors)),
            "mixing_radius": 1.0,
            "n_occurrences": len(credits),
            "theorem": "thm:realizability / thm:forward-backward at T_g",
        })
        print(
            f"[oracle T_g] D={depth:<3} ||q̄||={rows[-1]['fwd_mean']:.4f}  "
            f"||b̄||={rows[-1]['bwd_mean']:.4f}  credit={rows[-1]['credit_mean']:.4f}  "
            f"process floor={rows[-1]['process_floor_mean']:.4f}",
            flush=True,
        )
    return rows


def oracle_patches(depths, device, n_circuits=8, donors=(0, 5, 10, 15), probe_size=16):
    tokenizer = h.make_tokenizer()
    rows = []
    for depth in depths:
        set_seed(42 + depth)
        oracle = h.HandcodedOutcomeTransformer(tokenizer, depth).to(device).eval()
        train = c.unique_circuits(max(probe_size, 8), 123, depth)
        val = c.unique_circuits(n_circuits, 8000, depth, c.circuit_keys(train))
        probes, _, _ = evaluate_probes(
            oracle, train, val, tokenizer, device, tokenizer.n_states, 40, 0.05,
        )
        for prow in run_patches(
            oracle, val, tokenizer, device, probes=probes, model_name="oracle_outcome",
            donor_ids=list(donors),
        ):
            rows.append({"depth": depth, "theorem": "thm:realizability patch", **prow})
        by_method = {}
        for r in rows:
            if r["depth"] == depth:
                by_method.setdefault(r["method"], []).append(r["counterfactual_accuracy"])
        summary = {m: float(np.mean(v)) for m, v in by_method.items()}
        print(f"[oracle patch] D={depth} " + " ".join(f"{k}={v:.3f}" for k, v in summary.items()),
              flush=True)
    return rows


def init_induced(depths, device, seeds=(42, 43, 44), n_circuits=64, backgrounds=2):
    tokenizer = h.make_tokenizer()
    rows = []
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            model = h.build_random_trainable_outcome_architecture(
                tokenizer, depth, seed=seed, device=device,
            ).eval()
            circuits = c.unique_circuits(n_circuits, 8000 + depth, depth)
            measured = measure_induced_credit(
                model, tokenizer, circuits, device, depth, backgrounds=backgrounds,
            )
            rows.append({
                "depth": int(depth),
                "seed": int(seed),
                "step": 0,
                "condition": "init_outcome_architecture",
                "theorem": "cor:near-mixing / Def 20 at init",
                **measured,
            })
            print(
                f"[init] D={depth} seed={seed} mix={measured['mixing_radius_mean']:.3f}  "
                f"outcome credit={measured['outcome_rule_credit_mean']:.3e}  "
                f"process floor={measured['process_floor_mean']:.3f}",
                flush=True,
            )
            del model
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def draw_oracle_and_init(true_rows, patch_rows, init_rows):
    apply_style({
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot([r["depth"] for r in true_rows], [r["credit_mean"] for r in true_rows],
            marker="o", color=OUTCOME, lw=1.7, label=r"outcome Rule credit at $T_g$")
    ax.plot([r["depth"] for r in true_rows], [r["process_floor_mean"] for r in true_rows],
            marker="s", color=PROCESS, lw=1.7, ls="--", label=r"process floor at $T_g$")
    ax.set_xlabel("Composition depth $D$")
    ax.set_ylabel("credit")
    ax.set_title("Theorems 1–2: exact executor is not a mixing ball (credit stays $\\Theta(1)$)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "oracle_true_credit.pdf", bbox_inches="tight")
    fig.savefig(OUT / "oracle_true_credit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if patch_rows:
        methods = ["oracle_slot", "random_subspace", "probe_subspace"]
        depths = sorted({r["depth"] for r in patch_rows})
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        x = np.arange(len(depths))
        width = 0.25
        colors = {"oracle_slot": PROCESS, "random_subspace": CHANCE, "probe_subspace": OUTCOME}
        for i, method in enumerate(methods):
            ys = []
            for d in depths:
                vals = [r["counterfactual_accuracy"] for r in patch_rows
                        if r["depth"] == d and r["method"] == method]
                ys.append(float(np.mean(vals)) if vals else float("nan"))
            ax.bar(x + (i - 1) * width, ys, width, color=colors[method], label=method)
        ax.axhline(1.0 / 16, color=CHANCE, ls=":", lw=1, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Oracle depth $D$")
        ax.set_ylabel("counterfactual accuracy")
        ax.set_title("Theorem 1 construction: patching the known $s_t$ slot flips the answer")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(OUT / "oracle_patches.pdf", bbox_inches="tight")
        fig.savefig(OUT / "oracle_patches.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if init_rows:
        depths = sorted({r["depth"] for r in init_rows})
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        out_y, proc_y, mix_y = [], [], []
        for d in depths:
            sel = [r for r in init_rows if r["depth"] == d]
            out_y.append(float(np.mean([r["outcome_rule_credit_mean"] for r in sel])))
            proc_y.append(float(np.mean([r["process_floor_mean"] for r in sel])))
            mix_y.append(float(np.mean([r["mixing_radius_mean"] for r in sel])))
        axes[0].semilogy(depths, out_y, marker="o", color=OUTCOME, lw=1.7, label="outcome Rule credit")
        axes[0].plot(depths, proc_y, marker="s", color=PROCESS, lw=1.7, label="process floor")
        axes[0].set_xlabel("Architecture depth $D$")
        axes[0].set_ylabel("credit on $\\widehat P_g$")
        axes[0].set_title("Init (step 0): Def 20 tables on the D-block net")
        axes[0].legend(frameon=True)
        axes[1].plot(depths, mix_y, marker="o", color=CHANCE, lw=1.7)
        axes[1].set_xlabel("Architecture depth $D$")
        axes[1].set_ylabel(r"mean $\|\widehat P_g-U\|_2$")
        axes[1].set_title("Mixing radius (do not claim $\\varepsilon^{D-1}$ if this is large)")
        fig.tight_layout()
        fig.savefig(OUT / "init_gradient.pdf", bbox_inches="tight")
        fig.savefig(OUT / "init_gradient.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def maybe_train(args, device):
    if not args.train:
        return []
    from src.eval.local_credit import parse_args, run_experiment

    persists = []
    for depth in args.train_depths:
        out = OUT / f"early_D{depth}"
        argv = [
            "--config", str(ROOT / "configs/experiments/closure_matched_early.yaml"),
            "--depth", str(depth),
            "--seed", str(args.seed),
            "--output", str(out),
            "--device", str(device),
            "--matched-architecture",
            "--induced-credit",
            "--no-compile",
        ]
        ns = parse_args(argv)
        print(f"[early] launching matched D={depth} steps={ns.steps} → {out}", flush=True)
        persist = run_experiment(ns)
        persists.append(persist)
    return persists


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--patch-depths", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--init-depths", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--patch-circuits", type=int, default=8)
    ap.add_argument("--skip-patches", action="store_true")
    ap.add_argument("--skip-init", action="store_true")
    ap.add_argument("--train", action="store_true",
                    help="Also run short matched-architecture training (hours-scale).")
    ap.add_argument("--train-depths", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    device = configure_device(args.device or default_device(), compile=False, bf16=False)
    print(f"device={device}", flush=True)

    rng = np.random.default_rng(2001)
    print("=== Oracle TRUE tables (Thm 1 / Thm 2 at T_g) ===", flush=True)
    true_rows = oracle_true_tables(rng, args.depths, n=args.n)
    write_csv(OUT / "oracle_true.csv", true_rows)

    patch_rows = []
    if not args.skip_patches:
        print("\n=== Oracle nnsight patches ===", flush=True)
        patch_rows = oracle_patches(args.patch_depths, device, n_circuits=args.patch_circuits)
        write_csv(OUT / "oracle_patches.csv", patch_rows)

    init_rows = []
    if not args.skip_init:
        print("\n=== Init Def 20 credit vs D (untrained D-block net) ===", flush=True)
        init_rows = init_induced(args.init_depths, device, seeds=tuple(args.seeds))
        write_csv(OUT / "init_induced.csv", init_rows)

    draw_oracle_and_init(true_rows, patch_rows, init_rows)
    summary = {
        "device": str(device),
        "oracle_true": true_rows,
        "n_patch_rows": len(patch_rows),
        "n_init_rows": len(init_rows),
    }
    (OUT / "handcoded_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    maybe_train(args, device)
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
