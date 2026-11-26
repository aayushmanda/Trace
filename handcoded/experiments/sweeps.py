"""Training sweeps for Silent Routing's two-block counting experiments (Figure 1a, Tables 6 & 8, App. G.1).

Subcommands:
    uv run python handcoded/experiments/sweeps.py zero_readout        # Fig. 1a & Table 8: 10 paired seeds, n=16, V(0)=0
    uv run python handcoded/experiments/sweeps.py basin success       # Table 6 (tab:gap): 20 seeds per horizon n in {6..16}
    uv run python handcoded/experiments/sweeps.py lambda_grid --all   # App. G.1: 20-cell (n, |Sigma|) grid, Spearman r_S=0.91
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HANDCODED_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = HANDCODED_DIR.parent
for _path in (str(HANDCODED_DIR), str(ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import lettertrace as lt  # noqa: E402

ZERO_READOUT_DIR = ROOT_DIR / "results/paper/zero_readout_n16_8k"
BASIN_DIR = ROOT_DIR / "results/basin"
BASIN_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
GRID_HORIZONS = (6, 8, 10, 12, 16)
GRID_ALPHABETS = (4, 6, 10, 20)
GRID_SEEDS = (42, 43, 44)


# -----------------------------------------------------------------------------
# 1. zero_readout: Figure 1a & Table 8 (n = 16, 10 paired seeds, V(0) = 0)
# -----------------------------------------------------------------------------


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_zero_readout_seed(seed: int, out_dir: Path = ZERO_READOUT_DIR) -> None:
    """Run one paired step/outcome seed at n=16, mod=2 with zero initial readout V(0)=0."""
    torch.set_num_threads(1)
    cfg = lt.configure(task="count", word_len=16, mod=2, seed=seed, steps=8_000, eval_every=1_000)
    train, shown, test = lt.make_corpus(cfg)
    theta = lt.init_theta(cfg)

    upstream_hash = hashlib.sha256(
        b"".join(t.numpy().tobytes() for k, t in theta.items() if k != "readout")
    ).hexdigest()
    theta["readout"].zero_()
    assert torch.count_nonzero(theta["readout"]).item() == 0

    result: dict[str, Any] = {
        "seed": seed,
        "config": cfg.to_dict(),
        "zero_readout": True,
        "upstream_sha256": upstream_hash,
        "train_sha256": _json_sha256(train),
        "test_sha256": _json_sha256(test),
        "width": cfg.model_width,
        "heads": lt.NUM_HEADS,
        "head_dim": cfg.max_seq_len,
        "mlp_width": cfg.mlp_width,
        "modes": {},
    }

    device_theta = {k: t.to(cfg.device) for k, t in theta.items()}
    for mode in cfg.train_modes:
        init_loss = lt.mean_loss(device_theta, mode, test, cfg=cfg)
        assert abs(init_loss - math.log(cfg.vocab_size)) < 1e-5
        _, history = lt.train_mode(mode, theta, train, shown, test, disable_tqdm=True, cfg=cfg)
        result["modes"][mode] = {"initial_loss": init_loss, "history": history}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"s{seed}.json.tmp"
    tmp_path.write_text(json.dumps(result, indent=2))
    tmp_path.replace(out_dir / f"s{seed}.json")


def summarize_zero_readout(out_dir: Path = ZERO_READOUT_DIR, seeds: range = range(42, 52)) -> dict[str, Any]:
    """Aggregate completed zero-readout seed JSONs into per_seed.csv, summary.json, and accuracy.pdf."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [json.loads((out_dir / f"s{s}.json").read_text()) for s in seeds]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for mode, label, color in (("process", "Step", "#176b92"), ("outcome", "Outcome", "#b85830")):
        curves: list[list[float]] = []
        updates: list[int] = []
        for rec in results:
            evals = [r for r in rec["modes"][mode]["history"] if "exact" in r]
            assert evals[-1]["step"] == 8_000
            updates = [r["step"] for r in evals]
            curves.append([r["exact"] for r in evals])
            rows.append(
                {
                    "seed": rec["seed"],
                    "mode": mode,
                    **evals[-1],
                    "solved_by_8000": any(r["exact"] >= 0.9 for r in evals),
                }
            )

        arr = np.asarray(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(updates, 100.0 * mean, label=label, color=color)
        ax.fill_between(updates, 100.0 * (mean - std), 100.0 * (mean + std), alpha=0.16, color=color)

        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "final_exact_mean": float(mean[-1]),
            "final_exact_sample_std": float(std[-1]),
            "final_answer_mean": float(np.mean([r["answer"] for r in subset])),
            "solved_by_8000": sum(r["solved_by_8000"] for r in subset),
            "final_solved": sum(r["exact"] >= 0.9 for r in subset),
            "seeds": len(subset),
        }

    ax.axhline(50.0, color="gray", ls=":", lw=1.0, label="Chance answer accuracy")
    ax.set(
        xlabel="Training updates",
        ylabel="Exact continuation accuracy (%)",
        title=r"Zero readout: counting, $n=16$, $K=2$",
        ylim=(-2, 102),
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"accuracy.{ext}", dpi=200)
    plt.close(fig)

    with (out_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def cmd_zero_readout(args: argparse.Namespace) -> None:
    if args.seed is not None:
        run_zero_readout_seed(args.seed)
        return

    ZERO_READOUT_DIR.mkdir(parents=True, exist_ok=True)
    (ZERO_READOUT_DIR / "logs").mkdir(exist_ok=True)

    pending: list[int] = []
    for seed in range(42, 52):
        path = ZERO_READOUT_DIR / f"s{seed}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            assert rec["zero_readout"] and rec["seed"] == seed
            assert all(rec["modes"][m]["history"][-1]["step"] == 8_000 for m in ("process", "outcome"))
        else:
            pending.append(seed)

    gpu_slots = args.gpus

    def worker(slot_idx: int) -> None:
        gpu = gpu_slots[slot_idx]
        for seed in pending[slot_idx :: len(gpu_slots)]:
            print(f"Starting seed {seed} on GPU {gpu}", flush=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
            log_path = ZERO_READOUT_DIR / "logs" / f"s{seed}.log"
            with log_path.open("w") as log_file:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "zero_readout", "--seed", str(seed)],
                    cwd=ROOT_DIR,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"Completed seed {seed}", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=len(gpu_slots)) as pool:
            list(pool.map(worker, range(len(gpu_slots))))
    summarize_zero_readout()


# -----------------------------------------------------------------------------
# 2. basin: Table 6 (20-seed horizon sweep) & interpolation path probes
# -----------------------------------------------------------------------------


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def run_basin_success(horizons: list[int], num_seeds: int, steps: int, tag: str = "") -> None:
    """Train paired step/outcome models across horizons and seeds (Table 6 / tab:gap)."""
    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASIN_DIR / f"success{tag}.json"
    done = _load_json_list(out_path)
    base_path = BASIN_DIR / "success.json"
    if tag and base_path.exists():
        done.extend(r for r in _load_json_list(base_path) if r["n"] in horizons and r not in done)
    completed = {(r["n"], r["seed"], r["mode"]) for r in done}

    for word_len in horizons:
        for seed in range(42, 42 + num_seeds):
            if all((word_len, seed, m) in completed for m in ("process", "outcome")):
                continue
            cfg = lt.configure(task="count", word_len=word_len, mod=2, seed=seed, steps=steps, eval_every=steps // 4)
            train, shown, test = lt.make_corpus(cfg)
            theta_0 = lt.init_theta(cfg)
            for mode in ("process", "outcome"):
                _, history = lt.train_mode(mode, theta_0, train, shown, test, cfg=cfg)
                evals = [h for h in history if "exact" in h]
                final_exact = evals[-1]["exact"]
                best_exact = max(h["exact"] for h in evals)
                done.append(
                    {
                        "n": word_len,
                        "seed": seed,
                        "mode": mode,
                        "steps": steps,
                        "final_exact": final_exact,
                        "best_exact": best_exact,
                        "solved": final_exact >= 0.9,
                    }
                )
                print(f"n={word_len:2d} seed={seed} {mode:8s} final exact {final_exact:.3f}", flush=True)
            out_path.write_text(json.dumps(done, indent=2))


def run_basin_start_near(
    horizons: list[int],
    seeds: list[int],
    alphas: list[float],
    steps: int,
    tag: str,
    lr: float = 1e-3,
) -> None:
    """Train from interpolated initializations theta_a = theta_0 + a * (theta* - theta_0)."""
    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASIN_DIR / f"start{tag}.json"
    done = _load_json_list(out_path)
    completed = {(r["n"], r["seed"], r["mode"], r["alpha"]) for r in done}

    for word_len in horizons:
        for seed in seeds:
            cfg = lt.configure(
                task="count", word_len=word_len, mod=2, seed=seed, steps=steps, eval_every=steps // 4, lr=lr
            )
            train, shown, test = lt.make_corpus(cfg)
            theta_0 = {k: t.to(cfg.device) for k, t in lt.init_theta(cfg).items()}
            executors = lt.handcoded_params(cfg)
            for mode in ("process", "outcome"):
                for alpha in alphas:
                    if (word_len, seed, mode, alpha) in completed:
                        continue
                    start_params = {k: theta_0[k] + alpha * (executors[mode][k] - theta_0[k]) for k in theta_0}
                    start_exact = lt.free_run(start_params, test, mode, cfg)[0]
                    _, history = lt.train_mode(mode, start_params, train, shown, test, cfg=cfg)
                    final_exact = history[-1]["exact"]
                    done.append(
                        {
                            "n": word_len,
                            "seed": seed,
                            "mode": mode,
                            "alpha": alpha,
                            "steps": steps,
                            "lr": lr,
                            "exact_at_start": start_exact,
                            "final_exact": final_exact,
                            "solved": final_exact >= 0.9,
                        }
                    )
                    print(
                        f"n={word_len:2d} seed={seed} {mode:8s} alpha={alpha:.2f} "
                        f"start {start_exact:.3f} final {final_exact:.3f}",
                        flush=True,
                    )
                    out_path.write_text(json.dumps(done, indent=2))


def run_basin_align(horizons: list[int], seeds: list[int], num_rows: int = 2_048) -> None:
    """Measure negative-gradient alignment with (theta* - theta) along linear interpolation paths."""
    records: list[dict[str, Any]] = []
    for word_len in horizons:
        for seed in seeds:
            cfg = lt.configure(task="count", word_len=word_len, mod=2, seed=seed)
            train, shown, _ = lt.make_corpus(cfg)
            theta_0 = {k: t.to(cfg.device) for k, t in lt.init_theta(cfg).items()}
            executors = lt.handcoded_params(cfg)

            for mode in ("process", "outcome"):
                target_sol = executors[mode]
                rows = [lt.encode_row(ex, mode, sh, cfg) for ex, sh in zip(train[:num_rows], shown[:num_rows])]
                inputs, targets = lt.encode_batch(rows, mode, cfg)

                cos_vals: list[float] = []
                losses: list[float] = []
                for alpha in BASIN_ALPHAS:
                    params = {
                        k: (theta_0[k] + alpha * (target_sol[k] - theta_0[k])).detach().requires_grad_()
                        for k in target_sol
                    }
                    logits = lt.forward(params, inputs, cfg)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
                    grads = torch.autograd.grad(loss, list(params.values()))
                    flat_g = torch.cat([g.flatten() for g in grads])
                    flat_d = torch.cat([(target_sol[k] - params[k]).detach().flatten() for k in target_sol])
                    cos_vals.append(float(-(flat_g @ flat_d) / (flat_g.norm() * flat_d.norm() + 1e-30)))
                    losses.append(float(loss))

                records.append(
                    {"n": word_len, "seed": seed, "mode": mode, "alphas": BASIN_ALPHAS, "cos": cos_vals, "loss": losses}
                )
                print(
                    f"n={word_len:2d} seed={seed} {mode:8s} cos " + " ".join(f"{c:+.3f}" for c in cos_vals),
                    flush=True,
                )

    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    (BASIN_DIR / "align.json").write_text(json.dumps(records, indent=2))


def cmd_basin(args: argparse.Namespace) -> None:
    if args.what == "align":
        run_basin_align(args.n or [8, 12, 16, 24], args.seeds or [42, 43, 44])
    elif args.what == "success":
        run_basin_success(args.n or [6, 8, 10, 12, 14, 16], (args.seeds or [20])[0], args.steps, args.tag)
    else:
        run_basin_start_near(
            args.n or [8, 16],
            args.seeds or [42, 43],
            args.alphas or [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            args.steps,
            args.tag,
            args.lr,
        )


# -----------------------------------------------------------------------------
# 3. lambda_grid: Appendix G.1 (20-cell horizon x alphabet contraction grid)
# -----------------------------------------------------------------------------


def run_lambda_cell(word_len: int, alphabet: int, seed: int, steps: int, out_dir: Path) -> None:
    out_path = out_dir / f"n{word_len}_A{alphabet}_s{seed}.json"
    if out_path.exists():
        return

    cfg = lt.configure(
        task="count",
        alphabet=alphabet,
        word_len=word_len,
        mod=2,
        n_blocks=2,
        train_size=20_000,
        test_size=1_000,
        steps=steps,
        batch_size=256,
        lr=1e-3,
        init_std=0.02,
        seed=seed,
        eval_every=500,
    )
    t0 = time.monotonic()
    train, shown, test = lt.make_corpus(cfg)
    theta_0 = lt.init_theta(cfg)

    record: dict[str, Any] = {
        "n": word_len,
        "alphabet": alphabet,
        "p": 1.0 / alphabet,
        "lam": abs(1.0 - 2.0 / alphabet),
        "seed": seed,
        "steps": steps,
    }
    for mode in ("process", "outcome"):
        _, history = lt.train_mode(mode, theta_0, train, shown, test, cfg=cfg)
        evals = [h for h in history if "exact" in h]
        first_solved = next((h["step"] for h in evals if h["exact"] >= 0.9), None)
        record[mode] = {
            "final_exact": evals[-1]["exact"],
            "final_answer": evals[-1]["answer"],
            "solved": evals[-1]["exact"] >= 0.9,
            "first_step_exact_ge_0_9": first_solved,
            "curve": [(h["step"], h["exact"]) for h in evals],
        }
    record["elapsed_seconds"] = time.monotonic() - t0
    out_path.write_text(json.dumps(record, indent=2))
    print(
        f"n={word_len} A={alphabet} s={seed} "
        f"process={record['process']['final_exact']:.3f} outcome={record['outcome']['final_exact']:.3f}",
        flush=True,
    )


def cmd_lambda_grid(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        cells = list(itertools.product(GRID_HORIZONS, GRID_ALPHABETS, GRID_SEEDS))[args.worker :: args.workers]
    else:
        cells = [tuple(int(x) for x in spec.split(":")) for spec in args.cells]
    for word_len, alphabet, seed in cells:
        run_lambda_cell(word_len, alphabet, seed, args.steps, out_dir)


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-block counting sweeps for Silent Routing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_zr = sub.add_parser("zero_readout", help="Figure 1a & Table 8: n=16, 10 paired seeds, V(0)=0.")
    p_zr.add_argument("--seed", type=int, default=None, help="Run a single seed instead of all 10.")
    p_zr.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 3], help="GPU indices for parallel workers.")
    p_zr.set_defaults(func=cmd_zero_readout)

    p_basin = sub.add_parser("basin", help="Table 6: 20-seed horizon sweep and interpolation probes.")
    p_basin.add_argument("what", choices=("align", "success", "start"))
    p_basin.add_argument("--alphas", type=float, nargs="+")
    p_basin.add_argument("--tag", default="")
    p_basin.add_argument("--lr", type=float, default=1e-3)
    p_basin.add_argument("--n", type=int, nargs="+")
    p_basin.add_argument("--seeds", type=int, nargs="+")
    p_basin.add_argument("--steps", type=int, default=8_000)
    p_basin.set_defaults(func=cmd_basin)

    p_grid = sub.add_parser("lambda_grid", help="Appendix G.1: 20-cell (n, |Sigma|) contraction grid.")
    p_grid.add_argument("--cells", nargs="*", default=[], help="Explicit n:A:seed cells.")
    p_grid.add_argument("--all", action="store_true", help="Run all 20 cells x 3 seeds.")
    p_grid.add_argument("--worker", type=int, default=0)
    p_grid.add_argument("--workers", type=int, default=1)
    p_grid.add_argument("--steps", type=int, default=8_000)
    p_grid.add_argument("--output", default="results/paper/lambda_grid")
    p_grid.set_defaults(func=cmd_lambda_grid)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
