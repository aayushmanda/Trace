"""Matched counting rerun: change only the final readout initialization to zero.

Run with the project Python: python handcoded/zero_readout_rerun.py
Uses GPUs 0, 2, 3; never GPU 1. Existing completed seeds are checked and reused.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/zero_readout_n16_8k"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_seed(seed):
    import torch
    import lettertrace as L

    torch.set_num_threads(1)
    L.configure(task="count", word_len=16, mod=2, seed=seed,
                steps=8000, eval_every=1000)
    config = {k: getattr(L, k) for k in L.DEFAULTS}
    train, shown, test = L.make_corpus()
    theta = L.init_theta()
    upstream_hash = hashlib.sha256(b"".join(
        t.numpy().tobytes() for k, t in theta.items() if k != "readout"
    )).hexdigest()
    theta["readout"].zero_()
    assert torch.count_nonzero(theta["readout"]).item() == 0
    result = dict(seed=seed, config=config, zero_readout=True,
                  upstream_sha256=upstream_hash, train_sha256=digest(train),
                  test_sha256=digest(test), width=L.W, heads=L.H,
                  head_dim=L.P, mlp_width=L.F_WIDTH, modes={})
    # Suppress progress bars; keep the original evaluation and training code.
    original_tqdm = L.tqdm
    L.tqdm = lambda *args, **kw: original_tqdm(*args, **dict(kw, disable=True))
    L.tqdm.write = original_tqdm.write
    for mode in L.train_modes:
        initial_loss = L.mean_loss({k: t.to(L.device) for k, t in theta.items()},
                                   mode, test)
        assert abs(initial_loss - __import__('math').log(L.VOCAB)) < 1e-5
        _, history = L.train_mode(mode, theta, train, shown, test)
        result["modes"][mode] = dict(initial_loss=initial_loss, history=history)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"s{seed}.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(OUT / f"s{seed}.json")


def summarize():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [json.loads((OUT / f"s{s}.json").read_text()) for s in range(42, 52)]
    rows = []
    summary = {}
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for mode, label, color in [("process", "Step", "#176b92"),
                                ("outcome", "Outcome", "#b85830")]:
        curves = []
        for result in results:
            hist = [r for r in result["modes"][mode]["history"] if "exact" in r]
            assert hist[-1]["step"] == 8000
            curves.append([r["exact"] for r in hist])
            final = hist[-1]
            rows.append(dict(seed=result["seed"], mode=mode, **final,
                             solved_by_8000=any(r["exact"] >= .9 for r in hist)))
        curves = np.asarray(curves)
        updates = [r["step"] for r in hist]
        mean, std = curves.mean(0), curves.std(0, ddof=1)
        ax.plot(updates, 100 * mean, label=label, color=color)
        ax.fill_between(updates, 100*(mean-std), 100*(mean+std), alpha=.16, color=color)
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = dict(final_exact_mean=float(mean[-1]),
                             final_exact_sample_std=float(std[-1]),
                             final_answer_mean=float(np.mean([r["answer"] for r in subset])),
                             solved_by_8000=sum(r["solved_by_8000"] for r in subset),
                             final_solved=sum(r["exact"] >= .9 for r in subset), seeds=10)
    ax.axhline(50, color="gray", ls=":", lw=1, label="Chance answer accuracy")
    ax.set(xlabel="Training updates", ylabel="Exact continuation accuracy (%)",
           title=r"Zero readout: counting, $n=16$, $K=2$", ylim=(-2, 102))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"accuracy.{ext}", dpi=200)
    with (OUT / "per_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is not None:
        run_seed(args.seed)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    pending = []
    for seed in range(42, 52):
        p = OUT / f"s{seed}.json"
        if p.exists():
            result = json.loads(p.read_text())
            assert result["zero_readout"] and result["seed"] == seed
            assert all(result["modes"][m]["history"][-1]["step"] == 8000
                       for m in ("process", "outcome"))
        else:
            pending.append(seed)
    def worker(gpu):
        for seed in pending[gpu_slots.index(gpu)::len(gpu_slots)]:
            print(f"Starting seed {seed} on GPU {gpu}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            with (OUT / "logs" / f"s{seed}.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--seed", str(seed)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Completed seed {seed}", flush=True)
    gpu_slots = [0, 2, 3]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, gpu_slots))
    summarize()


if __name__ == "__main__":
    main()
