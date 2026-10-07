"""Outcome and step success over the horizon n and the contraction lambda.

Counting modulo 2 with letters uniform over an alphabet of size A: the match probability is
p = 1/A and lambda = |1 - 2/A|. Each cell (n, A, seed) trains both formats from one shared
initialization with the same AdamW budget as the paper's 120-run sweep, and records held-out
exact accuracy, success (exact >= 0.9 at the end), and the first evaluation reaching 0.9.

    python handcoded/lambda_grid.py --cells 6:4:42 8:10:43 ...
    python handcoded/lambda_grid.py --all --worker 0 --workers 4
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lettertrace as lt

NS = (6, 8, 10, 12, 16)
ALPHABETS = (4, 6, 10, 20)          # lambda = 0.5, 0.667, 0.8, 0.9
SEEDS = (42, 43, 44)


def run_cell(n, A, seed, steps, out_dir):
    path = Path(out_dir) / f"n{n}_A{A}_s{seed}.json"
    if path.exists():
        return
    lt.configure(task="count", alphabet=A, word_len=n, mod=2, n_blocks=2, train_size=20000,
                 test_size=1000, steps=steps, batch_size=256, lr=1e-3, init_std=0.02, seed=seed,
                 eval_every=500)
    t0 = time.monotonic()
    train, shown, test = lt.make_corpus()
    theta_0 = lt.init_theta()
    rec = dict(n=n, alphabet=A, p=1 / A, lam=abs(1 - 2 / A), seed=seed, steps=steps)
    for mode in ("process", "outcome"):
        _, hist = lt.train_mode(mode, theta_0, train, shown, test)
        evals = [h for h in hist if "exact" in h]
        first = next((h["step"] for h in evals if h["exact"] >= 0.9), None)
        rec[mode] = dict(final_exact=evals[-1]["exact"], final_answer=evals[-1]["answer"],
                         solved=evals[-1]["exact"] >= 0.9, first_step_exact_ge_0_9=first,
                         curve=[(h["step"], h["exact"]) for h in evals])
    rec["elapsed_seconds"] = time.monotonic() - t0
    path.write_text(json.dumps(rec, indent=2))
    print(f"n={n} A={A} s={seed} process={rec['process']['final_exact']:.3f} "
          f"outcome={rec['outcome']['final_exact']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--output", default="results/paper/lambda_grid")
    args = ap.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    cells = [tuple(int(x) for x in c.split(":")) for c in args.cells]
    if args.all:
        cells = list(itertools.product(NS, ALPHABETS, SEEDS))[args.worker::args.workers]
    for n, A, seed in cells:
        run_cell(n, A, seed, args.steps, args.output)
