"""Training sweeps behind the paper's counting experiments (all train handcoded/lettertrace.py models).

    python handcoded/experiments/sweeps.py zero_readout            # Fig. 1b: ten paired seeds, n = 16, K = 2, zero readout
    python handcoded/experiments/sweeps.py basin success           # tab:gap(b): success over n, 20 seeds per n
    python handcoded/experiments/sweeps.py lambda_grid --all       # Fig. 2b,c: horizon x contraction grid, 3 seeds per cell

zero_readout: matched counting rerun that changes only the final readout initialization to zero; spreads seeds 42-51 over
  GPUs 0, 2, 3, reuses completed seeds, then writes results/paper/zero_readout_n16_8k/{per_seed.csv,summary.json,accuracy.pdf}.
basin: `success` is the fraction of random initializations whose exact accuracy reaches 0.9 within the budget, under both
  losses (same theta_0, data and batches per seed); `align` and `start` probe the basin along theta_0 -> theta*.
  Results go to results/basin/.
lambda_grid: counting modulo 2 with letters uniform over A in {4, 6, 10, 20} (lambda = 1 - 2/A) and n in {6, 8, 10, 12, 16};
  each cell trains both formats from one initialization and records held-out exact accuracy and success.
"""
import sys
from pathlib import Path

HANDCODED = Path(__file__).resolve().parents[1]
ROOT_DIR = HANDCODED.parent
for _p in (str(HANDCODED), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_zero_readout(argv):
    __import__("sys").argv = ['sweeps.py zero_readout'] + list(argv)
    import argparse
    import csv
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    
    ROOT = Path(__file__).resolve().parents[2]
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
        parser = argparse.ArgumentParser()
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
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "zero_readout",
                                    "--seed", str(seed)], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"Completed seed {seed}", flush=True)
        gpu_slots = [0, 2, 3]
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(worker, gpu_slots))
        summarize()
    
    
    if __name__ == "__main__":
        main()


def run_basin(argv):
    __import__("sys").argv = ['sweeps.py basin'] + list(argv)
    import argparse
    import json
    import os
    import sys
    
    import torch
    from torch.nn import functional as F
    
    import lettertrace as L
    
    ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    OUT = "results/basin"
    
    
    def start_near(ns, seeds, alphas, steps, tag, lr=1e-3):
        """Basin along the path: train each loss from theta_a = theta_0 + a (theta* - theta_0)."""
        path = f"{OUT}/start{tag}.json"
        os.makedirs(OUT, exist_ok=True)
        done = json.load(open(path)) if os.path.exists(path) else []
        have = {(r["n"], r["seed"], r["mode"], r["alpha"]) for r in done}
        for n in ns:
            for s in seeds:
                L.configure(task="count", word_len=n, mod=2, seed=s, steps=steps, eval_every=steps // 4, lr=lr)
                train, shown, test = L.make_corpus()
                theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
                hand = L.handcoded_params()
                for mode in ("process", "outcome"):
                    for a in alphas:
                        if (n, s, mode, a) in have:
                            continue
                        start = {k: theta_0[k] + a * (hand[mode][k] - theta_0[k]) for k in theta_0}
                        exact0 = L.free_run(start, test, mode)[0]
                        p, hist = L.train_mode(mode, start, train, shown, test)
                        final = hist[-1]["exact"]
                        done.append(dict(n=n, seed=s, mode=mode, alpha=a, steps=steps, lr=lr,
                                         exact_at_start=exact0, final_exact=final, solved=final >= 0.9))
                        print(f"n={n:2d} seed={s} {mode:8s} alpha={a:.2f} start {exact0:.3f} final {final:.3f}", flush=True)
                        with open(path, "w") as f:
                            json.dump(done, f)
    
    
    def alignment_along_path(mode, theta_0, train, shown, alphas=ALPHAS, rows=2048):
        hand = L.handcoded_params()[mode]
        inputs, targets = L.encode([L.row(e, mode, s) for e, s in zip(train[:rows], shown[:rows])], mode)
        cos, loss_path = [], []
        for a in alphas:
            p = {k: (theta_0[k] + a * (hand[k] - theta_0[k])).detach().requires_grad_() for k in hand}
            loss = F.cross_entropy(L.forward(p, inputs).flatten(0, 1), targets.flatten(), ignore_index=-100)
            grads = torch.autograd.grad(loss, list(p.values()))
            g = torch.cat([x.flatten() for x in grads])
            d = torch.cat([(hand[k] - p[k]).detach().flatten() for k in hand])
            cos.append(float(-(g @ d) / (g.norm() * d.norm() + 1e-30)))
            loss_path.append(float(loss))
        return cos, loss_path
    
    
    def align(ns, seeds):
        out = []
        for n in ns:
            for s in seeds:
                L.configure(task="count", word_len=n, mod=2, seed=s)
                train, shown, _ = L.make_corpus()
                theta_0 = {k: t.to(L.device) for k, t in L.init_theta().items()}
                for mode in ("process", "outcome"):
                    cos, lp = alignment_along_path(mode, theta_0, train, shown)
                    out.append(dict(n=n, seed=s, mode=mode, alphas=ALPHAS, cos=cos, loss=lp))
                    print(f"n={n:2d} seed={s} {mode:8s} cos " + " ".join(f"{c:+.3f}" for c in cos), flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/align.json", "w") as f:
            json.dump(out, f)
    
    
    def success(ns, n_seeds, steps, tag=""):
        path = f"{OUT}/success{tag}.json"
        os.makedirs(OUT, exist_ok=True)
        done = json.load(open(path)) if os.path.exists(path) else []
        if tag and os.path.exists(f"{OUT}/success.json"):           # reuse runs of the single-GPU sweep
            done += [r for r in json.load(open(f"{OUT}/success.json")) if r["n"] in ns and r not in done]
        have = {(r["n"], r["seed"], r["mode"]) for r in done}
        for n in ns:
            for s in range(42, 42 + n_seeds):
                if all((n, s, m) in have for m in ("process", "outcome")):
                    continue
                L.configure(task="count", word_len=n, mod=2, seed=s, steps=steps, eval_every=steps // 4)
                train, shown, test = L.make_corpus()
                theta_0 = L.init_theta()
                for mode in ("process", "outcome"):
                    p, hist = L.train_mode(mode, theta_0, train, shown, test)
                    best = max(h["exact"] for h in hist if "exact" in h)
                    final = hist[-1]["exact"]
                    done.append(dict(n=n, seed=s, mode=mode, steps=steps, final_exact=final,
                                     best_exact=best, solved=final >= 0.9))
                    print(f"n={n:2d} seed={s} {mode:8s} final exact {final:.3f}", flush=True)
                with open(path, "w") as f:
                    json.dump(done, f)
    
    
    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("what", choices=["align", "success", "start"])
        ap.add_argument("--alphas", type=float, nargs="+")
        ap.add_argument("--tag", default="")
        ap.add_argument("--lr", type=float, default=1e-3)
        ap.add_argument("--n", type=int, nargs="+")
        ap.add_argument("--seeds", type=int, nargs="+")
        ap.add_argument("--steps", type=int, default=8000)
        a = ap.parse_args()
        if a.what == "align":
            align(a.n or [8, 12, 16, 24], a.seeds or [42, 43, 44])
        elif a.what == "success":
            success(a.n or [6, 8, 10, 12, 14, 16], (a.seeds or [20])[0], a.steps, a.tag)
        else:
            start_near(a.n or [8, 16], a.seeds or [42, 43], a.alphas or [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                       a.steps, a.tag, a.lr)


def run_lambda_grid(argv):
    __import__("sys").argv = ['sweeps.py lambda_grid'] + list(argv)
    import argparse
    import itertools
    import json
    import sys
    import time
    from pathlib import Path
    
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
        ap = argparse.ArgumentParser()
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



if __name__ == "__main__":
    COMMANDS = {"zero_readout": run_zero_readout, "basin": run_basin, "lambda_grid": run_lambda_grid}
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2:])
