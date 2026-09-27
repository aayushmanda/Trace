#!/usr/bin/env python3
"""Rerun Figure 2 / Table 4 init gradients. GPUs 0/2/3, never 1."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "fig2"
OUT = ROOT / "results" / "paper" / "fig2"
LOG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python"]
GPUS = [0, 0, 2, 2, 3, 3]  # two workers / GPU; never GPU 1


def job(n, seed):
    name = f"n{n}_s{seed}"
    return dict(
        name=name,
        out=OUT / f"{name}.json",
        args=[
            str(ROOT / "handcoded/spectrum.py"),
            "task=count", f"word_len={n}", "mod=2", f"seed={seed}",
            "layout=block", "grads_only=True",
            f"max_exact={10_000_000 if n >= 9 else 2_000_000}",
            f"out={OUT / f'{name}.json'}",
        ],
    )


JOBS = [job(n, seed) for n in (2, 4, 6, 8, 9, 10) for seed in (42, 43, 44)]


def run_one(gpu, item):
    name, args, out = item["name"], item["args"], item["out"]
    if out.exists():
        print(f"[skip] {name}", flush=True)
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    log_path = LOG / f"{name}.log"
    with log_path.open("w") as fh:
        fh.write(f"# {' '.join(PY + args)}\n")
        fh.flush()
        proc = subprocess.run(PY + args, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[done  gpu{gpu}] {name}  exit={proc.returncode}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return proc.returncode


def main():
    lock = threading.Lock()
    idx = [0]
    failed = []

    def worker(gpu):
        while True:
            with lock:
                if idx[0] >= len(JOBS):
                    return
                item = JOBS[idx[0]]
                idx[0] += 1
            if run_one(gpu, item):
                failed.append(item["name"])

    print(f"{len(JOBS)} fig2 jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
