#!/usr/bin/env python3
"""Queue the paper's empty-cell letter-task jobs across 4 GPUs."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "pending"
LOG.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python"]
GPUS = [0, 1, 2, 3]


def job(name, args, skip_if=None):
    return dict(name=name, args=args, skip_if=skip_if)


def lettertrace(**kw):
    return [str(ROOT / "handcoded/lettertrace.py")] + [f"{k}={v}" for k, v in kw.items()]


def spectrum(**kw):
    return [str(ROOT / "handcoded/spectrum.py")] + [f"{k}={v}" for k, v in kw.items()]


JOBS = []

# Table 2: count at longer horizons (n=8 already in the paper)
for n in (12, 16, 24):
    for seed in (42, 43):
        name = f"gap_count_n{n}_s{seed}"
        JOBS.append(job(name, lettertrace(task="count", word_len=n, mod=2, steps=8000, seed=seed)))

# Table 2: first occurrence. n=16 seeds 2,3 and n=24 seeds 1-3 already logged.
for n in (8, 12):
    for seed in (1, 2, 3):
        name = f"gap_first_n{n}_s{seed}"
        JOBS.append(job(name, lettertrace(task="first", word_len=n, steps=8000, seed=seed)))
JOBS.append(job("gap_first_n16_s1", lettertrace(task="first", word_len=16, steps=8000, seed=1)))

# Table 3: two more seeds for the count spectrum
for n in (2, 4, 6, 8, 9, 10):
    for seed in (43, 44):
        name = f"spec_count_n{n}_s{seed}"
        out = LOG / f"{name}.json"
        JOBS.append(job(name, spectrum(
            task="count", word_len=n, mod=2, seed=seed,
            max_exact=10_000_000 if n >= 9 else 2_000_000,
            out=str(out),
        ), skip_if=out))

# Table 3: first-occurrence spectrum
for n in (8, 12, 16, 24):
    name = f"spec_first_n{n}"
    out = LOG / f"{name}.json"
    JOBS.append(job(name, spectrum(
        task="first", word_len=n, exact=n <= 8,
        out=str(out),
    ), skip_if=out))

# Wrong-trace gradients at init, K=3
out = LOG / "spec_count_K3_n8.json"
JOBS.append(job("spec_count_K3_n8", spectrum(
    task="count", word_len=8, mod=3, out=str(out),
), skip_if=out))

# Table 4 + shift-vs-scatter: count K=3 reliability grid
for corrupt in ("scatter", "shift"):
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for seed in (42, 43):
            JOBS.append(job(
                f"rho8k_{corrupt}_r{rho}_s{seed}",
                lettertrace(
                    task="count", word_len=8, mod=3, rho=rho, corrupt=corrupt,
                    steps=8000, modes="process", seed=seed,
                ),
            ))

# Clean outcome baselines for the same K=3 setting
for seed in (42, 43):
    JOBS.append(job(
        f"out8k_K3_s{seed}",
        lettertrace(task="count", word_len=8, mod=3, steps=8000, modes="outcome", seed=seed),
    ))

# Longer budget (S=32k) after the 8k grid
for corrupt in ("scatter", "shift"):
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for seed in (42, 43):
            JOBS.append(job(
                f"rho32k_{corrupt}_r{rho}_s{seed}",
                lettertrace(
                    task="count", word_len=8, mod=3, rho=rho, corrupt=corrupt,
                    steps=32000, modes="process", seed=seed,
                ),
            ))
for seed in (42, 43):
    JOBS.append(job(
        f"out32k_K3_s{seed}",
        lettertrace(task="count", word_len=8, mod=3, steps=32000, modes="outcome", seed=seed),
    ))


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    log_path = LOG / f"{name}.log"
    if item.get("skip_if") and Path(item["skip_if"]).exists():
        print(f"[skip] {name}", flush=True)
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    with log_path.open("w") as fh:
        fh.write(f"# {' '.join(args)}\n")
        fh.flush()
        proc = subprocess.run(PY + args, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[done  gpu{gpu}] {name}  exit={proc.returncode}  {dt/60:.1f} min", flush=True)
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

    print(f"{len(JOBS)} jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
