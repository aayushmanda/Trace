#!/usr/bin/env python3
"""Remaining paper pendings that lettertrace can run. GPUs 0/2/3, never 1."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "remaining"
LOG.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python"]
GPUS = [0, 0, 2, 2, 3, 3]
QUEUE_LOG = LOG / "queue.log"
SEEDS = (42, 43, 44)


def job(name, args, skip_if=None):
    return dict(name=name, args=args, skip_if=skip_if)


def lettertrace(**kw):
    return [str(ROOT / "handcoded/lettertrace.py")] + [f"{k}={v}" for k, v in kw.items()]


JOBS = []

# count vs first at L=4,6 (n=8,12; n=16,24 if these finish)
for L in (4, 6):
    for task, extra in (("count", dict(mod=2)), ("first", {})):
        for n in (8, 12):
            for seed in SEEDS:
                name = f"depth_L{L}_{task}_n{n}_s{seed}"
                JOBS.append(job(name, lettertrace(
                    task=task, layout="block", word_len=n, n_blocks=L,
                    steps=8000, seed=seed, **extra,
                )))

# outcome vs alphabet at n=8 against (1+(1-2/A)^n)/2
for A in (8, 16, 32, 64):
    for seed in SEEDS:
        name = f"alpha_A{A}_n8_s{seed}"
        JOBS.append(job(name, lettertrace(
            task="count", layout="block", word_len=8, mod=2, alphabet=A,
            steps=8000, seed=seed, modes="process,outcome",
        )))

# CoT every k steps, n=8
for k in (2, 4, 8):
    for seed in SEEDS:
        name = f"everyk_k{k}_n8_s{seed}"
        JOBS.append(job(name, lettertrace(
            task="count", layout="block", word_len=8, mod=2, every_k=k,
            steps=8000, seed=seed, modes="process",
        )))

# AdamW vs SGD at two batch sizes: clean t_clean, K=3
for opt in ("adamw", "sgd"):
    for bs in (64, 256):
        for seed in SEEDS:
            name = f"opt_{opt}_b{bs}_K3_s{seed}"
            JOBS.append(job(name, lettertrace(
                task="count", layout="block", word_len=8, mod=3,
                opt=opt, batch_size=bs, steps=16000, eval_every=1000,
                seed=seed, modes="process",
            )))

# shell game, 5 cups, vs first-touched cup
for task in ("shell", "first_cup"):
    for seed in SEEDS:
        name = f"{task}_n8_s{seed}"
        JOBS.append(job(name, lettertrace(
            task=task, layout="block", word_len=8, steps=8000, seed=seed,
        )))

# light switches: full state vs parity of the lit count
for task in ("lights", "lights_parity"):
    for seed in SEEDS:
        name = f"{task}_n8_s{seed}"
        JOBS.append(job(name, lettertrace(
            task=task, layout="block", word_len=8, steps=8000, seed=seed,
            modes="outcome",
        )))


def already_done(name):
    log = LOG / f"{name}.log"
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    if "out of memory" in text.lower() or "CUDA error" in text:
        return False
    return "test exact" in text and "step" in text


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    log_path = LOG / f"{name}.log"
    if item.get("skip_if") and Path(item["skip_if"]).exists():
        print(f"[skip] {name}", flush=True)
        return 0
    if already_done(name):
        print(f"[skip done] {name}", flush=True)
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    with log_path.open("w") as fh:
        fh.write(f"# {' '.join(PY + args)}\n")
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

    header = f"{len(JOBS)} remaining jobs on GPUs {GPUS}\n"
    print(header, flush=True)
    with QUEUE_LOG.open("a") as fh:
        fh.write(header)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
