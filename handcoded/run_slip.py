#!/usr/bin/env python3
"""Single-slip grid: one wrong step, then local-correct. GPUs 0/2/3."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "slip"
LOG.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python", str(ROOT / "handcoded/lettertrace.py")]
GPUS = [0, 0, 2, 2, 3, 3]

JOBS = []
for rho in (0.3, 0.5, 0.7, 1.0):
    for slip_at in (1, 4, 8):
        for seed in (42, 43, 44):
            if rho == 1.0 and slip_at != 1:
                continue
            name = f"slip_r{rho}_j{slip_at}_s{seed}"
            JOBS.append(dict(
                name=name,
                args=[
                    "task=count", "layout=block", "word_len=8", "mod=3",
                    f"rho={rho}", "corrupt=slip", f"slip_at={slip_at}",
                    "modes=process", "steps=8000", f"seed={seed}",
                    "eval_every=2000",
                ],
            ))


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    log_path = LOG / f"{name}.log"
    if log_path.exists() and "process  step  8000" in log_path.read_text():
        print(f"[skip] {name}", flush=True)
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
    print(f"[done  gpu{gpu}] {name}  exit={proc.returncode}  {(time.time()-t0)/60:.1f} min", flush=True)
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

    print(f"{len(JOBS)} slip jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
