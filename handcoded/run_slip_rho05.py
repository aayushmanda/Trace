#!/usr/bin/env python3
"""Slip sweep at rho=0.5: almost-clean prediction is high gold accuracy at every j."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "slip_rho05"
LOG.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python", str(ROOT / "handcoded/lettertrace.py")]
# GPU 1 is off-limits. 0 is busy (~42GB); keep one worker there.
GPUS = [0, 2, 2, 3, 3]

JOBS = []
for slip_at in range(1, 9):
    for seed in (42, 43, 44):
        JOBS.append(dict(
            name=f"slip05_j{slip_at}_s{seed}",
            args=[
                "task=count", "layout=block", "word_len=8", "mod=3",
                "rho=0.5", "corrupt=slip", f"slip_at={slip_at}",
                "modes=process", "steps=32000", f"seed={seed}",
                "eval_every=4000",
            ],
        ))


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    log_path = LOG / f"{name}.log"
    if log_path.exists() and "step 32000" in log_path.read_text(errors="ignore"):
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
                with lock:
                    failed.append(item["name"])

    print(f"{len(JOBS)} slip rho=0.5 jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)
    (LOG / "FAILED.txt").write_text("\n".join(failed) + ("\n" if failed else "none\n"))


if __name__ == "__main__":
    main()
