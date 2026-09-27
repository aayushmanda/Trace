#!/usr/bin/env python3
"""Rerun the 32k reliability cells that OOM'd under 2-jobs-per-GPU."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "must_run"
PY = ["/home/hariguru/aayus/.venv/bin/python"]
GPUS = [0, 2, 3]

FAILED = [
    ("rho32k_scatter_r0.7_s42", dict(rho=0.7, corrupt="scatter", seed=42)),
    ("rho32k_scatter_r0.7_s43", dict(rho=0.7, corrupt="scatter", seed=43)),
    ("rho32k_scatter_r0.7_s44", dict(rho=0.7, corrupt="scatter", seed=44)),
    ("rho32k_scatter_r0.8_s42", dict(rho=0.8, corrupt="scatter", seed=42)),
    ("rho32k_scatter_r0.8_s43", dict(rho=0.8, corrupt="scatter", seed=43)),
    ("rho32k_scatter_r0.8_s44", dict(rho=0.8, corrupt="scatter", seed=44)),
    ("rho32k_scatter_r0.9_s42", dict(rho=0.9, corrupt="scatter", seed=42)),
    ("rho32k_scatter_r0.9_s43", dict(rho=0.9, corrupt="scatter", seed=43)),
    ("rho32k_scatter_r0.9_s44", dict(rho=0.9, corrupt="scatter", seed=44)),
    ("rho32k_scatter_r1.0_s42", dict(rho=1.0, corrupt="scatter", seed=42)),
    ("rho32k_scatter_r1.0_s43", dict(rho=1.0, corrupt="scatter", seed=43)),
    ("rho32k_scatter_r1.0_s44", dict(rho=1.0, corrupt="scatter", seed=44)),
    ("rho32k_shift_r0.0_s42", dict(rho=0.0, corrupt="shift", seed=42)),
    ("rho32k_shift_r0.0_s43", dict(rho=0.0, corrupt="shift", seed=43)),
    ("rho32k_shift_r0.0_s44", dict(rho=0.0, corrupt="shift", seed=44)),
    ("rho32k_shift_r0.2_s43", dict(rho=0.2, corrupt="shift", seed=43)),
    ("rho32k_shift_r0.2_s44", dict(rho=0.2, corrupt="shift", seed=44)),
    ("rho32k_shift_r0.25_s43", dict(rho=0.25, corrupt="shift", seed=43)),
    ("rho32k_shift_r0.25_s44", dict(rho=0.25, corrupt="shift", seed=44)),
    ("rho32k_shift_r0.3_s42", dict(rho=0.3, corrupt="shift", seed=42)),
    ("rho32k_shift_r0.3_s43", dict(rho=0.3, corrupt="shift", seed=43)),
    ("rho32k_shift_r0.35_s42", dict(rho=0.35, corrupt="shift", seed=42)),
]


def args_for(kw):
    ev = 2000 if kw["rho"] == 1.0 else 0
    return [
        str(ROOT / "handcoded/lettertrace.py"),
        "task=count", "layout=block", "word_len=8", "mod=3",
        f"rho={kw['rho']}", f"corrupt={kw['corrupt']}",
        "steps=32000", "modes=process", f"seed={kw['seed']}",
        f"eval_every={ev}",
    ]


def run_one(gpu, name, kw):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    with (LOG / f"{name}.log").open("w") as fh:
        proc = subprocess.run(PY + args_for(kw), cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[done  gpu{gpu}] {name}  exit={proc.returncode}  {(time.time()-t0)/60:.1f} min", flush=True)
    return proc.returncode


def main():
    lock = threading.Lock()
    idx = [0]
    failed = []

    def worker(gpu):
        while True:
            with lock:
                if idx[0] >= len(FAILED):
                    return
                name, kw = FAILED[idx[0]]
                idx[0] += 1
            if run_one(gpu, name, kw):
                failed.append(name)

    print(f"rerunning {len(FAILED)} OOM cells, 1 job/GPU", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
