#!/usr/bin/env python3
"""Finish remaining paper measurements. GPUs 0/2/3, never 1.

Letter probes first (minutes), then Boolean/register t_clean (hours).
"""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "finish"
OUT = ROOT / "results" / "paper"
LOG.mkdir(parents=True, exist_ok=True)
(OUT / "tclean").mkdir(parents=True, exist_ok=True)
(OUT / "probes").mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python"]
GPUS = [0, 2, 3]
CKPTS = [str(c) for c in range(250, 8001, 250)]


def job(name, args, skip_if=None):
    return dict(name=name, args=args, skip_if=skip_if)


JOBS = []

# --- A. local dependence on existing n=8 seed-42 process ckpts ---
for step in (1000, 3000, 6000, 8000, 12000, 16000, 32000):
    ckpt = ROOT / f"ckpt/n8_block_s42/s42_process_step{step}.pt"
    out = OUT / "probes" / f"local_s42_step{step}.json"
    if ckpt.exists():
        JOBS.append(job(f"local_s42_{step}", [
            str(ROOT / "handcoded/local_dependence.py"),
            f"ckpt={ckpt}", f"out={out}",
        ], skip_if=out))

# --- B. three-gradient residual + probe for Z, three seeds ---
for seed in (42, 43, 44):
    out = OUT / "probes" / f"residual_probe_s{seed}.json"
    JOBS.append(job(f"residual_s{seed}", [
        str(ROOT / "handcoded/probe_residual.py"),
        f"seed={seed}", f"steps=16000", f"eval_every=2000",
        f"save_dir={ROOT / f'ckpt/rho05_k3_s{seed}'}",
        f"out={out}",
    ], skip_if=out))

# --- C. Boolean / register t_clean ---
def reliability(name, task, layers, ntrain, seeds):
    out = OUT / "tclean" / f"{name}.csv"
    args = [
        "-m", "src", "reliability",
        "--task", task, "--rhos", "1.0",
        "--seeds", *[str(s) for s in seeds],
        "--checkpoints", *CKPTS,
        "--train-size", str(ntrain), "--val-size", "1000",
        "--train-seed", "501", "--val-seed", "101",
        "--ratio-seed", "777", "--batch-seed", "12345",
        "--batch-size", "128", "--eval-batch-size", "128",
        "--embedding", "128", "--heads", "4", "--layers", str(layers),
        "--dropout", "0", "--lr", "0.0003", "--weight-decay", "0",
        "--grad-clip", "1", "--bf16", "--no-compile",
        "--output", str(out),
    ]
    JOBS.append(job(name, args, skip_if=out))


for layers, ntrain in ((2, 20_000), (6, 20_000), (2, 100_000), (6, 100_000)):
    reliability(f"bool_L{layers}_N{ntrain}", "boolean_circuit_8", layers, ntrain, (2001, 2002, 2003))
for layers, ntrain in ((2, 100_000), (4, 100_000), (6, 100_000), (2, 25_000), (2, 50_000)):
    reliability(f"reg_L{layers}_N{ntrain}", "register_machine_16", layers, ntrain, (2001, 2002, 2003))


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    skip = item.get("skip_if")
    if skip and Path(skip).exists():
        print(f"[skip] {name}", flush=True)
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TRACE_COMPILE"] = "0"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    log_path = LOG / f"{name}.log"
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

    print(f"{len(JOBS)} finish jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
