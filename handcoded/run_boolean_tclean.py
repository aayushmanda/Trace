#!/usr/bin/env python3
"""Clean-run t_clean for tab:thresholds. One job per GPU; leave GPU 1 alone."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "tclean"
OUT = ROOT / "results" / "paper" / "tclean"
LOG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python", "-m", "src"]
GPUS = [0, 2, 3]

# eval every 250 through 8000
CKPTS = list(range(250, 8001, 250))


def job(name, args):
    return dict(name=name, args=args)


JOBS = []
# Boolean-8, L=2/6, N=20k, three seeds
for layers, ntrain in ((2, 20_000), (6, 20_000), (2, 100_000), (6, 100_000)):
    name = f"bool_L{layers}_N{ntrain}"
    JOBS.append(job(name, [
        "reliability", "--task", "boolean_circuit_8", "--rhos", "1.0",
        "--seeds", "2001", "2002", "2003",
        "--checkpoints", *[str(c) for c in CKPTS],
        "--train-size", str(ntrain), "--val-size", "500",
        "--layers", str(layers), "--batch-size", "128",
        "--output", str(OUT / f"{name}.csv"),
    ]))

# Register, L=2/4/6 N=100k and N=25/50/100k L=2 — one seed first
for layers, ntrain in ((2, 100_000), (4, 100_000), (6, 100_000),
                       (2, 25_000), (2, 50_000)):
    name = f"reg_L{layers}_N{ntrain}"
    JOBS.append(job(name, [
        "reliability", "--task", "register_machine_16", "--rhos", "1.0",
        "--seeds", "2001",
        "--checkpoints", *[str(c) for c in CKPTS],
        "--train-size", str(ntrain), "--val-size", "500",
        "--layers", str(layers), "--batch-size", "128",
        "--output", str(OUT / f"{name}.csv"),
    ]))


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    out = Path(args[args.index("--output") + 1])
    if out.exists():
        print(f"[skip] {name}", flush=True)
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TRACE_COMPILE", "0")
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

    print(f"{len(JOBS)} t_clean jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
