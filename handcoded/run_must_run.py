#!/usr/bin/env python3
"""Must-run paper jobs: block-layout Table 3/11, three-seed Table 4, K=3 gradients, reliability."""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "must_run"
LOG.mkdir(parents=True, exist_ok=True)
PY = ["/home/hariguru/aayus/.venv/bin/python"]
# GPU 1 is occupied. Two workers on each free GPU — jobs are tiny in VRAM.
GPUS = [0, 0, 2, 2, 3, 3]
QUEUE_LOG = LOG / "queue.log"


def job(name, args, skip_if=None):
    return dict(name=name, args=args, skip_if=skip_if)


def lettertrace(**kw):
    return [str(ROOT / "handcoded/lettertrace.py")] + [f"{k}={v}" for k, v in kw.items()]


def spectrum(**kw):
    return [str(ROOT / "handcoded/spectrum.py")] + [f"{k}={v}" for k, v in kw.items()]


JOBS = []

# --- 3. Table 4: init spectrum, three seeds, block layout ---
for n in (2, 4, 6, 8, 9, 10):
    for seed in (42, 43, 44):
        name = f"tab4_count_n{n}_s{seed}"
        out = LOG / f"{name}.json"
        JOBS.append(job(name, spectrum(
            task="count", word_len=n, mod=2, seed=seed, layout="block",
            max_exact=10_000_000 if n >= 9 else 2_000_000,
            out=str(out),
        ), skip_if=out))

# --- 4. K=3 wrong-trace gradients at init, three seeds ---
for seed in (42, 43, 44):
    name = f"wrong_K3_n8_s{seed}"
    out = LOG / f"{name}.json"
    JOBS.append(job(name, spectrum(
        task="count", word_len=8, mod=3, seed=seed, layout="block",
        out=str(out),
    ), skip_if=out))

# --- 1+2. Table 3 block training; n=8 seed 42 also supplies Table 11 checkpoints ---
for n in (8, 12, 16, 24):
    for seed in (42, 43, 44):
        kw = dict(task="count", layout="block", word_len=n, mod=2, steps=32000,
                  seed=seed, eval_every=2000)
        if n == 8 and seed == 42:
            kw.update(save_every=500, save_dir="ckpt/n8_block_s42", tag="s42",
                      eval_every=1000)
        JOBS.append(job(f"tab3_count_n{n}_s{seed}", lettertrace(**kw)))

for n in (8, 12, 16, 24):
    for seed in (42, 43, 44):
        JOBS.append(job(
            f"tab3_first_n{n}_s{seed}",
            lettertrace(task="first", layout="block", word_len=n, steps=16000,
                        seed=seed, eval_every=2000),
        ))

# --- 5. K=3 reliability grid, block layout ---
RHOS = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0)
for steps in (8000, 32000):
    ev = 500 if steps == 8000 else 2000
    for corrupt in ("scatter", "shift"):
        for rho in RHOS:
            for seed in (42, 43, 44):
                kw = dict(
                    task="count", layout="block", word_len=8, mod=3, rho=rho,
                    corrupt=corrupt, steps=steps, modes="process", seed=seed,
                    eval_every=ev if rho == 1.0 else 0,
                )
                if steps == 8000 and corrupt == "shift" and rho == 0.5 and seed == 42:
                    kw.update(save_every=8000, save_dir="ckpt/rho05_shift_block", tag="s42")
                JOBS.append(job(
                    f"rho{steps//1000}k_{corrupt}_r{rho}_s{seed}",
                    lettertrace(**kw),
                ))
    for seed in (42, 43, 44):
        JOBS.append(job(
            f"out{steps//1000}k_K3_s{seed}",
            lettertrace(task="count", layout="block", word_len=8, mod=3,
                        steps=steps, modes="outcome", seed=seed),
        ))

# --- 2. Table 11: spectrum on n=8 block checkpoints (queued after the training job) ---
for site, mode in (("state", "process"), ("answer", "outcome")):
    for step in (1000, 3000, 6000, 8000, 12000, 16000, 32000):
        ckpt = ROOT / f"ckpt/n8_block_s42/s42_{mode}_step{step}.pt"
        name = f"tab11_{mode}_step{step}"
        out = LOG / f"{name}.json"
        JOBS.append(job(name, spectrum(
            task="count", word_len=8, mod=2, layout="block", site=site,
            ckpt=str(ckpt), out=str(out),
        ), skip_if=out))


def already_done(item):
    skip = item.get("skip_if")
    if skip and Path(skip).exists():
        return True
    name = item["name"]
    if QUEUE_LOG.exists() and f"] {name}  exit=0" in QUEUE_LOG.read_text():
        return True
    log_path = LOG / f"{name}.log"
    if not log_path.exists():
        return False
    text = log_path.read_text()
    if "Traceback" in text and "Error" in text.split("Traceback")[-1][:400]:
        return False
    steps = next((a.split("=", 1)[1] for a in item["args"] if a.startswith("steps=")), None)
    modes = next((a.split("=", 1)[1] for a in item["args"] if a.startswith("modes=")), "process,outcome")
    last = modes.split(",")[-1]
    if steps and f"{last:8s} step {int(steps):5d}" in text:
        return True
    return False


def already_running(item):
    needle = " ".join(item["args"][-6:])
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any(needle in line and "lettertrace.py" in line for line in out.splitlines())


def run_one(gpu, item):
    name, args = item["name"], item["args"]
    log_path = LOG / f"{name}.log"
    if already_done(item):
        print(f"[skip] {name}", flush=True)
        return 0
    if already_running(item):
        print(f"[skip-running] {name}", flush=True)
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
