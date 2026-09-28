#!/usr/bin/env python3
"""Score-8 leftover jobs: slip sweep, extra seeds, local-J residual, SmolLM coherent.

GPUs 0/2/3 only (never 1). Two workers per GPU.
"""
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/hariguru/aayus/trace")
LOG = ROOT / "logs" / "score8"
LOG.mkdir(parents=True, exist_ok=True)
PY = "/home/hariguru/aayus/.venv/bin/python"
GPUS = [0, 0, 2, 2, 3, 3]
QUEUE_LOG = LOG / "queue.log"


def job(name, cmd, done_if=None):
    return dict(name=name, cmd=cmd, done_if=done_if)


def lettertrace(**kw):
    return [PY, str(ROOT / "handcoded/lettertrace.py")] + [f"{k}={v}" for k, v in kw.items()]


def local_j(seed):
    out = ROOT / "results/paper/probes" / f"local_j_s{seed}.json"
    return job(
        f"localj_s{seed}",
        [PY, str(ROOT / "handcoded/probe_local_j.py"), f"seed={seed}", f"out={out}"],
        done_if=out,
    )


def lora_coherent(rho, seed):
    out = ROOT / "results/paper/smollm/coherent" / f"r{rho}_s{seed}.csv"
    run_dir = ROOT / "results/paper/smollm/artifacts_coherent" / f"r{rho}_s{seed}"
    return job(
        f"smollm_coh_r{rho}_s{seed}",
        [
            PY, "-m", "src", "lora",
            "--model", "HuggingFaceTB/SmolLM2-135M",
            "--task", "boolean_circuit_8_coherent",
            "--rhos", str(rho),
            "--seeds", str(seed),
            "--no-include-outcome",
            "--no-include-answer-first",
            "--include-process",
            "--train-size", "12000",
            "--val-size", "1000",
            "--local-eval-size", "1000",
            "--steps", "800",
            "--batch-size", "16",
            "--lora-rank", "8",
            "--output", str(out),
            "--run-dir", str(run_dir),
        ],
        done_if=out,
    )


JOBS = []

# 1. Residual with J restricted to the local triple (init only).
for seed in (42, 43, 44):
    JOBS.append(local_j(seed))

# 2. Slip-position sweep at 32k (eval every 4k covers 16k and 32k).
for slip_at in range(1, 9):
    for seed in (42, 43, 44):
        JOBS.append(job(
            f"slip32k_j{slip_at}_s{seed}",
            lettertrace(
                task="count", layout="block", word_len=8, mod=3, rho=0.3,
                corrupt="slip", slip_at=slip_at, modes="process",
                steps=32000, seed=seed, eval_every=4000,
            ),
        ))
for seed in (42, 43, 44):
    JOBS.append(job(
        f"slip32k_clean_s{seed}",
        lettertrace(
            task="count", layout="block", word_len=8, mod=3, rho=1.0,
            modes="process", steps=32000, seed=seed, eval_every=4000,
        ),
    ))

# 3. Ten seeds on the bimodal cells (existing 42-44; add 45-51).
EXTRA = (45, 46, 47, 48, 49, 50, 51)
for seed in EXTRA:
    JOBS.append(job(
        f"tab5_rho04_n16_s{seed}",
        lettertrace(
            task="count", layout="block", word_len=16, mod=3, rho=0.4,
            corrupt="scatter", modes="process", steps=32000, seed=seed,
            eval_every=4000,
        ),
    ))
    JOBS.append(job(
        f"tab5_out_n16_s{seed}",
        lettertrace(
            task="count", layout="block", word_len=16, mod=3,
            modes="outcome", steps=32000, seed=seed, eval_every=4000,
        ),
    ))
    JOBS.append(job(
        f"tab4_cot_n16_s{seed}",
        lettertrace(
            task="count", layout="block", word_len=16, mod=2,
            modes="process", steps=16000, seed=seed, eval_every=2000,
        ),
    ))
    JOBS.append(job(
        f"tab4_out_n16_s{seed}",
        lettertrace(
            task="count", layout="block", word_len=16, mod=2,
            modes="outcome", steps=32000, seed=seed, eval_every=4000,
        ),
    ))

# 4. Symmetric vs coherent on SmolLM, larger held-out.
for rho in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
    for seed in (2001, 2002, 2003):
        JOBS.append(lora_coherent(rho, seed))


def finished(item):
    marker = item.get("done_if")
    return bool(marker) and Path(marker).exists() and Path(marker).stat().st_size > 0


def run_one(gpu, item):
    name, cmd = item["name"], item["cmd"]
    log_path = LOG / f"{name}.log"
    if finished(item):
        print(f"[skip] {name}", flush=True)
        return 0
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        if "step 32000" in text or (name.startswith("tab4_cot") and "step 16000" in text):
            print(f"[skip-log] {name}", flush=True)
            return 0
        if "saved aggregate results" in text:
            print(f"[skip-log] {name}", flush=True)
            return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["TRACE_COMPILE"] = "0"
    t0 = time.time()
    print(f"[start gpu{gpu}] {name}", flush=True)
    with log_path.open("w") as fh:
        fh.write(f"# {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
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

    print(f"{len(JOBS)} jobs on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"failed: {failed or 'none'}", flush=True)
    (LOG / "FAILED.txt").write_text("\n".join(failed) + ("\n" if failed else "none\n"))


if __name__ == "__main__":
    main()
