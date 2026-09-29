"""GPU job queues. Never uses GPU 1, except the old pending list.

    python handcoded/run.py spotlight
    python handcoded/run.py fig2
"""
import sys

_COMMANDS = (
    'spotlight',
    'fig2',
    'slip',
    'slip05',
    'score8',
    'must',
    'pending',
    'remaining',
    'finish',
    'tclean',
    'oom',
)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"usage: python {sys.argv[0]} <command> [args...]")
        print("commands:", ", ".join(_COMMANDS))
        raise SystemExit(2)
    _cmd = sys.argv[1]
    sys.argv = [_cmd] + sys.argv[2:]
    if _cmd == 'spotlight':
        #!/usr/bin/env python3
        """Lambda sweep and W_r intervention. GPUs 0/2/3, never 1. Two jobs per GPU."""
        import os
        import subprocess
        import threading
        import time
        from pathlib import Path

        ROOT = Path("/home/hariguru/aayus/trace")
        PY = ["/home/hariguru/aayus/.venv/bin/python"]
        GPUS = [0, 0, 2, 2, 3, 3]

        LAM = ROOT / "results" / "paper" / "lambda_sweep"
        WR = ROOT / "results" / "paper" / "wr_intervention"
        for d in (LAM, WR, ROOT / "logs" / "lambda_sweep", ROOT / "logs" / "wr_intervention"):
            d.mkdir(parents=True, exist_ok=True)


        def lam_job(n, p, seed):
            name = f"n{n}_p{p:.2f}_s{seed}"
            return dict(
                name=name,
                kind="lambda",
                out=LAM / f"{name}.json",
                args=[
                    str(ROOT / "handcoded/spectrum.py"),
                    "task=count", f"word_len={n}", "mod=2", f"seed={seed}",
                    "layout=block", "grads_only=True", f"match_p={p}",
                    "max_exact=2000000", f"out={LAM / f'{name}.json'}",
                ],
            )


        def wr_job(R, align, seed, gamma=1.0):
            tag = "off" if align == "off" else f"R{R}_{align}_g{gamma:g}"
            name = f"{tag}_s{seed}"
            return dict(
                name=name,
                kind="wr",
                out=WR / f"{name}.json",
                args=[
                    str(ROOT / "handcoded/spectrum.py"), "job=wr",
                    f"R={R}", f"align={align}", f"seed={seed}", f"gamma={gamma}",
                    "steps=8000", "eval_every=1000",
                    f"out={WR / f'{name}.json'}",
                ],
            )


        JOBS = [lam_job(n, p, seed)
                for p in (0.05, 0.10, 0.15, 0.25, 0.35, 0.40)
                for n in (2, 4, 6, 8)
                for seed in (42, 43, 44)]
        JOBS += [wr_job(0, "off", seed) for seed in (42, 43, 44)]
        JOBS += [wr_job(R, "aligned", seed) for R in (0, 1, 2, 4, 8) for seed in (42, 43, 44)]
        JOBS += [wr_job(R, "shuffle", seed) for R in (1, 2, 4, 8) for seed in (42, 43, 44)]
        JOBS += [wr_job(4, "aligned", seed, gamma) for gamma in (0.25, 4.0) for seed in (42, 43, 44)]


        def run_one(gpu, item):
            name, args, out, kind = item["name"], item["args"], item["out"], item["kind"]
            if out.exists() and out.stat().st_size > 0:
                print(f"[skip] {name}", flush=True)
                return 0
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            log = ROOT / "logs" / ("lambda_sweep" if kind == "lambda" else "wr_intervention") / f"{name}.log"
            t0 = time.time()
            print(f"[start gpu{gpu}] {name}", flush=True)
            with log.open("w") as fh:
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

            print(f"{len(JOBS)} jobs on GPUs {GPUS}", flush=True)
            threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            print(f"failed: {failed or 'none'}", flush=True)


        if __name__ == "__main__":
            main()
    elif _cmd == 'fig2':
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
    elif _cmd == 'slip':
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
    elif _cmd == 'slip05':
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
    elif _cmd == 'score8':
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
                [PY, str(ROOT / "handcoded/probes.py"), "local_j", f"seed={seed}", f"out={out}"],
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
    elif _cmd == 'must':
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
    elif _cmd == 'pending':
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
    elif _cmd == 'remaining':
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
    elif _cmd == 'finish':
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
                    str(ROOT / "handcoded/probes.py"), "dependence",
                    f"ckpt={ckpt}", f"out={out}",
                ], skip_if=out))

        # --- B. three-gradient residual + probe for Z, three seeds ---
        for seed in (42, 43, 44):
            out = OUT / "probes" / f"residual_probe_s{seed}.json"
            JOBS.append(job(f"residual_s{seed}", [
                str(ROOT / "handcoded/probes.py"), "residual",
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
    elif _cmd == 'tclean':
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
    elif _cmd == 'oom':
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
