"""Handcoded depth replication of the executor comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from src.training.config import load_yaml
from src.training.progress import progress

ROOT = Path(__file__).resolve().parents[2]


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/experiments/executor_depths.yaml")
    pre_args, _ = pre.parse_known_args()
    cfg = {}
    cfg_path = ROOT / pre_args.config if not Path(pre_args.config).is_absolute() else Path(pre_args.config)
    if cfg_path.exists():
        cfg = load_yaml(cfg_path)
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--output", type=Path, default=ROOT / str(cfg.get("output", "results/executor_comparison/depth_replication")))
    p.add_argument("--workers", type=int, default=int(cfg.get("workers", 2)))
    p.add_argument("--steps", type=int, default=int(cfg.get("steps", 2000)))
    p.add_argument("--seeds", nargs="+", type=int, default=list(cfg.get("seeds", [42, 43, 44, 45, 46])))
    p.add_argument("--depths", nargs="+", type=int, default=list(cfg.get("depths", [2, 4, 6])))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "depths": args.depths, "seeds": args.seeds, "steps": args.steps,
        "train_size": 10000, "test_size": 256, "probe_size": 64, "batch_size": 128,
        "d_model": 96, "d_ff": 192, "lr": 0.002, "device": "cpu", "threads_per_run": 1,
        "backgrounds": 2, "checkpoints": [0, 100, 500, 1000, args.steps],
        "purpose": "depth replication of local readout; not architecture_controls",
    }
    mf = args.output / "protocol.json"
    if mf.exists() and json.loads(mf.read_text()) != protocol:
        raise ValueError("Protocol changed: select a new output directory")
    mf.write_text(json.dumps(protocol, indent=2) + "\n")

    def run_one(depth, seed):
        directory = args.output / f"depth{depth}_seed{seed}"
        manifest = directory / "manifest.json"
        if manifest.exists() and "elapsed_seconds" in json.loads(manifest.read_text()):
            return depth, seed, "completed previously"
        if directory.exists():
            raise RuntimeError(f"Incomplete run exists: {directory}")
        command = [
            sys.executable, "-m", "src", "executor",
            "--output", str(directory), "--depth", str(depth), "--seed", str(seed),
            "--steps", str(args.steps), "--device", "cpu", "--threads", "1",
        ]
        log = args.output / f"depth{depth}_seed{seed}.log"
        with log.open("w") as f:
            result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Run failed: {log}")
        return depth, seed, "complete"

    started = time.monotonic()
    jobs = [(d, s) for d in args.depths for s in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, d, s) for d, s in jobs]
        for future in progress(as_completed(futs), total=len(futs), desc="executor depths"):
            print(future.result(), "elapsed", round(time.monotonic() - started), flush=True)
    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            with (args.output / f"depth{depth}_seed{seed}" / "metrics.csv").open() as f:
                rows.extend({"depth": depth, **r} for r in csv.DictReader(f))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.output / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("All runs complete", args.output, flush=True)


if __name__ == "__main__":
    main()
