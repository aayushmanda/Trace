
"""
main.py
Entry point: loads config.yml, sweeps over rho values, writes results to CSV.

Usage:
    python main.py --config config.yml
    python main.py --config config.yml --rhos 0.0,0.5,1.0   # override sweep
"""
import argparse
import csv
import os

import torch
import yaml

from data import Tokenizer
from train import run_experiment_for_rho


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--rhos", type=str, default=None,
                         help="Comma-separated rho values to override config.yml sweep")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    tokenizer = Tokenizer(cfg["tokenizer"]["extra_chars"])

    rhos = cfg["experiment"]["rhos"]
    if args.rhos:
        rhos = [float(r) for r in args.rhos.split(",")]

    output_csv = cfg["experiment"]["output_csv"]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []
    for rho in rhos:
        final_accuracy, _ = run_experiment_for_rho(rho, tokenizer, cfg, device)
        results.append({
            "rho": rho,
            "seed": cfg["training"]["seed_model"],
            "final_accuracy": final_accuracy,
            "stop_step": cfg["training"]["max_iters"],
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rho", "seed", "final_accuracy", "stop_step"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results for {len(results)} rho values to {output_csv}")


if __name__ == "__main__":
    main()