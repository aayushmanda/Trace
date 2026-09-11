"""Does each objective's gradient carry the other's executor information?

Alignment against the constructed oracle theta* is not measurable in this
architecture: permuting the 832 MLP lookup units of a block leaves the function
unchanged, so theta* is one point in an orbit of size (832!)^D and the cosine of
any gradient with theta* - theta_0 is at chance by construction (see
results/oracle_alignment).  This script measures a quantity that is invariant to
every such symmetry, because it only ever evaluates losses.

At the randomized initialization theta_0 of the deep outcome-shaped architecture
-- which sits at the uniform-prediction point, L = log V, the network analogue of
complete mixing -- take a unit step along each negative population gradient and
record what happens to *both* losses:

    transfer(proc -> out) = [L_out(theta_0 - eta g_proc/|g_proc|) - L_out(theta_0)]
                            / [L_out(theta_0 - eta g_out /|g_out| ) - L_out(theta_0)]

and symmetrically for out -> proc.  A ratio near one means the other objective's
gradient buys as much as the objective's own; a ratio near zero means it buys
nothing.  The shared-kernel claim is that at complete mixing the outcome gradient
is marginal -- it raises some destination for every source at once -- while the
process gradient is conditional.  A marginal update cannot improve
source-conditioned predictions, so the prediction is an asymmetry: process
gradients should transfer to the outcome loss, outcome gradients should not
transfer to the process loss, and the asymmetry should grow with depth.

Writes results/gradient_transfer/summary.json.  No training is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.eval import executor_comparison as c
from src.training.seed import set_seed
from src.experiments.oracle_alignment import (CHUNK, DATA_SEED, N_CIRCUITS, flat, loss_at,
                              population_gradient)

h = c.h
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "gradient_transfer"

DEPTHS = (2, 4, 6, 8)
SEEDS = (42, 43, 44, 45, 46)
ETAS = (0.1, 1.0, 10.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--etas", type=float, nargs="+", default=list(ETAS))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tok = h.make_tokenizer()
    rows = []
    print(f"device={device} depths={args.depths} seeds={args.seeds}", flush=True)

    for depth in args.depths:
        circuits = c.unique_circuits(N_CIRCUITS, DATA_SEED, depth)
        data = {m: h.encode_dataset(circuits, tok, m).to(device) for m in ("outcome", "process")}
        for seed in args.seeds:
            set_seed(seed)
            model = h.build_random_trainable_outcome_architecture(
                tok, depth, seed=seed, device=device)
            shapes = [p.shape for p in model.parameters()]
            sizes = [p.numel() for p in model.parameters()]
            theta0 = flat(model.parameters()).clone()

            def assign(vec):
                off = 0
                with torch.no_grad():
                    for p, s, n in zip(model.parameters(), shapes, sizes):
                        p.copy_(vec[off:off + n].view(s)); off += n

            g = {m: population_gradient(model, data[m]) for m in ("outcome", "process")}
            base = {m: loss_at(model, data[m]) for m in ("outcome", "process")}
            unit = {m: g[m] / (g[m].norm() + 1e-30) for m in g}
            cos_gg = float(torch.dot(unit["outcome"], unit["process"]))

            for eta in args.etas:
                delta = {}
                for step_mode in ("outcome", "process"):
                    assign(theta0 - eta * unit[step_mode])
                    for eval_mode in ("outcome", "process"):
                        delta[(step_mode, eval_mode)] = loss_at(model, data[eval_mode]) - base[eval_mode]
                assign(theta0)
                own_out = delta[("outcome", "outcome")]
                own_proc = delta[("process", "process")]
                rows.append({
                    "depth": depth, "seed": seed, "eta": eta, "cos_grad_grad": cos_gg,
                    "base_L_out": base["outcome"], "base_L_proc": base["process"],
                    "d_out_by_out": own_out, "d_out_by_proc": delta[("process", "outcome")],
                    "d_proc_by_proc": own_proc, "d_proc_by_out": delta[("outcome", "process")],
                    "transfer_proc_to_out": delta[("process", "outcome")] / own_out if own_out else float("nan"),
                    "transfer_out_to_proc": delta[("outcome", "process")] / own_proc if own_proc else float("nan"),
                    "gradnorm_outcome": float(g["outcome"].norm()),
                    "gradnorm_process": float(g["process"].norm()),
                })
                r = rows[-1]
                print(f"[T] D={depth} seed={seed} eta={eta:<5} "
                      f"proc->out={r['transfer_proc_to_out']:+.4f} "
                      f"out->proc={r['transfer_out_to_proc']:+.4f} "
                      f"cos(g,g)={cos_gg:+.5f}", flush=True)
            del model, g, unit
            torch.cuda.empty_cache()

    (OUT / "summary.json").write_text(json.dumps(
        {"config": {"depths": args.depths, "seeds": args.seeds, "etas": args.etas,
                    "n_circuits": N_CIRCUITS, "data_seed": DATA_SEED}, "rows": rows},
        indent=2) + "\n")
    print(f"\nwrote {OUT/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
