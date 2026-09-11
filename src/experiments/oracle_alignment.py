"""Gradient alignment and loss profile against the exact oracle, in real parameter space.

The shared-kernel theorems are claims about where gradient mass goes, not about a
trained network being a transition kernel.  This script tests them in the
coordinates a trained network actually has, using the construction of
Theorem 1 as the reference direction: the oracle theta* and a randomized copy
theta_0 occupy the same parameterization (Appendix G), so theta* - theta_0 is an
exact direction toward a zero-error executor with no decoded table involved.

Part A (alignment).  At theta_0 in the deep outcome-shaped architecture, compute
the population gradients of the outcome and the process objective on the same
circuits and report cos(-grad, theta* - theta_0) against depth.  A random
direction of the same shape gives the chance level, which matters: in ~10^6
dimensions chance cosine is ~10^-3, so an apparently small alignment can still be
orders of magnitude above chance.

Alignment is reported on three subspaces.  theta* is the *outcome*-format oracle,
so the embedding and readout coordinates are format-specific and comparing the
two objectives there is confounded; the block coordinates are not, since both
objectives need the same internal transitions.  The block subspace is therefore
the headline, the full parameter vector is reported alongside it, and the
per-block breakdown tests the sharper prediction that credit reaching an early
block is suppressed more than credit reaching a late one.

Part B (loss profile).  Evaluate both objectives at points along
theta(a) = theta_0 + a (theta* - theta_0).  This is a straight line in real
parameter space, not the affine kernel family of Theorem 4; it is a descriptive
probe of the landscape the optimizer walks, and is reported as such.

Writes results/oracle_alignment/summary.json.  No training is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.eval import executor_comparison as c
from src.training.seed import set_seed

h = c.h
from handcoded.models import HandcodedOutcomeTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "oracle_alignment"

DEPTHS = (2, 4, 6, 8)
SEEDS = (42, 43, 44, 45, 46)          # the seeds of the five-seed depth study
N_CIRCUITS = 2000                      # population estimate for the gradient
CHUNK = 200
DATA_SEED = 123


def _buffers_to_parameters(module):
    """Re-register every buffer as a Parameter of identical value (Appendix G)."""
    for name, buf in list(module._buffers.items()):
        if buf is None:
            continue
        del module._buffers[name]
        value = buf.detach().clone().float()
        module.register_parameter(name, nn.Parameter(value))
    for child in module.children():
        _buffers_to_parameters(child)
    return module


def build_oracle(tok, depth, device):
    """theta*: the exact outcome executor, in the trainable parameterization."""
    model = HandcodedOutcomeTransformer(tok, depth, positions=3 * depth + 4)
    return _buffers_to_parameters(copy.deepcopy(model).cpu()).to(device)


def flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def population_gradient(model, batch, chunk=CHUNK):
    """Full-batch gradient of the mean token cross-entropy, accumulated in chunks."""
    model.zero_grad(set_to_none=True)
    n = batch.inputs.shape[0]
    total_tokens = int((batch.targets != -100).sum())
    for start in range(0, n, chunk):
        sub = h.LanguageBatch(batch.inputs[start:start + chunk],
                              batch.targets[start:start + chunk])
        logits = model(sub.inputs)
        # sum-reduction, so chunking is exact; normalise once at the end
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), sub.targets.flatten(), ignore_index=-100, reduction="sum"
        ) / total_tokens
        loss.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                   for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return g


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


@torch.no_grad()
def loss_at(model, batch, chunk=CHUNK):
    n = batch.inputs.shape[0]
    total = int((batch.targets != -100).sum())
    acc = 0.0
    for start in range(0, n, chunk):
        logits = model(batch.inputs[start:start + chunk])
        acc += float(torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), batch.targets[start:start + chunk].flatten(),
            ignore_index=-100, reduction="sum"))
    return acc / total


def parameter_slices(model, depth):
    """Coordinate ranges for the whole vector, the executor blocks, and each block."""
    spans, off = {}, 0
    for name, p in model.named_parameters():
        spans[name] = (off, off + p.numel()); off += p.numel()
    total = off
    groups = {"all": [(0, total)],
              "blocks": [v for k, v in spans.items() if k.startswith("blocks.")]}
    for k in range(depth):
        groups[f"block{k + 1}"] = [v for n, v in spans.items() if n.startswith(f"blocks.{k}.")]
    return groups, total


def gather(vec, spans):
    return torch.cat([vec[a:b] for a, b in spans])


def part_a(tok, device, depths, seeds):
    rows = []
    for depth in depths:
        circuits = c.unique_circuits(N_CIRCUITS, DATA_SEED, depth)
        data = {m: h.encode_dataset(circuits, tok, m).to(device) for m in ("outcome", "process")}
        oracle = flat(build_oracle(tok, depth, device).parameters())
        for seed in seeds:
            set_seed(seed)
            model = h.build_random_trainable_outcome_architecture(tok, depth, seed=seed, device=device)
            groups, total = parameter_slices(model, depth)
            theta0 = flat(model.parameters())
            direction = oracle - theta0
            row = {"depth": depth, "seed": seed, "n_params": int(total),
                   "oracle_distance": float(direction.norm())}
            grads = {}
            for mode in ("outcome", "process"):
                g = population_gradient(model, data[mode])
                grads[mode] = g
                row[f"gradnorm_{mode}"] = float(g.norm())
                row[f"loss_{mode}"] = loss_at(model, data[mode])
            rng = torch.Generator(device="cpu").manual_seed(10_000 + seed)
            rand = torch.randn(total, generator=rng).to(device)
            for gname, spans in groups.items():
                d = gather(direction, spans)
                row[f"dim_{gname}"] = int(d.numel())
                for mode in ("outcome", "process"):
                    row[f"cos_{gname}_{mode}"] = cosine(-gather(grads[mode], spans), d)
                row[f"cos_{gname}_chance"] = cosine(gather(rand, spans), d)
            rows.append(row)
            print(f"[A] D={depth} seed={seed} "
                  f"blocks: out={row['cos_blocks_outcome']:+.5f} "
                  f"proc={row['cos_blocks_process']:+.5f} "
                  f"chance={row['cos_blocks_chance']:+.5f} | "
                  f"all: out={row['cos_all_outcome']:+.5f} proc={row['cos_all_process']:+.5f}",
                  flush=True)
            del model, grads
            torch.cuda.empty_cache()
    return rows


def part_b(tok, device, depths, n_points, seed):
    rows = []
    alphas = [i / (n_points - 1) for i in range(n_points)]
    for depth in depths:
        circuits = c.unique_circuits(N_CIRCUITS, DATA_SEED, depth)
        data = {m: h.encode_dataset(circuits, tok, m).to(device) for m in ("outcome", "process")}
        oracle_model = build_oracle(tok, depth, device)
        oracle = flat(oracle_model.parameters())
        set_seed(seed)
        model = h.build_random_trainable_outcome_architecture(tok, depth, seed=seed, device=device)
        theta0 = flat(model.parameters())
        shapes = [p.shape for p in model.parameters()]
        sizes = [p.numel() for p in model.parameters()]

        def assign(vec):
            off = 0
            with torch.no_grad():
                for p, s, n in zip(model.parameters(), shapes, sizes):
                    p.copy_(vec[off:off + n].view(s)); off += n

        for a in alphas:
            assign(theta0 + a * (oracle - theta0))
            rows.append({"depth": depth, "seed": seed, "a": a,
                         "L_out": loss_at(model, data["outcome"]),
                         "L_proc": loss_at(model, data["process"])})
        head = rows[-n_points]
        tail = rows[-1]
        print(f"[B] D={depth}: L_out {head['L_out']:.4f} -> {tail['L_out']:.4f}, "
              f"L_proc {head['L_proc']:.4f} -> {tail['L_proc']:.4f}", flush=True)
        del model, oracle_model
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--points", type=int, default=101)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tok = h.make_tokenizer()
    print(f"device={device} depths={args.depths} seeds={args.seeds}", flush=True)

    results = {"config": {"depths": args.depths, "seeds": args.seeds,
                          "n_circuits": N_CIRCUITS, "data_seed": DATA_SEED,
                          "points": args.points, "device": str(device)}}
    results["alignment"] = part_a(tok, device, args.depths, args.seeds)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if not args.skip_b:
        results["profile"] = part_b(tok, device, args.depths, args.points, args.seeds[0])
        (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
