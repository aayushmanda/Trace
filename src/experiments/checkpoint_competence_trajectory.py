"""Checkpoint-trajectory validation of the rho_c(a,D,beta) frontier (Fig. 3)
against a real trained Transformer, not the population model.

fig:prop8-frontier plots the closed-form stalling threshold
rho_c(a,D,beta=1/D) as a function of a population-model competence
parameter a, with the caption stating plainly that no model is trained for
that figure. This script asks the question the caption leaves open: does a
real network's own trajectory through (a, rho) space respect that
threshold -- stalling when it starts below the curve, climbing through it
when it starts above?

Competence proxy. P_g(a) puts probability (1-a)/K + a on the true
successor and (1-a)/K on every other state. A trained network has no such
table explicitly, but at a local-transition position (the 4 state-bit
characters right after a gate specifier), its own teacher-forced softmax
gives p_correct = product of its 4 per-bit probabilities of the true next
state, exactly the quantity P_g(a) assigns to the true successor. Solving
(1-a)/K + a = p_correct for a gives

    a_hat = (K * p_correct - 1) / (K - 1)

This is read off directly, the same *kind* of quantity (a probability) the
theory is about, not an indirect signal -- unlike the two earlier, now
abandoned, probe experiments. The one assumption the closed-form curve
itself makes that this readout does not need is that wrong mass is spread
uniformly over the K-1 wrong states; a real network's errors need not be
uniform. So even a clean match only confirms the qualitative sign-transfer
(stall below the curve, climb above it), not the exact curve shape.

Protocol: train boolean_circuit_8 under process supervision with a fixed
mixed_process rho (RatioDataset, matching reliability.py's own corruption
mechanism, i.e. table 5-6 / fig 3's own experimental family), at a few rho
values chosen to bracket the existing depth_reliability sweep's observed
stall/rise transition (results/paper/depth_reliability/boolean_circuit_8.csv:
rho=0.5 -> ~11% terminal accuracy, rho=0.8 -> ~55%, rho=1.0 -> ~88%),
checkpointing throughout an 8000-step run at the architecture and budget
already used for Table 1 (configs/experiments/e1_five_condition_rerun.yaml).
At each checkpoint, on a fixed held-out batch of CLEAN instances (disjoint
from every training set), measures a_hat via the model's own per-transition
teacher-forced probabilities (reusing gradient_alignment.py's local-position
masking machinery), recording the full (step, rho, seed, a_hat) trajectory.

Writes results/checkpoint_competence_trajectory/trajectory.csv and
results/checkpoint_competence_trajectory/trajectory_persist.json, plus a
comparison figure overlaying these trajectories on the closed-form
rho_c(a, D=8, beta=1/8) curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.dataclass import ANSWER_SEP
from src.data.datasets import RatioDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.gradient_alignment import _encode, _state_char_offsets, _tensors
from src.experiments.prop8_frontier import closed_form_rho_c
from src.training.io import write_csv
from src.training.loop import train_with_checkpoints
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import default_device, maybe_high_precision, set_seed

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checkpoint_competence_trajectory"
FIG = ROOT / "Paper" / "figures"

K = 16
D = 8


def build_per_t_probe_batch(instances, task, n_gates: int):
    """Per-transition (x, y, mask) triples isolating exactly the 4 state-bit
    characters after each of the D gates, on the CLEAN trace -- same
    construction as gradient_alignment.build_reference_batch's 'per_t', but
    this script only needs the encodings, not gradients."""
    tokenizer, block_size = task.tokenizer, task.block_size
    per_t_rows = [[] for _ in range(n_gates)]
    for inst in instances:
        proc_target = f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n"
        state_offsets = _state_char_offsets(inst.correct_trace)
        assert len(state_offsets) == 4 * n_gates, (len(state_offsets), n_gates)
        for t in range(n_gates):
            t_offsets = state_offsets[4 * t: 4 * t + 4]
            per_t_rows[t].append(_encode(tokenizer, inst.prompt, proc_target, block_size,
                                          local_offsets=[o + 1 for o in t_offsets]))
    return [_tensors(rows, tokenizer) for rows in per_t_rows]


@torch.no_grad()
def per_transition_p_correct(model, per_t_batch, device) -> list[float]:
    """Average, over the probe batch, of the joint probability the model's
    own teacher-forced softmax assigns to the true 4-bit successor state, one
    value per transition position t."""
    out = []
    for x, y, mask in per_t_batch:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        joint_logp = (token_logp * mask).sum(dim=1)
        p_correct = joint_logp.exp()
        out.append(float(p_correct.mean()))
    return out


def a_hat_from_p_correct(p_correct: float, k: int = K) -> float:
    return (k * p_correct - 1) / (k - 1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.5, 0.8, 1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--checkpoint-every", type=int, default=400)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--probe-seed", type=int, default=101)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--ratio-seed", type=int, default=777)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    args = p.parse_args(argv)

    n_gates = int(args.task.rsplit("_", 1)[-1])
    assert n_gates == D
    device = torch.device(default_device())
    device = maybe_high_precision(device, compile=args.compile, bf16=args.bf16)
    task = TASKS[args.task]

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    probe_instances = generate_unique(task, args.probe_size, args.probe_seed,
                                       {i.prompt for i in train_instances})
    probe_batch = build_per_t_probe_batch(probe_instances, task, n_gates)
    checkpoints = sorted(set(range(0, args.steps + 1, args.checkpoint_every)) | {args.steps})
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))

    rows = []
    for rho in sorted(set(args.rhos)):
        dataset = RatioDataset(train_instances, task, "mixed_process", rho=rho,
                                ratio_scores=ratio_scores)
        for seed in args.seeds:
            set_seed(seed)
            loader = make_loader(dataset, args, device)
            model = build_gpt(task, args, device)
            optimizer = make_optimizer(model, args, device)

            def on_checkpoint(step, model, loss, _rho=rho, _seed=seed):
                model.eval()
                pt = per_transition_p_correct(model, probe_batch, device)
                p_correct = float(np.mean(pt))
                a_hat = a_hat_from_p_correct(p_correct)
                row = {
                    "task": args.task, "rho": _rho, "seed": _seed, "step": step,
                    "train_loss": float(loss.detach()) if loss is not None else "",
                    "p_correct": p_correct, "a_hat": a_hat,
                }
                for t, v in enumerate(pt):
                    row[f"p_correct_t{t}"] = v
                rows.append(row)
                print(f"rho={_rho:.2f} seed={_seed} step={step:5d} "
                      f"p_correct={p_correct:.4f} a_hat={a_hat:+.4f}")
                model.train()

            train_with_checkpoints(model, loader, optimizer, device, checkpoints, on_checkpoint,
                                    grad_clip=args.grad_clip, desc=f"ckpt-traj/rho={rho:.2f}/s{seed}")
            del model, optimizer, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del dataset

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "trajectory.csv"
    write_csv(csv_path, rows)
    persist = {
        "experiment": "checkpoint_competence_trajectory",
        "task": args.task, "D": D, "K": K,
        "rhos": sorted(set(args.rhos)), "seeds": list(args.seeds),
        "steps": args.steps, "checkpoints": checkpoints,
        "train_size": args.train_size, "probe_size": args.probe_size,
        "probe_seed": args.probe_seed, "train_seed": args.train_seed,
        "ratio_seed": args.ratio_seed, "batch_size": args.batch_size,
        "lr": args.lr, "embedding": args.embedding, "heads": args.heads,
        "layers": args.layers, "csv": str(csv_path), "n_rows": len(rows), "rows": rows,
        "note": "a_hat = (K*p_correct-1)/(K-1) from the model's own per-transition "
                "teacher-forced softmax on a held-out clean probe batch; compare "
                "against closed_form_rho_c(a, D=8, beta=1/8) from prop8_frontier.py.",
    }
    (OUT / "trajectory_persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
