"""Softmax-logit rule table Z under process vs. outcome supervision, from Z=0.

Direct empirical test of body.tex's optimization-landscape remark (near
thm:complete-mixing): P_g(Z) = softmax_row(Z_g) is app:shared-kernel-embedding's
own construction, Prop. shared-kernel-embedding, at its proven C -> infinity
limit (the finite-attention-score correction is separately shown there to
vanish as O(e^{-C}); this script exercises the exact limit directly rather
than re-simulating attention noise). Z is the *only* trainable object --
routing is not a separate mechanism to freeze here, since composing D belief
updates via Z alone already matches L_out's matrix-product definition
exactly (no attention/retrieval layer sits between them). Z is initialized
at exactly zero (P_g = uniform for every g).

Two losses, same Z, same circuits:
  process: teacher-forced per-step cross-entropy, -log P_{g_t}(s_{t-1}, s_t),
    summed over all D revealed transitions (encodes\ every gold intermediate
    state as a target).
  outcome: -log(e_{s_0}^T P_{g_1}(Z) ... P_{g_D}(Z) e_y), composing the
    model's OWN belief distribution through all D steps with no teacher
    forcing of intermediate states -- literally eq:high-order-flatness's
    matrix product.

Full-batch (not mini-batch) AdamW each step, on a large fixed circuit sample,
so the computed gradient closely approximates the population gradient
thm:complete-mixing describes (avoids Adam amplifying pure sampling noise
into steps at a population-zero point).

Prediction: outcome logits stay near Z=0 (population-stationary point,
Thm. thm:complete-mixing); process logits escape immediately and converge
toward the true executor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from handcoded.gates import make_circuits, make_gate_names
from src.training.io import write_csv
from src.training.seed import default_device, set_seed

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "results" / "zero_init_softmax_table"
N_BITS = 4
N_STATES = 2 ** N_BITS


class SharedSoftmaxTable(nn.Module):
    """Z in R^{M x K x K}; P_g = softmax_row(Z_g). `init_std=0` -> every P_g exactly
    uniform (Thm. thm:complete-mixing's literal Z=0); `init_std>0` -> small random
    perturbation around it, testing whether the separation survives without exact
    symmetry (a real network is never initialized at exactly Z=0)."""

    def __init__(self, n_gates, n_states, init_std=0.0, generator=None):
        super().__init__()
        z = torch.zeros(n_gates, n_states, n_states)
        if init_std > 0:
            z = z + init_std * torch.randn(z.shape, generator=generator)
        self.logits = nn.Parameter(z)
        self.n_states = n_states

    def transition(self, gate_idx):
        return F.softmax(self.logits[gate_idx], dim=-1)


def _encode_circuits(circuits, gate_to_idx, n_bits=N_BITS):
    starts = torch.tensor([c.start for c in circuits], dtype=torch.long)
    gates = torch.tensor([[gate_to_idx[g] for g in c.gates] for c in circuits], dtype=torch.long)
    states = torch.tensor([list(c.states) for c in circuits], dtype=torch.long)
    answers = torch.tensor([c.answer for c in circuits], dtype=torch.long)
    return starts, gates, states, answers


def process_loss(model, starts, gates, states):
    """Teacher-forced per-transition CE, mean over all (instance, step) pairs."""
    prev = torch.cat([starts[:, None], states[:, :-1]], dim=1)  # (N, D): s_0..s_{D-1}
    depth = gates.shape[1]
    losses = []
    for t in range(depth):
        P = model.transition(gates[:, t])  # (N, K, K), one row-stochastic matrix per instance
        p_true = P[torch.arange(P.shape[0]), prev[:, t], states[:, t]]
        losses.append(-torch.log(p_true.clamp_min(1e-12)))
    return torch.cat(losses).mean()


def outcome_loss(model, starts, gates, answers):
    """Compose D belief updates with no teacher forcing; CE on the final answer."""
    n = starts.shape[0]
    belief = F.one_hot(starts, num_classes=model.n_states).float()  # (N, K)
    depth = gates.shape[1]
    for t in range(depth):
        P = model.transition(gates[:, t])  # (N, K, K)
        belief = torch.einsum("nk,nkj->nj", belief, P)
    p_true = belief[torch.arange(n), answers]
    return -torch.log(p_true.clamp_min(1e-12))


def answer_accuracy(model, starts, gates, answers):
    with torch.no_grad():
        n = starts.shape[0]
        belief = F.one_hot(starts, num_classes=model.n_states).float()
        depth = gates.shape[1]
        for t in range(depth):
            P = model.transition(gates[:, t])
            belief = torch.einsum("nk,nkj->nj", belief, P)
        pred = belief.argmax(dim=-1)
        return float((pred == answers).float().mean())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--train-size", type=int, default=50000)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--init-std", type=float, default=0.0,
                    help="0 = exact Z=0 (deterministic); >0 = small random perturbation, seeded per run.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--data-seed", type=int, default=123)
    p.add_argument("--test-seed", type=int, default=9000)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.steps, args.train_size, args.test_size, args.log_every = 40, 2000, 500, 5
    return args


def run_seed_mode(seed, mode, args, gate_to_idx, train, test, device, checkpoints):
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = SharedSoftmaxTable(len(gate_to_idx), N_STATES, init_std=args.init_std, generator=generator).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    starts, gates, states, answers = train
    t_starts, t_gates, t_states, t_answers = test

    rows = []

    def checkpoint_row(step):
        with torch.no_grad():
            row = {
                "mode": mode, "seed": seed, "step": step,
                "train_answer_accuracy": answer_accuracy(model, starts, gates, answers),
                "test_answer_accuracy": answer_accuracy(model, t_starts, t_gates, t_answers),
                "logit_norm": float(model.logits.norm()),
            }
        return row

    rows.append(checkpoint_row(0))
    checkpoint_set = set(checkpoints)
    progress = tqdm(range(1, args.steps + 1), desc=f"softmax-table/{mode}/s{seed}")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = (
            process_loss(model, starts, gates, states) if mode == "process"
            else outcome_loss(model, starts, gates, answers).mean()
        )
        loss.backward()
        optimizer.step()
        if step not in checkpoint_set:
            continue
        row = checkpoint_row(step)
        rows.append(row)
        progress.set_postfix(test_acc=f"{row['test_answer_accuracy']:.3f}", logit_norm=f"{row['logit_norm']:.3f}")
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or default_device())
    print(f"device={device} depth={args.depth} steps={args.steps} seeds={args.seeds}", flush=True)

    gate_names = make_gate_names(N_BITS)
    gate_to_idx = {name: i for i, name in enumerate(gate_names)}
    train_circuits = make_circuits(args.train_size, args.data_seed, depth=args.depth, n_bits=N_BITS)
    test_circuits = make_circuits(args.test_size, args.test_seed, depth=args.depth, n_bits=N_BITS)
    train = tuple(t.to(device) for t in _encode_circuits(train_circuits, gate_to_idx))
    test = tuple(t.to(device) for t in _encode_circuits(test_circuits, gate_to_idx))
    checkpoints = sorted(set([0, 1, 2, 5, 10, 20]) | set(range(args.log_every, args.steps + 1, args.log_every)) | {args.steps})

    rows = []
    for mode in ("process", "outcome"):
        for seed in args.seeds:
            rows.extend(run_seed_mode(seed, mode, args, gate_to_idx, train, test, device, checkpoints))

    csv_path = write_csv(out / "metrics.csv", rows)
    final_step = max(r["step"] for r in rows)
    final = [r for r in rows if r["step"] == final_step]
    summary = {
        "experiment": "zero_init_softmax_table",
        "depth": args.depth, "steps": args.steps, "seeds": args.seeds,
        "train_size": args.train_size, "lr": args.lr, "init_std": args.init_std, "device": str(device),
        "smoke": bool(args.smoke), "csv": str(csv_path),
        "final_by_mode": {
            mode: {
                "test_answer_accuracy_mean": float(np.mean([r["test_answer_accuracy"] for r in final if r["mode"] == mode])),
                "test_answer_accuracy_std": float(np.std([r["test_answer_accuracy"] for r in final if r["mode"] == mode], ddof=1)) if len(args.seeds) > 1 else 0.0,
                "logit_norm_mean": float(np.mean([r["logit_norm"] for r in final if r["mode"] == mode])),
            }
            for mode in ("process", "outcome")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["final_by_mode"], indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    main()
