"""Per-parameter-group breakdown of the gradient-alignment diagnostic (App. F).

Same setup as gradient_alignment.py (same architecture, budget, reference
batch construction, and the three gradients g_local/g_proc/g_out), but
instead of one cosine/norm-ratio computed on the fully flattened parameter
vector, this computes the same statistics separately for each named
parameter group: embeddings, W_Q, W_K, W_V (the three row-blocks of the
fused c_attn weight), W_O (attention output projection), W_1/W_2 (the two
feed-forward layers), the readout W_U (lm_head), and LayerNorm gains/biases
(pooled across both blocks and both LayerNorms, for completeness; not
analyzed in the paper's discussion).

This tests where inside the Transformer the useful process credit and the
large-but-uncorrelated outcome credit actually live, a question the
fully-flattened diagnostic in gradient_alignment.py cannot answer on its
own. No model checkpoints are saved to disk (same as gradient_alignment.py);
this requires its own training run rather than reusing archived data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.alignment.gradient_alignment import _masked_loss, build_reference_batch
from src.training.io import write_csv
from src.training.loop import train_with_checkpoints
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import default_device, maybe_high_precision, set_seed

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "gradient_alignment_by_group"

N_EMBD_MARKER = "c_attn.weight"


def _param_groups(model, n_embd: int):
    """Maps each named parameter (or row-slice of one) to a role string.

    Returns a dict {role: [(name, row_slice_or_None), ...]}. c_attn.weight
    (shape 3*n_embd x n_embd, from `q, k, v = c_attn(x).chunk(3, dim=-1)`)
    is split into its three row-blocks: rows [0:n_embd)=W_Q,
    [n_embd:2*n_embd)=W_K, [2*n_embd:3*n_embd)=W_V.
    """
    groups: dict[str, list] = {
        "embed": [], "W_Q": [], "W_K": [], "W_V": [], "W_O": [],
        "W_1": [], "W_2": [], "W_U": [], "layernorm": [],
    }
    for name, _ in model.named_parameters():
        if "embedding_table" in name:
            groups["embed"].append((name, None))
        elif N_EMBD_MARKER in name:
            groups["W_Q"].append((name, slice(0, n_embd)))
            groups["W_K"].append((name, slice(n_embd, 2 * n_embd)))
            groups["W_V"].append((name, slice(2 * n_embd, 3 * n_embd)))
        elif ".sa.proj." in name:
            groups["W_O"].append((name, None))
        elif ".ffwd.net.0." in name:
            groups["W_1"].append((name, None))
        elif ".ffwd.net.2." in name:
            groups["W_2"].append((name, None))
        elif "lm_head" in name:
            groups["W_U"].append((name, None))
        elif name.startswith("ln") or ".ln1." in name or ".ln2." in name or "ln_f" in name:
            groups["layernorm"].append((name, None))
        else:
            raise ValueError(f"unclassified parameter: {name}")
    return groups


def _grouped_grad(model, loss, groups):
    """Returns {role: flat_grad_tensor} by backpropagating once and slicing."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    out = {}
    for role, entries in groups.items():
        pieces = []
        for name, row_slice in entries:
            g = named[name].grad
            if g is None:
                continue
            pieces.append((g[row_slice] if row_slice is not None else g).detach().reshape(-1))
        out[role] = torch.cat(pieces) if pieces else torch.zeros(0)
    model.zero_grad(set_to_none=True)
    return out


def _stats(g_local, g):
    dot = float(torch.dot(g_local, g))
    n_local = float(g_local.norm())
    n_g = float(g.norm())
    return {
        "dot": dot,
        "normalized_projection": dot / (n_local ** 2) if n_local > 0 else float("nan"),
        "cosine": dot / (n_local * n_g) if n_local > 0 and n_g > 0 else float("nan"),
        "grad_norm": n_g,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--ref-size", type=int, default=256)
    p.add_argument("--ref-seed", type=int, default=101)
    p.add_argument("--train-seed", type=int, default=501)
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
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    n_gates = int(args.task.rsplit("_", 1)[-1])
    device = torch.device(default_device())
    device = maybe_high_precision(device, compile=args.compile, bf16=args.bf16)
    task = TASKS[args.task]

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    ref_instances = generate_unique(task, args.ref_size, args.ref_seed,
                                     {i.prompt for i in train_instances})
    ref_batch = build_reference_batch(ref_instances, task, n_gates)
    checkpoints = sorted({0, args.steps // 4, args.steps // 2, args.steps})

    from src.data.datasets import SupervisionDataset

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        train_dataset = SupervisionDataset(train_instances, task, "process")
        loader = make_loader(train_dataset, args, device)
        model = build_gpt(task, args, device)
        groups = _param_groups(model, args.embedding)
        optimizer = make_optimizer(model, args, device)

        def on_checkpoint(step, model, loss, _seed=seed, _groups=groups):
            model.eval()
            g_local = _grouped_grad(model, _masked_loss(model, ref_batch["local"], device), _groups)
            g_proc = _grouped_grad(model, _masked_loss(model, ref_batch["proc"], device), _groups)
            g_out = _grouped_grad(model, _masked_loss(model, ref_batch["out"], device), _groups)
            summary = {}
            for role in _groups:
                row = {"seed": _seed, "step": step, "role": role,
                       "local_grad_norm": float(g_local[role].norm())}
                for name, g in [("proc", g_proc[role]), ("out", g_out[role])]:
                    s = _stats(g_local[role], g)
                    for k, v in s.items():
                        row[f"{name}_{k}"] = v
                rows.append(row)
                summary[role] = (row["proc_cosine"], row["out_cosine"])
            preview = " ".join(
                f"{role}:proc={summary[role][0]:+.2f}/out={summary[role][1]:+.2f}"
                for role in ("W_Q", "W_U")
            )
            print(f"seed={_seed} step={step:5d} {preview}")
            model.train()

        train_with_checkpoints(model, loader, optimizer, device, checkpoints, on_checkpoint,
                                grad_clip=args.grad_clip, desc=f"grad-align-by-group/s{seed}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output or (out_dir / "gradient_alignment_by_group.csv")
    csv_path = Path(csv_path)
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {csv_path}")
    write_csv(csv_path, rows)
    persist = {
        "experiment": "gradient_alignment_by_group",
        "task": args.task,
        "seeds": list(args.seeds),
        "steps": args.steps,
        "checkpoints": checkpoints,
        "train_size": args.train_size,
        "ref_size": args.ref_size,
        "ref_seed": args.ref_seed,
        "train_seed": args.train_seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding": args.embedding,
        "heads": args.heads,
        "layers": args.layers,
        "csv": str(csv_path),
        "n_rows": len(rows),
    }
    persist_path = csv_path.with_name(csv_path.stem + "_persist.json")
    persist_path.write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")
    print(f"persist: {persist_path}")


if __name__ == "__main__":
    main()
