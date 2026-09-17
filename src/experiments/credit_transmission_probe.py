"""Trained-network validation of Theorem 1's forward x backward credit
factorization.

Not a test of Corollary 2's chi_t upper bound, which submultiplicativity
gives no reason to expect is pointwise tight for a fixed prompt
distribution (misaligned singular subspaces generically make the true
norm decay faster than the product-of-norms bound). This instead tests
the underlying EXACT identity of Theorem 1 directly: outcome credit for
one transition is a forward signal times a backward signal, and process
supervision needs neither.

For each depth D in {2,4,6,8}, trains the same architecture/optimizer/
budget as the depth-capacity sweep (App. A.1, tab:depth-capacity) under
outcome-only supervision, 3 seeds, 8000 steps, 20000 train instances. At
the frozen final checkpoint, on a fresh held-out probe set, measures at
the residual-stream position right after gate 1 (fixed across D, so
composition depth is varied without moving the probed position itself):

  forward proxy:  held-out linear-probe accuracy for the true
                   intermediate state s_1 from that activation --
                   analogous to ||Pi q_bar||, how much source-state
                   information the network's own representation retains.
  backward proxy: mean L2 norm of d(outcome loss)/d(that same
                   activation) -- analogous to ||Pi b_bar||, how much
                   the eventual loss is sensitive to that position's
                   representation. Requires D-1 remaining gates' worth
                   of backprop to reach that position, so it is expected
                   to contract with D while the forward proxy (which
                   only depends on gate 1) should not.

Reports both proxies and their product per (D, seed), for comparison
against the already-measured outcome-only answer accuracy at each D
(results/paper/depth_reliability/boolean_circuit_{D}_outcome.csv).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.dataclass import ANSWER_SEP
from src.data.datasets import SupervisionDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.training.io import write_csv
from src.training.loop import train_steps
from src.training.optim import build_gpt, make_loader, make_optimizer
from src.training.seed import default_device, maybe_high_precision, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "credit_transmission_probe"

# "i" + 4 state-bit chars + ";" + "u" = 7 characters before gate 1 starts.
_GATE1_PREFIX_LEN = 7


def _gate1_end_index(inst) -> int:
    """0-indexed character offset, within inst.prompt, of gate 1's last char."""
    gate1 = inst.correct_trace.split(">", 1)[0]
    return _GATE1_PREFIX_LEN + len(gate1) - 1


def _state1_label(inst) -> int:
    state1_bits = inst.correct_trace.split(">", 1)[1][:4]
    return int(state1_bits, 2)


def _build_outcome_batch(instances, task):
    tokenizer, block_size = task.tokenizer, task.block_size
    xs, ys, masks, positions, labels = [], [], [], [], []
    for inst in instances:
        prompt_ids = tokenizer.encode(inst.prompt)
        target_ids = tokenizer.encode(f"{ANSWER_SEP}{inst.gold}\n")
        full = prompt_ids + target_ids
        width = block_size - 1
        x = [tokenizer.pad_id] * width
        y = [tokenizer.pad_id] * width
        mask = [0.0] * width
        n = len(full) - 1
        x[:n] = full[:-1]
        y[:n] = full[1:]
        start = len(prompt_ids) - 1
        for i in range(start, n):
            mask[i] = 1.0
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        pos = _gate1_end_index(inst)
        assert pos < len(prompt_ids), (pos, len(prompt_ids))
        positions.append(pos)
        labels.append(_state1_label(inst))
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.float32)
    return x, y, mask, np.array(positions), np.array(labels)


def _capture_forward_backward(model, x, y, mask, positions, device):
    """Runs one forward+backward pass, returns (h, grad_h) at each
    instance's own gate-1 position, from the residual stream between
    the two Transformer blocks (output of block 0 / input of block 1).

    Hooking the *final* output of model.blocks does not work: everything
    downstream of that point (ln_f, lm_head) is position-wise, so a loss
    masked to the answer positions has zero direct functional dependence
    there on any other position's row -- the cross-position mixing that
    actually carries information from an earlier position to a later
    one's loss happens *inside* the attention layers. Probing between
    the two blocks instead means the captured activation both (a) has
    already been processed by one full attention+FFN layer, so it is a
    meaningful forward representation to probe, and (b) still has to
    pass through the second block's attention for gradient to reach a
    later position's loss, so backward sensitivity is measurable.
    """
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    captured = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        captured["h"] = out

    handle = model.blocks[0].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, mask=mask)
        loss.backward()
        h = captured["h"]
        grad_h = h.grad
        if grad_h is None:
            raise RuntimeError("gradient did not reach model.blocks output")
        h_np = h.detach().cpu().numpy()
        g_np = grad_h.detach().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    b_idx = np.arange(len(positions))
    return h_np[b_idx, positions, :], g_np[b_idx, positions, :]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--depths", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--probe-size", type=int, default=2000)
    p.add_argument("--probe-seed", type=int, default=9101)
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
    args = p.parse_args(argv)

    device = torch.device(default_device())
    device = maybe_high_precision(device, compile=False, bf16=False)

    rows = []
    for D in args.depths:
        task = TASKS[f"boolean_circuit_{D}"]
        for seed in args.seeds:
            set_seed(seed)
            train_instances = generate_unique(task, args.train_size, args.train_seed)
            train_dataset = SupervisionDataset(train_instances, task, "outcome")
            loader = make_loader(train_dataset, args, device)
            model = build_gpt(task, args, device)
            optimizer = make_optimizer(model, args, device)

            train_steps(model, loader, optimizer, device, args.steps,
                        grad_clip=args.grad_clip, desc=f"probe-train D={D}/s{seed}")

            probe_instances = generate_unique(
                task, args.probe_size, args.probe_seed,
                {i.prompt for i in train_instances},
            )
            model.eval()
            x, y, mask, positions, labels = _build_outcome_batch(probe_instances, task)
            h, g = _capture_forward_backward(model, x, y, mask, positions, device)

            h_tr, h_te, y_tr, y_te = train_test_split(
                h, labels, test_size=0.3, random_state=0,
            )
            clf = LogisticRegression(max_iter=2000)
            clf.fit(h_tr, y_tr)
            forward_proxy = float(clf.score(h_te, y_te))
            backward_proxy = float(np.linalg.norm(g, axis=1).mean())

            row = {
                "depth": D,
                "seed": seed,
                "forward_probe_acc": forward_proxy,
                "backward_grad_norm": backward_proxy,
                "product": forward_proxy * backward_proxy,
            }
            rows.append(row)
            print(f"D={D:2d} seed={seed} forward_acc={forward_proxy:.3f} "
                  f"backward_grad_norm={backward_proxy:.4e} product={row['product']:.4e}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "credit_transmission_probe.csv"
    write_csv(csv_path, rows)
    persist = {
        "experiment": "credit_transmission_probe",
        "depths": list(args.depths),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "train_size": args.train_size,
        "probe_size": args.probe_size,
        "probe_seed": args.probe_seed,
        "train_seed": args.train_seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding": args.embedding,
        "heads": args.heads,
        "layers": args.layers,
        "csv": str(csv_path),
        "rows": rows,
    }
    (OUT / "credit_transmission_probe_persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
