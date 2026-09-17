"""Trained-network test of group-symmetry non-identifiability
(app:group-symmetry-nonidentifiability).

On shift_symmetry_8 (additive shifts on Z_16), the true gate-to-permutation
assignment {T_g} and the alternative {T'_g := T_g + 8 mod 16} produce
IDENTICAL 8-step compositions for every gate sequence (Z_16 is abelian and
8 has order 2, so the extra +8 per step cancels over any even number of
steps). Outcome supervision cannot distinguish {T_g} from {T'_g} even with
perfect coverage of every 8-gate word; process supervision, which displays
s_t under the canonical T-labeling, pins {T_g} down uniquely.

This trains one outcome-only and one process model on the same task (same
architecture/budget), then asks a linear probe at each of the 8 intermediate
positions to recover the state under BOTH labelings:
  true label:        s_t
  alternative label:  s_t' = (s_t + 8*t) mod 16   (agrees with s_t at every
                       even t, differs by a constant +8 at every odd t --
                       the two labelings are literally identical at half the
                       positions, so any global bias toward one or the other
                       shows up specifically at the odd positions)

Prediction: both models reach comparable, high final-answer accuracy (the
task is equally learnable regardless of which internal labeling is used);
the process model's probe accuracy against the TRUE label is high at every
position (it is directly supervised on it); the outcome model's probe
accuracy against the TRUE label should be markedly lower, especially at the
odd positions where "true" and "alternative" disagree, since outcome
supervision gives it no reason to prefer one over the other.
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
OUT = ROOT / "results" / "identifiability_probe"

SHIFT_MODULUS = 16
N_STEPS = 8


def _gate_end_positions(n_steps: int) -> list[int]:
    """0-indexed char offset of each gate's last char within the prompt.

    Prompt is "s{2 digits};u" (5 chars) then n_steps gates of "+{2 digits}"
    (3 chars each): gate t (1-indexed) ends at 5 + 3*t - 1 = 3t + 4.
    """
    return [3 * t + 4 for t in range(1, n_steps + 1)]


def _true_and_alt_labels(inst) -> tuple[list[int], list[int]]:
    """True state s_t and alternative s_t' = (s_t + 8t) mod 16 at each step."""
    true_states = [int(step.split(">", 1)[1]) for step in inst.correct_trace.split(" ")]
    alt_states = [(s + 8 * (t + 1)) % SHIFT_MODULUS for t, s in enumerate(true_states)]
    return true_states, alt_states


def _build_batch(instances, task, mode: str):
    tokenizer, block_size = task.tokenizer, task.block_size
    target_fn = {
        "outcome": lambda inst: f"{ANSWER_SEP}{inst.gold}\n",
        "process": lambda inst: f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n",
    }[mode]
    xs, ys, masks = [], [], []
    for inst in instances:
        prompt_ids = tokenizer.encode(inst.prompt)
        target_ids = tokenizer.encode(target_fn(inst))
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
    return (torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long),
            torch.tensor(masks, dtype=torch.float32))


def _final_answer_accuracy(model, task, instances, device, batch_size=256):
    """All shift_symmetry_8 prompts have identical length (fixed-width
    2-digit start state and 2-digit gate offsets), so no padding/bucketing
    is needed -- unlike src/eval/generate.py's general bucket-by-length
    handling for variable-length prompts."""
    tokenizer = task.tokenizer
    correct = 0
    for i in range(0, len(instances), batch_size):
        chunk = instances[i:i + batch_size]
        prompts = [tokenizer.encode(inst.prompt) for inst in chunk]
        prompt_len = len(prompts[0])
        assert all(len(p) == prompt_len for p in prompts), "expected fixed-length prompts"
        x = torch.tensor(prompts, dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(x, max_new_tokens=6, stop_id=tokenizer.newline_id, greedy=True)
        gen_text = [tokenizer.decode(row.tolist()) for row in gen[:, prompt_len:].cpu()]
        for inst, text in zip(chunk, gen_text):
            pred = task.extract_answer(text)
            if pred is not None and int(pred) == int(inst.gold):
                correct += 1
    return correct / len(instances)


def _capture_activations(model, x, positions):
    """Residual stream (final blocks output) at each fixed position, for
    every instance in the batch. `positions` is shared across the batch
    (same prompt structure for every instance in this task)."""
    captured = {}

    def hook(_module, _inp, out):
        captured["h"] = out.detach()

    handle = model.blocks.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    h = captured["h"].cpu().numpy()  # (B, T, C)
    return h[:, positions, :]  # (B, len(positions), C)


def _probe_accuracy(features, labels, n_classes=SHIFT_MODULUS):
    if len(set(labels)) < 2:
        return float("nan")
    h_tr, h_te, y_tr, y_te = train_test_split(features, labels, test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(h_tr, y_tr)
    return float(clf.score(h_te, y_te))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[2001, 2002, 2003])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--probe-size", type=int, default=2000)
    p.add_argument("--eval-size", type=int, default=1000)
    p.add_argument("--probe-seed", type=int, default=9101)
    p.add_argument("--eval-seed", type=int, default=7777)
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
    task = TASKS["shift_symmetry_8"]
    positions = _gate_end_positions(N_STEPS)

    rows = []
    for mode in ["outcome", "process"]:
        for seed in args.seeds:
            set_seed(seed)
            train_instances = generate_unique(task, args.train_size, args.train_seed)
            train_dataset = SupervisionDataset(train_instances, task, mode)
            loader = make_loader(train_dataset, args, device)
            model = build_gpt(task, args, device)
            optimizer = make_optimizer(model, args, device)

            train_steps(model, loader, optimizer, device, args.steps,
                        grad_clip=args.grad_clip, desc=f"identifiability-{mode}/s{seed}")

            model.eval()
            eval_instances = generate_unique(
                task, args.eval_size, args.eval_seed, {i.prompt for i in train_instances})
            answer_acc = _final_answer_accuracy(model, task, eval_instances, device)

            probe_instances = generate_unique(
                task, args.probe_size, args.probe_seed,
                {i.prompt for i in train_instances} | {i.prompt for i in eval_instances})
            x, _, _ = _build_batch(probe_instances, task, "process")
            x = x.to(device)
            acts = _capture_activations(model, x, positions)  # (B, D, C)

            true_labels = np.array([_true_and_alt_labels(inst)[0] for inst in probe_instances])
            alt_labels = np.array([_true_and_alt_labels(inst)[1] for inst in probe_instances])

            for t in range(N_STEPS):
                acc_true = _probe_accuracy(acts[:, t, :], true_labels[:, t])
                acc_alt = _probe_accuracy(acts[:, t, :], alt_labels[:, t])
                row = {
                    "mode": mode, "seed": seed, "position": t + 1,
                    "answer_accuracy": answer_acc,
                    "probe_acc_true_label": acc_true,
                    "probe_acc_alt_label": acc_alt,
                    "labels_agree_here": bool(np.array_equal(true_labels[:, t], alt_labels[:, t])),
                }
                rows.append(row)
                print(f"mode={mode:8s} seed={seed} t={t+1} answer_acc={answer_acc:.3f} "
                      f"probe(true)={acc_true:.3f} probe(alt)={acc_alt:.3f} "
                      f"agree={row['labels_agree_here']}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "identifiability_probe.csv"
    write_csv(csv_path, rows)
    persist = {
        "experiment": "identifiability_probe",
        "task": "shift_symmetry_8",
        "seeds": list(args.seeds),
        "steps": args.steps,
        "train_size": args.train_size,
        "probe_size": args.probe_size,
        "eval_size": args.eval_size,
        "csv": str(csv_path),
        "rows": rows,
    }
    (OUT / "identifiability_probe_persist.json").write_text(json.dumps(persist, indent=2) + "\n")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
