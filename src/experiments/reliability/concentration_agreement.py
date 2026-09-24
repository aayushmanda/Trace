"""Which rule does the trained concentration model follow? Agreement with reference predictors.

Reuses the checkpoints and corpora written by review_concentration_validation.py. On held-out gold
(clean) prefixes, compares the network's next-state prediction with the true rule T, the corpus table Q
(fitted to the same displayed transitions), and the coherent wrong rule W. Every corrupted trace is wrong
at every displayed step, so after one correct displayed transition the context-dependent optimum is T;
only the first step has no such evidence. Rows are therefore reported for step 1 and for steps 2..D.

Run from the repository root:
  uv run --offline --no-project --with 'torch==2.7.1' --with numpy --with pandas \
    python -m src.experiments.reliability.concentration_agreement \
    --archive results/review_validation/concentration --output results/paper/concentration_agreement.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from handcoded.data import encode_dataset
from handcoded.gates import make_gate_names
from handcoded.models import build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.experiments.circuit_match.handcoded_trace_gradients import continuation_slices
from src.experiments.circuit_match.review_functional_validation import keys, unique_circuits


@torch.no_grad()
def state_probabilities(model, batch, positions):
    chunks = [model(batch.inputs[i:i + 128])[:, list(positions), :16].softmax(-1).cpu().numpy()
              for i in range(0, len(batch), 128)]
    return np.concatenate(chunks)  # (circuits, steps, 16), renormalized over state tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    tokenizer = make_tokenizer()
    positions = continuation_slices(4).state
    gates = make_gate_names()
    gate_ids = {g: i for i, g in enumerate(gates)}
    rows = []
    for checkpoint in sorted(args.archive.glob("checkpoint_s*_rho*_lambda*.pt")):
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        seed, rho, lam = saved["seed"], saved["rho"], saved["concentration"]
        if seed not in args.seeds:
            continue
        tag = checkpoint.stem.removeprefix("checkpoint_")
        corpus = np.load(args.archive / f"corpus_{tag}.npz")
        q, truth = corpus["q"], corpus["truth"]
        train = unique_circuits(args.train_size, 123 + (seed - 42) * 1000)
        evaluation = unique_circuits(args.eval_size, 39000 + seed, keys(train))
        batch = encode_dataset(evaluation, tokenizer, "process").to(args.device)
        model = build_random_trainable_process_architecture(tokenizer, 4, seed=seed, init_std=.02, device=args.device)
        model.load_state_dict(saved["model"])
        model.eval()
        probs = state_probabilities(model, batch, positions)
        records = []
        for i, circuit in enumerate(evaluation):
            current = circuit.start
            for t, gate in enumerate(circuit.gates):
                op = gate_ids[gate]
                correct = int(truth[op, current])
                p = probs[i, t]
                q_row = q[op, current] / max(q[op, current].sum(), 1e-12)
                records.append(dict(
                    first=t == 0,
                    agree_T=p.argmax() == correct,
                    agree_Q=p.argmax() == q_row.argmax(),
                    agree_W=p.argmax() == correct ^ _coherent_wrong_mask(gate),
                    kl_T=-np.log(max(p[correct], 1e-12)),
                    kl_Q=float(np.sum(np.where(q_row > 0, q_row * np.log(q_row / np.maximum(p, 1e-12)), 0))),
                    table_correct=q_row.argmax() == correct,
                ))
                current = circuit.states[t]
        frame = pd.DataFrame(records)
        for first, part in frame.groupby("first"):
            rows.append(dict(seed=seed, rho=rho, concentration=lam, step="1" if first else "2..D",
                             positions=len(part), **{k: float(part[k].mean()) for k in
                             ("agree_T", "agree_Q", "agree_W", "kl_T", "kl_Q", "table_correct")}))
        print(checkpoint.name, rows[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
