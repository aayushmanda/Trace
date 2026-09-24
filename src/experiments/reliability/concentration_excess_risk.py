"""Measure the excess risk Delta, the context discrepancy, and the decoding bound 2 Delta / m^2.

Reuses the checkpoints written by review_concentration_validation.py. On held-out circuits, displayed
traces are redrawn with the training corruption process, so positions x follow the training law.
Two targets are scored:
  pooled      q_x = rho T + (1 - rho) R, the context-free row law of the theorem (known exactly);
  contextual  Q(.|x): the pooled row at t = 1; for t >= 2 a correct displayed prefix implies a clean
              trace (corrupted traces are wrong at every step), so Q = T, else Q = R.
Reports Delta = E KL(target || p_theta), the decoding error against argmax target, the bound 2 Delta/m^2
(pooled, when m > 0), and eps_ctx of the network: the largest sup-norm deviation of p_theta(.|x) from
its mean over positions that share (s, a).

Run from a tree holding the pre-f099952 handcoded package (see ARTIFACT_MANIFEST.md):
  python -m src.experiments.reliability.concentration_excess_risk \
    --archive <repo>/results/review_validation/concentration --output <repo>/results/paper/concentration_excess_risk.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from handcoded.gates import make_gate_names
from handcoded.models import build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.experiments.circuit_match.handcoded_trace_gradients import continuation_slices, encode_process_states
from src.experiments.circuit_match.review_functional_validation import keys, unique_circuits
from src.experiments.reliability.review_concentration_validation import make_states


def kl(target, p):
    support = target > 0
    return float(np.sum(target[support] * np.log(target[support] / np.maximum(p[support], 1e-30))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--redraws", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    tokenizer = make_tokenizer()
    positions = list(continuation_slices(4).state)
    gate_ids = {g: i for i, g in enumerate(make_gate_names())}
    rows = []
    for checkpoint in sorted(args.archive.glob("checkpoint_s*_rho*_lambda*.pt")):
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        seed, rho, lam = saved["seed"], saved["rho"], saved["concentration"]
        truth = np.load(args.archive / f"corpus_{checkpoint.stem.removeprefix('checkpoint_')}.npz")["truth"]
        train = unique_circuits(args.train_size, 123 + (seed - 42) * 1000)
        evaluation = unique_circuits(args.eval_size, 39000 + seed, keys(train))
        model = build_random_trainable_process_architecture(tokenizer, 4, seed=seed, init_std=.02, device=args.device)
        model.load_state_dict(saved["model"])
        model.eval()
        c = lam + (1 - lam) / 15
        margin = rho - (1 - rho) * c
        records = []
        for redraw in range(args.redraws):
            states, _ = make_states(evaluation, rho, lam, 50000 + 10 * seed + redraw)
            batch = encode_process_states(evaluation, tokenizer, states, [x.answer for x in evaluation]).to(args.device)
            with torch.no_grad():
                probs = np.concatenate([model(batch.inputs[i:i + 128])[:, positions].softmax(-1).double().cpu().numpy()
                                        for i in range(0, len(batch), 128)])
            vocabulary = probs.shape[-1]
            for i, circuit in enumerate(evaluation):
                current, prefix_clean = circuit.start, True
                for t, gate in enumerate(circuit.gates):
                    op = gate_ids[gate]
                    correct, wrong = int(truth[op, current]), int(truth[op, current]) ^ _coherent_wrong_mask(gate)
                    corruption = np.zeros(vocabulary)
                    corruption[:16] = (1 - lam) / 15
                    corruption[wrong] += lam
                    corruption[correct] = 0
                    clean = np.zeros(vocabulary)
                    clean[correct] = 1
                    pooled = rho * clean + (1 - rho) * corruption
                    contextual = pooled if t == 0 else (clean if prefix_clean else corruption)
                    p = probs[i, t]
                    records.append(dict(row=op * 16 + current, t=t, prefix_clean=prefix_clean, p=p,
                                        kl_pooled=kl(pooled, p), kl_ctx=kl(contextual, p),
                                        err_pooled=p.argmax() != pooled.argmax(),
                                        err_ctx=p.argmax() != contextual.argmax()))
                    prefix_clean = prefix_clean and states[i][t] == correct
                    current = states[i][t]
        frame = pd.DataFrame(records)
        vectors, row_ids = np.stack(frame.p.to_numpy()), frame.row.to_numpy()
        eps = max(np.abs(vectors[row_ids == r] - vectors[row_ids == r].mean(0)).max() for r in np.unique(row_ids))
        delta = frame.kl_pooled.mean()
        rows.append(dict(seed=seed, rho=rho, concentration=lam, margin=margin, positions=len(frame),
                         delta_pooled=delta, error_pooled=frame.err_pooled.mean(),
                         bound_pooled=2 * delta / margin ** 2 if margin > 0 else np.nan,
                         delta_ctx=frame.kl_ctx.mean(), error_ctx=frame.err_ctx.mean(),
                         error_ctx_clean_later=frame[(frame.t > 0) & frame.prefix_clean].err_ctx.mean(),
                         error_ctx_corrupt_later=frame[(frame.t > 0) & ~frame.prefix_clean].err_ctx.mean(),
                         eps_ctx_network=float(eps)))
        print(checkpoint.name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
