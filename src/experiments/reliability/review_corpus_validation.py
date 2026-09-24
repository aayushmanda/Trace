"""Fit local tables on exactly reconstructed canonical serialized corpora."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from handcoded.gates import make_gate_names, phi
from src.data.registry import TASKS
from src.data.sample import generate_unique


def arrays(instances, rho, scores, gate_ids):
    triples, starts, operations, answers = [], [], [], []
    payload = hashlib.sha256()
    for i, inst in enumerate(instances):
        trace = inst.correct_trace if scores[i] < rho else inst.wrong_trace
        payload.update(json.dumps([inst.prompt, trace, inst.gold]).encode())
        state = int(inst.prompt.split(';')[0][1:], 2)
        starts.append(state)
        ops = []
        for token in trace.split():
            gate, successor = token.split('>')
            successor = int(successor, 2)
            op = gate_ids[gate]
            triples.append((op, state, successor))
            ops.append(op)
            state = successor
        operations.append(ops)
        answers.append(int(inst.gold, 2))
    return np.asarray(triples), np.asarray(starts), np.asarray(operations), np.asarray(answers), payload.hexdigest()


def table_metrics(counts, truth, starts, operations, answers):
    totals = counts.sum(-1)
    q = np.divide(counts, totals[..., None], out=np.zeros_like(counts, dtype=float), where=totals[..., None] > 0)
    true_mass = np.take_along_axis(q, truth[..., None], -1)[..., 0]
    wrong = q.copy()
    np.put_along_axis(wrong, truth[..., None], -np.inf, -1)
    margins = true_mass - wrong.max(-1)
    decoded = q.argmax(-1)
    states = starts.copy()
    for t in range(operations.shape[1]):
        states = decoded[operations[:, t], states]
    return {
        'rule_accuracy': float((decoded == truth).mean()),
        'strict_rule_recovery': float(((margins > 0) & (totals > 0)).mean()),
        'answer_accuracy': float((states == answers).mean()),
        'n_min': int(totals.min()), 'n_median': float(np.median(totals)),
        'margin_min': float(margins.min()), 'margin_median': float(np.median(margins)),
        'unvisited': int((totals == 0).sum()),
    }, q, margins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-seeds', nargs='+', type=int, default=[501, 502, 503])
    p.add_argument('--train-size', type=int, default=20000)
    p.add_argument('--val-size', type=int, default=1000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gates = make_gate_names()
    ids = {g: i for i, g in enumerate(gates)}
    truth = np.array([[phi(s, g) for s in range(16)] for g in gates])
    rows = []
    for seed in args.train_seeds:
        train = generate_unique(TASKS['boolean_circuit_8'], args.train_size, seed)
        val = generate_unique(TASKS['boolean_circuit_8'], args.val_size, 101, {x.prompt for x in train})
        scores = np.random.default_rng(777).random(len(train))
        _, starts, ops, answers, _ = arrays(val, 1, np.zeros(len(val)), ids)
        for rho in [0, .0625, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            triples, _, _, _, digest = arrays(train, rho, scores, ids)
            counts = np.zeros((52, 16, 16), dtype=np.int64)
            np.add.at(counts, tuple(triples.T), 1)
            metrics, q, margins = table_metrics(counts, truth, starts, ops, answers)
            metrics.update(seed=seed, rho=rho, corpus_sha256=digest,
                           empirical_clean_fraction=float((scores < rho).mean()))
            rows.append(metrics)
            np.savez_compressed(args.output / f'rows_s{seed}_rho{rho:g}.npz', counts=counts,
                                q=q, margins=margins, truth=truth, gates=np.asarray(gates))
            print(json.dumps(metrics), flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'metrics.csv', index=False)
    (args.output / 'protocol.json').write_text(json.dumps({
        **vars(args), 'output': str(args.output), 'val_seed': 101, 'ratio_seed': 777,
        'canonical_match': 'seed 501 uses exactly the released canonical generator and reliability assignment',
        'decoder': 'full-state greedy table; argmax ties use lowest state index; strict recovery also reported',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
