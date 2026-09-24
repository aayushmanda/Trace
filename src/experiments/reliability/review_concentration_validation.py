"""Matched continuous-corruption experiment: semantic Transformer and local table.

The same prompt set, initialization, minibatch schedule, reliability scores,
and per-step random variates are reused across all conditions within a seed.
Wrong successors are generated recursively from the displayed prefix.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from handcoded.data import encode_dataset, make_batch_schedule
from handcoded.eval import free_run_metrics, make_generation_evaluation
from handcoded.gates import make_gate_names, phi
from handcoded.models import build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.experiments.circuit_match.handcoded_trace_gradients import continuation_slices, encode_process_states
from src.experiments.circuit_match.review_functional_validation import keys, local_metrics, loss_at, unique_circuits
from src.experiments.reliability.review_corpus_validation import table_metrics


def make_states(circuits, rho, concentration, seed):
    scores = np.random.default_rng(seed).random(len(circuits))
    selectors = np.random.default_rng(seed+1).random((len(circuits), 4))
    alternatives = np.random.default_rng(seed+2).integers(0, 15, (len(circuits), 4))
    states = []
    for i, circuit in enumerate(circuits):
        if scores[i] < rho:
            states.append(circuit.states)
            continue
        current, path = circuit.start, []
        for t, gate in enumerate(circuit.gates):
            correct = phi(current, gate)
            # Uniform integer among every successor other than correct.
            candidate = int(alternatives[i,t])
            candidate += candidate >= correct
            current = correct ^ _coherent_wrong_mask(gate) if selectors[i,t] < concentration else candidate
            assert current != correct
            path.append(current)
        states.append(path)
    return states, scores


def corpus_table(circuits, states, gate_ids):
    counts = np.zeros((52,16,16), dtype=np.int64)
    for circuit, path in zip(circuits, states):
        current = circuit.start
        for gate, successor in zip(circuit.gates, path):
            counts[gate_ids[gate], current, successor] += 1
            current = successor
    return counts


@torch.no_grad()
def row_diagnostics(model, batch, circuits, gate_ids, empirical_q, truth, positions):
    probs = []
    for start in range(0, len(batch), 128):
        probs.append(model(batch.inputs[start:start+128])[:,list(positions)].softmax(-1).cpu().numpy())
    probs = np.concatenate(probs)
    row_ids, vectors = [], []
    for i, circuit in enumerate(circuits):
        current = circuit.start
        for t, gate in enumerate(circuit.gates):
            row_ids.append(gate_ids[gate]*16+current)
            vectors.append(probs[i,t])
            current = circuit.states[t]
    row_ids, vectors = np.asarray(row_ids), np.asarray(vectors)
    rows = []
    for row_id in np.unique(row_ids):
        op, state = divmod(int(row_id),16)
        values = vectors[row_ids == row_id]
        mean = values.mean(0)
        target = truth[op,state]
        q = np.pad(empirical_q[op,state], (0,len(mean)-16))
        wrong = q.copy(); wrong[target] = -np.inf
        empirical_margin = q[target]-wrong.max()
        eps = np.abs(values-q[None]).max(axis=1)
        rows.append(dict(operation=op, state=state, contexts=len(values),
            empirical_margin=float(empirical_margin),
            mean_context_error=float(eps.mean()), max_context_error=float(eps.max()),
            mean_distribution_error=float(np.abs(mean-q).max()),
            context_std=float(values.std(axis=0).max()),
            invalid_token_mass=float(mean[16:].sum()),
            local_accuracy=float((values.argmax(-1)==target).mean()),
            mean_decodes_correct=bool(mean.argmax()==target),
            sufficient_bound_holds=bool(empirical_margin>2*eps.max())))
    frame = pd.DataFrame(rows)
    assert frame.loc[frame.sufficient_bound_holds, 'local_accuracy'].eq(1).all()
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--train-size', type=int, default=20000)
    p.add_argument('--eval-size', type=int, default=512)
    p.add_argument('--lambdas', type=float, nargs='+', default=[0,.5,1])
    p.add_argument('--rhos', type=float, nargs='+', default=[.3,.5,.7])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = make_tokenizer()
    slices = continuation_slices(4)
    gates = make_gate_names()
    gate_ids = {g:i for i,g in enumerate(gates)}
    truth = np.array([[phi(s,g) for s in range(16)] for g in gates])
    conditions = [(r,l) for l in args.lambdas for r in args.rhos]+[(1.,0.)]
    rows = []
    for seed in args.seeds:
        train_circuits = unique_circuits(args.train_size, 123+(seed-42)*1000)
        eval_circuits = unique_circuits(args.eval_size, 39000+seed, keys(train_circuits))
        eval_data = encode_dataset(eval_circuits, tokenizer, 'process').to(args.device)
        generation = make_generation_evaluation(eval_circuits, tokenizer, args.device)
        starts = np.array([c.start for c in eval_circuits])
        operations = np.array([[gate_ids[g] for g in c.gates] for c in eval_circuits])
        answers = np.array([c.answer for c in eval_circuits])
        schedule = make_batch_schedule(args.train_size, args.steps, 64, seed=2026+seed)
        for rho, concentration in conditions:
            tag = f's{seed}_rho{rho:g}_lambda{concentration:g}'
            states, scores = make_states(train_circuits, rho, concentration, seed+777)
            train = encode_process_states(train_circuits, tokenizer, states, [c.answer for c in train_circuits]).to(args.device)
            counts = corpus_table(train_circuits, states, gate_ids)
            tabular, q, margins = table_metrics(counts, truth, starts, operations, answers)
            digest = hashlib.sha256(train.inputs.cpu().numpy().tobytes()+train.targets.cpu().numpy().tobytes()).hexdigest()
            np.savez_compressed(args.output/f'corpus_{tag}.npz', counts=counts, q=q, margins=margins,
                                scores=scores, displayed_states=np.asarray(states), truth=truth)
            model = build_random_trainable_process_architecture(tokenizer,4,seed=seed,init_std=.02,device=args.device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0)
            for step, indices in enumerate(schedule,1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_at(model,train.select(indices),slices.continuation)
                loss.backward(); optimizer.step()
                if step not in {min(500,args.steps),args.steps}:
                    continue
                model.eval()
                metrics = local_metrics(model,eval_data,slices.state)
                metrics.update(free_run_metrics(model,generation,tokenizer,'process'))
                c = concentration+(1-concentration)/15
                metrics.update(seed=seed,step=step,rho=rho,concentration=concentration,
                    population_threshold=c/(1+c), population_margin=rho-(1-rho)*c,
                    empirical_clean_fraction=float((scores<rho).mean()),corpus_sha256=digest,
                    train_loss=float(loss.detach()),**{'tabular_'+k:v for k,v in tabular.items()})
                if step==args.steps:
                    diagnostics = row_diagnostics(model,eval_data,eval_circuits,gate_ids,q,truth,slices.state)
                    diagnostics.to_csv(args.output/f'neural_rows_{tag}.csv',index=False)
                    metrics.update(probed_rows=len(diagnostics),
                        mean_context_error=float(diagnostics.mean_context_error.mean()),
                        mean_context_std=float(diagnostics.context_std.mean()),
                        fraction_rows_certified=float(diagnostics.sufficient_bound_holds.mean()))
                    torch.save({'model':model.state_dict(),'seed':seed,'step':step,'rho':rho,
                                'concentration':concentration},args.output/f'checkpoint_{tag}.pt')
                rows.append(metrics)
                pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False)
                print(json.dumps(metrics),flush=True)
            del model,optimizer,train
    (args.output/'protocol.json').write_text(json.dumps({**vars(args),'output':str(args.output),
        'architecture':'semantic-token process construction, all parameters trainable',
        'depth':4,'lr':.002,'batch_size':64,'weight_decay':0,
        'precision':'float32, no TF32, no autocast, no compilation',
        'paired_conditions':True,'independent_corpus_per_seed':True,
        'row_probe':'gold prefixes; full vocabulary probabilities; contexts are repeated measurements',
        'bound_interpretation':'post-hoc sufficient condition, not independent prediction of neural frontier',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    },indent=2)+'\n')


if __name__=='__main__':
    main()
