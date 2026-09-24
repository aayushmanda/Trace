"""Independent-batch functional updates, token ablations, and AdamW controls.

Uses the semantic-token diagnostic architecture. No projection onto constructed
weights is used as a competence metric. Evaluation uses float64 to resolve small
finite loss changes; training uses float32 with TF32 disabled.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from handcoded.data import encode_dataset, make_batch_schedule
from handcoded.eval import free_run_metrics, make_generation_evaluation
from handcoded.gates import make_circuits
from handcoded.models import build_random_trainable_process_architecture
from handcoded.tokenizer import make_tokenizer
from src.experiments.circuit_match.handcoded_trace_gradients import (
    continuation_slices, make_matched_eval_batches,
)


def loss_at(model, batch, positions):
    logits = model(batch.inputs)[:, list(positions)]
    targets = batch.targets[:, list(positions)]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def conditional_state_loss(model, batch, positions):
    logits = model(batch.inputs)[:, list(positions), :16]
    targets = batch.targets[:, list(positions)]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def local_metrics(model, batch, positions):
    logits = model(batch.inputs)[:, list(positions)]
    targets = batch.targets[:, list(positions)]
    gold = logits.gather(-1, targets[..., None])[..., 0]
    competitors = logits.clone()
    competitors.scatter_(-1, targets[..., None], -torch.inf)
    return {
        'local_loss': float(F.cross_entropy(logits.flatten(0, 1), targets.flatten())),
        'conditional_state_loss': float(F.cross_entropy(logits[..., :16].flatten(0, 1), targets.flatten())),
        'valid_state_mass': float(logits.softmax(-1)[..., :16].sum(-1).mean()),
        'local_accuracy': float((logits.argmax(-1) == targets).double().mean()),
        'local_margin': float((gold - competitors.max(-1).values).mean()),
    }


def unique_circuits(n, seed, excluded=()):
    seen = set(excluded)
    out = []
    while len(out) < n:
        for c in make_circuits(n, seed, depth=4):
            key = (c.start, tuple(c.gates))
            if key not in seen:
                seen.add(key)
                out.append(c)
                if len(out) == n:
                    break
        seed += 100000
    return out


def keys(circuits):
    return {(c.start, tuple(c.gates)) for c in circuits}


def grads(loss, params):
    result = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, result)]


def group_indices(names):
    return {
        'all': list(range(len(names))),
        'MLP': [i for i, name in enumerate(names) if name.startswith('mlp_')],
        'QK': [i for i, name in enumerate(names) if name.startswith('heads.') and name.endswith(('query', 'key'))],
        'OV': [i for i, name in enumerate(names) if name.startswith('heads.') and name.endswith('value')],
    }


def probe(model, optimizer, update_circuits, evaluation, tokenizer, device, seed, step, etas):
    model.eval()
    diagnostic = copy.deepcopy(model).double()
    params = list(diagnostic.parameters())
    originals = [p.detach().clone() for p in params]
    groups = group_indices([name for name, _ in diagnostic.named_parameters()])
    slices = continuation_slices(4)
    base = local_metrics(diagnostic, evaluation, slices.state)
    oracle = grads(loss_at(diagnostic, evaluation, slices.state), params)
    conditional_oracle = grads(conditional_state_loss(diagnostic, evaluation, slices.state), params)
    rows = []
    for corruption in ['symmetric', 'coherent']:
        plus, minus = make_matched_eval_batches(update_circuits, tokenizer,
                corruption=corruption, seed=4242 + seed, device=device)
        specifications = {
            'clean_full': [(1., plus, slices.continuation)],
            'clean_state': [(1., plus, slices.state)],
            'wrong_full': [(1., minus, slices.continuation)],
            'wrong_state': [(1., minus, slices.state)],
            'uniform_state_wrong_prefix': [],
            'gold_answer_wrong_prefix': [(1., minus, slices.answer)],
            'other_correct_tokens_wrong_prefix': [(1., minus, slices.gate + slices.syntax)],
            'mixed_state_rho03': [(.3, plus, slices.state), (.7, minus, slices.state)],
            'mixed_state_rho07': [(.7, plus, slices.state), (.3, minus, slices.state)],
        }
        cached = {}
        for source, terms in specifications.items():
            loss = (-diagnostic(minus.inputs)[:, list(slices.state)].log_softmax(-1)[..., :16].mean()
                    if source == 'uniform_state_wrong_prefix'
                    else sum(w * loss_at(diagnostic, batch, positions) for w, batch, positions in terms))
            gradient = grads(loss, params)
            cached[source] = gradient
            for group, indices in groups.items():
                norm = float(sum(gradient[i].square().sum() for i in indices).sqrt())
                alignment = float(sum((gradient[i] * oracle[i]).sum() for i in indices))
                conditional_alignment = float(sum((gradient[i] * conditional_oracle[i]).sum() for i in indices))
                for eta in etas:
                    with torch.no_grad():
                        for i in indices:
                            params[i].copy_(originals[i] - eta * gradient[i])
                    moved = local_metrics(diagnostic, evaluation, slices.state)
                    with torch.no_grad():
                        for i in indices:
                            params[i].copy_(originals[i])
                    delta = moved['local_loss'] - base['local_loss']
                    predicted = -eta * alignment
                    rows.append(dict(seed=seed, step=step, corruption=corruption, source=source,
                        group=group, optimizer='euclidean', eta=eta, grad_norm=norm,
                        grad_dot_oracle=alignment, predicted_delta=predicted,
                        delta_loss=delta, taylor_residual=delta-predicted,
                        delta_conditional_state_loss=moved["conditional_state_loss"]-base["conditional_state_loss"],
                        predicted_conditional_delta=-eta*conditional_alignment,
                        delta_valid_state_mass=moved["valid_state_mass"]-base["valid_state_mass"],
                        delta_accuracy=moved['local_accuracy']-base['local_accuracy'],
                        delta_margin=moved['local_margin']-base['local_margin'], **base))
        # Exact weighted loss decomposition catches positional-mask/normalization mistakes.
        n = len(slices.continuation)
        for i in range(len(params)):
            reconstructed = (len(slices.state)*cached['wrong_state'][i]
                + len(slices.answer)*cached['gold_answer_wrong_prefix'][i]
                + (len(slices.gate)+len(slices.syntax))*cached['other_correct_tokens_wrong_prefix'][i])/n
            if not torch.allclose(reconstructed, cached['wrong_full'][i], atol=1e-9, rtol=1e-7):
                raise AssertionError('wrong full gradient does not equal token-weighted components')
        # Momentum control: AdamW can move even when the current gradient is zero.
        adam_deltas = {}
        for source in ['zero_gradient', 'clean_full', 'wrong_full', 'wrong_state', 'gold_answer_wrong_prefix', 'uniform_state_wrong_prefix']:
            candidate = copy.deepcopy(model)
            candidate_params = list(candidate.parameters())
            candidate_optimizer = torch.optim.AdamW(candidate_params, lr=optimizer.param_groups[0]['lr'], weight_decay=0)
            candidate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            for i, p in enumerate(candidate_params):
                p.grad = torch.zeros_like(p) if source == 'zero_gradient' else cached[source][i].to(p.dtype).clone()
            candidate_optimizer.step()
            moved = local_metrics(candidate.double(), evaluation, slices.state)
            delta = moved['local_loss'] - base['local_loss']
            adam_deltas[source] = delta
            rows.append(dict(seed=seed, step=step, corruption=corruption, source=source, group='all',
                optimizer='adamw', eta=optimizer.param_groups[0]['lr'], delta_loss=delta,
                delta_vs_zero_momentum=delta-adam_deltas['zero_gradient'],
                delta_conditional_state_loss=moved['conditional_state_loss']-base['conditional_state_loss'],
                delta_valid_state_mass=moved['valid_state_mass']-base['valid_state_mass'],
                delta_accuracy=moved['local_accuracy']-base['local_accuracy'],
                delta_margin=moved['local_margin']-base['local_margin'], **base))
            del candidate, candidate_optimizer
    # Every probe must leave the live training model AND optimizer untouched.
    assert all(torch.equal(p, original) for p, original in zip(params, originals))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--train-size', type=int, default=20000)
    p.add_argument('--probe-size', type=int, default=128)
    p.add_argument('--eval-size', type=int, default=256)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[0, 25, 100, 300, 1000, 2000])
    p.add_argument('--checkpoint-root', type=Path, help='Reuse saved models and AdamW states without training.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    tokenizer = make_tokenizer()
    slices = continuation_slices(4)
    all_rows, all_metrics = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        # Independent corpus, update, and evaluation draws per model seed.
        train_circuits = unique_circuits(args.train_size, 123 + (seed-42)*1000)
        update_circuits = unique_circuits(args.probe_size, 19000 + seed, keys(train_circuits))
        eval_circuits = unique_circuits(args.eval_size, 29000 + seed, keys(train_circuits)|keys(update_circuits))
        assert not (keys(train_circuits)&keys(update_circuits) or keys(train_circuits)&keys(eval_circuits)
                    or keys(update_circuits)&keys(eval_circuits))
        train = encode_dataset(train_circuits, tokenizer, 'process').to(device)
        evaluation = encode_dataset(eval_circuits, tokenizer, 'process').to(device)
        generation = make_generation_evaluation(eval_circuits, tokenizer, device)
        model = build_random_trainable_process_architecture(tokenizer, 4, seed=seed, init_std=.02, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
        schedule = make_batch_schedule(len(train), args.steps, 64, seed=2026+seed)
        marks = set(args.checkpoints) | {args.steps}
        for step in range(args.steps+1):
            if args.checkpoint_root:
                if step not in marks:
                    continue
                saved = torch.load(args.checkpoint_root / f"checkpoint_s{seed}_step{step}.pt", map_location=device, weights_only=True)
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
            if step in marks:
                model.eval()
                before = copy.deepcopy(model.state_dict())
                rows = probe(model, optimizer, update_circuits, evaluation, tokenizer, device,
                             seed, step, [1e-4, 1e-3, 1e-2])
                assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
                all_rows.extend(rows)
                metrics = local_metrics(model, evaluation, slices.state)
                metrics.update(free_run_metrics(model, generation, tokenizer, 'process'))
                metrics.update(seed=seed, step=step)
                all_metrics.append(metrics)
                if not args.checkpoint_root:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'seed': seed, 'step': step}, args.output / f'checkpoint_s{seed}_step{step}.pt')
                pd.DataFrame(all_rows).to_csv(args.output/'updates.csv', index=False)
                pd.DataFrame(all_metrics).to_csv(args.output/'training.csv', index=False)
                print(json.dumps(metrics), flush=True)
            if step == args.steps:
                break
            if args.checkpoint_root:
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_at(model, train.select(schedule[step]), slices.continuation).backward()
            optimizer.step()
        del model, optimizer
    frame = pd.DataFrame(all_rows)
    frame.groupby(['step','corruption','source','group','optimizer','eta']).delta_loss.agg(['mean','std','count']).to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps({**vars(args), 'output':str(args.output),
        'checkpoint_root': str(args.checkpoint_root) if args.checkpoint_root else None,
        'architecture':'randomized semantic-token process construction; all parameters trainable',
        'training_precision':'float32, no TF32, no autocast, no compilation',
        'probe_precision':'float64 Euclidean; actual float32 AdamW step evaluated in float64',
        'loss_normalization':'each component averaged over its own target positions',
        'disjoint_prompts':True, 'independent_corpus_per_seed':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_version':torch.__version__, 'numpy_version':np.__version__,
    },indent=2)+'\n')


if __name__ == '__main__':
    main()
