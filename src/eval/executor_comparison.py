"""Compare learned semantic-token executors with exact Boolean gate rules.

Local tables are gold-prefix readouts, not identified hidden mechanisms.
Outcome-only trace queries are omitted. An optional mixed-format training
condition supports both query families for actual-vs-surrogate gradient tests.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import handcoded as h
from src.training.config import load_yaml
from src.training.optim import make_adamw
from src.training.progress import progress
from src.training.seed import add_compile_bf16_flags, autocast_context, configure_device, prepare_train_model, set_seed
from src.plot_style import apply_style

COLORS = {'process': '#247ba0', 'outcome': '#dd8452', 'both': '#7252a2'}
DISPLAY_GATES = ['x0', 'c01', 't012', 's03']


def truth_tables(tok):
    out = np.zeros((len(tok.gates), tok.n_states, tok.n_states))
    for gi, g in enumerate(tok.gates):
        for s in range(tok.n_states):
            out[gi, s, h.phi(s, g)] = 1
    return out


def circuit_keys(circuits):
    return {(c.start, tuple(c.gates)) for c in circuits}


def unique_circuits(size, seed, depth, excluded=()):
    rng, seen, out = random.Random(seed), set(excluded), []
    for _ in range(200 * size):
        c = h.Circuit(*h.sample_example(rng, depth))
        key = (c.start, tuple(c.gates))
        if key not in seen:
            out.append(c)
            seen.add(key)
        if len(out) == size:
            return out
    raise ValueError('Not enough unique prompts for the requested split')


def display_circuits(depth):
    gates = [DISPLAY_GATES[i % len(DISPLAY_GATES)] for i in range(depth)]
    out = []
    for start in range(16):
        states, current = [], start
        for g in gates:
            current = h.phi(current, g)
            states.append(current)
        out.append(h.Circuit(start, gates, states))
    return out


def local_contexts(tok, depth, step, background):
    """All (gate, source) pairs at a zero-indexed step, with valid gold prefixes."""
    rows = []
    for gate in tok.gates:
        gates = list(background)
        gates[step] = gate
        for source in range(tok.n_states):
            start = source
            # The Boolean gates in this experiment are involutions.
            for prev in reversed(gates[:step]):
                start = h.phi(start, prev)
            row, current = tok.prompt(h.Circuit(start, gates, [])), start
            for prev in gates[:step]:
                current = h.phi(current, prev)
                row += [tok.ids[prev], current]
            assert current == source
            rows.append(row + [tok.ids[gate]])
    return torch.tensor(rows, dtype=torch.long)


def probabilities(model, contexts, device, batch_size=128):
    """Differentiable full-vocabulary probabilities."""
    return torch.cat([model(contexts[i:i + batch_size].to(device))[:, -1].softmax(-1)
                      for i in range(0, len(contexts), batch_size)])


def read_local_rules(model, tok, depth, device, backgrounds=2, seed=8127,
                     steps=None, batch_size=128):
    """Process/mixed models only. Output shape: context, gate, source, successor.

    Return state-conditional tables and full-vocabulary state mass separately.
    Gradients are retained unless the caller disables them.
    """
    k, m = tok.n_states, len(tok.gates)
    rng, tables, masses = random.Random(seed), [], []
    steps = list(range(depth) if steps is None else steps)
    for _ in range(backgrounds):
        background = [h.sample_gate(rng) for _ in range(depth)]
        for step in steps:
            ctx = local_contexts(tok, depth, step, background)
            raw = probabilities(model, ctx, device, batch_size)[:, :k]
            mass = raw.sum(-1, keepdim=True)
            tables.append((raw / mass.clamp_min(1e-30)).reshape(m, k, k))
            masses.append(mass.reshape(m, k))
    return torch.stack(tables), torch.stack(masses)


def compose_tables(tables, circuits, tok):
    gate_ids = {g: i for i, g in enumerate(tok.gates)}
    device = tables.device
    starts = torch.tensor([c.start for c in circuits], device=device)
    p = F.one_hot(starts, tok.n_states).to(tables.dtype)
    for t in range(len(circuits[0].gates)):
        ids = torch.tensor([gate_ids[c.gates[t]] for c in circuits], device=device)
        p = torch.bmm(p[:, None], tables[ids]).squeeze(1)
    return p


def direct_probabilities(model, circuits, tok, device):
    # COLON selects the trained direct-answer branch; no gold trace/answer.
    ctx = torch.tensor([tok.prompt(c) + [tok.colon] for c in circuits])
    return probabilities(model, ctx, device)


@torch.no_grad()
def terminal_probabilities(model, circuits, tok, device, mode):
    if mode in ('outcome', 'both'):
        return (direct_probabilities(model, circuits, tok, device)[:, :tok.n_states],
                'direct answer conditioned on COLON')
    rows = []
    for i in range(0, len(circuits), 128):
        group = circuits[i:i + 128]
        prompts = torch.tensor([tok.prompt(c) for c in group], device=device)
        prefix = h.generate(model, prompts, 2 * len(group[0].gates) + 1, tok.eos)
        valid = (prefix[:, -1] == tok.colon) & ~(
            prefix[:, prompts.shape[1]:] == tok.eos).any(1)
        rows.append(model(prefix)[:, -1].softmax(-1)[:, :tok.n_states] * valid[:, None])
    return torch.cat(rows), 'after greedy trace; not a marginal over traces'


@torch.no_grad()
def generation_metrics(model, circuits, tok, device, mode):
    answer, exact = 0, 0
    for i in range(0, len(circuits), 128):
        group = circuits[i:i + 128]
        prompts = torch.tensor([tok.prompt(c) for c in group], device=device)
        budget = 3 if mode == 'outcome' else 2 * len(group[0].gates) + 3
        out = h.generate(model, prompts, budget, tok.eos)
        for c, seq in zip(group, out[:, prompts.shape[1]:].cpu().tolist()):
            seq = h.strip_after_eos(seq, tok.eos)
            answer += h.generated_answer(seq, tok) == c.answer
            formats = ('outcome', 'process') if mode == 'both' else (mode,)
            exact += any(seq == tok.continuation(c, fmt) for fmt in formats)
    return {'free_answer_accuracy': answer / len(circuits),
            'free_exact_continuation': exact / len(circuits)}


@torch.no_grad()
def diagnose(model, tok, depth, circuits, device, mode, backgrounds=2):
    model.eval()
    metrics = generation_metrics(model, circuits, tok, device, mode)
    terminal, query = terminal_probabilities(model, circuits, tok, device, mode)
    gold = torch.tensor([c.answer for c in circuits], device=device)
    mass = terminal.sum(-1)
    metrics.update(terminal_query=query, terminal_state_mass=mass.mean().item(),
                   terminal_state_argmax_accuracy=((terminal.argmax(-1) == gold)
                                                  & (mass > 0)).float().mean().item())
    arrays = {'terminal_raw': terminal.cpu().numpy()}
    if mode == 'outcome':
        metrics['local_query_status'] = 'omitted: trace query is out of distribution'
        return metrics, arrays
    contexts, masses = read_local_rules(model, tok, depth, device, backgrounds)
    tables = contexts.mean(0)
    target = torch.tensor(truth_tables(tok), dtype=tables.dtype, device=device)
    raw = (contexts * masses[..., None]).mean(0)
    c = (tables - 1 / tok.n_states).mean(1)
    centered = tables - 1 / tok.n_states - c[:, None, :]
    product = compose_tables(tables, circuits, tok)
    conditional = terminal / mass[:, None].clamp_min(1e-30)
    tv = 0.5 * (product - terminal).abs().sum(-1) + 0.5 * (1 - mass).clamp_min(0)
    conditional_tv = 0.5 * (product - conditional).abs().sum(-1)
    conditional_tv = torch.where(mass > 0, conditional_tv, torch.nan)
    metrics.update(
        local_query_status='gold-prefix trace readout; conditioned on valid state token',
        local_rule_accuracy=(tables.argmax(-1) == target.argmax(-1)).float().mean().item(),
        local_rule_tv=0.5 * (tables - target).abs().sum(-1).mean().item(),
        local_state_mass=masses.mean().item(), local_state_mass_min=masses.min().item(),
        local_context_variation_tv=0.5 * (contexts - tables).abs().sum(-1).mean().item(),
        epsilon_rule=torch.linalg.matrix_norm(centered, ord=2).max().item(),
        gamma=torch.linalg.vector_norm(c, dim=-1).max().item(),
        composition_tv_with_invalid=tv.mean().item(),
        composition_tv_conditional=torch.nanmean(conditional_tv).item(),
        product_answer_accuracy=(product.argmax(-1) == gold).float().mean().item())
    arrays.update(local_conditional=tables.cpu().numpy(), local_raw=raw.cpu().numpy(),
                  local_contexts=contexts.cpu().numpy(), local_state_mass=masses.cpu().numpy(),
                  product=product.cpu().numpy(), composition_tv=tv.cpu().numpy())
    return metrics, arrays


def flatten_gradient(loss, parameters, retain_graph=False):
    grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, grads)])


def cosine(a, b):
    denominator = (a.norm() * b.norm()).item()
    return (a @ b).item() / denominator if denominator > 0 else float('nan')


def gradient_agreement(model, tok, depth, circuits, device, backgrounds=2):
    """Mixed-format models only: actual direct-answer CE vs table-product CE.

    First-step local tables, averaged across backgrounds, are shared across
    product occurrences. This is not the model's total serialized token loss.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    tables, _ = read_local_rules(model, tok, depth, device, backgrounds, steps=[0])
    tables = tables.mean(0)
    product = compose_tables(tables, circuits, tok)
    gold = torch.tensor([c.answer for c in circuits], device=device)
    predicted_loss = -product.gather(1, gold[:, None]).clamp_min(1e-30).log().mean()
    actual_p = direct_probabilities(model, circuits, tok, device)
    actual_loss = -actual_p.gather(1, gold[:, None]).clamp_min(1e-30).log().mean()
    actual = flatten_gradient(actual_loss, params)
    predicted = flatten_gradient(predicted_loss, params, retain_graph=True)
    table_grad = torch.autograd.grad(predicted_loss, tables, retain_graph=True)[0]
    pi = torch.eye(tok.n_states, device=device) - 1 / tok.n_states
    rule_credit = pi @ table_grad.detach() @ pi
    rule_predicted = flatten_gradient((tables * rule_credit).sum(), params, retain_graph=True)
    target = torch.tensor(truth_tables(tok), dtype=tables.dtype, device=device)
    oracle = flatten_gradient((tables * (target - 1 / tok.n_states)).sum(), params, retain_graph=True)
    random_cosines = []
    rng = torch.Generator(device=device).manual_seed(4581)
    for j in range(8):
        w = pi @ torch.randn(target.shape, device=device, generator=rng) @ pi
        w *= (target - 1 / tok.n_states).norm() / w.norm()
        random_pullback = flatten_gradient((tables * w).sum(), params, retain_graph=j < 7)
        random_cosines.append(cosine(-actual, random_pullback))
    an = actual.norm().item()
    return {
        'gradient_query': 'mixed only; direct answer CE; first-step local tables',
        'gradient_actual_norm': an, 'gradient_predicted_norm': predicted.norm().item(),
        'gradient_cosine': cosine(actual, predicted),
        'gradient_relative_error': (actual - predicted).norm().item() / an if an else None,
        'gradient_rule_pullback_norm': rule_predicted.norm().item(),
        'gradient_rule_cosine': cosine(actual, rule_predicted),
        'descent_oracle_rule_cosine': cosine(-actual, oracle),
        'descent_random_rule_cosine_mean': float(np.mean(random_cosines)),
        'descent_random_rule_cosine_std': float(np.std(random_cosines, ddof=1)),
        'gradient_actual_loss': actual_loss.item(), 'gradient_predicted_loss': predicted_loss.item()}


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(path).with_suffix('.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_results(outdir, tok, depth, records, snapshots, fixed):
    apply_style()
    outdir = Path(outdir)
    exact = truth_tables(tok)
    cols = [('Exact rule', exact), ('Hand-coded process', fixed['process']['local_raw'])]
    cols += [('Trained ' + ('mixed format' if mode == 'both' else mode), snapshots[mode]['local_raw'])
             for mode in ('process', 'both') if mode in snapshots]
    fig, axes = plt.subplots(4, len(cols), figsize=(3.2 * len(cols), 11), layout='constrained')
    for ri, gate in enumerate(['x0', 'c01', 's03', 't012']):
        for ci, (label, matrices) in enumerate(cols):
            ax = axes[ri, ci]
            im = ax.imshow(matrices[tok.gates.index(gate)], vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
            ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
            if ri == 0: ax.set_title(label)
            if ci == 0: ax.set_ylabel(f'{gate}\nSource state')
            if ri == 3: ax.set_xlabel('Successor state')
    fig.colorbar(im, ax=axes, shrink=0.7, label='Raw next-token probability (full vocabulary)')
    fig.suptitle('Learned local transitions versus the exact executor\nGold prefixes; averaged over positions and surrounding circuits', fontsize=14)
    save_figure(fig, outdir / 'transition_tables')

    exact_circuit = np.eye(16)[[c.answer for c in display_circuits(depth)]]
    cols = [('Exact composition', exact_circuit), ('Hand-coded outcome', fixed['outcome']['terminal_raw'][:16])]
    cols += [('Trained ' + mode, snapshots[mode]['terminal_raw'][:16])
             for mode in ('outcome', 'process', 'both') if mode in snapshots]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 4.4), layout='constrained')
    for ax, (label, mat) in zip(axes, cols):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
        ax.set(title=label, xlabel='Answer state', xticks=[0, 5, 10, 15], yticks=[0, 5, 10, 15])
    axes[0].set_ylabel('Initial state')
    fig.colorbar(im, ax=axes, shrink=0.65, label='Raw state probability')
    fig.suptitle('Same circuit, all 16 starting states\nProcess: generated trace. Outcome/mixed: direct branch conditioned on COLON.', fontsize=12)
    save_figure(fig, outdir / 'circuit_comparison')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
    panels = [('free_answer_accuracy', 'Free-running answer accuracy'),
              ('local_rule_accuracy', 'Local-rule accuracy (gold prefixes)'),
              ('composition_tv_with_invalid', 'Composition error (TV, including invalid mass)'),
              ('epsilon_rule', 'Induced rule strength'),
              ('local_state_mass', 'Probability assigned to valid state tokens'),
              ('local_context_variation_tv', 'Local context variation (mean TV)')]
    for ax, (field, title) in zip(axes.flat, panels):
        for mode, color in COLORS.items():
            rows = [r for r in records if r['mode'] == mode and field in r]
            if rows:
                ax.plot([r['step'] for r in rows], [r[field] for r in rows], 'o-', color=color, label=mode)
        ax.set(title=title, xlabel='Training update'); ax.grid(alpha=0.2)
        if field != 'epsilon_rule': ax.set_ylim(-0.03, 1.03)
    axes[0, 0].legend()
    fig.suptitle('Checkpoint diagnostics — one seed; update counts do not equalize computation', fontsize=13)
    save_figure(fig, outdir / 'training_diagnostics')
    rows = [r for r in records if 'gradient_cosine' in r]
    if not rows: return
    x = [r['step'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for key, label in [('gradient_cosine', 'Full surrogate gradient'), ('gradient_rule_cosine', 'Rule component')]:
        axes[0].plot(x, [r[key] for r in rows], 'o-', label=label)
    axes[0].axhline(0, color='gray', lw=0.8)
    axes[0].set(title='Agreement with actual answer gradient', ylabel='Cosine', ylim=(-1.05, 1.05))
    axes[0].legend(fontsize=8)
    for key, label in [('gradient_actual_norm', 'Actual'), ('gradient_predicted_norm', 'Surrogate')]:
        axes[1].plot(x, [r[key] for r in rows], 'o-', label=label)
    axes[1].set(title='Parameter-gradient magnitude', ylabel='L2 norm', yscale='log'); axes[1].legend()
    axes[2].plot(x, [r['descent_oracle_rule_cosine'] for r in rows], 'o-', label='True-rule direction')
    axes[2].errorbar(x, [r['descent_random_rule_cosine_mean'] for r in rows],
                     yerr=[r['descent_random_rule_cosine_std'] for r in rows], fmt='o--', label='8 random fields: mean ± SD')
    axes[2].set(title='Does actual descent improve true rules?', ylabel='Cosine', ylim=(-1.05, 1.05)); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Training update'); ax.grid(alpha=0.2)
    fig.suptitle('Mixed-format diagnostic: both query families trained; direct answer sees no gold trace', fontsize=12)
    save_figure(fig, outdir / 'gradient_agreement')


def write_report(outdir, args, records):
    last = {r['mode']: r for r in records}
    lines = ['# Trained versus hand-coded executor comparison', '',
             f'Fresh pilot: depth {args.depth}, seed {args.seed}, {args.steps:,} updates, '
             f'{args.train_size:,} unique training circuits and {args.test_size:,} disjoint held-out circuits.', '',
             'This semantic-token one-layer Transformer run does not reconstruct the archived RegisterMachine '
             'or character-token experiments. Learned conditions share initial weights, the circuit pool, '
             'minibatches and optimizer settings. Mixed format assigns direct answers to half the examples '
             'and process traces to half, using a fixed random assignment. Equal updates do not equalize '
             'tokens or FLOPs. Hand-coded references have different capacities.', '',
             '## Final held-out results', '',
             '| Condition | Free answer accuracy | Local rule accuracy | Composition TV |',
             '|---|---:|---:|---:|']
    for mode, r in last.items():
        local = f"{r['local_rule_accuracy']:.1%}" if 'local_rule_accuracy' in r else 'not queried'
        comp = f"{r['composition_tv_with_invalid']:.3f}" if 'composition_tv_with_invalid' in r else 'not defined'
        lines.append(f"| {mode} | {r['free_answer_accuracy']:.1%} | {local} | {comp} |")
    lines += ['', 'Local-rule and composition measurements use separate diagnostic circuits. '
              'The displayed circuit is excluded from training and the main accuracy test. '
              'Local tables average execution positions and circuit backgrounds. Gradient readouts '
              'use first-step tables only; this distinction is recorded in the CSV.', '',
              '## Figures', '', '- [Local transition tables](transition_tables.pdf)',
              '- [Whole-circuit comparison](circuit_comparison.pdf)',
              '- [Checkpoint diagnostics](training_diagnostics.pdf)']
    gradients = [r for r in records if 'gradient_cosine' in r]
    if gradients:
        r = gradients[-1]
        lines += ['- [Actual versus predicted gradients](gradient_agreement.pdf)', '',
                  f"Final mixed-format full-gradient cosine: **{r['gradient_cosine']:.3f}**; "
                  f"relative gradient error: **{r['gradient_relative_error']:.3f}**. "
                  'These measurements do not by themselves establish a shared mechanism.']
    lines += ['', '## Scope', '',
              '- Gold-prefix rule recovery is a readout diagnostic, not a hidden-state or causal-intervention result.',
              '- Outcome-only trace tables are omitted because those queries were not trained.',
              '- Process composition uses the answer distribution after a greedy generated trace, not the marginal over all traces.',
              '- Mixed-format composition and gradients condition on the direct-answer COLON branch, without a gold trace or answer in the input.',
              '- Heatmaps retain full-vocabulary probability mass. Conditional tables and invalid-token mass are saved separately.',
              '- One seed cannot establish robustness. Correct transitions, low composition error and gradient agreement are distinct requirements.',
              '- No layer-to-execution-step correspondence or raw parameter similarity is assumed.', '',
              '## Reproduce', '', '```bash',
              'python -m src executor --output results/executor_comparison/reproduction '
              f'--depth {args.depth} --seed {args.seed} --steps {args.steps} --train-size {args.train_size} '
              f'--test-size {args.test_size} --probe-size {args.probe_size} --batch-size {args.batch_size} '
              f'--d-model {args.d_model} --d-ff {args.d_ff} --lr {args.lr} --device {args.device} '
              f'--threads {args.threads} --backgrounds {args.backgrounds} '
              f"--checkpoints {' '.join(map(str, args.checkpoints))}", '```', '']
    (outdir / 'report.md').write_text('\n'.join(lines))


def compare_models(models, tok, depth, circuits, output, device='cpu', backgrounds=2):
    """Analyze live notebook models: mapping mode -> model. No training performed.

    Allowed modes: process, outcome, both. Use both ONLY for a model trained on
    direct-answer and process formats. Save into a new directory.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    if not models or set(models) - set(COLORS):
        raise ValueError('Expected process, outcome and/or both models')
    device = torch.device(device)
    diagnostic = display_circuits(depth) + list(circuits)
    fixed, snapshots, rows = {}, {}, []
    for mode, cls in [('process', h.HandcodedProcessTransformer), ('outcome', h.HandcodedOutcomeTransformer)]:
        ref = cls(tok, depth).to(device).eval()
        _, fixed[mode] = diagnose(ref, tok, depth, diagnostic, device, mode, backgrounds)
        np.savez_compressed(output / f'fixed_{mode}.npz', **fixed[mode])
    for mode, model in models.items():
        previous = model.training
        try:
            stats, arrays = diagnose(model, tok, depth, diagnostic, device, mode, backgrounds)
            stats.update(generation_metrics(model, circuits, tok, device, mode))
            if mode == 'both':
                stats.update(gradient_agreement(model, tok, depth, circuits[:16], device, backgrounds))
            rows.append({'mode': mode, 'step': 0, **stats})
            snapshots[mode] = arrays
            np.savez_compressed(output / f'{mode}.npz', **arrays)
        finally:
            model.train(previous)
    save_csv(output / 'metrics.csv', rows)
    plot_results(output, tok, depth, rows, snapshots, fixed)
    return rows


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='configs/experiments/executor_comparison.yaml')
    pre_args, _ = pre.parse_known_args()
    cfg_path = Path(pre_args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--output', type=Path, default=ROOT / str(cfg.get('output', 'results/executor_comparison/pilot')))
    for name, default in [('depth', int(cfg.get('depth', 4))), ('seed', int(cfg.get('seed', 42))),
                          ('steps', int(cfg.get('steps', 2000))),
                          ('train-size', int(cfg.get('train_size', 10000))),
                          ('test-size', int(cfg.get('test_size', 256))),
                          ('probe-size', int(cfg.get('probe_size', 64))),
                          ('batch-size', int(cfg.get('batch_size', 128))),
                          ('d-model', int(cfg.get('d_model', 96))),
                          ('d-ff', int(cfg.get('d_ff', 192))),
                          ('threads', int(cfg.get('threads', 2))),
                          ('backgrounds', int(cfg.get('backgrounds', 2)))]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=float(cfg.get('lr', 0.002)))
    parser.add_argument('--device', default=cfg.get('device', 'cpu'))
    parser.add_argument('--checkpoints', type=int, nargs='+', default=list(cfg.get('checkpoints', [0, 100, 500, 1000, 2000])))
    add_compile_bf16_flags(parser, cfg)
    args = parser.parse_args()
    if min(args.depth, args.steps, args.train_size, args.test_size, args.probe_size,
           args.batch_size, args.threads, args.backgrounds) < 1 or args.depth < 2:
        parser.error('positive sizes required; depth must be >= 2')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory is not empty; choose a new run directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'checkpoints').mkdir()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    tok = h.make_tokenizer()
    device = configure_device(args.device, compile=getattr(args, "compile", None),
                              bf16=getattr(args, "bf16", None),
                              distributed=getattr(args, "distributed", None))
    shown = display_circuits(args.depth)
    train = unique_circuits(args.train_size, 123, args.depth, circuit_keys(shown))
    test = unique_circuits(args.test_size, 9000, args.depth, circuit_keys(train + shown))
    probes = unique_circuits(args.probe_size, 12000, args.depth, circuit_keys(train + test + shown))
    diagnostic = shown + probes
    data = {mode: h.encode_dataset(train, tok, mode).to(device) for mode in ('outcome', 'process')}
    base = h.build_random_learned_model(tok, args.depth, seed=args.seed,
                                       d_model=args.d_model, d_ff=args.d_ff, device=device)
    models = {mode: copy.deepcopy(base) for mode in COLORS}
    train_models = {mode: prepare_train_model(models[mode], device, compile=getattr(args, "compile", None),
                                             distributed=getattr(args, "distributed", None))
                    for mode in COLORS}
    optimizers = {mode: make_adamw(model.parameters(), args.lr, weight_decay=0.01, device=device)
                  for mode, model in models.items()}
    schedule = h.make_batch_schedule(len(train), args.steps, args.batch_size, 2026)
    assignment = torch.rand(len(train), generator=torch.Generator().manual_seed(6721)).to(device) < 0.5
    checkpoints = sorted({0, args.steps} | {x for x in args.checkpoints if 0 <= x <= args.steps})
    args.checkpoints = checkpoints
    sources = [Path(__file__), Path(h.__file__)]
    manifest = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'output': str(args.output), 'torch_version': str(torch.__version__),
                'numpy_version': np.__version__,
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'data_seeds': {'train': 123, 'test': 9000, 'probe': 12000, 'batch': 2026, 'format': 6721},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'parameter_count': sum(p.numel() for p in base.parameters()),
                'training_test_overlap': len(circuit_keys(train) & circuit_keys(test))}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.save({'train': [(c.start, c.gates, c.states) for c in train],
                'test': [(c.start, c.gates, c.states) for c in test],
                'probes': [(c.start, c.gates, c.states) for c in probes],
                'mixed_assignment': assignment.cpu()}, args.output / 'checkpoints/data.pt')
    records, snapshots, fixed, refs = [], {}, {}, []
    for mode, cls in [('process', h.HandcodedProcessTransformer), ('outcome', h.HandcodedOutcomeTransformer)]:
        model = cls(tok, args.depth).to(device).eval()
        stats, arrays = diagnose(model, tok, args.depth, diagnostic, device, mode, args.backgrounds)
        stats.update(generation_metrics(model, test, tok, device, mode))
        assert stats['free_answer_accuracy'] == 1, 'Fixed reference failed'
        if mode == 'process': assert stats['local_rule_accuracy'] == 1, 'Local readout failed'
        fixed[mode] = arrays
        refs.append({'mode': mode, **stats})
        np.savez_compressed(args.output / f'fixed_{mode}.npz', **arrays)
    save_csv(args.output / 'fixed_metrics.csv', refs)
    np.savez_compressed(args.output / 'exact_rules.npz', rules=truth_tables(tok), gates=tok.gates)
    started = time.monotonic()
    for step in progress(range(args.steps + 1), desc=f"executor D={args.depth} s={args.seed}", leave=True):
        if step in checkpoints:
            for mode, model in models.items():
                stats, arrays = diagnose(model, tok, args.depth, diagnostic, device, mode, args.backgrounds)
                stats.update(generation_metrics(model, test, tok, device, mode))
                if mode == 'both':
                    stats.update(gradient_agreement(model, tok, args.depth, probes[:16], device, args.backgrounds))
                records.append({'mode': mode, 'step': step, 'seed': args.seed, **stats})
                snapshots[mode] = arrays
                np.savez_compressed(args.output / f'{mode}_step{step:05d}.npz', **arrays)
                torch.save({'model': model.state_dict(), 'optimizer': optimizers[mode].state_dict(),
                            'step': step, 'mode': mode, 'config': manifest},
                           args.output / 'checkpoints' / f'{mode}_step{step:05d}.pt')
                print(json.dumps({'step': step, 'mode': mode, 'answer': round(stats['free_answer_accuracy'], 4),
                                  'local': stats.get('local_rule_accuracy'),
                                  'composition_tv': stats.get('composition_tv_with_invalid'),
                                  'gradient_cosine': stats.get('gradient_cosine'),
                                  'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
            save_csv(args.output / 'metrics.csv', records)
            plot_results(args.output, tok, args.depth, records, snapshots, fixed)
            write_report(args.output, args, records)
        if step == args.steps: break
        idx = torch.tensor(schedule[step], device=device)
        for mode, model in models.items():
            runner = train_models[mode]
            runner.train()
            optimizer = optimizers[mode]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                if mode == 'both':
                    mask = assignment[idx]
                    loss = sum(h.language_model_loss(runner, data[fmt].select(idx[select])) * select.float().mean()
                               for fmt, select in (('process', mask), ('outcome', ~mask)) if select.any())
                else:
                    loss = h.language_model_loss(runner, data[mode].select(idx))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss for {mode} at step {step + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    manifest['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Saved comparison to {args.output}', flush=True)


if __name__ == '__main__':
    main()
