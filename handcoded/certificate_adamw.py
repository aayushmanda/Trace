"""Basin certificate from an AdamW checkpoint (Theorem 4 and Corollary 1).

Phase 1 trains with full-batch AdamW on the exact weighted loss over every prompt.
Phase 2 switches to exact float64 Armijo gradient descent from that checkpoint, records the
first update j0 with L < pi_min/4, and keeps running for --post-updates to check that the
Armijo inequality, positive margins and nonzero ReLU pre-activations persist.
Theorem 4 does not depend on how theta_{j0} was reached, so AdamW may supply it.

Model, support and loss are those of basin_certificate.py: two blocks, width 16, three heads,
MLP width 16, no LayerNorm, sigma = 0.02 with the whole readout zero, binary letters with
match probability 1/4, K = 2, both start states. The step format supervises s_1..s_n, ':',
s_n and EOS; the outcome format supervises ':', s_n and EOS.

    python handcoded/certificate_adamw.py --mode step --n 6 --seed 42
    python handcoded/certificate_adamw.py --mode outcome --n 6 --seed 42
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from basin_certificate import COLON, EOS, enumerate_data, forward, greedy_accuracy, initialize, objective


def outcome_data(step_data):
    """Same prompts and law; gold continuation ':' s_n EOS."""
    prompt, gold = step_data['prompt'], step_data['gold']
    final = gold[:, -2:-1]                                  # s_n token in the step gold
    out = torch.cat([torch.full_like(final, COLON), final, torch.full_like(final, EOS)], 1)
    prob = step_data['weights'].sum(1)                      # prompt probabilities
    weights = prob[:, None].expand_as(out) / out.shape[1]
    assert abs(float(weights.sum()) - 1) < 1e-14
    return dict(prompt=prompt, gold=out, ids=torch.cat([prompt, out[:, :-1]], 1),
                weights=weights, start_position=prompt.shape[1] - 1)


def run(args):
    torch.set_default_dtype(torch.float64)
    data = enumerate_data(args.n, args.device)
    if args.mode == 'outcome':
        data = outcome_data(data)
    out = Path(args.output) / f'{args.mode}_n{args.n}_s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    pi_min = float(data['weights'].min())
    threshold = pi_min / 4
    p = initialize(args.seed, args.sigma, args.device)
    assert torch.count_nonzero(p['readout']).item() == 0
    params = list(p.values())
    t0 = time.monotonic()

    # Phase 1: full-batch AdamW on the exact loss.
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    adam_log, solved_at = [], None
    for update in range(1, args.adam_updates + 1):
        opt.zero_grad()
        loss = objective(p, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if update % args.eval_every == 0 or update == args.adam_updates:
            acc, correct = greedy_accuracy(p, data)
            L = float(objective(p, data).detach())
            adam_log.append(dict(update=update, loss=L, ratio=L / threshold, exact_accuracy=acc))
            if solved_at is None and correct == len(data['prompt']):
                solved_at = update
            if update % (10 * args.eval_every) == 0:
                print(f'{args.mode} n={args.n} adam j={update} R={L / threshold:.3g} acc={acc:.4f}', flush=True)
    for v in params:
        v.grad = None
    handoff_loss, handoff_diag, signs = objective(p, data, True)
    handoff = dict(loss=float(handoff_loss.detach()), ratio=float(handoff_loss.detach()) / threshold,
                   exact_accuracy=greedy_accuracy(p, data)[0], **handoff_diag)
    print(f'{args.mode} n={args.n} handoff R={handoff["ratio"]:.3g} acc={handoff["exact_accuracy"]:.4f}', flush=True)

    # Phase 2: exact Armijo gradient descent from the AdamW checkpoint.
    entry = 0 if handoff['ratio'] < 1 else None
    gd_log, termination, tau = [], 'search_budget_reached', 0.0
    min_post_slack, min_post_margin, min_post_pre = math.inf, math.inf, math.inf
    for update in range(1, args.gd_updates + args.post_updates + 1):
        loss = objective(p, data)
        L_prev = float(loss.detach())
        grads = torch.autograd.grad(loss, params)
        g2 = sum(float((g * g).sum()) for g in grads)
        old = [v.detach().clone() for v in params]
        eta = args.max_step
        for backtracks in range(args.max_backtracks):
            with torch.no_grad():
                for v, v_old, g in zip(params, old, grads):
                    v.copy_(v_old - eta * g)
                cand, diag, new_signs = objective(p, data, True)
            L_new = float(cand)
            slack = L_prev - L_new - eta * g2 / 2
            if math.isfinite(L_new) and slack >= 0:
                break
            eta *= args.shrink
        else:
            with torch.no_grad():
                for v, v_old in zip(params, old):
                    v.copy_(v_old)
            termination = 'backtracking_exhausted'
            break
        tau += eta
        ratio = L_new / threshold
        if entry is None and ratio < 1:
            entry = update
        if entry is not None:
            min_post_slack = min(min_post_slack, slack)
            min_post_margin = min(min_post_margin, diag['min_margin'])
            min_post_pre = min(min_post_pre, diag['min_abs_preactivation'])
        rec = dict(update=update, loss=L_new, ratio=ratio, eta=eta, tau=tau, armijo_slack=slack,
                   backtracks=backtracks, pattern_changes=int((new_signs != signs).sum()), **diag)
        signs = new_signs
        gd_log.append(rec)
        if update % 1000 == 0:
            print(f'{args.mode} n={args.n} gd j={update} R={ratio:.3g} entry={entry}', flush=True)
        if entry is None and update >= args.gd_updates:
            break
        if entry is not None and update >= entry + args.post_updates:
            termination = 'post_entry_complete'
            break

    final_acc, final_correct = greedy_accuracy(p, data)
    summary = dict(
        mode=args.mode, n=args.n, seed=args.seed, sigma=args.sigma, support_size=len(data['prompt']),
        supervised_positions=data['gold'].shape[1], pi_min=pi_min, threshold=threshold,
        adam=dict(updates=args.adam_updates, lr=args.lr, weight_decay=args.weight_decay,
                  first_update_all_prompts_correct=solved_at),
        handoff=handoff,
        gd=dict(entry_update=entry, updates=len(gd_log), termination=termination,
                final_ratio=gd_log[-1]['ratio'] if gd_log else handoff['ratio'],
                min_post_entry_armijo_slack=None if entry is None else min_post_slack,
                min_post_entry_margin=None if entry is None else min_post_margin,
                min_post_entry_abs_preactivation=None if entry is None else min_post_pre,
                post_entry_pattern_changes=sum(r['pattern_changes'] for r in gd_log if entry is not None and r['update'] > entry)),
        final_exact_accuracy=final_acc, final_correct_prompts=final_correct,
        elapsed_seconds=time.monotonic() - t0,
        scope='Finite enumerated support; float64 numerical checks, not interval arithmetic.')
    (out / 'adam_log.json').write_text(json.dumps(adam_log))
    (out / 'gd_log.json').write_text(json.dumps(gd_log))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['step', 'outcome'], required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sigma', type=float, default=0.02)
    ap.add_argument('--adam-updates', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--gd-updates', type=int, default=20000)
    ap.add_argument('--post-updates', type=int, default=1000)
    ap.add_argument('--max-step', type=float, default=100.0)
    ap.add_argument('--max-backtracks', type=int, default=80)
    ap.add_argument('--shrink', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', default='results/paper/basin_certificate_adamw')
    run(ap.parse_args())
