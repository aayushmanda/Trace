"""Full-format counting basin entry under exact weighted Armijo GD.

Example: python handcoded/basin_certificate.py --n 4 --seed 42
All parameters train; the entire seven-token readout starts at zero.
"""
import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

VOCAB, WIDTH, HEADS, HEAD_DIM, MLP_WIDTH = 7, 16, 3, 8, 16
SEP, COLON, EOS = 4, 5, 6


def enumerate_data(n, device):
    words = torch.tensor(list(itertools.product(range(2), repeat=n)), device=device)
    counts = (words == 0).sum(1)
    # Cast before powers: the item law is evaluated in float64 as well.
    counts = counts.to(torch.float64)
    probability = (.25 ** counts * .75 ** (n-counts)) / 2
    words = words.repeat(2, 1)
    probability = probability.repeat(2)
    start = torch.arange(2, device=device).repeat_interleave(2**n)
    states = (start[:, None] + (words == 0).cumsum(1)) % 2
    prompt = torch.cat([2+start[:, None], torch.zeros_like(start[:, None]), words,
                        torch.full_like(start[:, None], SEP)], 1)
    gold = torch.cat([2+states, torch.full_like(start[:, None], COLON),
                      2+states[:, -1:], torch.full_like(start[:, None], EOS)], 1)
    weights = probability[:, None].expand_as(gold) / gold.shape[1]
    assert abs(float(weights.sum())-1) < 1e-14
    assert torch.all(weights > 0)
    return dict(prompt=prompt, gold=gold, ids=torch.cat([prompt, gold[:, :-1]], 1),
                weights=weights, start_position=prompt.shape[1]-1)


def initialize(seed, sigma, device):
    generator = torch.Generator().manual_seed(seed)
    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64).to(device)
    p = dict(wte=normal(VOCAB, WIDTH), wpe=normal(40, WIDTH),
             readout=torch.zeros(VOCAB, WIDTH, dtype=torch.float64, device=device))
    for b in range(2):
        p[f'{b}.q'] = normal(HEADS, HEAD_DIM, WIDTH)
        p[f'{b}.k'] = normal(HEADS, HEAD_DIM, WIDTH)
        p[f'{b}.v'] = normal(HEADS, WIDTH, WIDTH)
        p[f'{b}.mi'] = normal(MLP_WIDTH, WIDTH)
        p[f'{b}.mo'] = normal(WIDTH, MLP_WIDTH)
        bias = normal(MLP_WIDTH)
        p[f'{b}.mb'] = bias.sign() * (1+bias.abs())
    return {k: (sigma*v).requires_grad_() for k, v in p.items()}


def forward(p, ids, diagnostics=False):
    length = ids.shape[1]
    x = p['wte'][ids] + p['wpe'][:length]
    mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
    preactivations = []
    for b in range(2):
        q = torch.einsum('btw,hdw->bhtd', x, p[f'{b}.q'])
        k = torch.einsum('btw,hdw->bhtd', x, p[f'{b}.k'])
        v = torch.einsum('btw,hvw->bhtv', x, p[f'{b}.v'])
        scores = (q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)).masked_fill(mask, -math.inf)
        x = x + (scores.softmax(-1) @ v).sum(1)
        pre = x @ p[f'{b}.mi'].T + p[f'{b}.mb']
        if diagnostics:
            preactivations.append(pre.detach())
        x = x + F.relu(pre) @ p[f'{b}.mo'].T
    logits = x @ p['readout'].T
    if diagnostics:
        return logits, torch.stack(preactivations)
    return logits


def objective(p, data, diagnostics=False):
    result = forward(p, data['ids'], diagnostics)
    logits, pre = result if diagnostics else (result, None)
    logits = logits[:, data['start_position']:]
    ce = F.cross_entropy(logits.flatten(0, 1), data['gold'].flatten(), reduction='none')
    loss = (ce.view_as(data['weights']) * data['weights']).sum()
    if not diagnostics:
        return loss
    gold_logits = logits.gather(-1, data['gold'][..., None]).squeeze(-1)
    wrong = logits.detach().clone()
    wrong.scatter_(-1, data['gold'][..., None], -math.inf)
    return loss, dict(min_margin=float((gold_logits-wrong.max(-1).values).detach().min()),
                      min_abs_preactivation=float(pre.abs().min())), pre > 0


@torch.no_grad()
def greedy_accuracy(p, data):
    ids = data['prompt']
    generated = []
    for _ in range(data['gold'].shape[1]):
        token = forward(p, ids)[:, -1].argmax(-1)
        generated.append(token)
        ids = torch.cat([ids, token[:, None]], 1)
    correct = (torch.stack(generated, 1) == data['gold']).all(1)
    return float(correct.double().mean()), int(correct.sum())


def save_checkpoint(path, p, args, update):
    torch.save(dict(params={k:v.detach().cpu() for k,v in p.items()},
                    config=vars(args), update=update), path)


def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    data = enumerate_data(args.n, args.device)
    p = initialize(args.seed, args.sigma, args.device)
    assert torch.count_nonzero(p['readout']).item() == 0
    out = Path(args.output) / f'n{args.n}_s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'history.csv').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    pi_min = float(data['weights'].min())
    threshold = pi_min/4
    torch.save({k:v.cpu() for k,v in data.items() if isinstance(v,torch.Tensor)}, out/'support.pt')
    (out/'config.json').write_text(json.dumps(dict(**vars(args), pi_min=pi_min,
        threshold=threshold, support_size=len(data['prompt']),
        supervised_positions=data['gold'].shape[1], width=WIDTH, heads=HEADS,
        head_dim=HEAD_DIM, mlp_width=MLP_WIDTH, full_readout_zero=True), indent=2))
    params = list(p.values())
    history = []
    loss, diagnostic, signs = objective(p, data, True)
    assert abs(float(loss.detach())-math.log(VOCAB)) < 1e-13
    accuracy, correct = greedy_accuracy(p, data)
    history.append(dict(update=0, loss=float(loss.detach()), ratio=float(loss.detach())/threshold,
        **diagnostic, eta=None, tau=0., grad_norm_sq=None, armijo_slack=None,
        backtracks=None, pattern_changes=0, exact_accuracy=accuracy, correct_prompts=correct))
    entry, tau, start_time = None, 0., time.monotonic()
    termination = 'search_budget_reached'
    with (out/'history.csv').open('w', buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerow(history[0])
        for update in range(1,args.max_updates+args.post_updates+1):
            loss = objective(p,data)
            previous_loss = float(loss.detach())
            gradients = torch.autograd.grad(loss, params)
            grad_norm_sq = sum(float((g*g).sum()) for g in gradients)
            if update == 1:
                assert float(gradients[list(p).index('readout')].norm()) > 0
                assert all(torch.count_nonzero(g).item()==0 for k,g in zip(p,gradients) if k!='readout')
            old = [v.detach().clone() for v in params]
            eta = args.max_step
            for backtracks in range(args.max_backtracks):
                with torch.no_grad():
                    for v,v_old,g in zip(params,old,gradients):
                        v.copy_(v_old-eta*g)
                    candidate, diagnostic, new_signs = objective(p,data,True)
                candidate_loss = float(candidate)
                slack = previous_loss-candidate_loss-eta*grad_norm_sq/2
                if math.isfinite(candidate_loss) and slack >= 0:
                    break
                eta *= args.shrink
            else:
                with torch.no_grad():
                    for v,v_old in zip(params,old):
                        v.copy_(v_old)
                termination = 'backtracking_exhausted'
                print(f'Backtracking exhausted at attempted update {update}; preserving the last accepted iterate.',flush=True)
                break
            tau += eta
            ratio = candidate_loss/threshold
            entering = entry is None and ratio < 1
            if entering:
                entry = update
                save_checkpoint(out/'entry.pt',p,args,update)
            if update % args.eval_every == 0 or entering or update == args.max_updates or (entry is not None and update == entry+args.post_updates):
                accuracy, correct = greedy_accuracy(p,data)
            else:
                accuracy, correct = None, None
            record = dict(update=update,loss=candidate_loss,ratio=ratio,**diagnostic,
                eta=eta,tau=tau,grad_norm_sq=grad_norm_sq,armijo_slack=slack,
                backtracks=backtracks,pattern_changes=int((new_signs != signs).sum()),
                exact_accuracy=accuracy,correct_prompts=correct)
            signs = new_signs
            history.append(record); writer.writerow(record)
            if update % 1000 == 0:
                save_checkpoint(out/'last_accepted.pt',p,args,update)
            if update % 1000 == 0 or entering:
                print(f'n={args.n} seed={args.seed} j={update} R={ratio:.4g} margin={diagnostic["min_margin"]:.4g} entry={entry}',flush=True)
            if entry is None and update >= args.max_updates:
                break
            if entry is not None:
                assert ratio < 1 and diagnostic['min_margin'] > 0
                assert diagnostic['min_abs_preactivation'] > 0
                if accuracy is not None:
                    assert correct == len(data['prompt'])
                if update >= entry+args.post_updates:
                    termination = 'post_entry_complete'
                    break
    update = history[-1]['update']
    save_checkpoint(out/'final.pt',p,args,update)
    suffix = [r for r in history if entry is not None and r['update'] >= entry]
    # D_j is indexed by its departing iterate; history rows index arriving iterates.
    certified_descent = [r['armijo_slack'] for r in history if entry is not None and r['update'] > entry]
    summary = dict(n=args.n,seed=args.seed,entry_update=entry,updates=update,
        termination=termination,last_greedy_evaluation_update=update,
        post_entry_updates=0 if entry is None else update-entry,pi_min=pi_min,
        final_ratio=history[-1]['ratio'],final_loss=history[-1]['loss'],
        final_min_margin=history[-1]['min_margin'],
        final_exact_accuracy=greedy_accuracy(p,data)[0],
        min_armijo_slack=min((r['armijo_slack'] for r in history[1:]),default=None),
        min_post_entry_armijo_slack=min(certified_descent) if certified_descent else None,
        min_certified_abs_preactivation=min(r['min_abs_preactivation'] for r in suffix) if suffix else None,
        elapsed_seconds=time.monotonic()-start_time,
        certificate_scope='Finite enumerated support; float64 numerical checks, not interval arithmetic.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plot(out,history,entry,args)
    print(json.dumps(summary,indent=2),flush=True)


def plot(out,history,entry,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(6,7),sharex=True)
    updates=[r['update'] for r in history]
    axes[0].semilogy(updates,[r['ratio'] for r in history])
    axes[0].axhline(1,color='black',ls='--',lw=1)
    axes[0].set_ylabel(r'Certificate ratio $4L/\pi_{\min}$')
    axes[1].plot(updates,[r['min_margin'] for r in history])
    axes[1].axhline(0,color='black',ls='--',lw=1)
    axes[1].set_ylabel('Minimum gold-token margin')
    evaluations=[r for r in history if r['exact_accuracy'] is not None]
    axes[2].plot([r['update'] for r in evaluations],[100*r['exact_accuracy'] for r in evaluations])
    axes[2].set(ylabel='Exhaustive exact accuracy (%)',xlabel='Accepted GD updates',ylim=(-2,102))
    for ax in axes:
        if entry is not None:
            ax.axvline(entry,color='#b85830',ls=':',label=r'$j_0$')
        ax.grid(alpha=.15)
    axes[0].set_title(f'Full-format counting: n={args.n}, seed={args.seed}')
    if entry is not None:axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'certificate.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,choices=range(2,17),default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sigma',type=float,default=.02)
    parser.add_argument('--max-updates',type=int,default=20000)
    parser.add_argument('--post-updates',type=int,default=1000)
    parser.add_argument('--max-step',type=float,default=100.)
    parser.add_argument('--max-backtracks',type=int,default=80)
    parser.add_argument('--shrink',type=float,default=.5)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output',default='results/paper/basin_certificate')
    args=parser.parse_args()
    assert 0 < args.shrink < 1 and args.max_step > 0 and args.sigma > 0
    assert args.max_updates > 0 and args.post_updates > 0 and args.eval_every > 0 and args.max_backtracks > 0
    run(args)
