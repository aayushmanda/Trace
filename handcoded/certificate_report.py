"""Validate and summarize the completed n=4,6 basin-certificate pilots."""
import csv
import json
from pathlib import Path

import torch
import basin_certificate as B


def report(root=Path('results/paper/basin_certificate')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex='col')
    summaries, checks = [], []
    for col,n in enumerate((4,6)):
        folder=root/f'n{n}_s42'
        summary=json.loads((folder/'summary.json').read_text())
        config=json.loads((folder/'config.json').read_text())
        data=B.enumerate_data(n,'cpu')
        support=torch.load(folder/'support.pt',weights_only=True)
        assert all(torch.equal(data[k],v) for k,v in support.items() if k!='weights')
        weight_relative_error=float(((data['weights']-support['weights']).abs()/data['weights']).max())
        assert weight_relative_error < 1e-14
        # Preserve the exact tensor used by the GPU objective, including roundoff.
        data['weights']=support['weights']
        assert float(data['weights'].min()) == config['pi_min'] == summary['pi_min']
        checkpoint_verified = False
        if (folder/'final.pt').exists():
            checkpoint=torch.load(folder/'final.pt',weights_only=True)
            p=checkpoint['params']
            loss,diag,_=B.objective(p,data,True)
            accuracy,_=B.greedy_accuracy(p,data)
            assert abs(float(loss)-summary['final_loss']) < 1e-10
            assert accuracy==summary['final_exact_accuracy']
            assert abs(diag['min_margin']-summary['final_min_margin']) < 1e-8
            checkpoint_verified = True
        else:
            assert summary['termination']=='backtracking_exhausted' and summary['entry_update'] is None
        with (folder/'history.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==summary['updates']+1
        assert all(int(r['update'])==j for j,r in enumerate(rows))
        assert all(float(r['armijo_slack'])>=0 for r in rows[1:])
        assert all(float(r['min_abs_preactivation'])>0 for r in rows)
        for old,new in zip(rows,rows[1:]):
            slack=float(old['loss'])-float(new['loss'])-float(new['eta'])*float(new['grad_norm_sq'])/2
            assert abs(slack-float(new['armijo_slack']))<1e-12
        entry=summary['entry_update']
        if entry is not None:
            assert float(rows[entry]['ratio'])<1
            assert all(float(r['ratio'])>=1 for r in rows[:entry])
            assert summary['post_entry_updates']>=1000
            assert all(float(r['ratio'])<1 and float(r['min_margin'])>0 for r in rows[entry:])
            assert all(int(r['correct_prompts'])==len(data['prompt']) for r in rows[entry:] if r['correct_prompts'])
            entry_checkpoint=torch.load(folder/'entry.pt',weights_only=True)
            entry_loss=B.objective(entry_checkpoint['params'],data)
            assert float(entry_loss)<config['threshold']
            assert B.greedy_accuracy(entry_checkpoint['params'],data)[1]==len(data['prompt'])
        updates=[int(r['update']) for r in rows]
        axes[0,col].semilogy(updates,[float(r['ratio']) for r in rows])
        axes[0,col].axhline(1,color='black',ls='--',lw=1)
        axes[1,col].plot(updates,[float(r['min_margin']) for r in rows])
        axes[1,col].axhline(0,color='black',ls='--',lw=1)
        evaluations=[r for r in rows if r['exact_accuracy']]
        axes[2,col].plot([int(r['update']) for r in evaluations],
                         [100*float(r['exact_accuracy']) for r in evaluations])
        axes[2,col].set(ylim=(-2,102),xlabel='Accepted GD updates')
        status = f'entry j₀={entry}' if entry is not None else ('backtracking stopped' if summary.get('termination')=='backtracking_exhausted' else 'no entry within budget')
        axes[0,col].set_title(f'n={n}, seed=42; {status}')
        for ax in axes[:,col]:
            ax.grid(alpha=.15)
            if entry is not None:ax.axvline(entry,color='#b85830',ls=':')
        summaries.append(summary)
        checks.append(dict(n=n,seed=42,all_item_weights_verified=True,
            independently_regenerated_weight_max_relative_error=weight_relative_error,
            all_armijo_slacks_nonnegative=True,all_slacks_recomputed=True,
            all_saved_iterates_have_nonzero_preactivations=True,
            checkpoint_loss_and_greedy_reproduced=checkpoint_verified,entry_update=entry))
    axes[0,0].set_ylabel(r'Certificate ratio $4L/\pi_{\min}$')
    axes[1,0].set_ylabel('Minimum gold-token margin')
    axes[2,0].set_ylabel('Exhaustive exact accuracy (%)')
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(root/f'certificate_pilots.{ext}',dpi=180)
    (root/'verification.json').write_text(json.dumps(checks,indent=2))
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    table=['| Horizon | Accepted updates | Entry update | Last certificate ratio | Last evaluated exact accuracy |',
           '| --- | --- | --- | --- | --- |']
    for s in summaries:
        accuracy = s['final_exact_accuracy']
        accuracy_text = f'{100*accuracy:.3f}%' if accuracy is not None else f'{100*s["last_evaluated_exact_accuracy"]:.3f}% (update {s["last_greedy_evaluation_update"]})'
        table.append(f'| {s["n"]} | {s["updates"]} | {s["entry_update"] if s["entry_update"] is not None else "Not reached"} | {s["final_ratio"]:.5g} | {accuracy_text} |')
    (root/'RESULTS.md').write_text('# Full-format basin-certificate pilots\n\n'+'\n'.join(table)+
        '\n\nEach horizon uses one seed (42), exact weighted full-batch Armijo GD, and the same fixed-width architecture. '
        'Entry means the first loss strictly below the actual minimum item weight divided by four. '
        'A missing entry is a negative result within the stated budget; it is not evidence of permanent failure. '
        'The n=4 pilot exhausted backtracking at attempted update 8,308, near a ReLU boundary; its accepted history ends at 8,307 and its last greedy evaluation is at 8,300. The initial runner did not save a final checkpoint on that error. The failure is preserved, and no negative Armijo tolerance was introduced. The current runner saves a diagnostic checkpoint when backtracking is exhausted. '
        'High accuracy alone is not a certificate. These are float64 numerical checks, not interval arithmetic. '
        'See README.md for the protocol, scope, and width caveat.\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    report()
