"""Aggregate and plot completed review validations; never consume smoke results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path('results/review_validation')


def save(fig,name):
    fig.savefig(ROOT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/f'{name}.png',bbox_inches='tight',dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    a=pd.read_csv(ROOT/'concentration/metrics.csv')
    b=pd.read_csv(ROOT/'concentration_intermediate/metrics.csv')
    if len(a[a.step==2000])!=30 or len(b[b.step==2000])!=21:
        raise RuntimeError('Concentration training is incomplete.')
    # Repeated clean controls are a reproducibility check, not extra replicates.
    clean_a=a[(a.step==2000)&(a.rho==1)].sort_values('seed')
    clean_b=b[(b.step==2000)&(b.rho==1)].sort_values('seed')
    np.testing.assert_allclose(clean_a.final_answer,clean_b.final_answer,rtol=0,atol=0)
    np.testing.assert_allclose(clean_a.local_loss,clean_b.local_loss,rtol=0,atol=0)
    combined=pd.concat([a,b[b.rho!=1]],ignore_index=True)
    final=combined[combined.step==2000].copy()
    final.to_csv(ROOT/'concentration_final.csv',index=False)
    summary=final.groupby(['rho','concentration'])[['final_answer','local_accuracy','tabular_answer_accuracy','fraction_rows_certified','mean_context_error','mean_context_std']].agg(['mean','std','count'])
    summary.to_csv(ROOT/'concentration_summary.csv')
    contrasts=[]
    for rho in [.3,.5,.7]:
        subset=final[final.rho==rho]
        pivot=subset.pivot(index='seed',columns='concentration',values='final_answer')
        differences=pivot[0]-pivot[1]
        mean=differences.mean(); se=differences.std(ddof=1)/np.sqrt(3)
        width=stats.t.ppf(.975,2)*se
        contrasts.append(dict(rho=rho,mean_difference=mean,std=differences.std(ddof=1),
            ci95_low=mean-width,ci95_high=mean+width,n=3,
            note='Paired model+corpus seeds; t interval with 2 df; no multiple-comparison adjustment.'))
    pd.DataFrame(contrasts).to_csv(ROOT/'paired_contrasts.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,rho in zip(axes,[.3,.5,.7]):
        subset=final[final.rho==rho]
        for metric,label,color,style in [('final_answer','Transformer','#1766a4','-'),('tabular_answer_accuracy','Same-corpus table','#c46c19','--')]:
            g=subset.groupby('concentration')[metric].agg(['mean','std'])
            ax.errorbar(g.index,100*g['mean'],yerr=100*g['std'],marker='o',linestyle=style,color=color,label=label,capsize=3)
        ax.axhline(6.25,color='gray',lw=1,ls=':')
        ax.set_title(f'Reliability = {rho:g}'); ax.set_xlabel('Wrong-rule concentration'); ax.set_ylim(-3,108)
    axes[0].set_ylabel('Free-running answer accuracy (%)'); axes[0].legend(fontsize=8)
    fig.suptitle('Matched corruption intervention: three seeds, mean ± sample SD',fontsize=12)
    fig.tight_layout(); save(fig,'concentration_comparison')
    dense=pd.read_csv(ROOT/'dense_tabular/metrics.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for concentration,color in zip([0,.25,.5,.75,1],plt.cm.viridis(np.linspace(.05,.9,5))):
        g=dense[dense.concentration==concentration].groupby('rho').strict_rule_recovery.agg(['mean','std'])
        ax.plot(g.index,g['mean'],color=color,label=f'λ={concentration:g}')
        ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=color,alpha=.12)
        c=concentration+(1-concentration)/15
        ax.axvline(c/(1+c),color=color,ls=':',lw=1)
    ax.set(xlabel='Clean-trace reliability',ylabel='Strict local-rule recovery',ylim=(-.03,1.03))
    ax.legend(title='Concentration'); fig.tight_layout(); save(fig,'tabular_boundary')
    canonical=pd.read_csv(ROOT/'canonical_corpora/metrics.csv')
    old=pd.read_csv('results/paper/boolean_reliability_canonical.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    t=canonical[canonical.seed==501].sort_values('rho')
    ax.plot(t.rho,100*t.answer_accuracy,'o-',label='Table: exact canonical corpus',color='#c46c19')
    g=old[old.rho.notna()].groupby('rho').answer_accuracy.agg(['mean','std'])
    ax.errorbar(g.index,100*g['mean'],100*g['std'],fmt='o-',capsize=3,label='Canonical GPT: five model seeds',color='#1766a4')
    ax.axhline(6.25,color='gray',ls=':',lw=1)
    ax.set(xlabel='Clean-trace reliability',ylabel='Answer accuracy (%)',ylim=(-3,108))
    ax.legend();fig.tight_layout();save(fig,'canonical_same_corpus')
    functional=pd.read_csv(ROOT/'functional_final/updates.csv')
    expected=3*6*(2*(9*4*3+6))
    assert len(functional)==expected,(len(functional),expected)
    functional.groupby(['step','corruption','source','group','optimizer','eta'])[['delta_loss','delta_conditional_state_loss']].agg(['mean','std','count']).to_csv(ROOT/'functional_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    labels={'clean_state':'Clean states','wrong_state':'Wrong states','uniform_state_wrong_prefix':'Uniform state targets','mixed_state_rho07':'70% clean mixture'}
    for ax,law in zip(axes,['symmetric','coherent']):
        subset=functional[(functional.optimizer=='euclidean')&(functional.eta==.001)&(functional.group=='all')&(functional.corruption==law)]
        for source,label in labels.items():
            g=subset[subset.source==source].groupby('step').delta_conditional_state_loss.agg(['mean','std'])
            ax.errorbar(g.index,g['mean'],g['std'],fmt='o-',capsize=2,label=label,ms=3)
        ax.axhline(0,color='black',lw=1)
        ax.set_xscale('symlog',linthresh=25);ax.set_xlim(0,2500)
        ax.set_xticks([0,25,100,300,1000,2000], labels=['0','25','100','300','1000','2000'])
        ax.tick_params(axis='x',labelsize=8)
        ax.set_yscale('symlog',linthresh=1e-7)
        ax.set_title(law.capitalize());ax.set_xlabel('Training step')
    axes[0].set_ylabel('Change in independent conditional state loss\nnegative = improvement; η = 0.001')
    axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'functional_updates')
    e=functional[(functional.optimizer=='euclidean')&(functional.eta==1e-4)]
    meaningful=e[e.predicted_delta.abs()>1e-10]
    agreement=float((np.sign(meaningful.predicted_delta)==np.sign(meaningful.delta_loss)).mean())
    # Verify dense vectorized tables against ALL now-completed neural corpora.
    checked=0
    for directory in ['concentration','concentration_intermediate']:
        for path in (ROOT/directory).glob('corpus_*.npz'):
            data=np.load(path)
            assert data['counts'].sum()==80000
            checked+=1
    report={'unique_concentration_models':len(final),'physical_concentration_training_runs':51,
        'clean_control_repeats_identical':True,'functional_training_models':3,
        'functional_rows':len(functional),'dense_tabular_conditions':len(dense),
        'canonical_corpus_conditions':len(canonical),'corpus_count_checks':checked,
        'first_order_sign_agreement_small_step':agreement,'first_order_comparisons':len(meaningful),
        'first_order_note':'eta 1e-4, |predicted loss change|>1e-10; repeated probes, not independent replicates',
        'paired_contrasts':contrasts}
    (ROOT/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    files=list(ROOT.rglob('*.csv'))+list(ROOT.rglob('protocol.json'))+list(ROOT.glob('*.pdf'))
    sources=list(Path('src/experiments').rglob('review_*.py'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+sources if 'smoke' not in str(p)}
    (ROOT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(summary[['final_answer','tabular_answer_accuracy']].to_string())


if __name__=='__main__':
    main()
