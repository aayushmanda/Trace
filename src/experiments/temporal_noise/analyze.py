"""Aggregate all temporal experiments, auditing the known trace endpoint."""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from src.data.registry import TASKS
from src.data.sample import generate_unique

ROOT=Path('results/novelty_extension')


def corrected_rollout(records,instances,skip=0):
    """Only score prescribed trace characters, not unsupervised trailing text.

    The raw no-answer generation may append arbitrary characters after the last
    state. Its suffix receives no training loss, so truncation at the known
    trace horizon is essential. This does not repair errors inside the trace.
    """
    valid=exact=endpoint=0
    for record,inst in zip(records,instances):
        expected=' '.join(inst.correct_trace.split()[skip:])
        text=record['generated'].lstrip()[:len(expected)]
        parts=text.split()
        gold=expected.split()
        parsed=[re.fullmatch(r'([xcst][0-3]{1,3})>([01]{4})',p) for p in parts]
        ok=len(parsed)==len(gold) and all(p and p[1]==g.split('>')[0] for p,g in zip(parsed,gold))
        valid+=bool(ok)
        exact+=text==expected
        endpoint+=bool(ok and parsed[-1][2]==inst.gold)
    n=len(records)
    assert n==len(instances)
    return dict(valid_trace=valid/n,exact_trace=exact/n,endpoint_accuracy=endpoint/n)


def paired(values):
    values=np.asarray(values);n=len(values)
    mean=float(values.mean());sd=float(values.std(ddof=1))
    half=float(t_distribution.ppf(.975,n-1)*sd/np.sqrt(n))
    return dict(mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half,n=n)


def main():
    rows=[];comparison=[];checks=[]; evaluation_cache={}
    for family in ['canonical','matched_patterns']:
        for path in sorted((ROOT/family).glob('s*')):
            result_name='metrics.json' if family=='canonical' else 'pattern_metrics.json'
            protocol_name='protocol.json' if family=='canonical' else 'pattern_protocol.json'
            if not (path/result_name).exists():
                continue
            result=json.loads((path/result_name).read_text())
            protocol=json.loads((path/protocol_name).read_text())
            task=TASKS['boolean_circuit_8']
            cache_key=(protocol['train_seed'],protocol['train_size'],protocol['eval_size'])
            if cache_key not in evaluation_cache:
                train=generate_unique(task,protocol['train_size'],protocol['train_seed'])
                evaluation_cache[cache_key]=generate_unique(task,protocol['eval_size'],101,{i.prompt for i in train})
            evaluation=evaluation_cache[cache_key]
            assert hashlib.sha256('\n'.join(i.prompt for i in evaluation).encode()).hexdigest()==protocol['eval_prompt_sha256']
            for label,skip in [('ordinary',0),('anchored',0),('anchor_two',2)]:
                file=path/f'{label}_generations.jsonl'
                if not file.exists():
                    continue
                records=[json.loads(line) for line in file.read_text().splitlines()]
                corrected=corrected_rollout(records,evaluation,skip)
                for metric,value in corrected.items():
                    key=f'{label}_{metric}'
                    comparison.append(dict(family=family,tag=result['tag'],metric=key,raw=result[key],corrected=value))
                    result[key]=value
            result.update(family=family,empirical_local_correctness=protocol['empirical_local_correctness'],
                          empirical_whole_trace_correctness=protocol['empirical_whole_trace_correctness'])
            probe=path/'same_row_prefix.json'
            if probe.exists():
                result.update(json.loads(probe.read_text()))
            result['conditional_prefix_gap']=result['correct_prefix_step2_conditional_probability']-result['wrong_prefix_step2_conditional_probability']
            archive=np.load(path/'corpus.npz')
            assert archive['counts'].sum()==protocol['train_size']*8
            assert np.isclose(archive['clean'].mean(),protocol['empirical_local_correctness'])
            checks.append(dict(family=family,tag=result['tag'],counts_verified=True))
            rows.append(result)
    df=pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(ROOT/'all_results.csv',index=False)
    pd.DataFrame(comparison).to_csv(ROOT/'trace_endpoint_audit.csv',index=False)
    # Pairing checks within each experimental family and seed.
    pair_checks=[]
    for family in ['canonical','matched_patterns']:
        for seed in [2001,2002,2003]:
            protocols=[]
            for path in (ROOT/family).glob(f's{seed}_*'):
                file=path/('protocol.json' if family=='canonical' else 'pattern_protocol.json')
                if file.exists():protocols.append(json.loads(file.read_text()))
            for field in ['train_prompt_sha256','eval_prompt_sha256','schedule_sha256','model_init_sha256']:
                values={p[field] for p in protocols if p.get(field)}
                assert len(values)<=1,(family,seed,field)
            pair_checks.append(dict(family=family,seed=seed,runs=len(protocols),pairing_verified=True))
    for seed in [2001,2002,2003]:
        paths=[ROOT/'matched_patterns'/f's{seed}_rho0.3_k{k}_answer0'/'corpus.npz' for k in [0,1]]
        if all(p.exists() for p in paths):
            first,second=[np.load(p)['clean'] for p in paths]
            np.testing.assert_array_equal(first.sum(1),second.sum(1))
            np.testing.assert_array_equal(first.all(1),second.all(1))
    contrasts=[]
    metrics=['same_row_gap','prefix_probability_gap','conditional_prefix_gap','anchored_exact_trace','ordinary_exact_trace','anchored_endpoint_accuracy']
    for rho,answer in [(.3,False),(.7,False),(.3,True)]:
        sub=df[(df.family=='canonical')&(df.rho==rho)&(df.answer_loss==answer)]
        for metric in metrics:
            pivot=sub.pivot(index='seed',columns='persistence',values=metric)
            if 0. in pivot and 1. in pivot and len(pivot.dropna(subset=[0.,1.]))==3:
                contrasts.append(dict(comparison='persistent_minus_independent',rho=rho,answer_loss=answer,metric=metric,**paired((pivot[1.]-pivot[0.]).dropna())))
    if 'pattern' in df:
        sub=df[df.family=='matched_patterns']
        for metric in metrics+['anchor_two_exact_trace']:
            pivot=sub.pivot(index='seed',columns='pattern',values=metric)
            if set(['blocked','interleaved']).issubset(pivot.columns) and len(pivot.dropna())==3:
                contrasts.append(dict(comparison='blocked_minus_interleaved',rho=.3,answer_loss=False,metric=metric,**paired(pivot.blocked-pivot.interleaved)))
    pd.DataFrame(contrasts).to_csv(ROOT/'paired_contrasts.csv',index=False)
    (ROOT/'integrity.json').write_text(json.dumps(dict(corpus_checks=checks,pair_checks=pair_checks),indent=2)+'\n')
    # Primary figure: learned state-law response and actual continuation quality.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for rho,color in [(.3,'#1766a4'),(.7,'#be681e')]:
        sub=df[(df.family=='canonical')&(df.rho==rho)&(~df.answer_loss)]
        if sub.empty:continue
        for ax,metric in zip(axes[:2],['prefix_probability_gap','anchored_exact_trace']):
            g=sub.groupby('persistence')[metric].agg(['mean','std'])
            ax.errorbar(g.index,g['mean'],g['std'],marker='o',capsize=3,color=color,label=f'ρ={rho:g}')
    axes[0].plot([0,1],[0,1],'k:',lw=1,label='Population optimum')
    axes[0].set(xlabel='Persistence κ',ylabel='Correct vs wrong prefix probability gap',ylim=(-.05,1.05))
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Persistence κ',ylabel='Exact trace with first step supplied',ylim=(-.02,1.02))
    sub=df[df.family=='matched_patterns'] if 'pattern' in df else pd.DataFrame()
    for pattern,color in [('blocked','#1766a4'),('interleaved','#be681e')]:
        z=sub[sub.pattern==pattern] if not sub.empty else sub
        if z.empty:continue
        vals=z[[f'gold_probability_step_{i}' for i in range(1,9)]]
        axes[2].errorbar(range(1,9),vals.mean(),vals.std(),marker='o',capsize=2,color=color,label=pattern.title())
    axes[2].set(xlabel='State position (gold prefix)',ylabel='Probability of correct state',ylim=(-.02,1.02))
    if not sub.empty:axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT/'temporal_validation.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'temporal_validation.png',dpi=180,bbox_inches='tight')
    print('Completed models:',len(df))
    print(df[['tag','family','prefix_probability_gap','ordinary_exact_trace','anchored_exact_trace']].to_string(index=False))
    if contrasts:print(pd.DataFrame(contrasts).to_string(index=False))


if __name__=='__main__':
    main()
