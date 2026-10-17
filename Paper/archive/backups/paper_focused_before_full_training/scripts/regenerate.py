"""Regenerate figures and tables from checked per-seed evidence (no training)."""
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; R=P.parent
E=P/'evidence';E.mkdir(exist_ok=True)
manifest=[]
def record(path,dest=None):
    data=path.read_bytes();manifest.append({'source':str(path.relative_to(R)),'sha256':hashlib.sha256(data).hexdigest()})
    if dest: shutil.copy2(path,E/dest)

def readjson(path,dest=None):
    record(path,dest);return json.loads(path.read_text())
fig,axs=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
allrows=[]
for ax,prefix,title in zip(axs,['both_zero_ball_step1e-6_s','both_zero_seed'],['Theorem-radius exit','Larger displacement']):
    runs=[]
    for seed in range(3):
        name=f'{prefix}{seed}.json';d=readjson(R/'results/theorem_regime'/name,name)
        runs.extend(d);allrows.extend(dict(radius=title,**row) for row in d)
    ns=sorted(set(x['n'] for x in runs))
    for mode,col in [('outcome','#D55E00'),('step','#0072B2')]:
        values=np.array([[x['T'] for x in runs if x['n']==n and x['mode']==mode] for n in ns])
        med=np.median(values,axis=1)
        ax.errorbar(ns,med,yerr=[med-values.min(1),values.max(1)-med],marker='o',ms=3,capsize=2,color=col,label=mode.capitalize())
        if mode=='outcome':ax.plot(ns,med[0]*2.**(np.array(ns)-ns[0]),'--',color='#009E73',lw=1,label=r'$\lambda^{-n}$ guide')
    ax.set(yscale='log',xlabel='Horizon n',ylabel='Accumulated flow time',title=title,xticks=[4,8,12,14]);ax.grid(alpha=.15)
axs[0].legend(fontsize=7)
fig.savefig(P/'figures/controlled.pdf');plt.close(fig)
(E/'controlled_rows.json').write_text(json.dumps(allrows,indent=2))
# Extract Boolean-8 CSVs, preserving the exact settings in each original row.
groups=[];lo=[]
for label,pattern in [('Boolean-8: step','main/r1.0_s*.csv'),('Boolean-8: outcome','main/outcome_s*.csv'),('Boolean-8: answer first','main/answer_first_s*.csv'),('Boolean-8: outcome, 8x','outcome_6400/s*.csv')]:
    rows=[]
    for path in sorted((R/'results/paper/qwen05').glob(pattern)):
        record(path,'qwen_'+path.parent.name+'_'+path.name)
        rr=list(csv.DictReader(path.open()));assert len(rr)==1;rows.extend(rr)
    assert len(rows)==3,(label,len(rows))
    vals=[100*float(x['answer_accuracy']) for x in rows]
    groups.append((label,vals));lo.extend(rows)
(E/'qwen_rows.json').write_text(json.dumps(lo,indent=2))
# Official BBH metrics, not generated-test metrics at the top level.
bbh=[]
for k in [3,5,7]:
    for mode in ['process','outcome']:
        vals=[]
        for seed in [2001,2002,2003]:
            suffix='' if seed==2001 else f'_s{seed}'
            path=R/f'results/shuffled/bbh/bbh_K{k}_{mode}{suffix}.json'
            d=readjson(path);assert d['seed']==seed and d['bbh']['size']==250
            compact={key:d[key] for key in ['model','K','n','condition','seed','train_size','steps','batch_size','lr']}
            compact['official']= {key:val for key,val in d['bbh'].items() if key!='samples'}
            compact['generated']={key:d[key] for key in ['test_size','answer_accuracy','exact_trace_accuracy']}
            bbh.append(compact);vals.append(100*d['bbh']['answer_accuracy'])
        groups.append((f'BBH {k} objects: '+('step' if mode=='process' else mode),vals))
(E/'bbh_rows.json').write_text(json.dumps(bbh,indent=2))
tex=[r'\begin{table}[t]',r'\centering\small',r'\caption{\textbf{Qwen2.5-0.5B LoRA transfer, with all three seeds.} Held-out greedy answer accuracy (\%). Columns identify seeds $2001,2002,2003$; SD is the sample standard deviation. Boolean-8 has $300$ held-out generated prompts; BBH has $250$ official items per size. The eightfold outcome control uses $6400$ rather than $800$ updates.}',r'\label{tab:transfer}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Condition & 2001 & 2002 & 2003 & Mean $\pm$ SD\\',r'\midrule']
for label,vals in groups:
    tex.append(label.replace('8x',r'$8\times$')+' & '+' & '.join(f'{v:.1f}' for v in vals)+f' & ${np.mean(vals):.1f}\\pm{np.std(vals,ddof=1):.1f}$'+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'figures/transfer_table.tex').write_text('\n'.join(tex)+'\n')
# Keep both versions of label control.
for suffix in ['summary.json','v2_30k/summary.json']:
    readjson(R/'Paper/research/label_budget'/suffix,'labels_'+suffix.replace('/','_'))
# seed table generated from per-run original metrics, cross-check summaries
labels=json.loads((E/'labels_v2_30k_summary.json').read_text())
for row in labels:
    values=[r['answer'] for r in row['per_seed']];assert abs(np.mean(values)-row['mean_answer'])<1e-12
# Provenance for extra empirical claims
for path in [R/'results/paper/zero_readout_n16_8k/summary.json',R/'results/paper/zero_readout_n16_8k/per_seed.csv']:
    record(path,'counting_'+path.name)
(E/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Regenerated counting figure and transfer table from',len(manifest),'source files.')
for label,vals in groups: print(label, f'{np.mean(vals):.3f} +/- {np.std(vals,ddof=1):.3f}',vals)
