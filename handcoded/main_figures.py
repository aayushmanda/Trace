"""Main-paper figures, with audited sources and print-size vector output.

python handcoded/main_figures.py --result k2-zero
python handcoded/main_figures.py --result k3-three
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'papers/silent_routing/figures'
DATA = ROOT/'results/paper/main_figure_revision'
from figure_style import (STEP, OUTCOME, THEORY, MODEL, INK, MUTED, STEP_LIGHT, OUTCOME_LIGHT,
    MARKER, CHANCE_LS, THEORY_LS, FS_TEXT, FS_SMALL, TEXT_WIDTH, style, panel, clean, band)


def save(fig,name):
    # Fixed figure size (no bbox_inches='tight'): the PDF is the printed size.
    for ext in ('pdf','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=300,metadata={'CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)


def setup(ax):
    """Diagram of the same modular-counting task under two supervision formats."""
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    panel(ax, 'a', 'Task and targets')
    ax.text(.0,.93, r'Count a, mod 3; start at 1', fontsize=FS_TEXT, va='center')
    xs=np.linspace(.06,.94,6)
    for x,letter in zip(xs,'banana'):
        ax.text(x,.78,letter,ha='center',va='center',fontsize=10,
                color=OUTCOME if letter=='a' else INK,
                fontweight='bold' if letter=='a' else 'normal')
    ax.text(.0,.635,'Step',color=STEP,fontweight='bold',va='center')
    for x,count in zip(xs,[1,2,2,0,0,1]):
        ax.annotate('',xy=(x,.44),xytext=(x,.56),
                    arrowprops=dict(arrowstyle='->',color=STEP,lw=.7,shrinkA=0,shrinkB=0))
        ax.text(x,.38,str(count),ha='center',va='center',color=STEP,fontsize=FS_TEXT+.5,
                bbox=dict(boxstyle='round,pad=.18',fc=STEP_LIGHT,ec='none'))
    ax.plot([xs[0],xs[0],xs[-1],xs[-1]],[.30,.26,.26,.30],color=OUTCOME,lw=.8)
    ax.annotate('',xy=(xs[-1],.155),xytext=(xs[-1],.26),
                arrowprops=dict(arrowstyle='->',color=OUTCOME,lw=.7,shrinkA=0,shrinkB=0))
    ax.text(.0,.11,'Outcome',color=OUTCOME,fontweight='bold',va='center')
    ax.text(xs[-1],.11,'1',ha='center',va='center',color=OUTCOME,fontsize=FS_TEXT+.5,
            bbox=dict(boxstyle='round,pad=.18',fc=OUTCOME_LIGHT,ec='none'))
    ax.text(.0,-.06,'Same two-block architecture;',fontsize=FS_SMALL+.5,va='center')
    ax.text(.0,-.16,'exact executor for each format',fontsize=FS_SMALL+.5,color=THEORY,va='center')


def histories(which):
    sources=[];curves={}
    if which=='k2-zero':
        seeds=list(range(42,52));K=2
        for mode in ('process','outcome'):
            curves[mode]=[]
            for seed in seeds:
                path=ROOT/f'results/paper/zero_readout_n16_8k/s{seed}.json'
                rec=json.loads(path.read_text())
                assert rec['zero_readout'] and rec['config']['word_len']==16 and rec['config']['mod']==2
                rows=[r for r in rec['modes'][mode]['history'] if 'exact' in r]
                steps=[r['step'] for r in rows]
                curves[mode].append([100*r['exact'] for r in rows])
                sources.append(str(path.relative_to(ROOT)))
    else:
        seeds=[42,43,44];K=3
        for mode in ('process','outcome'):
            curves[mode]=[]
            for seed in seeds:
                path=ROOT/'logs/n16_K3'/f'{"cot_r1.0" if mode=="process" else "outcome"}_s{seed}.log'
                rows=re.findall(rf'{mode}\s+step\s+(\d+)\s+loss\s+[\d.e+-]+\s+test exact\s+([\d.]+)',path.read_text())
                unique={int(j):float(a)*100 for j,a in rows}
                steps=sorted(unique)
                curves[mode].append([unique[j] for j in steps])
                sources.append(str(path.relative_to(ROOT)))
    arrays={m:np.asarray(v) for m,v in curves.items()}
    assert all(a.shape==(len(seeds),len(steps)) for a in arrays.values())
    return np.asarray(steps),arrays,K,seeds,sorted(set(sources))


def overview(which):
    steps,curves,K,seeds,sources=histories(which)
    spectrum_path=ROOT/'results/paper/exact_spectra/sigma_sweep/s42_sg0.02.txt'
    rec=json.loads((ROOT/'results/paper/fig2/n8_s42.json').read_text())
    target=np.asarray(rec['A_r']);target=target/target.sum()
    line=next(l for l in spectrum_path.read_text().splitlines() if l.startswith('W_r / W'))
    jacobian=np.asarray([float(x) for x in line.split(':')[1].split('(')[0].split()])
    jacobian=jacobian/jacobian.sum()
    fig=plt.figure(figsize=(TEXT_WIDTH,2.2))
    grid=fig.add_gridspec(1,3,width_ratios=(1.1,1,1),left=.02,right=.985,
                          top=.875,bottom=.19,wspace=.52)
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    setup(axes[0]);ax=axes[1]
    panel(ax,'b','Training')
    for mode,col,key in [('process',STEP,'step'),('outcome',OUTCOME,'outcome')]:
        band(ax,steps/1000,curves[mode],col,marker=MARKER[key],zorder=3)
    ax.axhline(100,color=THEORY,lw=.9,ls=THEORY_LS,zorder=2)
    ax.axhline(100/K,color=MUTED,ls=CHANCE_LS,lw=.8,zorder=2)
    ax.text(.03,.955,'Exact executor',transform=ax.transAxes,fontsize=FS_SMALL+.5,color=THEORY,va='top')
    ax.text(.03,.74,'Step',color=STEP,transform=ax.transAxes,fontweight='bold',va='center')
    ax.text(.03,.645,'8/10 seeds solve' if which=='k2-zero' else '3/3 seeds solve',color=STEP,
            transform=ax.transAxes,fontsize=FS_SMALL,va='center')
    ax.text(.03,(100/K+3)/112,'Outcome',color=OUTCOME,transform=ax.transAxes,fontweight='bold',va='bottom')
    ax.text(.03,(100/K-5)/112,'Chance',color=MUTED,fontsize=FS_SMALL,transform=ax.transAxes,va='top')
    ax.set(xlabel='Updates (thousands)',ylabel='Exact accuracy (%)',ylim=(0,112),
           xlim=(0,steps[-1]/1000),yticks=[0,50,100])
    ax.set_xticks([0,4,8] if which=='k2-zero' else [0,16,32]);clean(ax)
    ax=axes[2];degree=np.arange(len(target))
    panel(ax,'c','Degree mismatch')
    for values,color,marker in [(target,OUTCOME,MARKER['outcome']),(jacobian,MODEL,MARKER['model'])]:
        ax.semilogy(degree,np.where(values>0,values,np.nan),marker+'-',color=color)
    ax.text(3.0,.9,r'Target $A_r$',color=OUTCOME,fontsize=FS_TEXT,va='bottom')
    ax.text(2.6,2e-5,r'Jacobian $J_r$',color=MODEL,fontsize=FS_TEXT)
    ax.set(xlabel=r'Degree $r$',ylabel='Energy share',xlim=(-.4,8.4),
           ylim=(1e-7,10),xticks=[0,4,8],yticks=[1e-6,1e-3,1]);clean(ax)
    save(fig,'main_overview')
    return dict(result=which,K=K,n=16,seeds=seeds,steps=steps.tolist(),
        band='25th to 75th seed percentiles; line is pointwise median',history_sources=sources,
        final_exact={m:v[:,-1].tolist() for m,v in curves.items()},
        spectrum_source=str(spectrum_path.relative_to(ROOT)),
        spectrum_case='K=2, n=8, sigma=0.02; nonzero readout; normalized energies')


def grid_cells():
    cells=[]
    for path in sorted((ROOT/'results/paper/lambda_grid').glob('n*_A*_s*.json')):
        rec=json.loads(path.read_text());assert rec['steps']==8000
        cells.append(dict(n=rec['n'],alphabet=rec['alphabet'],seed=rec['seed'],
            contraction=rec['lam'],bias=rec['lam']**rec['n'],
            process=100*rec['process']['final_exact'],outcome=100*rec['outcome']['final_exact'],
            source=str(path.relative_to(ROOT))))
    assert len(cells)==60
    return cells


def spearman(x,y):
    # Midranks handle ties in the horizon grid without adding a scipy dependency.
    def ranks(values):
        values=np.asarray(values);return np.asarray([np.sum(values<v)+(np.sum(values==v)+1)/2 for v in values])
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def prediction_panels(axes,cells,letters=('b','c')):
    """(b) outcome accuracy against lambda^n with the state-only prior; (c) accuracy above that prior."""
    groups={key:[r for r in cells if (r['n'],r['alphabet'])==key]
            for key in sorted({(r['n'],r['alphabet']) for r in cells})}
    means=np.asarray([np.mean([r['outcome'] for r in rs]) for rs in groups.values()])
    ns=np.asarray([key[0] for key in groups]);bias=np.asarray([rs[0]['bias'] for rs in groups.values()])
    prior=50*(1+bias);excess=means-prior
    correlation={'n':spearman(ns,means),'lambda^n':spearman(bias,means),
                 'excess,lambda^n':spearman(bias,excess),'excess,n':spearman(ns,excess)}
    lows=np.asarray([min(r['outcome'] for r in rs) for rs in groups.values()])
    highs=np.asarray([max(r['outcome'] for r in rs) for rs in groups.values()])
    guide=np.geomspace(1e-5,.7,200)
    for ax,y,title,letter,key in zip(axes,[means,excess],['Answer bias','Above the prior'],letters,
                                     ['lambda^n','excess,lambda^n']):
        ax.errorbar(bias,y,yerr=[means-lows,highs-means],fmt=MARKER['outcome'],ms=3.5,
                    color=OUTCOME,ecolor='#EDB79C',elinewidth=.8,capsize=0,zorder=3)
        panel(ax,letter,title)
        ax.set(xscale='log',xlabel=r'Remaining bias $\lambda^n$',xlim=(1e-5,.8),xticks=[1e-5,1e-3,1e-1]);clean(ax)
        ax.text(.04,.97,rf'Spearman $\rho={correlation[key]:.2f}$',transform=ax.transAxes,
                fontsize=FS_TEXT,va='top')
        if key=='lambda^n':
            ax.axhline(50,color=MUTED,ls=CHANCE_LS,lw=.8,zorder=1)
            ax.plot(guide,50*(1+guide),color=THEORY,lw=1,ls=THEORY_LS,zorder=2)
            ax.set(ylim=(0,103),yticks=[0,50,100],ylabel='Outcome accuracy (%)')
            ax.text(.04,.80,'State-only prior',transform=ax.transAxes,color=THEORY,fontsize=FS_TEXT,va='top')
            ax.text(.04,.455,'Chance',color=MUTED,fontsize=FS_SMALL,transform=ax.transAxes,va='top')
        else:
            ax.axhline(0,color=THEORY,ls=THEORY_LS,lw=1,zorder=1)
            ax.set(ylim=(-25,25),yticks=[-20,0,20],ylabel='Above prior (points)')
            ax.text(.04,.40,'State-only prior',transform=ax.transAxes,color=THEORY,fontsize=FS_TEXT,va='top')
    return correlation


def theory_measurement():
    paths=sorted((ROOT/'results/theorem_regime').glob('both_zero_seed[0-9].json'))
    records=[r for p in paths for r in json.loads(p.read_text()) if r['sigma']==.002]
    assert all(r.get('zero_readout') for r in records)
    records=list({(r['n'],r['mode'],r['seed']):r for r in records}.values())
    fig,axes=plt.subplots(1,3,figsize=(TEXT_WIDTH,2.1))
    fig.subplots_adjust(left=.09,right=.985,bottom=.2,top=.87,wspace=.5)
    ax=axes[0];panel(ax,'a','Plateau duration')
    for mode,col in [('step',STEP),('outcome',OUTCOME)]:
        grouped={n:[r['T'] for r in records if r['n']==n and r['mode']==mode and 'T' in r]
                 for n in sorted({r['n'] for r in records})}
        ns=np.asarray([n for n,v in grouped.items() if v])
        med=band(ax,ns,np.asarray([grouped[n] for n in ns]).T,col,marker=MARKER[mode],zorder=3)
        if mode=='outcome':
            guide=med[0]*2.**(ns-ns[0]);ax.plot(ns,guide,color=THEORY,lw=1,ls=THEORY_LS,zorder=4)
            ax.text(ns[-1],guide[-1]/8,r'$\lambda^{-n}$',color=THEORY,fontsize=FS_TEXT,ha='right',va='top')
            ax.text(.04,.72,'Outcome',color=col,fontsize=FS_TEXT,transform=ax.transAxes)
        else:ax.text(ns[len(ns)//2],med[-1]*3,'Step',color=col,fontsize=FS_TEXT,ha='center',va='bottom')
    ax.set(yscale='log',xlabel=r'Horizon $n$',ylabel=r'Exit time $\tau(0.1)$',xticks=[4,8,14]);clean(ax)
    cells=grid_cells();correlation=prediction_panels(axes[1:],cells)
    save(fig,'main_theory_measurement')
    # Standalone paired comparison is also available for talks and reuse.
    fig,axes=plt.subplots(1,2,figsize=(TEXT_WIDTH,2.1))
    fig.subplots_adjust(left=.09,right=.985,bottom=.2,top=.87,wspace=.3)
    prediction_panels(axes,cells,letters=('a','b'));save(fig,'lambda_grid')
    return dict(exit_sources=[str(p.relative_to(ROOT)) for p in paths],exit_sigma=.002,
        exit_records=records,grid_cells=cells,spearman_cell_means=correlation,
        exit_band='median and interquartile range over three draws',
        grid_summary='cell means and full seed ranges; 20 cells, three seeds each',
        grid_reference='state-only prior: 50*(1+lambda**n), not a fitted training law')


def transfer():
    def csv_rows(path):return list(csv.DictReader(path.open()))
    boolean=ROOT/'results/paper/e1_five_condition_rerun/table1_rerun.csv'
    register=ROOT/'results/paper/other_substrates/register_machine_16.csv'
    bs=csv_rows(boolean);rs=csv_rows(register);sources=[str(p.relative_to(ROOT)) for p in [boolean,register]]
    groups=[]
    for label,rows,key,modekey,out,step,chance in [
        ('Boolean-8',bs,'accuracy','mode','outcome','process',100/16),
        ('Register',rs,'answer_accuracy','condition','outcome','rho=1.00',100/17)]:
        groups.append(dict(label=label,chance=chance,
            outcome=[100*float(r[key]) for r in rows if r[modekey]==out],
            process=[100*float(r[key]) for r in rows if r[modekey]==step]))
    qwen=dict(label='Boolean-8',chance=100/16)
    for mode,stem in [('outcome','outcome'),('process','r1.0')]:
        qwen[mode]=[]
        for seed in [2001,2002,2003]:
            path=ROOT/f'results/paper/qwen05/main/{stem}_s{seed}.csv'
            row=csv_rows(path)[-1];qwen[mode].append(100*float(row['answer_accuracy']));sources.append(str(path.relative_to(ROOT)))
    bbh=[]
    for K in [3,5,7]:
        group=dict(label=f'{K} objects',chance=100/K)
        for mode in ['outcome','process']:
            group[mode]=[]
            for seed,suffix in [(2001,''),(2002,'_s2002'),(2003,'_s2003')]:
                path=ROOT/f'results/shuffled/bbh/bbh_K{K}_{mode}{suffix}.json'
                rec=json.loads(path.read_text());assert rec['bbh']['size']==250
                group[mode].append(100*rec['bbh']['answer_accuracy']);sources.append(str(path.relative_to(ROOT)))
        bbh.append(group)
    fig,axes=plt.subplots(1,3,figsize=(TEXT_WIDTH,2.15),gridspec_kw={'width_ratios':[1, .72,1.25]})
    fig.subplots_adjust(left=.085,right=.99,top=.80,bottom=.12,wspace=.32)
    for ax,local,title,letter in zip(axes,[groups,[qwen],bbh],['GPT-2L','Qwen2.5 LoRA','Qwen2.5 on BBH'],'abc'):
        panel(ax,letter,title)
        for i,group in enumerate(local):
            ax.plot([i-.32,i+.32],[group['chance']]*2,color=MUTED,lw=.8,ls=CHANCE_LS,zorder=1)
            for mode,col,key,offset in [('outcome',OUTCOME,'outcome',-.11),('process',STEP,'step',.11)]:
                values=np.asarray(group[mode]);med=np.median(values);lo,hi=np.percentile(values,[25,75])
                ax.vlines(i+offset,values.min(),values.max(),color=col,lw=.7,alpha=.55,zorder=2)
                ax.errorbar(i+offset,med,yerr=[[med-lo],[hi-med]],fmt=MARKER[key],color=col,
                            ms=4,elinewidth=2.2,capsize=0,zorder=3,label=mode if i==0 else None)
        ax.set(ylim=(0,106),yticks=[0,50,100],xlim=(-.5,len(local)-.5),
               xticks=range(len(local)),xticklabels=[g['label'] for g in local]);clean(ax)
        ax.tick_params(axis='x',length=0)
    axes[0].set_ylabel('Answer accuracy (%)')
    for ax in axes[1:]:ax.tick_params(labelleft=False)
    axes[2].text(1.2,52,'seed\nvariation',color=MUTED,fontsize=FS_SMALL,va='center',ha='left',linespacing=1)
    handles,labels=axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    fig.legend(handles+[Line2D([0],[0],color=MUTED,lw=.8,ls=CHANCE_LS)],
               ['Outcome','Step','Chance'],loc='upper center',bbox_to_anchor=(.53,1.0),
               ncol=3,frameon=False,handletextpad=.4,columnspacing=1.8,borderaxespad=.2)
    save(fig,'main_transfer')
    return dict(sources=sources,GPT2L=groups,Qwen_boolean=qwen,BBH=bbh,
        summary='median, thick interquartile interval, thin full seed range; not confidence intervals',
        metric='answer accuracy; official 250-item BBH test per object count')


def appendix_figures():
    """Keep the detailed views readable at the same 5.5-inch print width."""
    rec=json.loads((ROOT/'results/paper/fig2/n8_s42.json').read_text())
    target=np.asarray(rec['A_r'])
    text=(ROOT/'results/paper/exact_spectra/sigma_sweep/s42_sg0.02.txt').read_text()
    line=next(l for l in text.splitlines() if l.startswith('W_r / W'))
    shares=np.asarray([float(v) for v in line.split(':')[1].split('(')[0].split()])
    jacobian=shares*float(line.split('W = ')[1].rstrip(') '))
    degree=np.arange(len(target))
    fig,axes=plt.subplots(1,3,figsize=(TEXT_WIDTH,2.0))
    fig.subplots_adjust(left=.085,right=.985,top=.87,bottom=.2,wspace=.5)
    for ax,values,color,marker,letter,title,ylabel,log in zip(axes,
        [target,jacobian,np.sqrt(target*jacobian)], [OUTCOME,MODEL,INK],
        [MARKER['outcome'],MARKER['model'],MARKER['other']],'abc',
        ['Target spectrum','Jacobian spectrum','Credit bound'],
        [r'Target energy $A_r$',r'Jacobian energy $J_r$',r'Credit bound $\sqrt{A_rJ_r}$'],[False,True,True]):
        ax.plot(degree,np.where(values>0,values,np.nan),marker+'-',color=color)
        ax.set(xlabel=r'Degree $r$',ylabel=ylabel,xticks=[0,4,8],xlim=(-.4,8.4))
        panel(ax,letter,title)
        if log:ax.set_yscale('log')
        clean(ax)
    save(fig,'overlap_spectra')

    fig,axes=plt.subplots(1,2,figsize=(TEXT_WIDTH,2.1),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.02,right=.985,top=.87,bottom=.2,wspace=.3)
    ax=axes[0];ax.set(xlim=(-.3,6.3),ylim=(-1.3,1.3));ax.axis('off')
    panel(ax,'a','One transition or the whole word')
    for t in range(7):
        ax.text(t,0,rf'$s_{t}$',ha='center',va='center',fontsize=FS_TEXT+.5,
            bbox=dict(boxstyle='circle,pad=.15',facecolor='white',edgecolor=MUTED,lw=.7))
        if t<6:ax.annotate('',xy=(t+.76,0),xytext=(t+.25,0),arrowprops=dict(arrowstyle='->',lw=.7,color=MUTED))
    ax.plot([3,3,4,4],[.28,.53,.53,.28],color=STEP)
    ax.text(3.5,.76,'Step: one transition',ha='center',color=STEP,fontsize=FS_TEXT)
    ax.plot([0,0,6,6],[-.28,-.55,-.55,-.28],color=OUTCOME)
    ax.text(3,-.87,r'Outcome: $n$ transitions',ha='center',color=OUTCOME,fontsize=FS_TEXT)
    ax.text(3,-1.25,r'Degree-$r$ credit shrinks by $\lambda^{n-r}$',ha='center',fontsize=FS_TEXT)
    ax=axes[1];ns=np.asarray([2,4,6,8,9,10])
    for mode,key,color in [('step','grad_process',STEP),('outcome','grad_outcome',OUTCOME)]:
        curves=[]
        for seed in [42,43,44]:
            values=[json.loads((ROOT/f'results/paper/fig2/n{n}_s{seed}.json').read_text())[key] for n in ns]
            curves.append(values);ax.semilogy(ns,values,MARKER[mode],color=color,ms=3,alpha=.45)
        ax.plot(ns,curves[0],color=color)
        if mode=='outcome':ax.plot(ns,curves[0][0]*.5**(ns-ns[0]),color=THEORY,lw=1,ls=THEORY_LS)
    panel(ax,'b','Measured gradient credit')
    ax.text(.5,.80,'Step',transform=ax.transAxes,color=STEP,fontsize=FS_TEXT,ha='center',va='top')
    ax.text(6.3,1.5e-4,'Outcome',color=OUTCOME,fontsize=FS_TEXT,va='bottom')
    ax.set(xlabel=r'Horizon $n$',ylabel='Gradient norm',xticks=[2,6,10]);clean(ax)
    save(fig,'credit_locality')

    fig,axes=plt.subplots(1,2,figsize=(TEXT_WIDTH,1.85))
    fig.subplots_adjust(left=.01,right=.99,top=.80,bottom=.02,wspace=.08)
    fig.text(.5,.965,'Shared block 1: copy the query and compute each match bit',ha='center',va='top',
             fontsize=FS_TEXT,color=MUTED)
    descriptions=[['Read displayed state and current bit',r'Lookup $(s_{t-1},b_t)$: $2K$ units',r'Write $(s_{t-1}+b_t)\;\mathrm{mod}\;K$'],
        [r'Average all $n$ match bits',r'Rescale by $n$; add start state $s_0$',r'Count lookup: $3K(n+1)$ units']]
    for ax,lines,color,letter,title in zip(axes,descriptions,[STEP,OUTCOME],'ab',['Step executor (block 2)','Outcome executor (block 2)']):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');panel(ax,letter,title)
        for y,label in zip([.84,.5,.16],lines):
            ax.add_patch(FancyBboxPatch((.015,y-.1),.97,.2,boxstyle='round,pad=.012',
                         facecolor='white',edgecolor=color,lw=.9))
            ax.text(.5,y,label,ha='center',va='center',fontsize=FS_TEXT)
        for y in [.84,.5]:ax.annotate('',xy=(.5,y-.34+.115),xytext=(.5,y-.115),
            arrowprops=dict(arrowstyle='->',color=color,lw=.8,shrinkA=0,shrinkB=0))
    save(fig,'executors')

    paths=sorted((ROOT/'results/theorem_regime').glob('both_zero_seed[0-9].json'))+[ROOT/'results/theorem_regime/both_zero_s02.json']
    rows=[r for path in paths for r in json.loads(path.read_text())]
    rows=list({(r['sigma'],r['n'],r['mode'],r.get('seed',0)):r for r in rows}.values())
    fig,axes=plt.subplots(1,2,figsize=(TEXT_WIDTH,2.3))
    fig.subplots_adjust(left=.1,right=.985,top=.89,bottom=.3,wspace=.35)
    for sigma in [.002,.02]:
        for mode,color in [('step',STEP),('outcome',OUTCOME)]:
            selected=[r for r in rows if r['sigma']==sigma and r['mode']==mode and 'T' in r]
            ns=sorted({r['n'] for r in selected})
            stats=[]
            for n in ns:
                group=[r for r in selected if r['n']==n]
                stats.append([np.percentile([r['T'] for r in group],[25,50,75]),
                              np.percentile([max(r['drop_over_sigma2'],1e-12)/r['T'] for r in group],[25,50,75])])
            for index,ax in enumerate(axes):
                lo,med,hi=np.asarray([r[index] for r in stats]).T
                ax.fill_between(ns,lo,hi,color=color,alpha=.16,lw=0)
                ax.semilogy(ns,med,MARKER[mode]+('-' if sigma==.002 else '--'),color=color,
                            mfc=color if sigma==.002 else 'white',mec=color,mew=.8,
                            label=rf'{mode.capitalize()}, $\sigma={sigma}$')
    ref=sorted([r for r in rows if r['sigma']==.002 and r['mode']=='outcome' and r.get('seed',0)==0 and 'T' in r],key=lambda r:r['n'])
    axes[0].plot([r['n'] for r in ref],[ref[0]['T']*2**(r['n']-ref[0]['n']) for r in ref],color=THEORY,lw=1,ls=THEORY_LS)
    for ax,letter,title,ylabel in zip(axes,'ab',['Exit time','Loss decrease per unit time'],
        [r'Exit time $\tau(0.1)$',r'Loss decrease $/(\sigma^2\tau)$']):
        panel(ax,letter,title)
        ax.set(xlabel=r'Horizon $n$',ylabel=ylabel,xticks=[4,8,12,14]);clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,
               bbox_to_anchor=(.5,0),columnspacing=1.2,handletextpad=.4)
    save(fig,'regime_both')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',choices=['k2-zero','k3-three'],default='k2-zero')
    args=parser.parse_args();style();DATA.mkdir(parents=True,exist_ok=True)
    manifest=dict(overview=overview(args.result),theory_measurement=theory_measurement(),transfer=transfer())
    appendix_figures()
    (DATA/'figure_sources.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote unified vector figures and their source manifest.')
