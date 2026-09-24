"""Build the revised paper's three main figures and appendix tables from archives."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'results/review_validation'
FIG=ROOT/'Paper/figures'
TABLE=ROOT/'Paper/tables'


def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',pad_inches=.03,dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True);TABLE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    old=pd.read_csv(ROOT/'results/paper/boolean_reliability_canonical.csv')
    canonical=pd.read_csv(DATA/'canonical_corpora/metrics.csv')
    fig,ax=plt.subplots(figsize=(4.2,2.15))
    tab=canonical[canonical.seed==501].sort_values('rho')
    ax.plot(tab.rho,100*tab.answer_accuracy,'o-',ms=3,color='#be681e',label='Fitted table')
    g=old[old.rho.notna()].groupby('rho').answer_accuracy.agg(['mean','std'])
    ax.errorbar(g.index,100*g['mean'],100*g['std'],fmt='o-',ms=3,capsize=2,color='#1766a4',label='GPT-2L')
    ax.axhline(6.25,color='gray',ls=':',lw=.8)
    ax.set(xlabel='Clean-trace reliability ρ',ylabel='Answer accuracy (%)',ylim=(-2,105))
    ax.legend(loc='upper left');fig.tight_layout(pad=.3);save(fig,'canonical_same_corpus')
    concentration=pd.read_csv(DATA/'concentration_final.csv')
    fig,axes=plt.subplots(1,3,figsize=(5.55,1.95),sharey=True)
    for ax,rho in zip(axes,[.3,.5,.7]):
        subset=concentration[concentration.rho==rho]
        for metric,label,color,style in [('final_answer','Tx-sem','#1766a4','-'),('tabular_answer_accuracy','Fitted table','#be681e','--')]:
            g=subset.groupby('concentration')[metric].agg(['mean','std'])
            ax.errorbar(g.index,100*g['mean'],100*g['std'],fmt='o',linestyle=style,ms=2.5,capsize=2,lw=1,color=color,label=label)
        ax.axhline(6.25,color='gray',ls=':',lw=.7)
        ax.set_title(f'ρ = {rho:g}',fontsize=9,pad=3)
        ax.set_xticks([0,.5,1]);ax.set_xlabel('Concentration λ',labelpad=2)
        ax.set_ylim(-2,105)
    axes[0].set_ylabel('Answer accuracy (%)',labelpad=2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1.03),ncol=2,frameon=False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.23,top=.79,wspace=.13)
    save(fig,'concentration_validation')
    updates=pd.read_csv(DATA/'functional_final/updates.csv')
    sources=['clean_state','wrong_state','uniform_state_wrong_prefix','wrong_full']
    labels=['Clean\nstates','Wrong\nstates','Uniform\nstates','Wrong\ntrace']
    fig,axes=plt.subplots(1,2,figsize=(5.55,2.1))
    for ax,step,column,scale,title in zip(axes,[100,2000],['delta_conditional_state_loss','delta_loss'],[1000,100000],['(a) Early: step 100','(b) Converged: step 2000']):
        for j,(law,color) in enumerate([('symmetric','#1766a4'),('coherent','#be681e')]):
            subset=updates[(updates.step==step)&(updates.group=='all')&(updates.optimizer=='euclidean')&(updates.eta==.001)&(updates.corruption==law)]
            means=[];stds=[]
            for source in sources:
                values=subset[subset.source==source][column]*scale
                means.append(values.mean());stds.append(values.std())
            xs=np.arange(len(sources))+(j-.5)*.32
            ax.bar(xs,means,yerr=stds,width=.29,color=color,alpha=.75,capsize=2,label=law.capitalize())
            for i,source in enumerate(sources):
                values=subset[subset.source==source].sort_values('seed')[column].to_numpy()*scale
                ax.scatter(xs[i]+np.linspace(-.055,.055,len(values)),values,s=8,color=color,zorder=4,edgecolors='white',linewidths=.3)
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(range(len(sources)),labels)
        ax.set_title(title,fontsize=9,pad=4)
        exponent=3 if step==100 else 5
        ax.set_ylabel(f'Loss change (×10⁻{exponent})',labelpad=2)
    axes[0].legend(loc='lower left',fontsize=6,frameon=False)
    fig.tight_layout(pad=.35,w_pad=1.4);save(fig,'functional_validation')
    # All concentration seeds; repeated clean controls are already deduplicated.
    lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',r'$\rho$ & $\lambda$ & Seed 42 & Seed 43 & Seed 44 & Mean $\pm$ SD\\',r'\midrule']
    for (rho,lam),group in concentration.groupby(['rho','concentration']):
        values=group.sort_values('seed').final_answer.to_numpy()*100
        assert len(values)==3
        lines.append(f'{rho:.2f} & {lam:.2f} & '+ ' & '.join(f'{v:.2f}' for v in values)+f' & ${values.mean():.2f}\\pm{values.std(ddof=1):.2f}$'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLE/'concentration_seeds.tex').write_text('\n'.join(lines)+'\n')
    sources2=['wrong_state','wrong_full','uniform_state_wrong_prefix','gold_answer_wrong_prefix']
    names=['Wrong states','Full wrong trace','Uniform state targets','Gold answer, wrong prefix']
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Update source & Symmetric & Coherent\\',r'\midrule']
    for source,name in zip(sources2,names):
        entries=[]
        for law in ['symmetric','coherent']:
            z=updates[(updates.step==2000)&(updates.group=='all')&(updates.optimizer=='euclidean')&(updates.eta==.001)&(updates.corruption==law)&(updates.source==source)].delta_loss*1e5
            entries.append(f'${z.mean():.2f}\\pm{z.std():.2f}$')
        lines.append(name+' & '+' & '.join(entries)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLE/'functional_final.tex').write_text('\n'.join(lines)+'\n')
    appendix_figures = {'tabular_boundary.pdf': 'validated_tabular_boundary.pdf',
                        'functional_updates.pdf': 'validated_functional_trajectory.pdf'}
    for source, target in appendix_figures.items():
        shutil.copyfile(DATA/source, FIG/target)
    files=[DATA/'canonical_corpora/metrics.csv',ROOT/'results/paper/boolean_reliability_canonical.csv',DATA/'concentration_final.csv',DATA/'functional_final/updates.csv',Path(__file__)]+list(TABLE.glob('*.tex'))
    files += [FIG/f'{name}.pdf' for name in ['canonical_same_corpus','concentration_validation','functional_validation']]
    files += [DATA/name for name in appendix_figures] + [FIG/name for name in appendix_figures.values()]
    (DATA/'manuscript_artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print('Built three main figures, two appendix figures, and two tables from archived results.')


if __name__=='__main__':
    main()
