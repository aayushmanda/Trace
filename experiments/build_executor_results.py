"""Build manuscript figures only from the complete five-seed depth replication."""
import argparse
import hashlib
import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plot_style import apply_style

ROOT=Path(__file__).resolve().parents[1]
COLORS={'process':'#247ba0','outcome':'#dd8452','both':'#7252a2'}
LABELS={'process':'Process','outcome':'Outcome','both':'Mixed format'}


def curve(ax,frame,mode,field,label=None,color=None,style='-',band=True):
    f=frame[frame['mode']==mode].dropna(subset=[field])
    if f.empty:return
    g=f.groupby('step')[field].agg(['mean','std','count'])
    if not (g['count']==5).all():raise ValueError(f'Incomplete five-seed series: {mode} {field}')
    x=g.index.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
    color=color or COLORS[mode]
    ax.plot(x,y,style,color=color,label=label or LABELS[mode],linewidth=1.7)
    if band:ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12)


def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/executor_comparison/depth_replication')
    p.add_argument('--paper',type=Path,default=ROOT/'Paper')
    args=p.parse_args();protocol=json.loads((args.input/'protocol.json').read_text())
    frame=pd.read_csv(args.input/'metrics.csv')
    keys=['depth','seed','mode','step']
    if frame.duplicated(keys).any():raise ValueError('Duplicate measurements')
    expected={(d,s,m,t) for d in protocol['depths'] for s in protocol['seeds']
              for m in ['process','outcome','both'] for t in sorted(set(protocol['checkpoints']))}
    if set(map(tuple,frame[keys].to_numpy()))!=expected:raise ValueError('Missing or unexpected measurements')
    provenance=[]
    for depth in protocol['depths']:
        for seed in protocol['seeds']:
            directory=args.input/f'depth{depth}_seed{seed}'
            manifest=json.loads((directory/'manifest.json').read_text())
            if 'elapsed_seconds' not in manifest:raise ValueError(f'Incomplete {directory}')
            for name in ['manifest.json','metrics.csv']:
                path=directory/name
                provenance.append({'file':str(path.relative_to(ROOT)),
                                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    json_path=args.paper/'data/trained_executor_sources.json'
    json_path.write_text(json.dumps({'protocol':protocol,'sources':provenance},indent=2)+'\n')
    apply_style()
    d4=frame[frame.depth==4]
    fig,axs=plt.subplots(2,2,figsize=(9.4,6.1),layout='constrained')
    for mode in ['process','both']:
        curve(axs[0,0],d4,mode,'epsilon_rule')
        curve(axs[0,1],d4,mode,'composition_tv_with_invalid')
    axs[0,0].set(title=r'(A) Induced rule strength',ylabel=r'$\widehat{\epsilon}_{\rm rule}$')
    axs[0,0].text(.02,.96,'Outcome-only local query omitted',transform=axs[0,0].transAxes,va='top',fontsize=8,color='.35')
    axs[0,0].legend(loc='lower right',fontsize=8)
    axs[0,1].set(title='(B) Composition error',ylabel='Total variation',ylim=(-.02,1.02))
    axs[0,1].legend(fontsize=8)
    curve(axs[1,0],d4,'both','gradient_cosine','Composed-table gradient','#7252a2')
    curve(axs[1,0],d4,'both','descent_oracle_rule_cosine','True-rule direction','#247ba0','--')
    curve(axs[1,0],d4,'both','descent_random_rule_cosine_mean','Random-rule mean','#777777',':')
    axs[1,0].axhline(0,color='.6',lw=.6)
    axs[1,0].set(title='(C) Gradient agreement: mixed format',ylabel='Cosine',ylim=(-.35,1.02))
    axs[1,0].legend(fontsize=7)
    for mode in ['process','outcome','both']:curve(axs[1,1],d4,mode,'free_answer_accuracy')
    axs[1,1].set(title='(D) Free-running answer accuracy',ylabel='Accuracy',ylim=(-.02,1.02))
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.set_xlabel('Training update');ax.grid(alpha=.16)
    save(fig,args.paper/'figures/trained_executor_bridge')
    fig,axes=plt.subplots(3,4,figsize=(13,8),layout='constrained')
    for row,depth in enumerate(protocol['depths']):
        d=frame[frame.depth==depth]
        for mode in ['process','both']:
            curve(axes[row,0],d,mode,'epsilon_rule')
            curve(axes[row,1],d,mode,'composition_tv_with_invalid')
        curve(axes[row,2],d,'both','gradient_actual_norm','Actual answer gradient','#7252a2')
        curve(axes[row,2],d,'both','gradient_rule_pullback_norm','Rule pullback','#247ba0','--')
        for mode in COLORS:curve(axes[row,3],d,mode,'free_answer_accuracy')
        axes[row,0].set_ylabel(f'D = {depth}')
        axes[row,2].set_yscale('log')
        for ax in axes[row]:ax.set_xlabel('Training update');ax.grid(alpha=.16)
        axes[row,1].set_ylim(-.02,1.02);axes[row,3].set_ylim(-.02,1.02)
    for ax,title in zip(axes[0],['Induced rule strength','Composition TV','Gradient norms (mixed)','Answer accuracy']):
        ax.set_title(title);ax.legend(fontsize=7)
    save(fig,args.paper/'figures/trained_executor_depths')
    final=frame[frame.step==protocol['steps']]
    fields=['free_answer_accuracy','local_rule_accuracy','epsilon_rule','gamma','local_state_mass',
            'composition_tv_with_invalid','gradient_cosine','gradient_relative_error',
            'gradient_rule_pullback_norm','descent_oracle_rule_cosine','descent_random_rule_cosine_mean']
    summary=final.groupby(['depth','mode'])[fields].agg(['mean','std','count'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary=summary.reset_index();summary.to_csv(args.input/'summary_final.csv',index=False)
    tex=[]
    for _,r in summary.iterrows():
        def cell(field,percent=False):
            if r[field+'_count']==0:return '---'
            scale=100 if percent else 1
            return rf"${scale*r[field+'_mean']:.3f}\pm{scale*r[field+'_std']:.3f}$"
        tex.append(f"{int(r.depth)} & {LABELS[r['mode']]} & "+' & '.join([
            cell('free_answer_accuracy',True),cell('local_rule_accuracy',True),
            cell('composition_tv_with_invalid'),cell('gradient_cosine')])+r' \\')
    (args.paper/'data/trained_executor_rows.tex').write_text('\n'.join(tex)+'\n')
    summary.to_json(args.paper/'data/trained_executor_summary.json',orient='records',indent=2)
    macros=[]
    for name,mode,field,scale in [
        ('BridgeProcessAcc','process','free_answer_accuracy',100),
        ('BridgeProcessLocal','process','local_rule_accuracy',100),
        ('BridgeMixedLocal','both','local_rule_accuracy',100),
        ('BridgeMixedTV','both','composition_tv_with_invalid',1),
        ('BridgeMixedCos','both','gradient_cosine',1),
        ('BridgeMixedGradError','both','gradient_relative_error',1)]:
        row=summary[(summary.depth==4)&(summary['mode']==mode)].iloc[0]
        mean=scale*row[field+'_mean'];sd=scale*row[field+'_std']
        precision=1 if scale==100 else 3
        value=f'{mean:.{precision}f}'+r'\pm'+f'{sd:.{precision}f}'
        macros.append(r'\newcommand{\'+name+'}{'+value+'}')
    (args.paper/'data/trained_executor_numbers.tex').write_text('\n'.join(macros)+'\n')
    print(summary[['depth','mode','free_answer_accuracy_mean','local_rule_accuracy_mean',
                   'composition_tv_with_invalid_mean','gradient_cosine_mean']].to_string(index=False))
    print('Wrote verified five-seed manuscript figures and summary')


if __name__=='__main__':main()
