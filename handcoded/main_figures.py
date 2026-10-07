"""Main-paper figures, with audited sources and print-size vector output.

python handcoded/main_figures.py --result k2-zero
python handcoded/main_figures.py --result k3-three
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'Paper/figures'
DATA = ROOT/'results/paper/main_figure_revision'
STEP, OUTCOME, INK, MUTED = '#0a7fad', '#c8680a', '#202020', '#666666'


def style():
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'font.family':'DejaVu Sans','mathtext.fontset':'stixsans',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'lines.linewidth':1.5})


def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=220)
    plt.close(fig)


def setup(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.set_title('(a) One task, two formats',loc='left',fontsize=8.5,pad=8)
    def box(y,height,color):
        ax.add_patch(FancyBboxPatch((.01,y),.95,height,boxstyle='round,pad=0.012',
            fc=color,ec='none',transform=ax.transAxes))
    box(.70,.24,'#f0f0f0')
    ax.text(.055,.88,'Prompt: count a, mod 3',fontsize=8.5,weight='bold')
    ax.text(.055,.79,'start 1  |  banana',fontfamily='DejaVu Sans Mono',fontsize=8.3)
    box(.40,.25,'#eef6fa')
    ax.text(.055,.58,'Step continuation',color=STEP,weight='bold')
    ax.text(.055,.49,'1 2 2 0 0 1',fontfamily='DejaVu Sans Mono',fontsize=8.5)
    ax.text(.055,.42,': 1 EOS',fontfamily='DejaVu Sans Mono',fontsize=8.5)
    box(.20,.15,'#fff3e8')
    ax.text(.055,.295,'Outcome continuation',color=OUTCOME,weight='bold')
    ax.text(.055,.225,': 1 EOS',fontfamily='DejaVu Sans Mono',fontsize=8.5)
    ax.text(.055,.11,'One two-block architecture.',fontsize=8.3)
    ax.text(.055,.045,'An exact executor for each.',fontsize=8.3,weight='bold')


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
    fig=plt.figure(figsize=(5.5,2.45))
    grid=fig.add_gridspec(1,3,width_ratios=(1.40,1,1),left=.015,right=.98,
                          top=.87,bottom=.22,wspace=.52)
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    setup(axes[0]);ax=axes[1]
    ax.set_title('(b) Step training learns',loc='left',fontsize=8.5,pad=8)
    for mode,col in [('process',STEP),('outcome',OUTCOME)]:
        values=curves[mode]
        ax.fill_between(steps/1000,values.min(0),values.max(0),color=col,alpha=.14,lw=0)
        ax.plot(steps/1000,np.median(values,axis=0),color=col)
    ax.axhline(100,color=INK,ls='--',lw=1)
    ax.axhline(100/K,color=MUTED,ls=':',lw=1)
    ax.text(.04,.90,'Exact executors',transform=ax.transAxes,fontsize=8,va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.9))
    ax.text(.05,.78,'Step',color=STEP,transform=ax.transAxes,weight='bold')
    ax.text(.40,.34,'Outcome',color=OUTCOME,transform=ax.transAxes,weight='bold')
    ax.text(.04,.07,f'n=16, K={K}\n{len(seeds)} paired seeds',transform=ax.transAxes,fontsize=8)
    ax.set(xlabel='Updates (thousands)',ylabel='Exact accuracy (%)',ylim=(-2,108),
           xlim=(0,steps[-1]/1000),yticks=[0,50,100])
    ax.set_xticks([0,4,8] if which=='k2-zero' else [0,16,32])
    ax.grid(axis='y',alpha=.15)
    ax=axes[2]
    degree=np.arange(len(target))
    ax.set_title('(c) Degree mismatch',loc='left',fontsize=8.5,pad=8)
    ax.semilogy(degree,target,'o-',color=OUTCOME,ms=2.5)
    ax.semilogy(degree,jacobian,'s-',color=STEP,ms=2.5)
    ax.text(3.4,.07,r'Target $A_r$',color=OUTCOME,fontsize=8)
    ax.text(1.5,1.1e-5,r'Jacobian $J_r$',color=STEP,fontsize=8)
    ax.set(xlabel=r'Interaction degree $r$',ylabel='Normalized energy',
           xlim=(-.25,8.3),ylim=(1e-7,2),xticks=[0,4,8],yticks=[1e-6,1e-3,1])
    ax.grid(axis='y',alpha=.15)
    save(fig,'main_overview')
    return dict(result=which,K=K,n=16,seeds=seeds,steps=steps.tolist(),
        band='minimum to maximum; line is pointwise median',history_sources=sources,
        final_exact={m:v[:,-1].tolist() for m,v in curves.items()},
        spectrum_source=str(spectrum_path.relative_to(ROOT)),
        spectrum_case='K=2, n=8, sigma=0.02; nonzero readout; normalized energies')


def theory_measurement():
    paths=sorted((ROOT/'results/theorem_regime').glob('both_zero_seed[0-9].json'))
    records=[r for p in paths for r in json.loads(p.read_text()) if r['sigma']==.002]
    assert all(r.get('zero_readout') for r in records)
    records=list({(r['n'],r['mode'],r['seed']):r for r in records}.values())
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.45))
    fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.85,wspace=.46)
    ax=axes[0]
    ax.set_title('(a) Outcome escapes slowly',loc='left',fontsize=9)
    for mode,col in [('step',STEP),('outcome',OUTCOME)]:
        grouped={n:[r['T'] for r in records if r['n']==n and r['mode']==mode and 'T' in r]
                 for n in sorted({r['n'] for r in records})}
        ns=np.asarray([n for n,v in grouped.items() if v])
        med=np.asarray([np.median(grouped[n]) for n in ns])
        for n in ns:ax.scatter([n]*len(grouped[n]),grouped[n],color=col,s=9,alpha=.4)
        ax.semilogy(ns,med,'o-',color=col,ms=3)
        if mode=='outcome':
            guide=med[0]*2.**(ns-ns[0])
            ax.plot(ns,guide,'--',color=MUTED,lw=1)
            ax.text(10.8,guide[-1]/7,r'$\lambda^{-n}$',color=MUTED,fontsize=8.5)
            ax.text(5.8,med[1]*2.3,'Outcome',color=col,weight='bold')
        else:
            ax.text(8,med[-1]*1.8,'Step',color=col,weight='bold')
    ax.set(xlabel=r'Horizon $n$',ylabel=r'Accumulated exit time $\tau(0.1)$',xticks=[4,8,12,14])
    ax.grid(axis='y',alpha=.15)
    ax=axes[1]
    ax.set_title('(b) Bias predicts accuracy',loc='left',fontsize=8.5)
    cells=[]
    for path in sorted((ROOT/'results/paper/lambda_grid').glob('n*_A*_s*.json')):
        rec=json.loads(path.read_text());assert rec['steps']==8000
        cells.append(dict(n=rec['n'],alphabet=rec['alphabet'],seed=rec['seed'],
            contraction=rec['lam'],bias=rec['lam']**rec['n'],
            process=100*rec['process']['final_exact'],outcome=100*rec['outcome']['final_exact'],
            source=str(path.relative_to(ROOT))))
    assert len(cells)==60
    for mode,col,marker in [('process',STEP,'s'),('outcome',OUTCOME,'o')]:
        ax.scatter([r['bias'] for r in cells],[r[mode] for r in cells],
                   color=col,marker=marker,s=11,alpha=.65,lw=0)
    x=np.geomspace(1e-5,.65,300)
    ax.plot(x,50*(1+x),'--',color=MUTED,lw=1)
    ax.axhline(50,color=MUTED,ls=':',lw=1)
    ax.text(8e-5,94,'Step',color=STEP,weight='bold')
    ax.text(2e-4,56,'Outcome',color=OUTCOME,weight='bold')
    ax.text(.36,.22,'Prior baseline',transform=ax.transAxes,color=MUTED,fontsize=8)
    ax.annotate('',xy=(.18,59),xytext=(.07,43),arrowprops=dict(arrowstyle='-',lw=.6,color=MUTED))
    ax.set(xscale='log',xlabel=r'Remaining answer bias $\lambda^n$',
           ylabel='Exact accuracy (%)',ylim=(-2,103),xlim=(1e-5,.65),
           xticks=[1e-5,1e-3,1e-1],yticks=[0,50,100])
    ax.grid(axis='y',alpha=.15)
    save(fig,'main_theory_measurement')
    return dict(exit_sources=[str(p.relative_to(ROOT)) for p in paths],
        exit_sigma=.002,exit_records=records,grid_cells=cells,
        grid_reference='state-only prior: 50*(1+lambda**n), not a fitted training law')


def appendix_figures():
    """Keep the detailed views readable at the same 5.5-inch print width."""
    rec=json.loads((ROOT/'results/paper/fig2/n8_s42.json').read_text())
    target=np.asarray(rec['A_r'])
    text=(ROOT/'results/paper/exact_spectra/sigma_sweep/s42_sg0.02.txt').read_text()
    line=next(l for l in text.splitlines() if l.startswith('W_r / W'))
    shares=np.asarray([float(v) for v in line.split(':')[1].split('(')[0].split()])
    jacobian=shares*float(line.split('W = ')[1].rstrip(') '))
    degree=np.arange(len(target))
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.35))
    fig.subplots_adjust(left=.10,right=.98,top=.84,bottom=.23,wspace=.55)
    for ax,values,color,title,ylabel,log in zip(axes,
        [target,jacobian,np.sqrt(target*jacobian)], [OUTCOME,STEP,INK],
        ['(a) Target spectrum','(b) Model spectrum','(c) Credit bound'],
        [r'$A_r$',r'$J_r$',r'$\sqrt{A_rJ_r}$'],[False,True,True]):
        ax.bar(degree,values,color=color,width=.75)
        ax.set(xlabel=r'Degree $r$',ylabel=ylabel,xticks=[0,4,8])
        ax.set_title(title,loc='left',fontsize=8.5)
        if log:ax.set_yscale('log')
        ax.grid(axis='y',alpha=.15)
    save(fig,'overlap_spectra')

    fig,axes=plt.subplots(1,2,figsize=(5.5,2.35),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.03,right=.98,top=.84,bottom=.23,wspace=.36)
    ax=axes[0];ax.set(xlim=(-.3,6.3),ylim=(-1.3,1.3));ax.axis('off')
    ax.set_title('(a) One transition or the whole word',loc='left',fontsize=8.5)
    for t in range(7):
        ax.text(t,0,rf'$s_{t}$',ha='center',va='center',fontsize=8.5,
            bbox=dict(boxstyle='circle,pad=.15',facecolor='white',edgecolor=MUTED,lw=.7))
        if t<6:ax.annotate('',xy=(t+.76,0),xytext=(t+.25,0),arrowprops=dict(arrowstyle='->',lw=.7,color=MUTED))
    ax.plot([3,3,4,4],[.28,.53,.53,.28],color=STEP)
    ax.text(3.1,.76,'Step: one transition',ha='center',color=STEP,fontsize=8.5)
    ax.plot([0,0,6,6],[-.28,-.55,-.55,-.28],color=OUTCOME)
    ax.text(3,-.87,'Outcome: n transitions',ha='center',color=OUTCOME,fontsize=8.5)
    ax.text(3,-1.25,r'Degree-$r$ credit shrinks by $\lambda^{n-r}$',ha='center',fontsize=8)
    ax=axes[1];ns=np.asarray([2,4,6,8,9,10])
    for mode,key,color in [('step','grad_process',STEP),('outcome','grad_outcome',OUTCOME)]:
        curves=[]
        for seed in [42,43,44]:
            values=[json.loads((ROOT/f'results/paper/fig2/n{n}_s{seed}.json').read_text())[key] for n in ns]
            curves.append(values);ax.semilogy(ns,values,'o',color=color,ms=3,alpha=.4)
        ax.plot(ns,curves[0],color=color)
        if mode=='outcome':ax.plot(ns,curves[0][0]*.5**(ns-ns[0]),'--',color=MUTED,lw=1)
    ax.set_title('(b) Measured gradient credit',loc='left',fontsize=8.5)
    ax.text(.5,.92,'Step',transform=ax.transAxes,color=STEP,fontsize=8.5)
    ax.text(6,8e-5,'Outcome',color=OUTCOME,fontsize=8.5)
    ax.set(xlabel=r'Horizon $n$',ylabel='Gradient norm',xticks=[2,6,10])
    ax.grid(axis='y',alpha=.15)
    save(fig,'credit_locality')

    fig,axes=plt.subplots(1,2,figsize=(5.5,2.45))
    fig.subplots_adjust(left=.03,right=.98,top=.80,bottom=.04,wspace=.15)
    fig.text(.5,.94,'Shared block 1: copy the query and compute each match bit',ha='center',fontsize=9)
    descriptions=[['Read displayed state and current bit',r'Lookup $(s_{t-1},b_t)$: $2K$ units',r'Write $(s_{t-1}+b_t)\;\mathrm{mod}\ K$'],
        ['Average all n match bits',r'Rescale by $n$; add start state $s_0$',r'Count lookup: $3K(n+1)$ units']]
    for ax,lines,color,title in zip(axes,descriptions,[STEP,OUTCOME],['(a) Step executor','(b) Outcome executor']):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,loc='left',fontsize=9)
        for y,label in zip([.77,.47,.17],lines):
            ax.add_patch(FancyBboxPatch((.015,y-.1),.96,.2,boxstyle='round,pad=.015',
                         facecolor='white',edgecolor=color,lw=1))
            ax.text(.495,y,label,ha='center',va='center',fontsize=8)
        for y in [.65,.35]:ax.annotate('',xy=(.495,y-.075),xytext=(.495,y),arrowprops=dict(arrowstyle='->',color=color))
    save(fig,'executors')

    paths=sorted((ROOT/'results/theorem_regime').glob('both_zero_seed[0-9].json'))+[ROOT/'results/theorem_regime/both_zero_s02.json']
    rows=[r for path in paths for r in json.loads(path.read_text())]
    rows=list({(r['sigma'],r['n'],r['mode'],r.get('seed',0)):r for r in rows}.values())
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.6))
    fig.subplots_adjust(left=.11,right=.98,top=.87,bottom=.32,wspace=.42)
    for sigma,marker in [(.002,'o'),(.02,'s')]:
        for mode,color in [('step',STEP),('outcome',OUTCOME)]:
            for seed in sorted({r.get('seed',0) for r in rows}):
                selected=sorted([r for r in rows if r['sigma']==sigma and r['mode']==mode and r.get('seed',0)==seed and 'T' in r],key=lambda r:r['n'])
                if not selected:continue
                ns=[r['n'] for r in selected]
                label=rf'{mode}, $\sigma={sigma}$' if seed==0 else None
                axes[0].semilogy(ns,[r['T'] for r in selected],marker+'-',color=color,ms=3,alpha=1 if seed==0 else .35,label=label)
                axes[1].semilogy(ns,[max(r['drop_over_sigma2'],1e-12)/r['T'] for r in selected],marker+'-',color=color,ms=3,alpha=1 if seed==0 else .35)
    ref=sorted([r for r in rows if r['sigma']==.002 and r['mode']=='outcome' and r.get('seed',0)==0 and 'T' in r],key=lambda r:r['n'])
    axes[0].plot([r['n'] for r in ref],[ref[0]['T']*2**(r['n']-ref[0]['n']) for r in ref],'--',color=MUTED,lw=1)
    for ax,title,ylabel in zip(axes,['(a) Exit time','(b) Loss decrease per unit time'],
        [r'$\tau(0.1)$',r'Loss decrease $/(\sigma^2\tau)$']):
        ax.set_title(title,loc='left',fontsize=8.5)
        ax.set(xlabel=r'Horizon $n$',ylabel=ylabel,xticks=[4,8,12,14]);ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8)
    save(fig,'regime_both')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',choices=['k2-zero','k3-three'],default='k2-zero')
    args=parser.parse_args();style();DATA.mkdir(parents=True,exist_ok=True)
    manifest=dict(overview=overview(args.result),theory_measurement=theory_measurement())
    appendix_figures()
    (DATA/'figure_sources.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote two vector PDF figures and their source manifest.')
