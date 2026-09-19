"""Compact main-paper figures from canonical saved rows; no training."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.experiments.plot_paper_figures import style, save, PROCESS, OUTCOME, BASELINE

ROOT = Path(__file__).resolve().parents[2]


def plot_reliability_structure():
    style()
    data = pd.read_csv(ROOT / 'results/paper/boolean_reliability_canonical.csv')
    stats = data[data.condition != 'outcome'].groupby('rho').answer_accuracy.agg(['mean', 'std'])
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.8), layout='constrained')
    ax = axes[0]
    ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                    color=OUTCOME, alpha=.20, linewidth=0)
    ax.plot(stats.index, stats['mean'], 'o-', color=OUTCOME, ms=4, label='Process targets')
    ax.axhline(data[data.condition == 'outcome'].answer_accuracy.mean(), color=BASELINE,
               ls='-.', lw=1.4, label='Outcome-only')
    ax.axhline(1/16, color=BASELINE, ls=':', lw=1.2, label='Chance')
    ax.set(title='(a) Boolean circuit, depth 8', xlabel=r'Trace reliability $\rho$',
           ylabel='Answer accuracy', xlim=(-.02, 1.02), ylim=(-.03, 1.02))
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1]
    for law, color, marker, label in [('symmetric', PROCESS, 'o', 'Symmetric'),
                                      ('coherent', '#b42318', 's', 'Coherent wrong rule')]:
        data = pd.read_csv(ROOT / f'results/paper/matched_corruption/register_machine_16_{law}.csv')
        data = data[data.step == 8000]
        stats = data.groupby('rho').answer_accuracy.agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], color=color,
                    marker=marker, ms=4, capsize=3, label=label)
    ax.axhline(1/17, color=BASELINE, ls=':', lw=1.2, label='Chance')
    ax.set(title='(b) Register machine, 16 steps', xlabel=r'Trace reliability $\rho$',
           ylabel='Answer accuracy', xlim=(.25, .75), ylim=(-.03, 1.02), xticks=[.3, .5, .7])
    ax.legend(loc='lower right', fontsize=8)
    save(fig, 'reliability_structure')


def plot_gradient_alignment():
    style()
    data = pd.read_csv(ROOT / 'results/gradient_alignment_archived_3seed/gradient_alignment.csv')
    fig, ax = plt.subplots(figsize=(5.0, 2.35), layout='constrained')
    for column, color, label in [('proc_cosine', PROCESS, 'Process'),
                                 ('out_cosine', OUTCOME, 'Outcome (counterfactual)')]:
        stats = data.groupby('step')[column].agg(['mean', 'std'])
        ax.errorbar(stats.index/1000, stats['mean'], yerr=stats['std'],
                    color=color, marker='o', ms=4, capsize=3, label=label)
    ax.axhline(0, color=BASELINE, ls=':', lw=1)
    ax.set(xlabel='Training steps (thousands)', ylabel='Cosine with local gradient',
           xticks=[0, 2, 4, 8], ylim=(-.22, 1.12))
    ax.legend(loc='center right', fontsize=8)
    save(fig, 'gradient_alignment_compact')


def plot_depth_control():
    """Aggregate the archived matched outcome/clean-process runs at each depth."""
    style()
    frames = []
    for depth in (2, 4, 6, 8):
        data = pd.read_csv(
            ROOT / f'results/paper/depth_reliability/boolean_circuit_{depth}_outcome.csv'
        )
        data = data[(data.step == 8000) & data.condition.isin(['outcome', 'rho=1.00'])].copy()
        for condition in ('outcome', 'rho=1.00'):
            seeds = sorted(data.loc[data.condition == condition, 'seed'].tolist())
            if seeds != [2001, 2002, 2003]:
                raise ValueError(f'Depth {depth}, {condition}: expected one row per archived seed')
        data['depth'] = depth
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.8, 2.55), layout='constrained')
    for condition, color, marker, label in [
        ('rho=1.00', PROCESS, 'o', r'Process ($\rho=1$)'),
        ('outcome', OUTCOME, 's', 'Outcome'),
    ]:
        stats = data[data.condition == condition].groupby('depth').answer_accuracy.agg(['mean', 'std'])
        ax.errorbar(stats.index, 100 * stats['mean'], yerr=100 * stats['std'],
                    color=color, marker=marker, ms=5, capsize=3, label=label)
    ax.axhline(100 / 16, color=BASELINE, ls=':', lw=1.2, label='Chance (6.25%)')
    ax.set(xlabel=r'Composition depth $D$', ylabel='Answer accuracy (%)',
           xticks=[2, 4, 6, 8], yticks=[0, 25, 50, 75, 100],
           xlim=(1.7, 8.3), ylim=(0, 105))
    ax.legend(loc='center right', fontsize=9)
    save(fig, 'depth_control')


def plot_credit_geometry():
    from matplotlib.patches import FancyBboxPatch
    style()
    fig, ax = plt.subplots(figsize=(8.0, 2.45), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 3.4))
    ax.axis('off')
    def box(x, y, w, title, color):
        ax.add_patch(FancyBboxPatch((x, y), w, .62, boxstyle='round,pad=0.05',
                                   edgecolor=color, facecolor='white', linewidth=1.5))
        ax.text(x+w/2, y+.31, title, ha='center', va='center', fontsize=10, color=color)
    def arrow(x, y, end_x, end_y, color):
        ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                    arrowprops={'arrowstyle':'->', 'color':color, 'lw':1.5})
    ax.text(.1, 3.1, 'OUTCOME: credit passes through the surrounding computation',
            color=OUTCOME, fontsize=11, weight='bold')
    box(.2, 2.1, 2.5, 'Forward state signal', OUTCOME)
    box(3.6, 2.1, 2.8, 'Local transition', BASELINE)
    box(7.3, 2.1, 2.5, 'Backward answer signal', OUTCOME)
    arrow(2.8, 2.41, 3.5, 2.41, OUTCOME)
    arrow(7.2, 2.41, 6.5, 2.41, OUTCOME)
    ax.text(5, 1.65, r'Rule credit $=\bar{q}_{t-1}\bar{b}_t^{\mathsf{T}}/p_y$',
            ha='center', fontsize=12, color=OUTCOME)
    ax.text(.1, 1.1, 'PROCESS: the target exposes the local transition directly',
            color=PROCESS, fontsize=11, weight='bold')
    box(.2, .1, 2.5, 'Observed source', PROCESS)
    box(3.6, .1, 2.8, 'Local transition', BASELINE)
    box(7.3, .1, 2.5, 'Target successor', PROCESS)
    arrow(2.8, .41, 3.5, .41, PROCESS)
    arrow(7.2, .41, 6.5, .41, PROCESS)
    save(fig, 'credit_geometry')


if __name__ == '__main__':
    plot_reliability_structure()
    plot_gradient_alignment()
    plot_credit_geometry()
    plot_depth_control()
