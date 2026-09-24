"""Build paper artifacts and an honest report from the full temporal experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.temporal_noise.analyze import ROOT, main as analyze


def ms(values,scale=100):
    return f'{values.mean()*scale:.2f} ± {values.std()*scale:.2f}'


def main():
    analyze()
    df=pd.read_csv(ROOT/'all_results.csv')
    assert len(df)==30
    contrast=pd.read_csv(ROOT/'paired_contrasts.csv')
    groups=[]
    for (rho,kappa,answer),z in df[df.family=='canonical'].groupby(['rho','persistence','answer_loss']):
        label='Answer loss' if answer else 'Clean' if rho==1 else 'No answer'
        groups.append((label,rho,kappa,z))
    for pattern,z in df[df.family=='matched_patterns'].groupby('pattern'):
        groups.append((pattern.title(),.3,None,z))
    table=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Condition & $\rho$ & $\kappa$ & Prefix gap & Exact trace & Supplied first\\',r'\midrule']
    markdown=['| Condition | ρ | κ | Prefix gap (pp) | Exact trace (%) | With first step (%) |',
              '|---|---:|---:|---:|---:|---:|']
    for label,rho,kappa,z in groups:
        values=[ms(z[k]) for k in ['prefix_probability_gap','ordinary_exact_trace','anchored_exact_trace']]
        k='--' if kappa is None else f'{kappa:g}'
        tex=['$'+v.replace(' ± ',r'\pm')+'$' for v in values]
        table.append(f'{label} & {rho:g} & {k} & '+' & '.join(tex)+r'\\')
        markdown.append(f'| {label} | {rho:g} | {k} | '+' | '.join(values)+' |')
    table += [r'\bottomrule',r'\end{tabular}']
    tables=Path('Paper/tables');tables.mkdir(exist_ok=True)
    (tables/'temporal_summary.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':7,'legend.fontsize':6,'xtick.labelsize':6,'ytick.labelsize':6})
    fig,axes=plt.subplots(1,3,figsize=(5.55,1.85))
    for rho,color in [(.3,'#1766a4'),(.7,'#be681e')]:
        sub=df[(df.family=='canonical')&(df.rho==rho)&(~df.answer_loss)]
        for ax,metric in zip(axes[:2],['prefix_probability_gap','anchored_exact_trace']):
            g=sub.groupby('persistence')[metric].agg(['mean','std'])
            ax.errorbar(g.index,g['mean']*100,g['std']*100,marker='o',ms=2.5,lw=1,capsize=2,color=color,label=f'ρ={rho:g}')
    axes[0].plot([0,1],[0,100],'k:',lw=.8,label='Population')
    axes[0].set(title='(a) Prefix response',xlabel='Persistence κ',ylabel='Probability gap (pp)',ylim=(-5,105),xticks=[0,.5,1])
    axes[0].legend(loc='upper left',frameon=False)
    axes[1].set(title='(b) Supplied first step',xlabel='Persistence κ',ylabel='Exact trace (%)',ylim=(-2,50),xticks=[0,.5,1])
    for i,(pattern,color) in enumerate([('blocked','#1766a4'),('interleaved','#be681e')]):
        z=df[(df.family=='matched_patterns')&(df.pattern==pattern)].prefix_probability_gap*100
        axes[2].bar(i,z.mean(),yerr=z.std(),width=.55,color=color,alpha=.7,capsize=2)
        axes[2].scatter(i+np.linspace(-.1,.1,len(z)),z,s=10,color=color,edgecolors='white',linewidths=.3,zorder=3)
    axes[2].axhline(0,color='black',lw=.6)
    axes[2].set(title='(c) Matched trace quality',ylabel='Probability gap (pp)',xticks=[0,1],xticklabels=['Blocked','Interleaved'])
    fig.tight_layout(pad=.35,w_pad=.7)
    for ext in ['pdf','png']:
        fig.savefig(Path('Paper/figures')/f'temporal_main.{ext}',bbox_inches='tight',dpi=200)
    plt.close(fig)
    primary=contrast[(contrast.comparison=='persistent_minus_independent')&(contrast.rho==.3)&(~contrast.answer_loss)&(contrast.metric=='prefix_probability_gap')].iloc[0]
    samerow=contrast[(contrast.comparison=='persistent_minus_independent')&(contrast.rho==.3)&(~contrast.answer_loss)&(contrast.metric=='same_row_gap')].iloc[0]
    matched=contrast[(contrast.comparison=='blocked_minus_interleaved')&(contrast.metric=='prefix_probability_gap')].iloc[0]
    text=f'''# Temporal corruption extension: completed results

Completed 2026-09-23. **30 new canonical character-token GPT training runs**:
24 initial conditions and six matched-pattern controls. Each uses 20,000 unique
training prompts and 8,000 updates; three independent paired model/corpus seeds.
All models, optimizer states, midpoint checkpoints, generations and predictions
are archived. No pretrained or larger-model replication was run.

## What is established

The population construction proves that local transition laws do not determine
contextual successor laws. With persistent correctness flags, a correct versus
wrong preceding transition changes the next clean probability by kappa even
though every local transition law is unchanged. A grouped construction also
matches whole-trace correctness and correct-step-count distributions while
changing the verified-prefix length needed for ideal greedy continuation.
See [the derivations](THEORY.md). These are elementary probability results,
not guarantees of finite neural training or claims of priority.

In the canonical GPT, the preregistered rho=.3 persistence contrast in the
prefix-response gap is **{primary['mean']*100:.2f} percentage points**, with
three-seed descriptive paired 95% t interval
**[{primary.ci_low*100:.2f}, {primary.ci_high*100:.2f}]**. The contrast is positive
in every seed. A post-hoc same-row probe, holding displayed state, next operation,
and target successor fixed, gives **{samerow['mean']*100:.2f} points
[{samerow.ci_low*100:.2f}, {samerow.ci_high*100:.2f}]**. This supports history-dependent
prediction beyond a change in the current state. It does not identify an exact
Bayesian computation inside the model.

## What is NOT established

- At rho=.3, a supplied correct first step yields only 1.03% exact traces under
  persistent corruption, versus 0% under independent corruption.
- At rho=.7, corresponding exact-trace accuracies are 30.03% and 31.60%:
  persistence has no consistent continuation benefit at this budget.
- The ideal advantage of two verified steps for interleaved corruption is NOT
  observed. Mean exact suffix accuracy is 0.87% blocked versus 0.07% interleaved.
- The blocked-minus-interleaved prefix-gap contrast is {matched['mean']*100:.2f}
  points [{matched.ci_low*100:.2f}, {matched.ci_high*100:.2f}]. It is positive in
  all three seeds, but its interval includes zero.
- Answer-supervision restoration has a positive mean prefix effect with a wide
  interval including zero. Neither probability effects nor per-prompt counts
  should be presented as extra independent training replications.

The evidence strengthens a narrow contextual-corruption contribution. It does
not justify claiming a validated neural recovery boundary or assigning 8/10
novelty as an achieved experimental outcome.

## Aggregate results

All entries are mean ± sample SD across three seeds. Prefix gap is the true-state
probability after a correct first transition minus that after a coherent wrong
first transition, each using its own displayed source. Raw and conditional-on-
valid-state gaps agree closely. Clean-control wrong-prefix probes are outside
the clean training distribution and are not covered by the noisy-law theorem.

'''+ '\n'.join(markdown)+'''

## Controls and audit

- Training corpora, initialization and minibatch schedules match within each seed.
- Table counts and correctness masks were checked for all 30 corpora.
- In every matched blocked/interleaved pair, clean-trace indicators and counts
  of correct steps agree EXACTLY per prompt. Local laws match in population;
  finite empirical row frequencies need not be identical.
- The stationary sweep changes whole-trace cleanliness; it is not claimed fixed.
- No-answer training uses a constant terminal value and masks the entire suffix.
- Full-state candidate scoring and free character-wise generation are separate.
- An evaluation correction truncates free generation at the prescribed trace
  endpoint. Previously, unsupervised characters after the final state could
  falsely invalidate a completed trace. No in-trace token is repaired. The audit
  CSV preserves both definitions; all final tables use the corrected endpoint.
- Three seeds underlie each comparison. Intervals have two degrees of freedom;
  secondary comparisons are exploratory, with no multiplicity adjustment.

## Artifacts and reproduction

- [Initial frozen protocol](PROTOCOL.md)
- [Matched-pattern follow-up protocol](MATCHED_PATTERN_PROTOCOL.md)
- [Targeted prior-work comparison](RELATED_WORK.md)
- [All final per-seed results](all_results.csv)
- [Paired contrasts](paired_contrasts.csv)
- [Endpoint-definition audit](trace_endpoint_audit.csv)
- [Corpus and pairing checks](integrity.json)
- [Exact population checks](exact_checks.json)
- [Validation figure](temporal_validation.pdf)

```bash
python -m src.experiments.temporal_noise.checks
python -m src.experiments.temporal_noise.run --output results/novelty_extension/canonical --device cuda:0 --shard 0 --shards 2
python -m src.experiments.temporal_noise.run --output results/novelty_extension/canonical --device cuda:1 --shard 1 --shards 2
python -m src.experiments.temporal_noise.run_patterns --output results/novelty_extension/matched_patterns --device cuda:0 --code 0
python -m src.experiments.temporal_noise.run_patterns --output results/novelty_extension/matched_patterns --device cuda:1 --code 1
python -m src.experiments.temporal_noise.probe_same_row --device cuda:0
python -m src.experiments.temporal_noise.report
```

Completed conditions are skipped. To retrain, use fresh output directories;
analysis/report default to the archive paths above. For matched-pattern runs,
`pattern_protocol.json` and `pattern_metrics.json` are the authoritative raw
metadata; the reused trainer's k0/k1 filenames are pattern codes, not Markov
persistence settings. `all_results.csv` is authoritative for corrected rollout.

The current paper integrates the limited positive effect and the negative
continuation results. Its earlier version is in `manuscript_before_extension/`.
'''
    (ROOT/'RESULTS.md').write_text(text)
    files=list(Path('src/experiments/temporal_noise').glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.csv'))
    files += [ROOT/'exact_checks.json',ROOT/'integrity.json',Path('Paper/figures/temporal_main.pdf'),tables/'temporal_summary.tex']
    for family in ['canonical','matched_patterns']:
        files += [p for p in (ROOT/family).glob('*/*') if p.is_file()]
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Wrote report, main-paper figure, appendix table, and',len(manifest),'artifact checksums.')


if __name__=='__main__':
    main()
