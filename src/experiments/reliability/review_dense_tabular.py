"""Dense, paired tabular concentration/reliability grid using neural-run corpora."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from handcoded.gates import make_gate_names, phi
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.experiments.circuit_match.review_functional_validation import unique_circuits, keys
from src.experiments.reliability.review_corpus_validation import table_metrics


def main():
    out=Path('results/review_validation/dense_tabular')
    out.mkdir(parents=True,exist_ok=False)
    gates=make_gate_names(); ids={g:i for i,g in enumerate(gates)}
    truth=np.array([[phi(s,g) for s in range(16)] for g in gates])
    masks=np.array([_coherent_wrong_mask(g) for g in gates])
    rows=[]; verified=0
    for seed in [42,43,44]:
        train=unique_circuits(20000,123+(seed-42)*1000)
        evaluation=unique_circuits(512,39000+seed,keys(train))
        ops=np.array([[ids[g] for g in c.gates] for c in train])
        start=np.array([c.start for c in train])
        eval_ops=np.array([[ids[g] for g in c.gates] for c in evaluation])
        eval_start=np.array([c.start for c in evaluation]); answers=np.array([c.answer for c in evaluation])
        scores=np.random.default_rng(seed+777).random(20000)
        selectors=np.random.default_rng(seed+778).random((20000,4))
        alternatives=np.random.default_rng(seed+779).integers(0,15,(20000,4))
        for concentration in [0.,.25,.5,.75,1.]:
            c=concentration+(1-concentration)/15
            for rho in sorted(set([round(v,2) for v in np.arange(0,1.001,.05)]+[c/(1+c)])):
                counts=np.zeros((52,16,16),dtype=np.int64); state=start.copy()
                for t in range(4):
                    correct=truth[ops[:,t],state]
                    diffuse=alternatives[:,t]+(alternatives[:,t]>=correct)
                    wrong=np.where(selectors[:,t]<concentration,correct ^ masks[ops[:,t]],diffuse)
                    successor=np.where(scores<rho,correct,wrong)
                    np.add.at(counts,(ops[:,t],state,successor),1)
                    state=successor
                metrics,_,_=table_metrics(counts,truth,eval_start,eval_ops,answers)
                metrics.update(seed=seed,rho=rho,concentration=concentration,
                               population_threshold=c/(1+c),population_margin=rho-(1-rho)*c)
                rows.append(metrics)
                for directory in ['concentration','concentration_intermediate']:
                    archive=out.parent/directory/f'corpus_s{seed}_rho{rho:g}_lambda{concentration:g}.npz'
                    if archive.exists():
                        np.testing.assert_array_equal(counts,np.load(archive)['counts'])
                        verified+=1
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'checks.json').write_text(json.dumps({'rows':len(rows),'identical_neural_counts_verified':verified,
        'note':'Independent vectorized implementation; exact comparison to available neural corpus archives.'},indent=2)+'\n')
    print(f'Wrote {len(rows)} rows; matched {verified} existing neural corpus tables exactly.',flush=True)


if __name__=='__main__':
    main()
