"""Independent exact-enumeration and serialization checks for temporal noise."""
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.experiments.temporal_noise.run import make_corpus, make_dataset


def enumerate_law(rho, kappa, k=4, depth=4):
    counts = np.zeros((depth,k,k))
    correct_mass, prefix_mass = np.zeros(depth), np.zeros(depth)
    wrong_first_mass = wrong_first_correct_second = total = whole = 0.
    for start in range(k):
        for path in itertools.product(range(k), repeat=depth):
            sources = (start,)+path[:-1]
            flags = [y == (x+1)%k for x,y in zip(sources,path)]
            weight = 1./k
            for t, flag in enumerate(flags):
                pi = rho if t == 0 else kappa*flags[t-1]+(1-kappa)*rho
                weight *= pi if flag else (1-pi)/(k-1)
            total += weight
            whole += weight*all(flags)
            if not flags[0]:
                wrong_first_mass += weight
                wrong_first_correct_second += weight*flags[1]
            for t, (x,y) in enumerate(zip(sources,path)):
                counts[t,x,y] += weight
                if all(flags[:t]):
                    prefix_mass[t] += weight
                    correct_mass[t] += weight*flags[t]
    q = counts/counts.sum(-1, keepdims=True)
    expected = np.full((k,k),(1-rho)/(k-1))
    expected[np.arange(k),(np.arange(k)+1)%k] = rho
    return dict(total=total, max_local_error=float(np.max(np.abs(q-expected))),
                clean_prefix_probabilities=(correct_mass/prefix_mass).tolist(),
                wrong_prefix_probability=wrong_first_correct_second/wrong_first_mass,
                whole_clean_probability=whole)


class Checks(unittest.TestCase):
    def test_exact_population_laws(self):
        records=[]
        for rho,kappa in itertools.product([.2,.3,.7],[0.,.25,.5,.75,1.]):
            result=enumerate_law(rho,kappa)
            alpha=kappa+(1-kappa)*rho
            self.assertAlmostEqual(result['total'],1.)
            self.assertLess(result['max_local_error'],1e-12)
            np.testing.assert_allclose(result['clean_prefix_probabilities'],[rho,alpha,alpha,alpha],atol=1e-12)
            self.assertAlmostEqual(result['wrong_prefix_probability'],(1-kappa)*rho)
            self.assertAlmostEqual(result['whole_clean_probability'],rho*alpha**3)
            records.append(dict(rho=rho,kappa=kappa,**result))
        target=Path('results/novelty_extension/exact_checks.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(records,indent=2)+'\n')

    def test_corruption_and_suffix_masks(self):
        task=TASKS['boolean_circuit_8']
        instances=generate_unique(task,256,4001)
        for persistence in [0.,.5,1.]:
            traces,states,clean,counts=make_corpus(instances,.3,persistence,88)
            self.assertEqual(int(counts.sum()),256*8)
            noanswer=make_dataset(instances,task,traces,False)
            answer=make_dataset(instances,task,traces,True)
            self.assertEqual(noanswer.x.shape,answer.x.shape)
            for i,(inst,trace) in enumerate(zip(instances,traces)):
                suffix=len(inst.prompt)+1+len(trace)
                self.assertTrue(noanswer.mask[i,len(inst.prompt)-1:suffix-1].all())
                self.assertFalse(noanswer.mask[i,suffix-1:].any())
                self.assertTrue(torch.equal(noanswer.x[i,:suffix],answer.x[i,:suffix]))
                self.assertTrue(answer.mask[i,suffix-1:suffix+6].all())
            if persistence == 1:
                self.assertTrue((clean==clean[:,:1]).all())
            traces2,states2,clean2,counts2=make_corpus(instances,.3,persistence,88)
            self.assertEqual(traces,traces2)
            for first,second in [(states,states2),(clean,clean2),(counts,counts2)]:
                np.testing.assert_array_equal(first,second)

    def test_matched_patterns_and_trace_endpoint(self):
        from src.experiments.temporal_noise.run_patterns import pattern_corpus
        from src.experiments.temporal_noise.analyze import corrected_rollout
        task=TASKS['boolean_circuit_8']
        instances=generate_unique(task,128,4051)
        first=pattern_corpus(instances,.3,0,777)
        second=pattern_corpus(instances,.3,1,777)
        np.testing.assert_array_equal(first[2].sum(1),second[2].sum(1))
        np.testing.assert_array_equal(first[2].all(1),second[2].all(1))
        self.assertEqual(int(first[3].sum()),128*8)
        records=[{'generated': ' '+i.correct_trace+'111extra'} for i in instances]
        metrics=corrected_rollout(records,instances)
        self.assertEqual(metrics['exact_trace'],1.)
        self.assertEqual(metrics['valid_trace'],1.)
        records[0]['generated']='broken '+instances[0].correct_trace
        metrics=corrected_rollout(records,instances)
        self.assertEqual(metrics['exact_trace'],127/128)
        self.assertEqual(metrics['valid_trace'],127/128)

    def test_clean_corpus_matches_original(self):
        task=TASKS['boolean_circuit_8']
        instances=generate_unique(task,128,4031)
        for persistence in [0.,.5,1.]:
            traces,_,clean,_=make_corpus(instances,1.,persistence,88)
            self.assertEqual(traces,[i.correct_trace for i in instances])
            self.assertTrue(clean.all())


if __name__=='__main__':
    unittest.main()
