"""Independent integrity checks for the new review experiments."""
import unittest
import numpy as np

from handcoded.gates import make_gate_names, phi
from src.data.datasets import RatioDataset
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.experiments.reliability.review_corpus_validation import arrays
from src.experiments.reliability.review_concentration_validation import make_states, corpus_table
from src.experiments.circuit_match.review_functional_validation import unique_circuits


class ValidationChecks(unittest.TestCase):
    def test_canonical_serialization_matches_actual_dataset(self):
        task = TASKS['boolean_circuit_8']
        instances = generate_unique(task, 40, 501)
        scores = np.random.default_rng(777).random(40)
        ids = {g:i for i,g in enumerate(make_gate_names())}
        for rho in [0., .3, 1.]:
            dataset = RatioDataset(instances, task, 'mixed_process', rho=rho, ratio_scores=scores)
            triples, _, _, _, _ = arrays(instances, rho, scores, ids)
            for i, inst in enumerate(instances):
                # Read the training targets, not a second invocation of the trace selector.
                emitted = task.tokenizer.decode(dataset.y[i][dataset.mask[i]].tolist())
                displayed = emitted.split(' : ')[0].strip()
                actual = triples[8*i:8*(i+1)]
                source = int(inst.prompt.split(';')[0][1:], 2)
                for j, token in enumerate(displayed.split()):
                    gate, state = token.split('>')
                    self.assertEqual(tuple(actual[j]), (ids[gate], source, int(state,2)))
                    source = int(state,2)
                self.assertEqual(emitted.split(' : ')[1].strip(), inst.gold)

    def test_all_mixture_kernels_preserve_uniform_sources(self):
        k = 16
        for gate in make_gate_names():
            successors = np.array([phi(s,gate) for s in range(k)])
            t = np.eye(k)[successors]
            coherent = np.eye(k)[successors ^ _coherent_wrong_mask(gate)]
            symmetric = (np.ones((k,k))-t)/(k-1)
            for concentration in [0.,.25,.5,.75,1.]:
                r = concentration*coherent+(1-concentration)*symmetric
                np.testing.assert_allclose(r.sum(0),1)
                np.testing.assert_allclose(r.sum(1),1)
                self.assertTrue(np.all(r[np.arange(k),successors]==0))
                c=concentration+(1-concentration)/(k-1)
                threshold=c/(1+c)
                for rho in [threshold-.001, threshold+.001]:
                    q=rho*t+(1-rho)*r
                    correct=q[np.arange(k),successors]
                    wrong=q.copy(); wrong[np.arange(k),successors]=-np.inf
                    self.assertTrue(np.all((correct-wrong.max(1)>0)==(rho>threshold)))

    def test_recursive_corruption_and_clean_pairing(self):
        circuits=unique_circuits(200,123)
        for rho in [0.,.3,1.]:
            for concentration in [0.,.25,.5,.75,1.]:
                states,scores=make_states(circuits,rho,concentration,819)
                for i,(c,path) in enumerate(zip(circuits,states)):
                    source=c.start
                    if scores[i]<rho:
                        self.assertEqual(path,c.states)
                    for gate,destination in zip(c.gates,path):
                        true=phi(source,gate)
                        if scores[i]>=rho:
                            self.assertNotEqual(destination,true)
                            if concentration==1:
                                self.assertEqual(destination,true ^ _coherent_wrong_mask(gate))
                        source=destination
                counts=corpus_table(circuits,states,{g:i for i,g in enumerate(make_gate_names())})
                self.assertEqual(counts.sum(),len(circuits)*4)

    def test_register_state_space_and_readout_are_distinct(self):
        from src.data.sequential_tasks import _apply_register_instruction
        states=[(x,y) for x in range(17) for y in range(17)]
        self.assertEqual(len(states),289)
        for instruction in 'abcde':
            self.assertEqual(len({_apply_register_instruction(x,y,instruction) for x,y in states}),289)
        self.assertAlmostEqual(1/289,0.0034602076124567475)
        self.assertNotEqual(1/289,1/17)


if __name__=='__main__':
    unittest.main()
