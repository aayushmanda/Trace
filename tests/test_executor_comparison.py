"""Numerical checks for the scientific readouts, not training-success tests."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.eval import executor_comparison as c


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = c.h.make_tokenizer()

    def test_fixed_rules_every_gate_state_and_position(self):
        model = c.h.HandcodedProcessTransformer(self.tok, 4).eval()
        with torch.no_grad():
            tables, mass = c.read_local_rules(model, self.tok, 4, 'cpu', backgrounds=1)
        true = c.truth_tables(self.tok)
        for table in tables.numpy():
            np.testing.assert_array_equal(table.argmax(-1), true.argmax(-1))
            np.testing.assert_allclose(table, true, atol=1e-5)
        self.assertGreater(mass.min().item(), 0.999)

    def test_exact_composition_matches_execution(self):
        examples = c.unique_circuits(50, 214, 4)
        tables = torch.tensor(c.truth_tables(self.tok))
        p = c.compose_tables(tables, examples, self.tok)
        np.testing.assert_array_equal(p.argmax(-1), [x.answer for x in examples])
        np.testing.assert_allclose(p.sum(-1), 1)

    def test_outcome_does_not_use_trace_queries(self):
        examples = c.unique_circuits(4, 1, 2)
        model = c.h.build_random_learned_model(self.tok, 2, d_model=16, d_ff=32)
        with patch.object(c, 'read_local_rules', side_effect=AssertionError('OOD query')):
            metrics, arrays = c.diagnose(model, self.tok, 2, examples, 'cpu', 'outcome')
        self.assertNotIn('local_conditional', arrays)
        self.assertIn('out of distribution', metrics['local_query_status'])

    def test_direct_query_does_not_include_gold_trace(self):
        model = c.h.build_random_learned_model(self.tok, 2, d_model=16, d_ff=32).eval()
        examples = c.unique_circuits(4, 18, 2)
        altered = [c.h.Circuit(x.start, x.gates, [15 - s for s in x.states]) for x in examples]
        with torch.no_grad():
            p = c.direct_probabilities(model, examples, self.tok, 'cpu')
            q = c.direct_probabilities(model, altered, self.tok, 'cpu')
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_readout_gradient_matches_finite_difference(self):
        model = c.h.build_random_learned_model(self.tok, 2, seed=17, d_model=16, d_ff=32).double().eval()
        examples = c.unique_circuits(4, 23, 2)
        def loss():
            tables, _ = c.read_local_rules(model, self.tok, 2, 'cpu', backgrounds=1, steps=[0])
            p = c.compose_tables(tables.mean(0), examples, self.tok)
            return -p[range(4), [x.answer for x in examples]].log().mean()
        g = torch.autograd.grad(loss(), model.readout.weight)[0]
        index = np.unravel_index(g.abs().argmax().item(), g.shape)
        original = model.readout.weight[index].item()
        eps = 1e-5
        with torch.no_grad():
            model.readout.weight[index] = original + eps
            plus = loss().item()
            model.readout.weight[index] = original - eps
            minus = loss().item()
            model.readout.weight[index] = original
        self.assertAlmostEqual((plus - minus) / (2 * eps), g[index].item(), places=7)

    def test_splits_are_disjoint(self):
        train = c.unique_circuits(100, 1, 2)
        test = c.unique_circuits(50, 2, 2, c.circuit_keys(train))
        self.assertFalse(c.circuit_keys(train) & c.circuit_keys(test))
        self.assertEqual(len(c.circuit_keys(train)), 100)


if __name__ == '__main__':
    unittest.main()
