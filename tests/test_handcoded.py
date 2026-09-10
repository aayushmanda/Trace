"""Exact-rule recovery for the semantic-token executors."""
import unittest

from handcoded import (
    HandcodedOutcomeTransformer, HandcodedProcessTransformer, make_circuits,
    make_generation_evaluation, free_run_metrics, make_tokenizer, phi,
)
from handcoded.eval import gold_answer_matrix
from handcoded.animate import animate_all_circuits
from src.plot_style import STYLE, apply_style


class HandcodedExactTests(unittest.TestCase):
    def test_fixed_process_recovers_gold_trace(self):
        tok = make_tokenizer()
        model = HandcodedProcessTransformer(tok, 4).eval()
        circuits = make_circuits(16, 7, 4)
        ev = make_generation_evaluation(circuits, tok, "cpu")
        metrics = free_run_metrics(model, ev, tok, "process")
        self.assertEqual(metrics["exact_continuation"], 1.0)
        self.assertEqual(metrics["final_answer"], 1.0)

    def test_fixed_outcome_recovers_answer(self):
        tok = make_tokenizer()
        model = HandcodedOutcomeTransformer(tok, 4).eval()
        circuits = make_circuits(16, 11, 4)
        ev = make_generation_evaluation(circuits, tok, "cpu")
        metrics = free_run_metrics(model, ev, tok, "outcome")
        self.assertEqual(metrics["exact_continuation"], 1.0)
        self.assertEqual(metrics["final_answer"], 1.0)

    def test_phi_composes(self):
        c = make_circuits(1, 3, 4)[0]
        s = c.start
        for g, gold in zip(c.gates, c.states):
            s = phi(s, g)
            self.assertEqual(s, gold)

    def test_encode_dataset_shapes(self):
        from handcoded.data import encode_dataset

        tok = make_tokenizer()
        circuits = make_circuits(8, 3, 4)
        for mode in ("process", "outcome"):
            batch = encode_dataset(circuits, tok, mode)
            self.assertEqual(batch.inputs.ndim, 2)
            self.assertEqual(batch.targets.shape, batch.inputs.shape)
            self.assertEqual(len(batch), 8)
            self.assertTrue((batch.targets[:, 0] == -100).all())

    def test_plot_style_is_central(self):
        apply_style()
        self.assertEqual(STYLE["axes.facecolor"], "#EAEAF2")
        import matplotlib.pyplot as plt
        self.assertEqual(plt.rcParams["axes.facecolor"], "#EAEAF2")

    def test_gold_answer_matrix_is_a_permutation(self):
        circuit = make_circuits(1, 3, 4)[0]
        matrix = gold_answer_matrix(circuit.gates)
        self.assertEqual(matrix.shape, (16, 16))
        self.assertEqual(int(matrix.sum()), 16)
        self.assertEqual(int(matrix[circuit.start].argmax()), circuit.answer)
        self.assertTrue((matrix.sum(axis=1) == 1).all())
        self.assertTrue((matrix.sum(axis=0) == 1).all())

    def test_animate_all_circuits_one_frame_per_circuit(self):
        tok = make_tokenizer()
        circuits = make_circuits(3, 0, 4)
        models = {
            "outcome": HandcodedOutcomeTransformer(tok, 4).eval(),
            "process": HandcodedProcessTransformer(tok, 4).eval(),
        }
        animation = animate_all_circuits(circuits, models, tok, "cpu", interval_ms=40, dpi=40)
        self.assertEqual(len(list(animation.new_frame_seq())), 3)


if __name__ == "__main__":
    unittest.main()
