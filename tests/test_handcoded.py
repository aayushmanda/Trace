"""Exact-rule recovery for the semantic-token executors."""
import unittest

import torch

from handcoded import (
    HandcodedOutcomeTransformer, HandcodedProcessTransformer, make_circuits,
    make_generation_evaluation, free_run_metrics, make_tokenizer, phi,
)
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

    def test_plot_style_is_central(self):
        apply_style()
        self.assertEqual(STYLE["axes.facecolor"], "#EAEAF2")
        import matplotlib.pyplot as plt
        self.assertEqual(plt.rcParams["axes.facecolor"], "#EAEAF2")


if __name__ == "__main__":
    unittest.main()
