"""Sanity checks for the revision-bridge plumbing (no GPU, no training)."""
import unittest

from src.data.boolean_circuit_tasks import BOOLEAN_CIRCUIT_DEPTHS, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS
from src.data.registry import TASKS


class RevisionBridgeTests(unittest.TestCase):
    def test_length_eval_depths_are_registered(self):
        for depth in (8, 10, 12, 16):
            self.assertIn(depth, BOOLEAN_CIRCUIT_DEPTHS)
            self.assertIn(f"boolean_circuit_{depth}", TASKS)
            self.assertIn(depth, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS)

    def test_induced_depths_have_samplers(self):
        for depth in (2, 4, 6, 8):
            self.assertIn(depth, BOOLEAN_CIRCUIT_DEPTHS)
            inst = TASKS[f"boolean_circuit_{depth}"].sample()
            self.assertEqual(len(inst.correct_trace.split()), depth)


if __name__ == "__main__":
    unittest.main()
