"""Local-credit mechanism test: no trace tokens; oracle patch is exact; random is not."""
import unittest

import torch

from handcoded import (
    Circuit, HandcodedOutcomeTransformer, attach_local_heads, encode_dataset, make_circuits,
    make_tokenizer,
)
from handcoded.data import language_model_loss
from handcoded.local_credit import (
    counterfactual_accuracy,
    has_gate_tokens,
    outcome_plus_local_loss,
    trained_target_ids,
)
from src.eval import executor_comparison as c


class OutcomeLocalCreditTests(unittest.TestCase):
    def test_outcome_local_targets_have_no_process_tokens(self):
        tok = make_tokenizer()
        circuits = make_circuits(8, 3, 2)
        batch = encode_dataset(circuits, tok, "outcome")
        trained = trained_target_ids(batch)
        self.assertFalse(has_gate_tokens(trained, tok))
        process = encode_dataset(circuits, tok, "process")
        self.assertTrue(has_gate_tokens(trained_target_ids(process), tok))

        model = HandcodedOutcomeTransformer(tok, 2)
        attach_local_heads(model, seed=0)
        gold = torch.tensor([c.states for c in circuits], dtype=torch.long)
        loss, terminal, local = outcome_plus_local_loss(model, batch, gold, lambda_local=1.0)
        self.assertGreater(local.detach().item(), 0.0)
        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertEqual(batch.targets.shape, batch.inputs.shape)
        self.assertFalse(has_gate_tokens(trained_target_ids(batch), tok))

        lied = [
            Circuit(c.start, c.gates, [(s + 1) % tok.n_states for s in c.states[:-1]] + [c.states[-1]])
            for c in circuits
        ]
        self.assertEqual([c.answer for c in lied], [c.answer for c in circuits])
        lied_gold = torch.tensor([c.states for c in lied], dtype=torch.long)
        _, terminal_lied, local_lied = outcome_plus_local_loss(model, batch, lied_gold, lambda_local=1.0)
        self.assertAlmostEqual(terminal.detach().item(), terminal_lied.detach().item(), places=5)
        self.assertNotAlmostEqual(local.detach().item(), local_lied.detach().item(), places=5)
        self.assertAlmostEqual(
            terminal.detach().item(), language_model_loss(model, batch).detach().item(), places=5,
        )

    def test_oracle_patch_is_exact_counterfactual(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(6, 11, depth)
        for layer in range(depth):
            acc = counterfactual_accuracy(model, circuits, tok, layer, method="oracle_slot", device="cpu")
            self.assertGreaterEqual(acc, 0.999, msg=f"oracle layer {layer} acc={acc}")

    def test_random_patch_is_not_exact(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(6, 11, depth)
        generator = torch.Generator().manual_seed(0)
        acc = counterfactual_accuracy(
            model, circuits, tok, 0, method="random_subspace", device="cpu", generator=generator,
        )
        self.assertLess(acc, 0.5, msg=f"random patch acc={acc}")


if __name__ == "__main__":
    unittest.main()
