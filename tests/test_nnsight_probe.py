"""NNsight probe + causal patch (CPU, oracle exact; random/wrong-layer not)."""
import unittest

import torch

from handcoded import HandcodedOutcomeTransformer, make_tokenizer
from handcoded.local_credit import counterfactual_accuracy
from handcoded.nnsight_probe import (
    collect_hiddens_nnsight,
    counterfactual_accuracy_nnsight,
    train_linear_probes,
)
from src.eval import executor_comparison as c


class NNsightProbeTests(unittest.TestCase):
    def test_collect_hiddens_matches_forward(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(4, 11, depth)
        h_nns, gold = collect_hiddens_nnsight(model, circuits, tok, device="cpu")
        ids = torch.tensor([tok.prompt(c) + [tok.colon] for c in circuits])
        _, h_fwd = model(ids, return_states=True)
        self.assertEqual(len(h_nns), depth)
        for step in range(depth):
            self.assertTrue(torch.allclose(h_nns[step], h_fwd[step], atol=1e-5))

    def test_linear_probe_trains(self):
        tok = make_tokenizer()
        model = HandcodedOutcomeTransformer(tok, 2).eval()
        circuits = c.unique_circuits(12, 22, 2)
        h, gold = collect_hiddens_nnsight(model, circuits, tok, device="cpu")
        probes, train_acc, eval_acc = train_linear_probes(h, gold, h, gold, tok.n_states, steps=10, lr=0.1)
        self.assertEqual(len(probes), 2)
        self.assertGreater(train_acc[0], 0.5)

    def test_oracle_patch_nnsight_exact(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(6, 11, depth)
        for layer in range(depth):
            acc = counterfactual_accuracy_nnsight(model, circuits, tok, layer, method="oracle_slot", device="cpu")
            self.assertGreaterEqual(acc, 0.999, msg=f"nnsight oracle layer {layer} acc={acc}")

    def test_nnsight_matches_hook_patch(self):
        tok = make_tokenizer()
        model = HandcodedOutcomeTransformer(tok, 2).eval()
        circuits = c.unique_circuits(6, 33, 2)
        generator = torch.Generator().manual_seed(0)
        for method in ("random_subspace", "unstructured"):
            hook = counterfactual_accuracy(
                model, circuits, tok, 0, method=method, device="cpu", generator=generator,
            )
            generator = torch.Generator().manual_seed(0)
            nns = counterfactual_accuracy_nnsight(
                model, circuits, tok, 0, method=method, device="cpu", generator=generator,
            )
            self.assertAlmostEqual(hook, nns, places=5, msg=method)

    def test_wrong_layer_below_oracle(self):
        tok = make_tokenizer()
        model = HandcodedOutcomeTransformer(tok, 2).eval()
        circuits = c.unique_circuits(6, 44, 2)
        oracle = counterfactual_accuracy_nnsight(model, circuits, tok, 0, method="oracle_slot", device="cpu")
        wrong = counterfactual_accuracy_nnsight(
            model, circuits, tok, 0, method="oracle_slot", device="cpu", wrong_layer=1,
        )
        self.assertGreaterEqual(oracle, 0.99)
        self.assertLess(wrong, oracle)


if __name__ == "__main__":
    unittest.main()
