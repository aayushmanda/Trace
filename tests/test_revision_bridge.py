"""Sanity checks for the revision-bridge plumbing (no GPU, no long training)."""
import inspect
import unittest

import numpy as np
import torch

from src.data.boolean_circuit_tasks import BOOLEAN_CIRCUIT_DEPTHS, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS, make_boolean_circuit_sampler
from src.data.datasets import encode_pair
from src.data.registry import TASKS
from src.eval.induced_rule import GATES, K, M, TOK, composition, render
from src.eval.pullback import outcome_grad, process_grad, serialized_lm_grad
from src.models.gpt import GPTModel
from src.training.seed import set_seed


class RevisionBridgeTests(unittest.TestCase):
    def test_length_eval_depths_are_registered(self):
        for depth in (8, 10, 12, 16):
            self.assertIn(depth, BOOLEAN_CIRCUIT_DEPTHS)
            self.assertIn(f"boolean_circuit_{depth}", TASKS)
            self.assertIn(depth, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS)

    def test_induced_depths_have_samplers(self):
        for depth in (2, 3, 4, 6, 8):
            self.assertIn(depth, BOOLEAN_CIRCUIT_DEPTHS)
            inst = TASKS[f"boolean_circuit_{depth}"].sample()
            self.assertEqual(len(inst.correct_trace.split()), depth)

    def test_pullback_uses_full_vocab_ce_not_16way(self):
        src = inspect.getsource(serialized_lm_grad)
        self.assertIn("encode_pair", src)
        self.assertIn("mask", src)
        self.assertNotIn("STATES", src)
        self.assertIn("serialized_lm_grad", inspect.getsource(outcome_grad))
        self.assertIn("serialized_lm_grad", inspect.getsource(process_grad))
        self.assertGreater(TOK.vocab_size, 16)

        set_seed(0)
        sampler = make_boolean_circuit_sampler(2)
        insts = [sampler() for _ in range(3)]
        block = 64
        model = GPTModel(
            vocab_size=TOK.vocab_size, block_size=block, pad_id=TOK.pad_id,
            n_embd=16, n_head=2, n_layer=1, dropout=0.0,
        )
        g = serialized_lm_grad(model, insts, ["direct"] * len(insts), "cpu", block, batch=2)
        model.zero_grad(set_to_none=True)
        chunks = []
        n_mask = 0.0
        for inst in insts:
            _, target = render(inst, "direct")
            x, y, mask = encode_pair(TOK, inst.prompt, target, block)
            chunks.append((x, y, mask))
            n_mask += sum(mask)
        xs = torch.tensor([c[0] for c in chunks], dtype=torch.long)
        ys = torch.tensor([c[1] for c in chunks], dtype=torch.long)
        mask = torch.tensor([c[2] for c in chunks], dtype=torch.float32)
        logits, loss = model(xs, targets=ys, mask=mask)
        self.assertEqual(logits.shape[-1], TOK.vocab_size)
        self.assertNotEqual(logits.shape[-1], 16)
        loss.backward()
        ref = -torch.cat([
            (p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel()))
            for p in model.parameters()
        ])
        self.assertTrue(torch.allclose(g, ref, atol=1e-4, rtol=1e-4))
        self.assertGreater(g.norm().item(), 0.0)

    def test_composition_uses_per_step_tables(self):
        t1 = np.zeros((M, K, K))
        t1[:, :, 0] = 1.0
        t2 = np.zeros((M, K, K))
        t2[:, :, 1] = 1.0
        gates = [GATES[0], GATES[1]]
        v = composition([t1, t2], 7, gates)
        self.assertEqual(int(v.argmax()), 1)
        reused = composition([t1, t1], 7, gates)
        self.assertEqual(int(reused.argmax()), 0)
        with self.assertRaises(TypeError):
            composition(t1, 7, gates)


if __name__ == "__main__":
    unittest.main()
