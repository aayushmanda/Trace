"""Sanity checks for the revision-bridge plumbing (no GPU, no long training)."""
import inspect
import math
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

    def test_pullback_credit_rejects_a_reused_table(self):
        from src.eval.pullback import measure_pullback, surrogate_credit
        t1 = np.zeros((M, K, K))
        t1[:, :, 0] = 1.0
        sampler = make_boolean_circuit_sampler(2)
        insts = [sampler() for _ in range(2)]
        with self.assertRaises(TypeError):
            surrogate_credit(t1, insts, 2)
        G = surrogate_credit([t1, t1], insts, 2)
        self.assertEqual(G.shape, (2, M, K, K))
        model = GPTModel(
            vocab_size=TOK.vocab_size, block_size=64, pad_id=TOK.pad_id,
            n_embd=16, n_head=2, n_layer=1, dropout=0.0,
        )
        with self.assertRaises(TypeError):
            measure_pullback(model, insts, 2, "cpu", 64, Phat=t1, fd=False)

    def test_skip_jacobian_still_uses_serialized_lm(self):
        from src.eval.pullback import measure_pullback
        set_seed(0)
        sampler = make_boolean_circuit_sampler(2)
        insts = [sampler() for _ in range(2)]
        t1 = np.broadcast_to(np.eye(K), (M, K, K)).copy()
        model = GPTModel(
            vocab_size=TOK.vocab_size, block_size=64, pad_id=TOK.pad_id,
            n_embd=16, n_head=2, n_layer=1, dropout=0.0,
        )
        row = measure_pullback(
            model, insts, 2, "cpu", 64, Phat=[t1, t1], fd=False, skip_jacobian=True,
        )
        self.assertEqual(row["pullback_objective"], "serialized_lm_ce")
        self.assertGreater(row["norm_outcome"], 0.0)
        self.assertTrue(np.isnan(row["cos_true_outcome"]))

    def test_pullback_jacobian_sums_per_step_not_step1_only(self):
        from src.eval.pullback import pullback_grad
        set_seed(0)
        model = GPTModel(
            vocab_size=TOK.vocab_size, block_size=64, pad_id=TOK.pad_id,
            n_embd=16, n_head=2, n_layer=1, dropout=0.0,
        )
        depth = 2
        W = np.zeros((depth, M, K, K))
        rng = np.random.default_rng(1)
        slab = rng.normal(size=(M, K, K))
        from src.eval.pullback import PI
        slab = np.einsum("ij,mjk,kl->mil", PI, slab, PI)
        W0 = np.zeros_like(W); W0[0] = slab
        W1 = np.zeros_like(W); W1[1] = slab
        both = np.stack([slab, slab], axis=0)
        pooled_on_t1 = np.zeros_like(W); pooled_on_t1[0] = 2 * slab
        kw = dict(chunk=64, max_gates=2, max_states=3)
        g0 = pullback_grad(model, depth, "cpu", W0, **kw)
        g1 = pullback_grad(model, depth, "cpu", W1, **kw)
        gboth = pullback_grad(model, depth, "cpu", both, **kw)
        gwrong = pullback_grad(model, depth, "cpu", pooled_on_t1, **kw)
        self.assertTrue(torch.allclose(gboth, g0 + g1, atol=1e-4, rtol=1e-4))
        self.assertGreater((g0 - g1).norm().item(), 1e-8)
        self.assertGreater((gboth - gwrong).norm().item(), 1e-8)
        with self.assertRaises(TypeError):
            pullback_grad(model, depth, "cpu", slab, **kw)

    def test_pullback_metrics_nan_on_zero_norm(self):
        from src.eval.pullback import cosine, grad_norm_ratio, relative_grad_residual
        z = torch.zeros(4)
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(math.isnan(cosine(z, v)))
        self.assertTrue(math.isnan(cosine(v, z)))
        self.assertTrue(math.isnan(grad_norm_ratio(v, z)))
        self.assertTrue(math.isnan(relative_grad_residual(v, z)))
        self.assertAlmostEqual(cosine(v, v), 1.0)
        self.assertAlmostEqual(grad_norm_ratio(2 * v, v), 2.0)

    def test_split_verdict_builds_from_csv_rows(self):
        import pandas as pd
        from src.eval.split_verdict import build_table
        import json
        df = pd.DataFrame([{
            "depth": 2, "seed": 1, "step": 2, "probe_step": 1,
            "state_on_set_mass": 0.9, "delta_comp_tv": 0.1,
            "cos_true_outcome": 0.2, "rel_grad_err_outcome": 0.5,
            "eps_rule_hat": 1.0, "eps_step_std": 0.01, "table_step_tv": 0.02,
            "background_eps_std": 0.0, "condition": "both",
            "eps_by_lambda_conditional": json.dumps({"1": 0.4, "0.5": 0.2, "0.25": 0.1, "0.125": 0.05}),
            "credit_by_lambda_conditional": json.dumps({"1": 0.16, "0.5": 0.04, "0.25": 0.01, "0.125": 0.0025}),
        }])
        tbl = build_table(df)
        self.assertEqual(int(tbl.iloc[0].predicted_exponent), 1)
        self.assertFalse(pd.isna(tbl.iloc[0].fitted_exponent_in_ball))


if __name__ == "__main__":
    unittest.main()
