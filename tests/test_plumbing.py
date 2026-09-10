"""Unit tests for the restructured package. Quiet tqdm via TRACE_TQDM=0 in unittest argv."""
import json
import tempfile
import unittest
from pathlib import Path

from src.data.boolean_circuit_tasks import BOOLEAN_CIRCUIT_DEPTHS, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS
from src.data.registry import TASKS
from src.models.gpt import GPTModel
from src.models.handcoded import h
from src.training.handcoded_lm import handcoded_lm_loss
from src.training.loop import gpt_lm_loss, train_steps
from src.training.seed import set_seed


class PlumbingTests(unittest.TestCase):
    def test_length_and_induced_depths_registered(self):
        for depth in (2, 4, 6, 8, 10, 12, 16, 20):
            self.assertIn(depth, BOOLEAN_CIRCUIT_DEPTHS)
            self.assertIn(f"boolean_circuit_{depth}", TASKS)
            self.assertIn(depth, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS)
        inst = TASKS["boolean_circuit_8"].sample()
        self.assertEqual(len(inst.correct_trace.split()), 8)

    def test_both_stacks_import(self):
        self.assertTrue(hasattr(GPTModel, "generate"))
        tok = h.make_tokenizer()
        self.assertGreater(tok.n_states, 0)
        self.assertTrue(callable(handcoded_lm_loss))
        self.assertTrue(callable(gpt_lm_loss))
        self.assertTrue(callable(train_steps))
        set_seed(0)

    def test_architecture_plan_does_not_clobber(self):
        from src.eval.architecture_controls import plan
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            proto = {"rates": [0.001], "confirmation_seeds": [1]}
            (out / "protocol.json").write_text(json.dumps(proto))
            args = Namespace(output=out, depth=4, rates=[0.002], confirmation_seeds=[9],
                             calibration_steps=1, steps=1, train_size=1, val_size=1,
                             test_size=1, batch_size=1)
            got = plan(args)
            self.assertEqual(got["rates"], [0.001])

    def test_compile_disabled_under_unittest(self):
        from src.training.seed import compile_enabled, maybe_compile
        self.assertFalse(compile_enabled(device="cuda"))
        self.assertFalse(compile_enabled(explicit=True, device="cuda"))
        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1)
        forward = model.forward
        compiled = maybe_compile(model, "cpu", enabled=True)
        self.assertIs(compiled, model)
        self.assertIs(model.forward, forward)
        self.assertFalse(getattr(compiled, "_trace_compiled", False))

    def test_maybe_compile_does_not_replace_forward(self):
        from src.training.seed import maybe_compile
        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1)
        original = model.forward
        maybe_compile(model, "cuda", enabled=True)
        self.assertIs(model.forward, original)

    def test_yaml_configs_exist(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("induced_rule", "architecture_controls", "length_generalization",
                     "margin_histograms", "executor_comparison", "smoke"):
            self.assertTrue((root / "configs" / "experiments" / f"{name}.yaml").exists())


if __name__ == "__main__":
    unittest.main()
