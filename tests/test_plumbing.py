"""Unit tests for the restructured package. Quiet tqdm via TRACE_TQDM=0 in unittest argv."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.boolean_circuit_tasks import BOOLEAN_CIRCUIT_DEPTHS, BOOLEAN_CIRCUIT_MAX_NEW_TOKENS
from src.data.registry import TASKS
from src.models.gpt import GPTModel
from src.models.handcoded import h
from src.training.handcoded_lm import handcoded_lm_loss
from src.training.loop import gpt_lm_loss, train_steps
from src.training.seed import set_seed


class PlumbingTests(unittest.TestCase):
    def test_length_and_induced_depths_registered(self):
        for depth in (2, 3, 4, 6, 8, 10, 12, 16, 20):
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
        forward = model.forward.__func__
        compiled = maybe_compile(model, "cpu", enabled=True)
        self.assertIs(compiled, model)
        self.assertIs(model.forward.__func__, forward)
        self.assertNotIn("forward", model.__dict__)
        self.assertFalse(getattr(compiled, "_trace_compiled", False))

    def test_maybe_compile_does_not_replace_forward(self):
        from src.training.seed import maybe_compile
        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1)
        original = model.forward.__func__
        maybe_compile(model, "cuda", enabled=True)
        self.assertIs(model.forward.__func__, original)
        self.assertNotIn("forward", model.__dict__)

    def test_generate_path_is_not_compiled(self):
        from unittest.mock import patch

        from src.training import seed as seed_mod
        from src.training.seed import maybe_compile

        class CompiledStub(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return self.inner(*args, **kwargs)

        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1)
        original_forward = model.forward.__func__
        original_generate = model.generate.__func__

        def fake_compile(m, **_kwargs):
            stub = CompiledStub(m)
            stub._trace_compiled = True
            return stub

        with patch.object(seed_mod, "_in_tests", return_value=False), \
             patch.object(seed_mod, "compile_enabled", return_value=True), \
             patch.object(torch, "compile", side_effect=fake_compile):
            compiled = maybe_compile(model, "cuda", enabled=True)
        self.assertIsNot(compiled, model)
        self.assertIs(model.forward.__func__, original_forward)
        self.assertNotIn("forward", model.__dict__)
        self.assertIs(model.generate.__func__, original_generate)
        idx = torch.zeros(1, 2, dtype=torch.long)
        out = model.generate(idx, max_new_tokens=2)
        self.assertEqual(tuple(out.shape), (1, 4))

    def test_yaml_compile_flags_parse(self):
        from src.training.config import load_experiment, load_yaml

        root = Path(__file__).resolve().parents[1]
        default = load_yaml(root / "configs" / "train" / "default.yaml")
        self.assertIs(default["compile"], False)
        ns, cfg = load_experiment(root / "configs" / "experiments" / "smoke.yaml")
        nested = (cfg.get("train") or {}).get("compile")
        self.assertFalse(bool(getattr(ns, "compile", nested if nested is not None else False)))
        hand = load_yaml(root / "configs" / "handcoded_smoke.yaml")
        self.assertIs(hand["compile"], False)
        exec_cfg = load_yaml(root / "configs" / "experiments" / "executor_comparison.yaml")
        self.assertIs(exec_cfg["compile"], True)

    def test_architecture_yaml_compile_off(self):
        from src.training.config import load_yaml
        from pathlib import Path
        cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs/experiments/architecture_controls.yaml")
        self.assertFalse(bool(cfg.get("compile")))
        self.assertEqual(cfg.get("devices"), ["cuda:2", "cuda:3"])
        root = Path(__file__).resolve().parents[1]
        for name in ("induced_rule", "architecture_controls", "length_generalization",
                     "margin_histograms", "executor_comparison", "smoke", "pullback",
                     "split_verdict", "escape_time", "lora_transfer"):
            self.assertTrue((root / "configs" / "experiments" / f"{name}.yaml").exists())

    def test_lora_model_flag_without_download(self):
        from src.training.config import load_yaml
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        root = Path(__file__).resolve().parents[1]
        cfg = load_yaml(root / "configs" / "experiments" / "lora_transfer.yaml")
        self.assertEqual(cfg["model"], "HuggingFaceTB/SmolLM2-135M")
        self.assertTrue(callable(AutoModelForCausalLM.from_pretrained))
        self.assertTrue(callable(get_peft_model))
        self.assertEqual(LoraConfig(task_type="CAUSAL_LM", r=8).r, 8)


if __name__ == "__main__":
    unittest.main()
