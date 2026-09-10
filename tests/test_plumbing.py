"""Unit tests for the restructured package. Quiet tqdm via TRACE_TQDM=0 in unittest argv."""
import json
import math
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

    def test_yaml_distributed_off_by_default(self):
        from src.training.config import load_yaml
        from src.training.distributed import distributed_enabled, maybe_data_parallel

        root = Path(__file__).resolve().parents[1]
        default = load_yaml(root / "configs" / "train" / "default.yaml")
        self.assertIs(default["distributed"], False)
        self.assertFalse(distributed_enabled(explicit=False))
        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1)
        self.assertIs(maybe_data_parallel(model, "cpu", enabled=True), model)
        self.assertIs(maybe_data_parallel(model, "cuda:2", enabled=False), model)

    def test_dataparallel_two_step_or_skip(self):
        from src.training.distributed import distributed_device_ids, maybe_data_parallel, unwrap_model
        from src.training.loop import train_steps
        from src.training.optim import make_adamw

        ids = distributed_device_ids(enabled=True)
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if ids is None or n < 2:
            self.skipTest(f"need 2+ CUDA devices for DataParallel smoke; have {n}")
        device = torch.device(f"cuda:{ids[0]}")
        model = GPTModel(vocab_size=8, block_size=8, pad_id=0, n_embd=8, n_head=2, n_layer=1).to(device)
        train_model = maybe_data_parallel(model, device, enabled=True)
        self.assertIsInstance(train_model, torch.nn.DataParallel)
        self.assertIs(unwrap_model(train_model), model)
        x = torch.randint(0, 8, (8, 4), device=device)
        y = torch.randint(0, 8, (8, 4), device=device)
        mask = torch.ones(8, 4, device=device)

        class _Once:
            def __iter__(self):
                while True:
                    yield (x, y, mask)

        opt = make_adamw(model.parameters(), 1e-3, device=device)
        loss = train_steps(train_model, _Once(), opt, device, 2, desc="dp-smoke")
        self.assertTrue(math.isfinite(loss))
        idx = torch.zeros(1, 2, dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=1)
        self.assertEqual(tuple(out.shape), (1, 3))

    def test_architecture_yaml_compile_off(self):
        from src.training.config import load_yaml
        from pathlib import Path
        cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs/experiments/architecture_controls.yaml")
        self.assertFalse(bool(cfg.get("compile")))
        self.assertEqual(cfg.get("devices"), ["cuda:2", "cuda:3"])
        self.assertEqual(len(cfg.get("rates") or []), 5)
        self.assertEqual(len(cfg.get("confirmation_seeds") or []), 10)
        root = Path(__file__).resolve().parents[1]
        for name in ("induced_rule", "architecture_controls", "length_generalization",
                     "margin_histograms", "executor_comparison", "smoke", "pullback",
                     "split_verdict", "escape_time", "lora_transfer",
                     "e1_five_condition", "e2_architecture", "e3_mask_trace", "e4_projected_kernel"):
            self.assertTrue((root / "configs" / "experiments" / f"{name}.yaml").exists())

    def test_architecture_summarize_counts_unstable_as_failure(self):
        from argparse import Namespace
        from src.eval.architecture_controls import summarize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "confirmation" / "process_process_s42_lr0.001_clip1"
            cell.mkdir(parents=True)
            (cell / "result.json").write_text(json.dumps({
                "status": "complete",
                "config": {"architecture": "process", "mode": "process", "seed": 42,
                           "lr": 0.001, "clip": 1.0},
                "selected_test": {"free_answer_accuracy": 1.0},
            }))
            bad = root / "confirmation" / "process_process_s43_lr0.001_clip1"
            bad.mkdir(parents=True)
            (bad / "result.json").write_text(json.dumps({
                "status": "unstable",
                "config": {"architecture": "process", "mode": "process", "seed": 43,
                           "lr": 0.001, "clip": 1.0},
                "reason": "Non-finite training loss",
            }))
            (root / "protocol.json").write_text(json.dumps({
                "confirmation_seeds": [42, 43],
                "clips": [1.0],
            }))
            summary = summarize(Namespace(output=root))
            self.assertEqual(len(summary), 1)
            row = summary[0]
            self.assertEqual(row["n_attempted"], 2)
            self.assertEqual(row["n_success"], 1)
            self.assertEqual(row["n_unstable"], 1)
            self.assertEqual(row["fraction_gt_95"], 0.5)
            payload = json.loads((root / "success_fraction.json").read_text())
            self.assertEqual(payload[0]["fraction_gt_95"], 0.5)

    def test_lora_model_flag_without_download(self):
        from src.training.config import load_yaml

        root = Path(__file__).resolve().parents[1]
        cfg = load_yaml(root / "configs" / "experiments" / "lora_transfer.yaml")
        self.assertEqual(cfg["model"], "HuggingFaceTB/SmolLM2-135M")
        try:
            from transformers import AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("transformers/peft not installed in this environment")
        self.assertTrue(callable(AutoModelForCausalLM.from_pretrained))
        self.assertTrue(callable(get_peft_model))
        self.assertEqual(LoraConfig(task_type="CAUSAL_LM", r=8).r, 8)


if __name__ == "__main__":
    unittest.main()
