"""Stack, modular, and word-index sequential-executor families.

These are extra mechanism tasks beyond Boolean circuits, FSMs, and registers.
Each sampler emits an exact local trace, a same-length corrupted trace, and a
gold terminal that stays correct under corruption.
"""
import random
import unittest

import numpy as np
import torch

from src.data.dataclass import GLOBAL_TOKENIZER, Instance
from src.data.datasets import ContinuationDataset, RatioDataset, encode_pair
from src.data.hard_word_index_tasks import WORD_INDEX_LENGTHS
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.data.sequential_tasks import DIFFICULTY_STEPS, MODULUS, _apply_modular, _num


PAPER_FAMILIES = ("boolean_circuit_8", "state_machine_8", "register_machine_8")
MECHANISM_HORIZONS = {
    "modular_program": DIFFICULTY_STEPS,
    "stack_machine": DIFFICULTY_STEPS,
}
REPRESENTATIVE = ("modular_program_8", "stack_machine_8", "word_index_len16")


def _parse_modular_prompt(prompt: str):
    start = int(prompt[1:3])
    program = prompt.split(";u", 1)[1]
    ops = []
    i = 0
    while i < len(program):
        ops.append((program[i], int(program[i + 1:i + 3])))
        i += 3
    return start, ops


def _parse_stack_prompt(prompt: str):
    body = prompt[1:]
    ops = []
    i = 0
    while i < len(body):
        if body[i] == "p":
            ops.append(("p", int(body[i + 1:i + 3])))
            i += 3
        elif body[i] == "o":
            ops.append(("o", None))
            i += 1
        else:
            raise AssertionError(f"unexpected stack opcode {body[i]!r} in {prompt!r}")
    return ops


def _assert_instance_well_formed(test, task, inst, n_steps):
    test.assertEqual(len(inst.correct_trace), len(inst.wrong_trace))
    test.assertNotEqual(inst.correct_trace, inst.wrong_trace)
    test.assertEqual(len(inst.correct_trace.split()), n_steps)
    test.assertEqual(len(inst.wrong_trace.split()), n_steps)
    unknown = set(inst.prompt + inst.correct_trace + inst.gold) - set(GLOBAL_TOKENIZER.stoi)
    test.assertFalse(unknown)
    for mode in ("correct_think", "wrong_think"):
        prompt, target = task.render(inst, mode)
        encode_pair(task.tokenizer, prompt, target + "\n", task.block_size)
        test.assertLessEqual(len(target) + 8, task.max_new_tokens)
        test.assertTrue(target.endswith(inst.gold))


class MechanismTaskTests(unittest.TestCase):
    def test_paper_families_still_registered(self):
        for name in PAPER_FAMILIES:
            self.assertIn(name, TASKS)

    def test_families_registered(self):
        for family, horizons in MECHANISM_HORIZONS.items():
            for steps in horizons:
                self.assertIn(f"{family}_{steps}", TASKS)
        for length in WORD_INDEX_LENGTHS:
            self.assertIn(f"word_index_len{length}", TASKS)

    def test_chance_acc(self):
        self.assertAlmostEqual(TASKS["modular_program_8"].chance_acc, 1 / MODULUS)
        self.assertAlmostEqual(TASKS["stack_machine_8"].chance_acc, 1 / MODULUS)
        self.assertAlmostEqual(TASKS["word_index_len16"].chance_acc, 1 / 16)

    def test_sample_prompt_uniqueness_and_trace_length(self):
        expected_steps = {
            "modular_program_8": 8,
            "stack_machine_8": 8,
            "word_index_len16": 16,
        }
        for name, n_steps in expected_steps.items():
            task = TASKS[name]
            items = generate_unique(task, 256, seed=0)
            prompts = [inst.prompt for inst in items]
            self.assertEqual(len(prompts), 256)
            self.assertEqual(len(set(prompts)), 256)
            for inst in items[:32]:
                _assert_instance_well_formed(self, task, inst, n_steps)

    def test_all_horizons_fit_block(self):
        names = (
            [f"modular_program_{s}" for s in DIFFICULTY_STEPS]
            + [f"stack_machine_{s}" for s in DIFFICULTY_STEPS]
            + [f"word_index_len{n}" for n in WORD_INDEX_LENGTHS]
        )
        for name in names:
            n_steps = int("".join(ch for ch in name.split("_")[-1] if ch.isdigit()))
            task = TASKS[name]
            random.seed(8)
            for _ in range(8):
                _assert_instance_well_formed(self, task, task.sample(), n_steps)

    def test_modular_gold_replays_the_program(self):
        task = TASKS["modular_program_8"]
        random.seed(3)
        for _ in range(32):
            inst = task.sample()
            current, ops = _parse_modular_prompt(inst.prompt)
            self.assertEqual(len(ops), 8)
            steps = inst.correct_trace.split()
            for (operator, operand), step in zip(ops, steps):
                nxt = _apply_modular(current, operator, operand)
                self.assertEqual(step, f"{_num(current)}{operator}{_num(operand)}>{_num(nxt)}")
                current = nxt
            self.assertEqual(inst.gold, _num(current))

    def test_stack_gold_is_top_after_replay(self):
        task = TASKS["stack_machine_8"]
        random.seed(4)
        for _ in range(32):
            inst = task.sample()
            ops = _parse_stack_prompt(inst.prompt)
            self.assertEqual(len(ops), 8)
            stack = []
            for (kind, value), step in zip(ops, inst.correct_trace.split()):
                if kind == "p":
                    stack.append(value)
                    self.assertTrue(step.startswith(f"p{_num(value)}>"))
                else:
                    stack.pop()
                    self.assertTrue(step.startswith("o>"))
                self.assertEqual(step.split(">", 1)[1], "".join(_num(item) for item in stack))
            self.assertTrue(stack)
            self.assertEqual(inst.gold, _num(stack[-1]))

    def test_word_index_gold_is_first_occurrence(self):
        task = TASKS["word_index_len16"]
        random.seed(5)
        for _ in range(32):
            inst = task.sample()
            word, query = inst.prompt.split(";")
            self.assertEqual(len(word), 16)
            self.assertEqual(len(query), 1)
            self.assertEqual(inst.gold, str(word.index(query)))
            steps = inst.correct_trace.split()
            self.assertEqual(len(steps), 16)
            for i, (step, ch) in enumerate(zip(steps, word)):
                self.assertEqual(step, f"{i}{ch}")

    def test_corrupt_trace_keeps_gold(self):
        for name in REPRESENTATIVE:
            task = TASKS[name]
            random.seed(6)
            inst = task.sample()
            _, correct_target = task.render(inst, "correct_think")
            _, wrong_target = task.render(inst, "wrong_think")
            self.assertTrue(correct_target.endswith(inst.gold))
            self.assertTrue(wrong_target.endswith(inst.gold))
            self.assertNotEqual(inst.correct_trace, inst.wrong_trace)

    def test_reliability_ratio_dataset_encodes(self):
        task = TASKS["stack_machine_8"]
        items = generate_unique(task, 8, seed=7)
        scores = np.linspace(0, 1, 8, endpoint=False)
        RatioDataset(items, task, "mixed_process", rho=0.5, ratio_scores=scores)
        RatioDataset(items, task, "outcome")

    def test_continuation_dataset_crops_to_max_used_length(self):
        tok = GLOBAL_TOKENIZER
        block_size = 64
        short = Instance(prompt="ab", correct_trace="c", wrong_trace="d", gold="1")
        long = Instance(prompt="ab", correct_trace="cccc", wrong_trace="dddd", gold="12")
        short_target = " : 1\n"
        long_target = " cccc : 12\n"
        used = []
        for prompt, target in ((short.prompt, short_target), (long.prompt, long_target)):
            full = tok.encode(prompt) + tok.encode(target)
            self.assertLess(len(full), block_size)
            used.append(len(full) - 1)
        max_used = max(used)
        self.assertLess(max_used, block_size - 1)
        self.assertNotEqual(used[0], used[1])

        dataset = ContinuationDataset([short, long], tok, block_size, [short_target, long_target])
        self.assertEqual(tuple(dataset.x.shape), (2, max_used))
        self.assertEqual(dataset.y.shape, dataset.x.shape)
        self.assertEqual(dataset.mask.shape, dataset.x.shape)

        x0, y0, m0 = dataset[0]
        n_short = used[0]
        self.assertEqual(x0.shape[0], max_used)
        self.assertTrue(torch.equal(x0[n_short:], torch.full((max_used - n_short,), tok.pad_id, dtype=torch.uint8)))
        self.assertTrue(torch.equal(y0[n_short:], torch.full((max_used - n_short,), tok.pad_id, dtype=torch.uint8)))
        self.assertFalse(m0[n_short:].any())
        x_s, y_s, m_s = encode_pair(tok, short.prompt, short_target, block_size)
        self.assertEqual(len(x_s), n_short)
        self.assertTrue(torch.equal(x0[:n_short], torch.tensor(x_s, dtype=torch.uint8)))
        self.assertTrue(torch.equal(y0[:n_short], torch.tensor(y_s, dtype=torch.uint8)))
        self.assertTrue(torch.equal(m0[:n_short], torch.tensor(m_s, dtype=torch.bool)))
        prompt_len = len(tok.encode(short.prompt))
        self.assertEqual(int(m0.sum()), n_short - (prompt_len - 1))
        self.assertTrue(m0[: prompt_len - 1].eq(False).all())
        self.assertTrue(m0[prompt_len - 1:n_short].all())

        x1, y1, m1 = dataset[1]
        x_l, y_l, m_l = encode_pair(tok, long.prompt, long_target, block_size)
        self.assertEqual(len(x_l), max_used)
        self.assertTrue(torch.equal(x1, torch.tensor(x_l, dtype=torch.uint8)))
        self.assertTrue(torch.equal(y1, torch.tensor(y_l, dtype=torch.uint8)))
        self.assertTrue(torch.equal(m1, torch.tensor(m_l, dtype=torch.bool)))

        with self.assertRaises(ValueError):
            encode_pair(tok, "ab", "c" * block_size, block_size)


if __name__ == "__main__":
    unittest.main()
