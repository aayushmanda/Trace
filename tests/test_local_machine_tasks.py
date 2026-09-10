import random
import unittest

from src.data.dataclass import GLOBAL_TOKENIZER
from src.data.datasets import encode_pair
from src.data.local_machine_tasks import (
    GRID_CELLS,
    TAPE_LENGTH,
    TAPE_SYMBOLS,
    _apply_grid,
    _apply_tape,
    _tape_op_text,
)
from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.data.sequential_tasks import DIFFICULTY_STEPS, MODULUS


class LocalMachineTaskTests(unittest.TestCase):
    def test_paper_families_still_registered(self):
        for name in ("boolean_circuit_8", "state_machine_8", "register_machine_8"):
            self.assertIn(name, TASKS)

    def test_new_families_registered(self):
        for family in ("tape_machine", "queue_machine", "grid_walk"):
            for steps in DIFFICULTY_STEPS:
                self.assertIn(f"{family}_{steps}", TASKS)

    def test_prompt_uniqueness_and_trace_length(self):
        for name, n_steps in (
            ("tape_machine_8", 8),
            ("queue_machine_8", 8),
            ("grid_walk_8", 8),
        ):
            task = TASKS[name]
            items = generate_unique(task, 128, seed=0)
            self.assertEqual(len({inst.prompt for inst in items}), 128)
            for inst in items[:16]:
                self.assertEqual(len(inst.correct_trace.split()), n_steps)
                self.assertEqual(len(inst.wrong_trace.split()), n_steps)
                self.assertEqual(len(inst.correct_trace), len(inst.wrong_trace))
                self.assertNotEqual(inst.correct_trace, inst.wrong_trace)

    def test_chance_and_traces_fit_block(self):
        expected = {
            "tape_machine_8": 1 / TAPE_SYMBOLS,
            "queue_machine_8": 1 / MODULUS,
            "grid_walk_8": 1 / GRID_CELLS,
        }
        for name, chance in expected.items():
            task = TASKS[name]
            self.assertAlmostEqual(task.chance_acc, chance)
            random.seed(0)
            for _ in range(32):
                inst = task.sample()
                self.assertEqual(len(inst.correct_trace), len(inst.wrong_trace))
                self.assertNotEqual(inst.correct_trace, inst.wrong_trace)
                unknown = set(inst.prompt + inst.correct_trace + inst.gold) - set(GLOBAL_TOKENIZER.stoi)
                self.assertFalse(unknown)
                prompt, target = task.render(inst, "correct_think")
                encode_pair(task.tokenizer, prompt, target + "\n", task.block_size)
                self.assertLessEqual(len(target) + 8, task.max_new_tokens)

    def test_tape_gold_is_symbol_under_head(self):
        task = TASKS["tape_machine_8"]
        random.seed(1)
        inst = task.sample()
        tape = list(inst.prompt[1:1 + TAPE_LENGTH])
        head = int(inst.prompt.split(";h")[1][0])
        program = inst.prompt.split(";u", 1)[1]
        ops = []
        i = 0
        while i < len(program):
            if program[i] == "m":
                ops.append(("l" if program[i + 1] == "l" else "r", None))
                i += 2
            else:
                ops.append(("w", int(program[i + 1])))
                i += 2
        for op in ops:
            head, tape = _apply_tape(head, tape, op)
        self.assertEqual(inst.gold, tape[head])
        self.assertEqual(len(inst.correct_trace.split()), 8)
        self.assertTrue(all(step.startswith(_tape_op_text(op)) for step, op in zip(inst.correct_trace.split(), ops)))

    def test_grid_stays_off_walls(self):
        task = TASKS["grid_walk_8"]
        random.seed(2)
        inst = task.sample()
        mask = inst.prompt[1:1 + GRID_CELLS]
        walls = {i for i, ch in enumerate(mask) if ch == "1"}
        start = int(inst.prompt.split(";s")[1][:2])
        actions = inst.prompt.split(";u", 1)[1]
        cell = start
        self.assertNotIn(start, walls)
        for action in actions:
            cell = _apply_grid(cell, action, walls)
            self.assertNotIn(cell, walls)
        self.assertEqual(inst.gold, f"{cell:02d}")


if __name__ == "__main__":
    unittest.main()
