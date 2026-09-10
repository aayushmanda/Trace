"""Local-transition machines that are not Boolean circuits, FSMs, or registers.

Each sampler emits an exact checkable trace, a same-length corrupted trace, and a
gold terminal answer that stays correct under corruption. Chance accuracy is
1 / |answer alphabet|.
"""
import random

from src.data.dataclass import Instance
from src.data.sequential_tasks import DIFFICULTY_STEPS, MODULUS, _different_value, _num

TAPE_LENGTH = 8
TAPE_SYMBOLS = 10
GRID_SIDE = 4
GRID_CELLS = GRID_SIDE * GRID_SIDE
QUEUE_MAX_DEPTH = 4
N_WALLS = 4


def _tape_op_text(operation) -> str:
    kind, value = operation
    if kind == "l":
        return "ml"
    if kind == "r":
        return "mr"
    return f"w{value}"


def _apply_tape(head: int, tape: list[str], operation) -> tuple[int, list[str]]:
    kind, value = operation
    tape = list(tape)
    if kind == "l":
        return (head - 1) % TAPE_LENGTH, tape
    if kind == "r":
        return (head + 1) % TAPE_LENGTH, tape
    tape[head] = str(value)
    return head, tape


def _corrupt_tape(head: int, tape: list[str]) -> tuple[int, list[str]]:
    tape = list(tape)
    if random.random() < 0.5:
        head = (head + random.randrange(1, TAPE_LENGTH)) % TAPE_LENGTH
    else:
        index = random.randrange(TAPE_LENGTH)
        tape[index] = str(_different_value(int(tape[index]), modulus=TAPE_SYMBOLS))
    return head, tape


def make_tape_machine_sampler(n_steps: int):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    def sample() -> Instance:
        tape = [str(random.randrange(TAPE_SYMBOLS)) for _ in range(TAPE_LENGTH)]
        head = random.randrange(TAPE_LENGTH)
        operations = []
        for _ in range(n_steps):
            kind = random.choice(("l", "r", "w", "w"))
            value = random.randrange(TAPE_SYMBOLS) if kind == "w" else None
            operations.append((kind, value))
        program = "".join(_tape_op_text(op) for op in operations)
        prompt = f"t{''.join(tape)};h{head};u{program}"
        cur_head, cur_tape = head, list(tape)
        wrong_head, wrong_tape = head, list(tape)
        correct_steps, wrong_steps = [], []
        for operation in operations:
            cur_head, cur_tape = _apply_tape(cur_head, cur_tape, operation)
            true_head, true_tape = _apply_tape(wrong_head, wrong_tape, operation)
            wrong_head, wrong_tape = _corrupt_tape(true_head, true_tape)
            op_text = _tape_op_text(operation)
            correct_steps.append(f"{op_text}>{cur_head}{''.join(cur_tape)}")
            wrong_steps.append(f"{op_text}>{wrong_head}{''.join(wrong_tape)}")
        gold = cur_tape[cur_head]
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


def _sample_queue_program(n_steps: int, max_depth: int = QUEUE_MAX_DEPTH):
    operations = []
    depth = 0
    for _ in range(n_steps):
        must_push = depth == 0
        can_push = depth < max_depth
        push = must_push or (can_push and random.random() < 0.60)
        if push:
            operations.append(("e", random.randrange(MODULUS)))
            depth += 1
        else:
            operations.append(("o", None))
            depth -= 1
    if depth == 0:
        operations[-1] = ("e", random.randrange(MODULUS))
    return operations


def make_queue_machine_sampler(n_steps: int, max_depth: int = QUEUE_MAX_DEPTH):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if max_depth < 2:
        raise ValueError("max_depth must be at least 2")

    def operation_text(operation):
        kind, value = operation
        return f"e{_num(value)}" if kind == "e" else "o"

    def sample() -> Instance:
        operations = _sample_queue_program(n_steps, max_depth=max_depth)
        prompt = "u" + "".join(operation_text(operation) for operation in operations)
        queue, correct_steps, wrong_steps = [], [], []
        for operation in operations:
            kind, value = operation
            if kind == "e":
                queue.append(value)
            else:
                queue.pop(0)
            op_text = operation_text(operation)
            correct_state = "".join(_num(item) for item in queue)
            wrong_state = "".join(_num(_different_value(item)) for item in queue)
            correct_steps.append(f"{op_text}>{correct_state}")
            wrong_steps.append(f"{op_text}>{wrong_state}")
        gold = _num(queue[0])
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


def _xy(cell: int) -> tuple[int, int]:
    return divmod(cell, GRID_SIDE)


def _cell(row: int, col: int) -> int:
    return row * GRID_SIDE + col


def _apply_grid(cell: int, action: str, walls: set[int]) -> int:
    row, col = _xy(cell)
    if action == "n":
        row -= 1
    elif action == "s":
        row += 1
    elif action == "e":
        col += 1
    elif action == "w":
        col -= 1
    else:
        raise ValueError(f"Unknown grid action: {action}")
    if not (0 <= row < GRID_SIDE and 0 <= col < GRID_SIDE):
        return cell
    nxt = _cell(row, col)
    if nxt in walls:
        return cell
    return nxt


def make_grid_walk_sampler(n_steps: int, n_walls: int = N_WALLS):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if not 1 <= n_walls < GRID_CELLS:
        raise ValueError("n_walls must leave at least one free cell")

    def sample() -> Instance:
        walls = set(random.sample(range(GRID_CELLS), n_walls))
        free = [cell for cell in range(GRID_CELLS) if cell not in walls]
        start = random.choice(free)
        actions = [random.choice("nsew") for _ in range(n_steps)]
        mask = "".join("1" if cell in walls else "0" for cell in range(GRID_CELLS))
        prompt = f"b{mask};s{_num(start)};u{''.join(actions)}"
        current = start
        correct_steps = []
        for action in actions:
            current = _apply_grid(current, action, walls)
            correct_steps.append(f"{action}>{_num(current)}")
        gold = _num(current)
        wrong_current = start
        wrong_steps = []
        for action in actions:
            true_next = _apply_grid(wrong_current, action, walls)
            wrong_next = random.choice([v for v in range(GRID_CELLS) if v != true_next])
            wrong_steps.append(f"{action}>{_num(wrong_next)}")
            wrong_current = wrong_next
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


TAPE_MACHINE_SAMPLERS = {s: make_tape_machine_sampler(s) for s in DIFFICULTY_STEPS}
QUEUE_MACHINE_SAMPLERS = {s: make_queue_machine_sampler(s) for s in DIFFICULTY_STEPS}
GRID_WALK_SAMPLERS = {s: make_grid_walk_sampler(s) for s in DIFFICULTY_STEPS}

# prompt + process target for the longest registered horizon (20 steps), plus margin.
TAPE_MACHINE_BLOCK_SIZE = 384
QUEUE_MACHINE_BLOCK_SIZE = 384
GRID_WALK_BLOCK_SIZE = 256
TAPE_MACHINE_MAX_NEW_TOKENS = {2: 48, 4: 80, 8: 128, 12: 192, 16: 240, 20: 280}
QUEUE_MACHINE_MAX_NEW_TOKENS = {2: 64, 4: 96, 8: 160, 12: 224, 16: 272, 20: 320}
GRID_WALK_MAX_NEW_TOKENS = {2: 32, 4: 48, 8: 64, 12: 80, 16: 96, 20: 128}
