import random

from src.data.dataclass import Instance

DIFFICULTY_STEPS = (2, 4, 8, 12, 16, 20)
MODULUS = 17


def _num(value: int) -> str:
    return f"{value:02d}"


def _different_value(value: int, modulus: int = MODULUS) -> int:
    return (value + random.randrange(1, modulus)) % modulus


def _apply_modular(value: int, operator: str, operand: int) -> int:
    if operator == "+":
        return (value + operand) % MODULUS
    if operator == "-":
        return (value - operand) % MODULUS
    if operator == "*":
        return (value * operand) % MODULUS
    raise ValueError(f"Unknown modular operator: {operator}")


def make_modular_program_sampler(n_steps: int):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    def sample() -> Instance:
        start = random.randrange(MODULUS)
        instructions = []
        for _ in range(n_steps):
            operator = random.choice(("+", "-", "*"))
            operand = random.randrange(MODULUS) if operator != "*" else random.randrange(1, MODULUS)
            instructions.append((operator, operand))
        program = "".join(f"{operator}{_num(operand)}" for operator, operand in instructions)
        prompt = f"s{_num(start)};u{program}"
        current = start
        correct_steps = []
        for operator, operand in instructions:
            nxt = _apply_modular(current, operator, operand)
            correct_steps.append(f"{_num(current)}{operator}{_num(operand)}>{_num(nxt)}")
            current = nxt
        gold = _num(current)
        wrong_current = start
        wrong_steps = []
        for operator, operand in instructions:
            true_next = _apply_modular(wrong_current, operator, operand)
            wrong_next = _different_value(true_next)
            wrong_steps.append(f"{_num(wrong_current)}{operator}{_num(operand)}>{_num(wrong_next)}")
            wrong_current = wrong_next
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong)
        return Instance(prompt, correct, wrong, gold)

    return sample


def _apply_register_instruction(x: int, y: int, instruction: str) -> tuple[int, int]:
    if instruction == "a":
        return (x + y) % MODULUS, y
    if instruction == "b":
        return x, (x + y) % MODULUS
    if instruction == "c":
        return y, x
    if instruction == "d":
        return (x + 1) % MODULUS, y
    if instruction == "e":
        return x, (y + 1) % MODULUS
    raise ValueError(f"Unknown register instruction: {instruction}")


def make_register_machine_sampler(n_steps: int):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    def sample() -> Instance:
        start_x, start_y = random.randrange(MODULUS), random.randrange(MODULUS)
        instructions = [random.choice("abcde") for _ in range(n_steps)]
        prompt = f"x{_num(start_x)};y{_num(start_y)};u{''.join(instructions)}"
        x, y = start_x, start_y
        correct_steps = []
        for instruction in instructions:
            x, y = _apply_register_instruction(x, y, instruction)
            correct_steps.append(f"{instruction}{_num(x)}{_num(y)}")
        gold = _num(x)
        wrong_x, wrong_y = start_x, start_y
        wrong_steps = []
        for instruction in instructions:
            true_x, true_y = _apply_register_instruction(wrong_x, wrong_y, instruction)
            while True:
                candidate_x = random.randrange(MODULUS)
                candidate_y = random.randrange(MODULUS)
                if (candidate_x, candidate_y) != (true_x, true_y):
                    break
            wrong_x, wrong_y = candidate_x, candidate_y
            wrong_steps.append(f"{instruction}{_num(wrong_x)}{_num(wrong_y)}")
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong)
        return Instance(prompt, correct, wrong, gold)

    return sample


def _sample_stack_program(n_steps: int, max_depth: int = 4):
    operations = []
    depth = 0
    for _ in range(n_steps):
        must_push = depth == 0
        can_push = depth < max_depth
        push = must_push or (can_push and random.random() < 0.60)
        if push:
            operations.append(("p", random.randrange(MODULUS)))
            depth += 1
        else:
            operations.append(("o", None))
            depth -= 1
    if depth == 0:
        operations[-1] = ("p", random.randrange(MODULUS))
    return operations


def make_stack_machine_sampler(n_steps: int, max_depth: int = 4):
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if max_depth < 2:
        raise ValueError("max_depth must be at least 2")

    def operation_text(operation):
        kind, value = operation
        return f"p{_num(value)}" if kind == "p" else "o"

    def sample() -> Instance:
        operations = _sample_stack_program(n_steps, max_depth=max_depth)
        prompt = "u" + "".join(operation_text(operation) for operation in operations)
        stack, correct_steps, wrong_steps = [], [], []
        for operation in operations:
            kind, value = operation
            if kind == "p":
                stack.append(value)
            else:
                stack.pop()
            op_text = operation_text(operation)
            correct_state = "".join(_num(item) for item in stack)
            wrong_state = "".join(_num(_different_value(item)) for item in stack)
            correct_steps.append(f"{op_text}>{correct_state}")
            wrong_steps.append(f"{op_text}>{wrong_state}")
        gold = _num(stack[-1])
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


MODULAR_PROGRAM_SAMPLERS = {s: make_modular_program_sampler(s) for s in DIFFICULTY_STEPS}
REGISTER_MACHINE_SAMPLERS = {s: make_register_machine_sampler(s) for s in DIFFICULTY_STEPS}
STACK_MACHINE_SAMPLERS = {s: make_stack_machine_sampler(s) for s in DIFFICULTY_STEPS}
