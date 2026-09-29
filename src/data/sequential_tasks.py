import random

from src.data.dataclass import Instance

DIFFICULTY_STEPS = (2, 4, 8, 12, 16, 20)
MODULUS = 17


def _num(value: int) -> str:
    return f"{value:02d}"


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


_COHERENT_REGISTER_MASK_SEED = "register_machine_coherent_wrong_mask_v1"
_coherent_register_mask_cache: dict[str, tuple[int, int]] = {}


def _coherent_register_mask(instruction: str) -> tuple[int, int]:
    """A fixed, nonzero (dx, dy) shift per instruction, deterministic across
    the whole run. wrong(x, y) := ((true_x + dx) % MODULUS, (true_y + dy) %
    MODULUS) is then a bijection on Z_MODULUS x Z_MODULUS (translation) that
    disagrees with the true instruction at every (x, y), i.e. exactly one
    coherent wrong rule per instruction, matching
    src.data.boolean_circuit_tasks._coherent_wrong_mask."""
    if instruction not in _coherent_register_mask_cache:
        rng = random.Random(f"{_COHERENT_REGISTER_MASK_SEED}:{instruction}")
        while True:
            dx, dy = rng.randrange(MODULUS), rng.randrange(MODULUS)
            if (dx, dy) != (0, 0):
                break
        _coherent_register_mask_cache[instruction] = (dx, dy)
    return _coherent_register_mask_cache[instruction]


def make_register_machine_sampler_coherent(n_steps: int):
    """Like make_register_machine_sampler, but wrong traces follow one fixed
    wrong permutation per instruction (via _coherent_register_mask) instead
    of a fresh uniform draw over all 288 wrong (x, y) pairs at each
    occurrence, matching the "one coherent wrong rule" corruption law of
    app:noise-threshold's tabular study and boolean_circuit_tasks'
    make_boolean_circuit_sampler_coherent."""
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
            dx, dy = _coherent_register_mask(instruction)
            wrong_x, wrong_y = (true_x + dx) % MODULUS, (true_y + dy) % MODULUS
            wrong_steps.append(f"{instruction}{_num(wrong_x)}{_num(wrong_y)}")
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


REGISTER_MACHINE_SAMPLERS = {s: make_register_machine_sampler(s) for s in DIFFICULTY_STEPS}
