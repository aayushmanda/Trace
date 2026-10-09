"""Boolean-8 and the register machine for the character-token GPT.

Counting lives in handcoded/lettertrace.py.
"""
import random
import string
from dataclasses import dataclass
from typing import Callable, List

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi.get("\n", None)

    def encode(self, s: str) -> List[int]:
        unknown = set(s) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters in input string: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


ANSWER_SEP = " : "
# The vocabulary sets the embedding size, and with it the model initialization: do not change it.
GLOBAL_CHARS = string.ascii_lowercase + string.digits + " ;.:->+=*\n"
GLOBAL_TOKENIZER = CharTokenizer(GLOBAL_CHARS)


@dataclass(frozen=True)
class Instance:
    prompt: str
    correct_trace: str
    # Unused by the paper's runs. The samplers still draw it, because those draws advance the global RNG
    # and so fix which prompts generate_unique returns.
    wrong_trace: str
    gold: str


@dataclass
class Task:
    name: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float

    @property
    def tokenizer(self) -> CharTokenizer:
        return GLOBAL_TOKENIZER


# --- Boolean circuits: four bits, NOT / CNOT / SWAP / Toffoli on random wires --------------------------

N_BITS = 4


def _bits(value: int) -> list[int]:
    return [int(bit) for bit in f"{value:0{N_BITS}b}"]


def _state_text(state: list[int]) -> str:
    return "".join(str(bit) for bit in state)


def _sample_gate() -> str:
    operation = random.choice("xcst")
    if operation == "x":
        return f"x{random.randrange(N_BITS)}"
    if operation in "cs":
        first, second = random.sample(range(N_BITS), 2)
        return f"{operation}{first}{second}"
    control_a, control_b, target = random.sample(range(N_BITS), 3)
    return f"t{control_a}{control_b}{target}"


def _apply_gate(state: list[int], gate: str) -> list[int]:
    result = state.copy()
    operation = gate[0]
    if operation == "x":
        result[int(gate[1])] ^= 1
    elif operation == "c":
        control, target = int(gate[1]), int(gate[2])
        result[target] ^= result[control]
    elif operation == "s":
        first, second = int(gate[1]), int(gate[2])
        result[first], result[second] = result[second], result[first]
    elif operation == "t":
        control_a, control_b, target = int(gate[1]), int(gate[2]), int(gate[3])
        result[target] ^= result[control_a] & result[control_b]
    else:
        raise ValueError(f"unknown Boolean gate: {gate}")
    return result


def make_boolean_circuit_sampler(n_gates: int):
    def sample() -> Instance:
        start = _bits(random.randrange(2 ** N_BITS))
        gates = [_sample_gate() for _ in range(n_gates)]
        prompt = f"i{_state_text(start)};u{''.join(gates)}"
        state = start
        correct_steps = []
        for gate in gates:
            state = _apply_gate(state, gate)
            correct_steps.append(f"{gate}>{_state_text(state)}")
        gold = _state_text(state)
        wrong_state = start
        wrong_steps = []
        for gate in gates:
            true_value = int(_state_text(_apply_gate(wrong_state, gate)), 2)
            wrong_state = _bits(random.choice([v for v in range(2 ** N_BITS) if v != true_value]))
            wrong_steps.append(f"{gate}>{_state_text(wrong_state)}")
        return Instance(prompt, " ".join(correct_steps), " ".join(wrong_steps), gold)

    return sample


# --- Register machine: (x, y) in Z_17^2, five instructions, answer x ---------------------------------

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
            true_xy = _apply_register_instruction(wrong_x, wrong_y, instruction)
            while True:
                wrong_x, wrong_y = random.randrange(MODULUS), random.randrange(MODULUS)
                if (wrong_x, wrong_y) != true_xy:
                    break
            wrong_steps.append(f"{instruction}{_num(wrong_x)}{_num(wrong_y)}")
        return Instance(prompt, " ".join(correct_steps), " ".join(wrong_steps), gold)

    return sample


TASKS: dict[str, Task] = {
    "boolean_circuit_8": Task("boolean_circuit_8", block_size=320, max_new_tokens=96,
                              sample=make_boolean_circuit_sampler(8), chance_acc=1 / 16),
    "register_machine_16": Task("register_machine_16", block_size=192, max_new_tokens=128,
                                sample=make_register_machine_sampler(16), chance_acc=1 / MODULUS),
}


def generate_unique(task: Task, size: int, seed: int, excluded=None):
    """Sample unique prompts without permanently mutating the global RNG."""
    excluded = set() if excluded is None else set(excluded)
    state = random.getstate()
    random.seed(seed)
    items, prompts = [], set()
    while len(items) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        items.append(inst)
        prompts.add(inst.prompt)
    random.setstate(state)
    return items


# --- Training sets -------------------------------------------------------------------------------------

TARGET_BUILDERS = {
    "outcome": lambda inst: f"{ANSWER_SEP}{inst.gold}\n",
    "answer_first": lambda inst: f"{ANSWER_SEP}{inst.gold} ; {inst.correct_trace}\n",
    "process": lambda inst: f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n",
}


def encode_pair(tokenizer, prompt: str, target: str, block_size: int):
    prompt_ids = tokenizer.encode(prompt)
    full = prompt_ids + tokenizer.encode(target)
    if len(full) > block_size:
        raise ValueError(f"{len(full)} tokens exceeds block_size={block_size}")
    n = len(full) - 1
    mask = [0.0] * n
    for i in range(len(prompt_ids) - 1, n):
        mask[i] = 1.0
    return full[:-1], full[1:], mask


class SupervisionDataset(Dataset):
    """Prompt plus target for one supervision mode; the loss is on the target tokens only."""

    def __init__(self, instances, task, mode: str):
        if mode not in TARGET_BUILDERS:
            raise ValueError(f"Unknown mode: {mode}")
        tokenizer = task.tokenizer
        width = task.block_size - 1
        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)
        max_len = 0
        for row, inst in enumerate(instances):
            x, y, mask = encode_pair(tokenizer, inst.prompt, TARGET_BUILDERS[mode](inst), task.block_size)
            n = len(x)
            self.x[row, :n] = torch.tensor(x, dtype=torch.uint8)
            self.y[row, :n] = torch.tensor(y, dtype=torch.uint8)
            self.mask[row, :n] = torch.tensor(mask, dtype=torch.bool)
            max_len = max(max_len, n)
        self.x = self.x[:, :max_len]
        self.y = self.y[:, :max_len]
        self.mask = self.mask[:, :max_len]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]
