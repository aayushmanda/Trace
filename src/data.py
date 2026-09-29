"""Boolean circuits, the register machine, and the state machine.

Counting, shell, and lights live in handcoded/lettertrace.py.
"""
from typing import List
import string

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
        ids = [self.stoi[c] for c in s]
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("encoded id outside vocab")
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

from dataclasses import dataclass
import hashlib
import random
import re
import string
from typing import Callable, Optional

ANSWER_SEP = " : "
DATASET_SEED = 12345
VAL_SEED = 9999
DATASET_VERSION = "1.0.0"
GLOBAL_CHARS = string.ascii_lowercase + string.digits + " ;.:->+=*\n"
GLOBAL_TOKENIZER = CharTokenizer(GLOBAL_CHARS)

@dataclass(frozen=True)
class Instance:
    prompt: str
    correct_trace: str
    wrong_trace: str
    gold: str

@dataclass
class Task:
    name: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float
    ceiling_acc: float
    description: str = ""
    answer_pattern: str = r"-?\d+"
    bayes_prob: Optional[Callable[[Instance], float]] = None

    @property
    def tokenizer(self) -> CharTokenizer:
        return GLOBAL_TOKENIZER

    def render(self, inst: Instance, mode: str = "correct_think"):
        if mode == "correct_think":
            return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "wrong_think":
            return inst.prompt, f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "no_think":
            return inst.prompt, f" {inst.gold}"
        raise ValueError(f"unknown mode {mode}")

    def context(self, inst: Instance, mode: str) -> str:
        if mode == "free":
            return inst.prompt
        if mode == "forced_correct":
            return f"{inst.prompt} {inst.correct_trace}{ANSWER_SEP}"
        if mode == "forced_wrong":
            return f"{inst.prompt} {inst.wrong_trace}{ANSWER_SEP}"
        if mode == "direct":
            return f"{inst.prompt} "
        raise ValueError(f"unknown eval mode {mode}")

    def extract_answer(self, generated_tail: str, first: bool = False) -> Optional[str]:
        tail = generated_tail.split("\n")[0]
        segment = tail.split(":")[-1] if ":" in tail else tail
        found = re.findall(self.answer_pattern, segment)
        if not found:
            return None
        return found[0] if first else found[-1]

    def fingerprint(self, n: int = 64) -> str:
        state = random.getstate()
        random.seed(12345)
        h = hashlib.sha1()
        h.update(DATASET_VERSION.encode())
        h.update(self.name.encode())
        h.update(f"{DATASET_SEED}|{VAL_SEED}".encode())
        for _ in range(n):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                h.update(f"{prompt}{answer}\n".encode())
        random.setstate(state)
        h.update(f"{self.block_size}|{''.join(self.tokenizer.chars)}".encode())
        return h.hexdigest()[:10]

import random

N_BITS = 4
BOOLEAN_CIRCUIT_DEPTHS = (2, 3, 4, 6, 8, 10, 12, 16, 20)

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
    if n_gates < 1:
        raise ValueError("n_gates must be positive")

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
            true_next = _apply_gate(wrong_state, gate)
            true_value = int(_state_text(true_next), 2)
            wrong_value = random.choice([v for v in range(2 ** N_BITS) if v != true_value])
            wrong_state = _bits(wrong_value)
            wrong_steps.append(f"{gate}>{_state_text(wrong_state)}")
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample

_COHERENT_WRONG_MASK_SEED = "boolean_circuit_coherent_wrong_mask_v1"
_coherent_wrong_mask_cache: dict[str, int] = {}

def _coherent_wrong_mask(gate: str) -> int:
    """A fixed, nonzero per-gate XOR mask, deterministic across the whole run.

    wrong(s) := correct(s) XOR mask is then itself a permutation (XOR by a
    fixed value is a bijection) that disagrees with the correct gate at
    every state, i.e. exactly one coherent wrong rule per gate, not a fresh
    uniform draw per occurrence as in make_boolean_circuit_sampler above.
    """
    if gate not in _coherent_wrong_mask_cache:
        rng = random.Random(f"{_COHERENT_WRONG_MASK_SEED}:{gate}")
        _coherent_wrong_mask_cache[gate] = rng.randrange(1, 2 ** N_BITS)
    return _coherent_wrong_mask_cache[gate]

def make_boolean_circuit_sampler_coherent(n_gates: int):
    """Like make_boolean_circuit_sampler, but wrong traces follow one fixed
    wrong permutation per gate (via _coherent_wrong_mask) instead of a fresh
    uniform-over-K-1 draw at each occurrence, matching the "one coherent
    wrong rule" corruption law of app:noise-threshold's tabular study."""
    if n_gates < 1:
        raise ValueError("n_gates must be positive")

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
            true_next = _apply_gate(wrong_state, gate)
            true_value = int(_state_text(true_next), 2)
            wrong_value = true_value ^ _coherent_wrong_mask(gate)
            wrong_state = _bits(wrong_value)
            wrong_steps.append(f"{gate}>{_state_text(wrong_state)}")
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample

BOOLEAN_CIRCUIT_SAMPLERS = {
    depth: make_boolean_circuit_sampler(depth) for depth in BOOLEAN_CIRCUIT_DEPTHS
}
BOOLEAN_CIRCUIT_BLOCK_SIZE = 320
BOOLEAN_CIRCUIT_MAX_NEW_TOKENS = {
    2: 48, 3: 56, 4: 64, 6: 80, 8: 96, 10: 120, 12: 144, 16: 184, 20: 224,
}

import random

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
    _coherent_wrong_mask."""
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

import random

def _derangement(n: int) -> list[int]:
    while True:
        values = list(range(n))
        random.shuffle(values)
        if all(i != value for i, value in enumerate(values)):
            return values

def make_state_machine_sampler(n_states: int = 16, n_steps: int = 8):
    if n_states < 4:
        raise ValueError("n_states must be at least 4")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    width = max(2, len(str(n_states - 1)))
    action_chars = "ab"

    def state(value: int) -> str:
        return f"{value:0{width}d}"

    def sample() -> Instance:
        transitions = [_derangement(n_states), _derangement(n_states)]
        start = random.randrange(n_states)
        actions = [random.randrange(2) for _ in range(n_steps)]
        table_a = "".join(state(transitions[0][source]) for source in range(n_states))
        table_b = "".join(state(transitions[1][source]) for source in range(n_states))
        program = "".join(action_chars[action] for action in actions)
        prompt = f"a{table_a};b{table_b};s{state(start)};u{program}"
        current = start
        correct_steps = []
        for action in actions:
            nxt = transitions[action][current]
            correct_steps.append(f"{state(current)}{action_chars[action]}{state(nxt)}")
            current = nxt
        gold = state(current)
        wrong_current = start
        wrong_steps = []
        for action in actions:
            true_next = transitions[action][wrong_current]
            wrong_next = random.choice([v for v in range(n_states) if v != true_next])
            wrong_steps.append(f"{state(wrong_current)}{action_chars[action]}{state(wrong_next)}")
            wrong_current = wrong_next
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong)
        return Instance(prompt, correct, wrong, gold)

    return sample

STATE_MACHINE_SAMPLERS = {
    steps: make_state_machine_sampler(n_states=16, n_steps=steps)
    for steps in (2, 4, 8, 12, 16, 20)
}

TASKS: dict[str, Task] = {
    f"boolean_circuit_{depth}": Task(
        name=f"boolean_circuit_{depth}",
        block_size=BOOLEAN_CIRCUIT_BLOCK_SIZE,
        max_new_tokens=BOOLEAN_CIRCUIT_MAX_NEW_TOKENS[depth],
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a reversible four-bit Boolean circuit with {depth} gates",
        answer_pattern=r"[01]{4}",
    )
    for depth, sampler in BOOLEAN_CIRCUIT_SAMPLERS.items()
}
TASKS["boolean_circuit_8_coherent"] = Task(
    name="boolean_circuit_8_coherent",
    block_size=BOOLEAN_CIRCUIT_BLOCK_SIZE,
    max_new_tokens=BOOLEAN_CIRCUIT_MAX_NEW_TOKENS[8],
    sample=make_boolean_circuit_sampler_coherent(8), chance_acc=1 / 16, ceiling_acc=1.0,
    description="depth-8 Boolean circuit, corrupted traces follow one fixed wrong permutation per gate",
    answer_pattern=r"[01]{4}",
)
TASKS.update({
    f"register_machine_{steps}": Task(
        name=f"register_machine_{steps}", block_size=192, max_new_tokens=128,
        sample=sampler, chance_acc=1 / MODULUS, ceiling_acc=1.0,
        description=f"execute {steps} updates on two registers modulo 17",
        answer_pattern=r"\d+",
    )
    for steps, sampler in REGISTER_MACHINE_SAMPLERS.items()
})
TASKS["register_machine_16_coherent"] = Task(
    name="register_machine_16_coherent", block_size=192, max_new_tokens=128,
    sample=make_register_machine_sampler_coherent(16), chance_acc=1 / MODULUS, ceiling_acc=1.0,
    description="16-step register machine, corrupted traces follow one fixed wrong permutation per instruction",
    answer_pattern=r"\d+",
)
TASKS.update({
    f"state_machine_{steps}": Task(
        name=f"state_machine_{steps}", block_size=256, max_new_tokens=128,
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a random state machine for {steps} transitions",
        answer_pattern=r"\d+",
    )
    for steps, sampler in STATE_MACHINE_SAMPLERS.items()
})

def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"unknown task {name!r}; known: {known}") from None

import random

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

def sample_instances(sampler, n, seed, exclude=None):
    """Same uniqueness rule for a raw sampler (Boolean induced-rule path)."""
    state = random.getstate()
    random.seed(seed)
    seen = set() if exclude is None else set(exclude)
    out, tries = [], 0
    while len(out) < n:
        inst = sampler()
        tries += 1
        if tries > 200 * n:
            raise RuntimeError(f"prompt space too small for {n} unique instances")
        if inst.prompt in seen:
            continue
        seen.add(inst.prompt)
        out.append(inst)
    random.setstate(state)
    return out

import torch
from torch.utils.data import Dataset

TARGET_BUILDERS = {
    "outcome": lambda inst: f"{ANSWER_SEP}{inst.gold}\n",
    "answer_first": lambda inst: f"{ANSWER_SEP}{inst.gold} ; {inst.correct_trace}\n",
    "filler": lambda inst: f" {'.' * len(inst.correct_trace)}{ANSWER_SEP}{inst.gold}\n",
    "process": lambda inst: f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}\n",
    "corrupted": lambda inst: f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}\n",
}

def encode_pair(tokenizer, prompt: str, target: str, block_size: int):
    prompt_ids = tokenizer.encode(prompt)
    full = prompt_ids + tokenizer.encode(target)
    if len(full) > block_size:
        raise ValueError(f"{len(full)} tokens exceeds block_size={block_size}")
    n = len(full) - 1
    x = full[:-1]
    y = full[1:]
    mask = [0.0] * n
    for i in range(len(prompt_ids) - 1, n):
        mask[i] = 1.0
    if not (len(x) == len(y) == len(mask) == n):
        raise RuntimeError(f"encode_pair ranks {len(x), len(y), len(mask)} != {n}")
    return x, y, mask

class ContinuationDataset(Dataset):
    """Supervised continuation; loss is on the target tokens only."""

    def __init__(self, instances, tokenizer, block_size, targets):
        if len(instances) != len(targets):
            raise ValueError("instances and targets must align")
        if tokenizer.vocab_size > 256:
            raise ValueError("uint8 storage requires tokenizer.vocab_size <= 256")
        width = block_size - 1
        self.x = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.y = torch.full((len(instances), width), tokenizer.pad_id, dtype=torch.uint8)
        self.mask = torch.zeros((len(instances), width), dtype=torch.bool)
        max_len = 0
        for row, (inst, target) in enumerate(zip(instances, targets)):
            if isinstance(target, Instance):
                raise TypeError("pass target strings, not Instance")
            x, y, mask = encode_pair(tokenizer, inst.prompt, target, block_size)
            n = len(x)
            if n:
                self.x[row, :n] = torch.tensor(x, dtype=torch.uint8)
                self.y[row, :n] = torch.tensor(y, dtype=torch.uint8)
                self.mask[row, :n] = torch.tensor(mask, dtype=torch.bool)
            max_len = max(max_len, n)
        self.x = self.x[:, :max_len]
        self.y = self.y[:, :max_len]
        self.mask = self.mask[:, :max_len]
        if self.x.ndim != 2 or self.x.shape != self.y.shape or self.mask.shape != self.x.shape:
            raise RuntimeError(f"dataset tensors must be (N, T), got x={tuple(self.x.shape)}")

    def to(self, device):
        """Optional GPU-resident copies for small GPT sets (`data_on_device: true`)."""
        self.x = self.x.to(device, non_blocking=True)
        self.y = self.y.to(device, non_blocking=True)
        self.mask = self.mask.to(device, non_blocking=True)
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]

class SupervisionDataset(ContinuationDataset):
    def __init__(self, instances, task, mode: str):
        if mode not in TARGET_BUILDERS:
            raise ValueError(f"Unknown mode: {mode}")
        targets = [TARGET_BUILDERS[mode](inst) for inst in instances]
        super().__init__(instances, task.tokenizer, task.block_size, targets)

class RatioDataset(ContinuationDataset):
    """`answer_loss="clean"` drops the loss on the answer tokens of corrupted traces (trace-only loss)."""

    def __init__(self, instances, task, condition: str, rho=None, ratio_scores=None, answer_loss="all"):
        if answer_loss not in {"all", "clean"}:
            raise ValueError(f"unknown answer_loss: {answer_loss}")
        if condition not in {"outcome", "mixed_process"}:
            raise ValueError(f"unknown condition: {condition}")
        if condition == "mixed_process" and (rho is None or ratio_scores is None):
            raise ValueError("mixed_process requires rho and ratio_scores")
        targets = []
        for row, inst in enumerate(instances):
            if condition == "outcome":
                targets.append(f"{ANSWER_SEP}{inst.gold}\n")
            else:
                trace = inst.correct_trace if ratio_scores[row] < rho else inst.wrong_trace
                targets.append(f" {trace}{ANSWER_SEP}{inst.gold}\n")
        super().__init__(instances, task.tokenizer, task.block_size, targets)
        if condition == "mixed_process" and answer_loss == "clean":
            for row, inst in enumerate(instances):
                if ratio_scores[row] >= rho:
                    end = int(self.mask[row].nonzero().max()) + 1
                    self.mask[row, end - len(task.tokenizer.encode(f"{inst.gold}\n")):end] = False
