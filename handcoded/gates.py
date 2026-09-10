"""Reversible 4-bit gates: the exact Boolean rules that label traces."""
from dataclasses import dataclass
from itertools import permutations
import random

from handcoded.config import DEPTH, N_BITS


def bits(value, n_bits=N_BITS):
    """Integer → bit list; index 0 is the leftmost bit (paper convention)."""
    return [int(bit) for bit in f"{value:0{n_bits}b}"]


def state_text(state):
    """Bits → string such as `1000`."""
    return "".join(map(str, state))


def make_gate_names(n_bits=N_BITS):
    """All legal gate strings (52 at n_bits=4); `s01` and `s10` are distinct tokens."""
    gates = [f"x{i}" for i in range(n_bits)]
    for operation in ("c", "s"):
        gates.extend(f"{operation}{i}{j}" for i, j in permutations(range(n_bits), 2))
    gates.extend(f"t{i}{j}{k}" for i, j, k in permutations(range(n_bits), 3))
    return gates


def sample_gate(rng, n_bits=N_BITS):
    """Uniform family, then distinct bit indices."""
    operation = rng.choice("xcst")
    arity = {"x": 1, "c": 2, "s": 2, "t": 3}[operation]
    indices = rng.sample(range(n_bits), arity)
    return operation + "".join(map(str, indices))


def apply_gate(state, gate):
    """One reversible gate on a bit list (does not mutate `state`)."""
    next_state = state.copy()
    operation = gate[0]
    indices = list(map(int, gate[1:]))
    if operation == "x":
        (target,) = indices
        next_state[target] ^= 1
    elif operation == "c":
        control, target = indices
        next_state[target] ^= next_state[control]
    elif operation == "s":
        left, right = indices
        next_state[left], next_state[right] = next_state[right], next_state[left]
    elif operation == "t":
        control_a, control_b, target = indices
        next_state[target] ^= next_state[control_a] & next_state[control_b]
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return next_state


def phi(state_int, gate, n_bits=N_BITS):
    """Exact integer transition φ(s, g) used as the gold local rule."""
    return int(state_text(apply_gate(bits(state_int, n_bits), gate)), 2)


def sample_example(rng, depth=DEPTH, n_bits=N_BITS):
    """Start state, gate names, and gold state after every gate."""
    n_states = 2 ** n_bits
    start = rng.randrange(n_states)
    gates = [sample_gate(rng, n_bits) for _ in range(depth)]
    states = []
    current = start
    for gate in gates:
        current = phi(current, gate, n_bits)
        states.append(current)
    return start, gates, states


GATES = make_gate_names(N_BITS)


@dataclass
class Circuit:
    """One Boolean circuit example: start, gates, and intermediate states."""
    start: int
    gates: list[str]
    states: list[int]

    @property
    def answer(self):
        return self.states[-1]


def make_circuits(size, seed, depth=DEPTH, n_bits=N_BITS):
    rng = random.Random(seed)
    return [Circuit(*sample_example(rng, depth, n_bits)) for _ in range(size)]
