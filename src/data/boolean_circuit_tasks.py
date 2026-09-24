import random

from src.data.dataclass import Instance

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


def canonical_gate_names():
    """One representative name per distinct permutation, in a fixed order.

    The 52 operation strings realize only 34 distinct permutations: swaps and
    control-symmetric Toffolis are aliased (`s01`/`s10`). Taking the first name
    of each permutation gives an alias-free set, so a subset of size M has
    exactly M operation strings AND M distinct transition functions.
    """
    from handcoded.gates import make_gate_names
    seen, out = {}, []
    for name in make_gate_names(N_BITS):
        key = tuple(
            int(_state_text(_apply_gate(_bits(v), name)), 2) for v in range(2 ** N_BITS)
        )
        if key not in seen:
            seen[key] = name
            out.append(name)
    return out


def make_boolean_circuit_sampler_subset(n_gates: int, gate_names):
    """Symmetric-corruption sampler restricted to a fixed operation subset."""
    gate_names = list(gate_names)
    if n_gates < 1 or not gate_names:
        raise ValueError("need a positive depth and a non-empty gate set")

    def sample() -> Instance:
        start = _bits(random.randrange(2 ** N_BITS))
        gates = [random.choice(gate_names) for _ in range(n_gates)]
        prompt = f"i{_state_text(start)};u{''.join(gates)}"
        state = start
        correct_steps = []
        for gate in gates:
            state = _apply_gate(state, gate)
            correct_steps.append(f"{gate}>{_state_text(state)}")
        gold = _state_text(state)
        wrong_state, wrong_steps = start, []
        for gate in gates:
            true_next = _apply_gate(wrong_state, gate)
            true_value = int(_state_text(true_next), 2)
            wrong_value = random.choice(
                [v for v in range(2 ** N_BITS) if v != true_value]
            )
            wrong_state = _bits(wrong_value)
            wrong_steps.append(f"{gate}>{_state_text(wrong_state)}")
        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


_CONCENTRATION_SELECTOR_SEED = "boolean_circuit_concentration_selector_v1"


def make_boolean_circuit_sampler_mixture(n_gates: int, concentration: float):
    """Wrong traces mix the symmetric and coherent corruption laws.

    At every corrupted step the wrong successor is the gate's fixed XOR image
    with probability `concentration` and a uniform wrong state otherwise, so
    the induced wrong-successor law is the R^(lambda) of the concentration
    interpolation, with strongest wrong mass
    c_lambda = lambda + (1 - lambda) / (2 ** N_BITS - 1).

    The uniform candidate is drawn from the global stream at every step
    regardless of which branch is taken, so make_boolean_circuit_sampler's
    draw sequence is reproduced exactly and every concentration shares one
    prompt sequence at a fixed train seed. The mixture selector therefore has
    to come from somewhere else: it uses its own prompt-keyed RNG, which also
    keeps a given prompt's corruption pattern fixed across concentrations.
    """
    if n_gates < 1:
        raise ValueError("n_gates must be positive")
    if not 0.0 <= concentration <= 1.0:
        raise ValueError("concentration must lie in [0, 1]")

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
        selector = random.Random(f"{_CONCENTRATION_SELECTOR_SEED}:{prompt}")
        wrong_state = start
        wrong_steps = []
        for gate in gates:
            true_next = _apply_gate(wrong_state, gate)
            true_value = int(_state_text(true_next), 2)
            uniform_wrong = random.choice(
                [v for v in range(2 ** N_BITS) if v != true_value]
            )
            coherent_wrong = true_value ^ _coherent_wrong_mask(gate)
            wrong_value = (
                coherent_wrong if selector.random() < concentration else uniform_wrong
            )
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
