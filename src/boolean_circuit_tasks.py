import random

from src.dataclass import Instance


N_BITS = 4
BOOLEAN_CIRCUIT_DEPTHS = (4, 8, 12, 16, 20)


def _bits(value: int) -> list[int]:
    return [int(bit) for bit in f"{value:0{N_BITS}b}"]


def _state_text(state: list[int]) -> str:
    return "".join(str(bit) for bit in state)


def _sample_gate() -> str:
    """Sample a reversible Boolean gate over four wires.

    xk: flip wire k
    cij: controlled-NOT with control i and target j
    sij: swap wires i and j
    tijk: Toffoli with controls i,j and target k
    """
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
        target = int(gate[1])
        result[target] ^= 1
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
    """Execute a reversible Boolean circuit and return the final four bits.

    The initial state is uniform over 16 values. Every gate is a permutation,
    so the final state is also exactly uniform and chance accuracy is 1/16.
    """
    if n_gates < 1:
        raise ValueError("n_gates must be positive")

    def sample() -> Instance:
        start = _bits(random.randrange(2**N_BITS))
        gates = [_sample_gate() for _ in range(n_gates)]
        prompt = f"i{_state_text(start)};u{''.join(gates)}"

        state = start
        correct_steps = []
        for gate in gates:
            state = _apply_gate(state, gate)
            correct_steps.append(f"{gate}>{_state_text(state)}")
        gold = _state_text(state)

        # Follow a corrupted state trajectory. At every step, choose a state
        # different from the gate's correct output given the corrupted input
        # state. Thus every displayed transition is locally invalid.
        wrong_state = start
        wrong_steps = []
        for gate in gates:
            true_next = _apply_gate(wrong_state, gate)
            true_value = int(_state_text(true_next), 2)
            wrong_value = random.choice([value for value in range(2**N_BITS) if value != true_value])
            wrong_state = _bits(wrong_value)
            wrong_steps.append(f"{gate}>{_state_text(wrong_state)}")

        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong)
        assert correct != wrong
        return Instance(prompt, correct, wrong, gold)

    return sample


BOOLEAN_CIRCUIT_SAMPLERS = {
    depth: make_boolean_circuit_sampler(depth) for depth in BOOLEAN_CIRCUIT_DEPTHS
}


# A shared block size keeps model capacity identical across circuit depths.
BOOLEAN_CIRCUIT_BLOCK_SIZE = 320
BOOLEAN_CIRCUIT_MAX_NEW_TOKENS = {
    4: 64,
    8: 96,
    12: 144,
    16: 184,
    20: 224,
}


if __name__ == "__main__":
    random.seed(7)
    for depth, sampler in BOOLEAN_CIRCUIT_SAMPLERS.items():
        inst = sampler()
        print(f"\nboolean_circuit_{depth}")
        print(f"prompt: {inst.prompt}")
        print(f"correct_trace: {inst.correct_trace}")
        print(f"wrong_trace: {inst.wrong_trace}")
        print(f"gold: {inst.gold}")
        