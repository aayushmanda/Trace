import random

from src.dataclass import Instance


def _derangement(n: int) -> list[int]:
    """Random permutation with no self-transition."""
    while True:
        values = list(range(n))
        random.shuffle(values)
        if all(i != value for i, value in enumerate(values)):
            return values


def make_state_machine_sampler(n_states: int = 16, n_steps: int = 8):
    """Build a sampler for deterministic finite-state-machine execution.

    Every example contains two newly sampled transition tables, a start state,
    and an action sequence. The correct trace exposes each local transition.
    The wrong trace has identical syntax and length but violates every local
    transition; its terminal answer remains correct through Instance.gold.
    """
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

        # Compact prompt: each action table lists the destination of states
        # 00, 01, ..., followed by the start state and action program.
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
            wrong_next = random.choice([value for value in range(n_states) if value != true_next])
            wrong_steps.append(f"{state(wrong_current)}{action_chars[action]}{state(wrong_next)}")
            wrong_current = wrong_next

        correct = " ".join(correct_steps)
        wrong = " ".join(wrong_steps)
        assert len(correct) == len(wrong)
        return Instance(prompt, correct, wrong, gold)

    return sample


_sample_state_machine_2 = make_state_machine_sampler(n_states=16, n_steps=2)
_sample_state_machine_4 = make_state_machine_sampler(n_states=16, n_steps=4)
_sample_state_machine_8 = make_state_machine_sampler(n_states=16, n_steps=8)
_sample_state_machine_12 = make_state_machine_sampler(n_states=16, n_steps=12)
_sample_state_machine_16 = make_state_machine_sampler(n_states=16, n_steps=16)
_sample_state_machine_20 = make_state_machine_sampler(n_states=16, n_steps=20)

STATE_MACHINE_SAMPLERS = {
    2: _sample_state_machine_2,
    4: _sample_state_machine_4,
    8: _sample_state_machine_8,
    12: _sample_state_machine_12,
    16: _sample_state_machine_16,
    20: _sample_state_machine_20,
}