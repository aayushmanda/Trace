import random

from src.data.dataclass import Instance


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
