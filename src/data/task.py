import random
import string

from src.data.dataclass import Instance


def _sample_word_index() -> Instance:
    L = random.randint(7, 14)
    gold = random.randrange(L)
    c = random.choice(string.ascii_lowercase)
    chars = [random.choice([ch for ch in string.ascii_lowercase if ch != c]) for _ in range(gold)]
    chars.append(c)
    chars.extend(random.choice(string.ascii_lowercase) for _ in range(gold + 1, L))
    word = "".join(chars)
    prompt = f"{word};{c}"
    correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))
    wrong_indices = list(range(L))
    while True:
        random.shuffle(wrong_indices)
        if all(wrong_indices[i] != i for i in range(L)):
            break
    wrong_pairs = []
    for i, true_char in enumerate(word):
        wrong_idx = wrong_indices[i]
        forbidden = {true_char, word[wrong_idx]}
        wrong_char = random.choice([ch for ch in string.ascii_lowercase if ch not in forbidden])
        wrong_pairs.append(f"{wrong_idx}{wrong_char}")
    return Instance(prompt, correct, " ".join(wrong_pairs), str(gold))


def _perturb_like(n: int) -> int:
    digits = str(n)
    d = len(digits)
    zeros = d - len(digits.rstrip("0"))
    step = 10 ** zeros
    lo = 10 ** (d - 1) if d > 1 else 0
    hi = 10 ** d - 1
    for _ in range(40):
        m = random.randint(lo, hi) // step * step
        s = str(m)
        if m != n and len(s) == d and len(s) - len(s.rstrip("0")) == zeros:
            return m
    return n + 1


def _sample_multiply() -> Instance:
    a = random.randint(10, 99)
    tens, units = random.randint(1, 9) * 10, random.randint(1, 9)
    b = tens + units
    prompt = f"{a}*{b}"
    p_u, p_t = units * a, tens * a
    total = a * b
    correct = f"{units}*{a}={p_u} {tens}*{a}={p_t} {p_u}+{p_t}={total}"
    w_total = total
    for _ in range(20):
        w_u, w_t = _perturb_like(p_u), _perturb_like(p_t)
        w_total = w_u + w_t
        if w_total != total:
            break
    wrong = f"{units}*{a}={w_u} {tens}*{a}={w_t} {w_u}+{w_t}={w_total}"
    return Instance(prompt, correct, wrong, str(total))


def _sample_count_char() -> Instance:
    L = random.randint(6, 14)
    word = "".join(random.choice(string.ascii_lowercase) for _ in range(L))
    query = random.choice(word)
    running, counts = 0, []
    for ch in word:
        running += int(ch == query)
        counts.append(running)
    correct = " ".join(f"{ch}" for ch, n in zip(word, counts))
    wrong_counts = [0] + counts[:-1]
    if wrong_counts == counts:
        wrong_counts = counts[1:] + [counts[-1]]
    wrong = " ".join("x" for _ in wrong_counts)
    return Instance(f"{word};{query}", correct, wrong, str(running))


def _sample_sort_letters() -> Instance:
    letters = random.sample(string.ascii_lowercase, 10)
    word = "".join(letters)
    gold = "".join(sorted(letters))
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        chosen = min(remaining)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    correct = " ".join(steps)
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        correct_choice = min(remaining)
        chosen = random.choice([x for x in remaining if x != correct_choice])
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    return Instance(word, correct, " ".join(steps), gold)
