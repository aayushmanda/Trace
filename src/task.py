import random
import string
from src.dataclass import Instance


### word_index


def _sample_word_index() -> Instance:
    L = random.randint(7, 14)  

    # Sample the desired first-occurrence index uniformly
    gold = random.randrange(L)

    # Sample the queried character
    c = random.choice(string.ascii_lowercase)

    # Build the word so c cannot appear before gold
    chars = []

    for i in range(gold):
        chars.append(random.choice([ch for ch in string.ascii_lowercase if ch != c]))

    # First occurrence of c
    chars.append(c)

    # Anything is allowed after gold, including c appearing again
    for i in range(gold + 1, L):
        chars.append(random.choice(string.ascii_lowercase))

    word = "".join(chars)

    # Sanity check
    assert word.index(c) == gold

    prompt = f"{word};{c}"

    # Correct trace: true index-character mapping
    correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))

    # Wrong trace: same indices/order, but every character is incorrect
    wrong_pairs = []

    for i, true_char in enumerate(word):
        wrong_char = random.choice([ch for ch in string.ascii_lowercase if ch != true_char])
        wrong_pairs.append(f"{i}{wrong_char}")

    wrong = " ".join(wrong_pairs)

    return Instance(prompt, correct, wrong, str(gold))

### Multiplcation task

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
    for _ in range(20):
        w_u, w_t = _perturb_like(p_u), _perturb_like(p_t)
        w_total = w_u + w_t
        if w_total != total:
            break
    wrong = f"{units}*{a}={w_u} {tens}*{a}={w_t} {w_u}+{w_t}={w_total}"
    return Instance(prompt, correct, wrong, str(total))


### count_character
def _sample_count_char() -> Instance:
    L = random.randint(6, 14)
    alphabet = string.ascii_lowercase
    word = "".join(random.choice(alphabet) for _ in range(L))
    query = random.choice(word)
    prompt = f"{word};{query}"
    running, counts = 0, []
    for ch in word:
        running += int(ch == query)
        counts.append(running)
    correct = " ".join(f"{ch}" for ch, n in zip(word, counts))


    wrong_counts = [0] + counts[:-1]
    if wrong_counts == counts:
        wrong_counts = counts[1:] + [counts[-1]]
    wrong = " ".join(f"{"x"}" for ch, n in zip(word, wrong_counts))

    assert wrong_counts != counts, "Wrong trace collided with correct trace"
    return Instance(prompt, correct, wrong, str(running))


### sort chars

def _sample_sort_letters() -> Instance:
    L = 10 #random.randint(7, 10)
    letters = random.sample(string.ascii_lowercase, L)
    word = "".join(letters)
    gold = "".join(sorted(letters))

    remaining, steps = list(letters), []
    while len(remaining) > 1:
        chosen = min(remaining)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    correct = " ".join(steps)

    # Same state-transition structure; every selected item violates selection-sort.
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        correct_choice = min(remaining)
        wrong_choices = [x for x in remaining if x != correct_choice]
        chosen = random.choice(wrong_choices)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    wrong = " ".join(steps)

    return Instance(word, correct, wrong, gold)


if __name__ == "__main__":
    for k, v in vars(_sample_word_index()).items():
        print(f"{k}: {v}")
        print("-"*3)
    for k, v in vars(_sample_multiply()).items():
        print(f"{k}: {v}")
        print("-"*3)        
    for k, v in vars(_sample_count_char()).items():
        print(f"{k}: {v}")
        print("-"*3)
    for k, v in vars(_sample_sort_letters()).items():
        print(f"{k}: {v}")
