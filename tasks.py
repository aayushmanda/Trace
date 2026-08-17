"""The task family: tokenizer, surface form, samplers, reference levels.

Every task presents the same surface form, so the ratio mix, the loss mask, the
evaluator and the answer-forcing probes are literally the same code across the
family:

    <prompt> <trace> : <gold>\\n

`correct_trace` is the genuine step-by-step computation; `wrong_trace` keeps the
same shape, length and token statistics but destroys the computation. In both
cases the text after " : " is the *true* answer, so the only thing the ratio
sweep varies is whether the reasoning that precedes the answer is valid.

Each task owns a tokenizer over exactly the characters it needs, so adding a
task with new symbols (`*`, `=`, `>`) never perturbs the vocabulary -- and
therefore the parameter count -- of the tasks already measured.

Adding a task = write a sampler returning an Instance, register it in TASKS,
then run `python main.py verify` to measure its chance floor and Bayes ceiling.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

ANSWER_SEP = " : "

# The three renderings of one instance. `no_think` is condition C: no trace.
TRACE_MODES = ("correct_think", "wrong_think", "no_think")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer over a fixed alphabet, plus one PAD id."""

    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi["\n"]

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One sampled problem, with both trace variants sharing the same gold."""

    prompt: str
    correct_trace: str
    wrong_trace: str
    gold: str


@dataclass
class Task:
    name: str
    chars: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float
    ceiling_acc: float
    description: str = ""
    # Regex for a well-formed answer token, used to parse generations.
    answer_pattern: str = r"\d+"
    # Bayes-optimal success probability for one instance; None means the prompt
    # fully determines the answer (probability 1).
    bayes_prob: Optional[Callable[[Instance], float]] = None
    tokenizer: CharTokenizer = field(init=False)

    def __post_init__(self):
        self.tokenizer = CharTokenizer(self.chars)

    # -- surface form --------------------------------------------------------

    def render(self, inst: Instance, mode: str = "correct_think"):
        """(prompt, answer_part) for one of TRACE_MODES. Training sees both."""
        if mode == "correct_think":
            return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "wrong_think":
            return inst.prompt, f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "no_think":
            return inst.prompt, f" {inst.gold}"
        raise ValueError(f"unknown mode {mode!r}; have {TRACE_MODES}")

    def context(self, inst: Instance, mode: str) -> str:
        """The exact prefix the model is conditioned on at evaluation time.

        This is the single definition of what each evaluation mode means:

          free           -- the bare prompt; the model must produce its own
                            trace, the delimiter and then the answer (the
                            end-to-end metric).
          forced_correct -- prompt + the clean trace + " : ": *answer forcing*.
                            Trace generation, delimiter emission and the token
                            budget are all removed from the measurement, so
                            what is left is the answer-position computation.
          forced_wrong   -- the same, with the corrupted trace. If accuracy
                            here tracks forced_correct, the model is not
                            reading the trace it was given.
          direct         -- prompt + " ": the condition-C form, where the
                            answer follows the prompt with no trace at all.
        """
        if mode == "free":
            return inst.prompt
        if mode == "forced_correct":
            return f"{inst.prompt} {inst.correct_trace}{ANSWER_SEP}"
        if mode == "forced_wrong":
            return f"{inst.prompt} {inst.wrong_trace}{ANSWER_SEP}"
        if mode == "direct":
            return f"{inst.prompt} "
        raise ValueError(f"unknown eval mode {mode!r}")

    def extract_answer(self, generated_tail: str, first: bool = False) -> Optional[str]:
        """Pulls the predicted answer out of everything the model emitted.

        Under free generation the tail contains the model's own trace, so the
        rule is: look after the last ':', then take the *last* token matching
        the answer pattern. Under answer forcing the tail starts at the answer
        position already, so the *first* match is the answer and anything after
        it is overrun -- hence `first`.
        """
        tail = generated_tail.split("\n")[0]
        segment = tail.split(":")[-1] if ":" in tail else tail
        found = re.findall(self.answer_pattern, segment)
        if not found:
            return None
        return found[0] if first else found[-1]

    # -- introspection used by the cache and the pre-flight checks -----------

    def fingerprint(self, n: int = 64) -> str:
        """Short hash of this task's rendered output.

        Keys the dataset cache, so editing a sampler, the trace format or the
        vocabulary invalidates the cached pools instead of silently reusing
        data generated by the old definition.
        """
        state = random.getstate()
        random.seed(12345)
        h = hashlib.sha1()
        for _ in range(n):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                h.update(f"{prompt}{answer}\n".encode())
        random.setstate(state)
        h.update(f"{self.block_size}|{''.join(self.tokenizer.chars)}".encode())
        return h.hexdigest()[:10]

    def max_example_len(self, n_probe: int = 4000) -> int:
        """Longest encoded example over n_probe samples (truncation check)."""
        state = random.getstate()
        random.seed(0)
        longest = 0
        for _ in range(n_probe):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                longest = max(
                    longest, len(self.tokenizer.encode(prompt + answer + "\n"))
                )
        random.setstate(state)
        return longest


# ---------------------------------------------------------------------------
# 1. word_index -- report the position of a queried letter.
# ---------------------------------------------------------------------------

WORD_INDEX_LENGTHS = (4, 10)


def _sample_word_index() -> Instance:
    L = random.randint(*WORD_INDEX_LENGTHS)
    word = "".join(random.choice(string.ascii_lowercase) for _ in range(L))
    gold = random.randrange(L)
    prompt = f"{word};{word[gold]}"

    correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))

    fake_indices = list(range(L))
    random.shuffle(fake_indices)
    wrong = " ".join(
        f"{i}{random.choice(string.ascii_lowercase)}" for i in fake_indices
    )

    return Instance(prompt, correct, wrong, str(gold))


def _word_index_bayes(inst: Instance) -> float:
    """The queried letter may repeat; even a perfect reader must then guess."""
    word, _, c = inst.prompt.partition(";")
    return 1.0 / word.count(c)


# ---------------------------------------------------------------------------
# 2. sort_letters -- selection sort over the letters of a word.
# ---------------------------------------------------------------------------

SORT_LENGTHS = (4, 7)


def _sample_sort_letters() -> Instance:
    L = random.randint(*SORT_LENGTHS)
    letters = random.sample(string.ascii_lowercase, L)  # distinct -> unique ranks
    word = "".join(letters)
    gold = "".join(sorted(letters))

    # Correct: repeatedly show the remaining multiset and extract its minimum.
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        chosen = min(remaining)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    correct = " ".join(steps)

    # Wrong: same step count and lengths, but the displayed remainder is
    # scrambled and the extracted letter is arbitrary rather than minimal.
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        shown = remaining[:]
        random.shuffle(shown)
        chosen = random.choice(remaining)
        steps.append(f"{''.join(shown)}>{chosen}")
        remaining.remove(chosen)
    wrong = " ".join(steps)

    return Instance(word, correct, wrong, gold)


# ---------------------------------------------------------------------------
# 3. multiply -- two-digit multiplication via partial products.
# ---------------------------------------------------------------------------


def _perturb_number(n: int) -> int:
    """A different number with the same digit count, so shape is preserved."""
    d = len(str(n))
    lo = 10 ** (d - 1) if d > 1 else 0
    hi = 10**d - 1
    for _ in range(20):
        m = random.randint(lo, hi)
        if m != n:
            return m
    return n


def _sample_multiply() -> Instance:
    a = random.randint(10, 99)
    # Both digits of b are non-zero so every instance needs two real partial
    # products; a units digit of 0 would hand the model a free step.
    tens, units = random.randint(1, 9) * 10, random.randint(1, 9)
    b = tens + units
    prompt = f"{a}*{b}"

    p_u, p_t = units * a, tens * a
    total = a * b
    correct = f"{units}*{a}={p_u} {tens}*{a}={p_t} {p_u}+{p_t}={total}"

    w_u, w_t = _perturb_number(p_u), _perturb_number(p_t)
    w_total = _perturb_number(total)
    wrong = f"{units}*{a}={w_u} {tens}*{a}={w_t} {w_u}+{w_t}={w_total}"

    return Instance(prompt, correct, wrong, str(total))


# ---------------------------------------------------------------------------
# 4. count_char -- running tally of a queried letter.
# ---------------------------------------------------------------------------

# A small alphabet over long words spreads the tally out; with 26 letters the
# answer is nearly always 1 and a constant predictor sits at ~50%.
COUNT_LENGTHS = (6, 14)
COUNT_ALPHABET = string.ascii_lowercase[:3]


def _sample_count_char() -> Instance:
    L = random.randint(*COUNT_LENGTHS)
    word = "".join(random.choice(COUNT_ALPHABET) for _ in range(L))
    # Query a letter that actually occurs (as word_index does), otherwise most
    # golds are 0 and a constant predictor clears the floor by a mile.
    query = random.choice(word)
    prompt = f"{word};{query}"

    running, counts = 0, []
    for ch in word:
        running += ch == query
        counts.append(running)
    correct = " ".join(f"{ch}{n}" for ch, n in zip(word, counts))

    # Same multiset of counts, so the wrong trace matches the correct one
    # character for character in length and digit statistics -- only the
    # letter/tally correspondence and the monotonicity are destroyed.
    shuffled = counts[:]
    random.shuffle(shuffled)
    wrong = " ".join(f"{random.choice(COUNT_ALPHABET)}{n}" for n in shuffled)

    return Instance(prompt, correct, wrong, str(running))


# ---------------------------------------------------------------------------
# Registry.
#
#   chance  = accuracy of the best CONSTANT predictor (the marginal modal
#             answer). This is what a model that ignores its input can reach,
#             so it is where the rho=0 end of the curve should land. For
#             word_index this reduces to P(gold=0) = E[1/L] = 0.1565, matching
#             the closed-form chance level of Corollary 2.
#
#   ceiling = E[ Bayes-optimal success probability ]. It is 1.0 whenever the
#             prompt determines the answer, and below 1.0 only for word_index,
#             where the queried letter may repeat: E[1/multiplicity] = 0.8924,
#             matching the 0.89259 of Corollary 2.
#
# Both are measured by estimate_reference_levels(); `main.py verify` re-measures
# them and fails loudly if a sampler edit has made the declared values stale.
# ---------------------------------------------------------------------------

DIGITS = string.digits
LOWER = string.ascii_lowercase

TASKS: Dict[str, Task] = {
    "word_index": Task(
        name="word_index",
        chars=LOWER + DIGITS + " ;:\n",
        block_size=64,
        max_new_tokens=35,
        sample=_sample_word_index,
        chance_acc=0.15652,
        ceiling_acc=0.89259,
        description="report the index of a queried letter; trace enumerates (i, char)",
        bayes_prob=_word_index_bayes,
    ),
    "sort_letters": Task(
        name="sort_letters",
        chars=LOWER + " :>\n",
        block_size=72,
        max_new_tokens=60,
        sample=_sample_sort_letters,
        chance_acc=0.00005,
        ceiling_acc=1.0,
        description="alphabetize a word; trace is selection sort (remainder > min)",
        answer_pattern=r"[a-z]+",
    ),
    "multiply": Task(
        name="multiply",
        chars=DIGITS + " :*+=\n",
        block_size=64,
        max_new_tokens=48,
        sample=_sample_multiply,
        chance_acc=0.00183,
        ceiling_acc=1.0,
        description="two-digit multiplication; trace is partial products then their sum",
    ),
    "count_char": Task(
        name="count_char",
        chars=LOWER + DIGITS + " ;:\n",
        block_size=76,
        max_new_tokens=52,
        sample=_sample_count_char,
        chance_acc=0.23124,
        ceiling_acc=1.0,
        description="count occurrences of a letter; trace is the running tally",
    ),
}


def get_task(name_or_task) -> Task:
    """Resolves a name to a Task. Unknown names raise -- never silently default."""
    if isinstance(name_or_task, Task):
        return name_or_task
    if name_or_task not in TASKS:
        raise KeyError(f"unknown task {name_or_task!r}; have {sorted(TASKS)}")
    return TASKS[name_or_task]


def estimate_reference_levels(task, n: int = 200_000):
    """Monte-Carlo (chance, ceiling) for a task -- the two ends of its y-axis."""
    task = get_task(task)
    state = random.getstate()
    random.seed(1234)

    marginal: Counter = Counter()
    bayes_total = 0.0
    for _ in range(n):
        inst = task.sample()
        marginal[inst.gold] += 1
        bayes_total += 1.0 if task.bayes_prob is None else task.bayes_prob(inst)
    random.setstate(state)

    return marginal.most_common(1)[0][1] / n, bayes_total / n
