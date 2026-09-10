import random
import string

from src.data.dataclass import Instance

WORD_INDEX_LENGTHS = (16, 24, 32, 48, 64)
REPEAT_PROBABILITY = 0.35


def make_hard_word_index_sampler(length: int, repeat_probability: float = REPEAT_PROBABILITY):
    if length < 2:
        raise ValueError("length must be at least 2")
    if not 0.0 <= repeat_probability <= 1.0:
        raise ValueError("repeat_probability must lie in [0, 1]")

    def sample() -> Instance:
        query = random.choice(string.ascii_lowercase)
        gold = random.randrange(length)
        non_query = [ch for ch in string.ascii_lowercase if ch != query]
        chars = [random.choice(non_query) for _ in range(gold)] + [query]
        for _ in range(gold + 1, length):
            chars.append(query if random.random() < repeat_probability else random.choice(non_query))
        word = "".join(chars)
        prompt = f"{word};{query}"
        correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))
        shift = random.randrange(1, length)
        wrong_indices = [(i + shift) % length for i in range(length)]
        wrong_pairs = []
        for i, true_char in enumerate(word):
            wrong_index = wrong_indices[i]
            forbidden = {true_char, word[wrong_index]}
            wrong_char = random.choice([ch for ch in string.ascii_lowercase if ch not in forbidden])
            wrong_pairs.append(f"{wrong_index}{wrong_char}")
        wrong = " ".join(wrong_pairs)
        return Instance(prompt, correct, wrong, str(gold))

    return sample


HARD_WORD_INDEX_SAMPLERS = {n: make_hard_word_index_sampler(n) for n in WORD_INDEX_LENGTHS}
HARD_WORD_INDEX_BLOCK_SIZE = 384
HARD_WORD_INDEX_MAX_NEW_TOKENS = {16: 64, 24: 96, 32: 128, 48: 192, 64: 256}
