import random
import string

from src.dataclass import Instance


WORD_INDEX_LENGTHS = (16, 24, 32, 48, 64)
REPEAT_PROBABILITY = 0.35


def make_hard_word_index_sampler(length: int, repeat_probability: float = REPEAT_PROBABILITY):
    """Pure first-occurrence lookup with controlled length and distractors."""
    if length < 2:
        raise ValueError("length must be at least 2")
    if not 0.0 <= repeat_probability <= 1.0:
        raise ValueError("repeat_probability must lie in [0, 1]")

    def sample() -> Instance:
        query = random.choice(string.ascii_lowercase)
        gold = random.randrange(length)
        non_query = [ch for ch in string.ascii_lowercase if ch != query]

        # The query cannot occur before gold, occurs exactly at gold, and is
        # deliberately repeated after gold to create distracting matches.
        chars = [random.choice(non_query) for _ in range(gold)] + [query]
        for _ in range(gold + 1, length):
            chars.append(query if random.random() < repeat_probability else random.choice(non_query))
        word = "".join(chars)
        assert len(word) == length and word.index(query) == gold

        prompt = f"{word};{query}"
        correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))

        # A cyclic index shift is a guaranteed derangement. Characters are
        # also changed, while the multiset of index widths keeps trace length
        # identical to the correct trace.
        shift = random.randrange(1, length)
        wrong_indices = [(i + shift) % length for i in range(length)]
        wrong_pairs = []
        for i, true_char in enumerate(word):
            wrong_index = wrong_indices[i]
            forbidden = {true_char, word[wrong_index]}
            wrong_char = random.choice([ch for ch in string.ascii_lowercase if ch not in forbidden])
            wrong_pairs.append(f"{wrong_index}{wrong_char}")
        wrong = " ".join(wrong_pairs)

        assert all(wrong_indices[i] != i for i in range(length))
        assert sorted(wrong_indices) == list(range(length))
        assert len(correct) == len(wrong) and correct != wrong
        return Instance(prompt, correct, wrong, str(gold))

    return sample


HARD_WORD_INDEX_SAMPLERS = {
    length: make_hard_word_index_sampler(length) for length in WORD_INDEX_LENGTHS
}


# Same context window for every length keeps model capacity comparable.
HARD_WORD_INDEX_BLOCK_SIZE = 384
HARD_WORD_INDEX_MAX_NEW_TOKENS = {
    16: 64,
    24: 96,
    32: 128,
    48: 192,
    64: 256,
}


if __name__ == "__main__":
    random.seed(7)
    for length, sampler in HARD_WORD_INDEX_SAMPLERS.items():
        inst = sampler()
        print(f"\nword_index_len{length}")
        print(f"prompt: {inst.prompt}")
        print(f"correct_trace: {inst.correct_trace}")
        print(f"wrong_trace: {inst.wrong_trace}")
        print(f"gold: {inst.gold}")