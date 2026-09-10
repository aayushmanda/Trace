from typing import List
import string


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi.get("\n", None)

    def encode(self, s: str) -> List[int]:
        unknown = set(s) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters in input string: {unknown}")
        ids = [self.stoi[c] for c in s]
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("encoded id outside vocab")
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


if __name__ == "__main__":
    chars = string.ascii_lowercase + string.digits + " ;:.->+=*\n"
    tokenizer = CharTokenizer(chars)
    encoded = tokenizer.encode("hello world")
    print(len(chars), encoded, tokenizer.decode(encoded))
