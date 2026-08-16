"""Character-level tokenizer for the word-indexing task."""

import string

chars = sorted(list(set(string.ascii_lowercase + string.digits + " ;:\n")))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

PAD_ID = len(chars)
vocab_size = len(chars) + 1
NEWLINE_ID = stoi["\n"]


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids if i in itos)
