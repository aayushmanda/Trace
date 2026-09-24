"""Semantic tokens: one token per 4-bit state and per gate (not character GPT)."""
import math

from handcoded import config
from handcoded.gates import active_gate_names


class CircuitTokenizer:
    """Map states `Sxxxx`, gates, and separators to ids. Prompt is start + gates + SEP."""

    def __init__(self, gates, n_states):
        self.n_states = n_states
        self.gates = list(gates)
        width = max(1, int(math.log2(n_states)))
        self.tokens = [f"S{state:0{width}b}" for state in range(n_states)] + self.gates
        self.tokens += ["<SEP>", "<COLON>", "<EOS>", "<PAD>"]
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.sep, self.colon, self.eos, self.pad = [
            self.ids[token] for token in ("<SEP>", "<COLON>", "<EOS>", "<PAD>")
        ]

    def prompt(self, circuit):
        return [circuit.start] + [self.ids[gate] for gate in circuit.gates] + [self.sep]

    def continuation(self, circuit, mode):
        ending = [self.colon, circuit.answer, self.eos]
        if mode == "outcome":
            return ending
        if mode == "process":
            trace = []
            for gate, state in zip(circuit.gates, circuit.states):
                trace.extend([self.ids[gate], state])
            return trace + ending
        raise ValueError(f"Unknown supervision mode: {mode}")

    def decode(self, ids):
        return " ".join(self.tokens[int(index)] for index in ids)


def make_tokenizer(n_bits=None, n_gates=None):
    n_bits = config.N_BITS if n_bits is None else n_bits
    return CircuitTokenizer(active_gate_names(n_bits, n_gates), 2 ** n_bits)
