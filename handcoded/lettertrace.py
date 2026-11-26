"""Two-block causal Transformer for sequential counting & exact executors (Proposition 1, Appendix B).

Implements the shared normalization-free two-block architecture (Assumption A.1), explicit exact
direct-answer (theta*_out) and step-by-step (theta*_step) parameter constructions, and paired step vs.
outcome training from identical random initializations.

CLI usage (key=value overrides):
    uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
    uv run python handcoded/lettertrace.py task=count word_len=1000 mod=2 handcoded_only=True train_size=1 test_size=532
"""

from __future__ import annotations

import ast
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm as _tqdm

TASKS = ("count", "first", "shell", "first_cup", "lights", "lights_parity")
SHELL_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_HEADS = 3

STRING_KEYS = frozenset(
    {
        "device",
        "task",
        "corrupt",
        "layout",
        "save_dir",
        "tag",
        "modes",
        "opt",
        "init",
        "seed_signs",
        "seed_bias",
    }
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_tqdm(iterable: Any, **kwargs: Any) -> Any:
    disable = kwargs.pop("disable", os.environ.get("TRACE_TQDM") == "0")
    return _tqdm(iterable, disable=disable, **kwargs)


@dataclass(frozen=True)
class ResidualSlots:
    """Named coordinate slices inside the shared residual stream."""

    tok_letter: slice
    tok_value: slice
    pos: slice
    query: slice
    match: slice
    prev: slice
    match_token: slice
    count: slice
    s0: slice
    out: slice
    total_width: int

    @classmethod
    def build(cls, alphabet: int, num_values: int, max_seq_len: int) -> ResidualSlots:
        sizes = (
            ("tok_letter", alphabet),
            ("tok_value", num_values),
            ("pos", max_seq_len),
            ("query", alphabet),
            ("match", 1),
            ("prev", num_values),
            ("match_token", 1),
            ("count", 1),
            ("s0", num_values),
            ("out", num_values),
        )
        slices: dict[str, slice] = {}
        offset = 0
        for name, width in sizes:
            slices[name] = slice(offset, offset + width)
            offset += width
        return cls(**slices, total_width=offset)


@dataclass
class Config:
    """Complete configuration and derived dimensions for lettertrace experiments."""

    task: str = "count"
    alphabet: int = 4
    p_match: float = 0.0
    word_len: int = 12
    mod: int = 2
    rho: float = 1.0
    corrupt: str = "scatter"
    slip_at: int = 0
    layout: str = "block"
    n_blocks: int = 2
    every_k: int = 1
    opt: str = "adamw"
    train_size: int = 20_000
    test_size: int = 1_000
    steps: int = 8_000
    batch_size: int = 256
    lr: float = 1e-3
    init_std: float = 0.02
    zero_readout: bool = False
    init: str = "random"
    seed_signs: str = "handcoded"
    seed_bias: str = "zero"
    seed: int = 42
    save_every: int = 0
    save_dir: str = "ckpt"
    tag: str = ""
    modes: str = "process,outcome"
    eval_every: int = 0
    clean_only: bool = False
    gap: float = 0.0
    perturb: tuple[float, ...] = ()
    perturb_draws: int = 5
    handcoded_only: bool = False
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.task in ("shell", "first_cup"):
            self.alphabet, self.mod = 10, 5
        elif self.task == "lights":
            self.alphabet, self.mod = 4, 16
        elif self.task == "lights_parity":
            self.alphabet, self.mod = 4, 2

        self._validate()

    def _validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; expected one of {TASKS}")
        if self.corrupt not in ("scatter", "shift", "slip"):
            raise ValueError(f"Unknown corrupt mode {self.corrupt!r}")
        if self.n_blocks < 2:
            raise ValueError(f"n_blocks must be >= 2 (got {self.n_blocks})")
        if self.layout not in ("stream", "block"):
            raise ValueError(f"Unknown layout {self.layout!r}")
        if self.opt not in ("adamw", "sgd", "adam"):
            raise ValueError(f"Unknown optimizer {self.opt!r}")
        if self.init not in ("random", "handcoded_dirs"):
            raise ValueError(f"Unknown init {self.init!r}")
        if self.seed_signs not in ("handcoded", "random"):
            raise ValueError(f"Unknown seed_signs {self.seed_signs!r}")
        if self.seed_bias not in ("zero", "solution"):
            raise ValueError(f"Unknown seed_bias {self.seed_bias!r}")
        if self.p_match != 0 and not (0.0 < self.p_match < 1.0 and self.alphabet >= 2):
            raise ValueError("p_match must lie in (0, 1) with alphabet >= 2")
        if not self.train_modes or not set(self.train_modes) <= {"process", "outcome"}:
            raise ValueError(f"modes must be process and/or outcome (got {self.modes!r})")
        if self.task != "count" and self.rho != 1.0:
            raise ValueError("Corruption is only defined for task='count'")
        if self.every_k < 1 or self.word_len % self.every_k != 0:
            raise ValueError("every_k must be >= 1 and divide word_len")
        if self.every_k != 1 and self.layout != "block":
            raise ValueError("every_k > 1 is only defined for layout='block'")

    @property
    def train_modes(self) -> tuple[str, ...]:
        return tuple(m.strip() for m in self.modes.split(",") if m.strip())

    @property
    def num_values(self) -> int:
        return self.word_len + 1 if self.task == "first" else self.mod

    @property
    def sep_token(self) -> int:
        return self.alphabet + self.num_values

    @property
    def colon_token(self) -> int:
        return self.alphabet + self.num_values + 1

    @property
    def eos_token(self) -> int:
        return self.alphabet + self.num_values + 2

    @property
    def vocab_size(self) -> int:
        return self.alphabet + self.num_values + 3

    @property
    def n_trace(self) -> int:
        return self.word_len // self.every_k

    @property
    def max_seq_len(self) -> int:
        return 2 * self.word_len + 6

    @property
    def is_stream(self) -> bool:
        return self.layout == "stream"

    @property
    def mlp_width(self) -> int:
        return max(self.alphabet, 3 * self.num_values * (self.word_len + 1), 2 * self.word_len + 1)

    @property
    def slots(self) -> ResidualSlots:
        return ResidualSlots.build(self.alphabet, self.num_values, self.max_seq_len)

    @property
    def model_width(self) -> int:
        return self.slots.total_width

    @property
    def routing_gap(self) -> float:
        return self.gap if self.gap else max(16.0, 3.0 * math.log(8.0 * (self.word_len + 3)))

    @property
    def routing_scale(self) -> float:
        return self.routing_gap * math.sqrt(self.max_seq_len)

    def to_dict(self, *, exclude_device: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if exclude_device:
            data.pop("device", None)
        return data


DEFAULTS: dict[str, Any] = {f.name: getattr(Config(), f.name) for f in fields(Config)}
ACTIVE_CONFIG = Config()


def configure(**overrides: Any) -> Config:
    """Create and activate a Config, seeding Python and PyTorch RNGs deterministically."""
    global ACTIVE_CONFIG
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown setting(s): {sorted(unknown)}")
    cfg = Config(**overrides)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ACTIVE_CONFIG = cfg
    return cfg


def parse_cli_overrides(argv: list[str]) -> dict[str, Any]:
    """Parse key=value CLI tokens into typed Config keyword overrides."""
    overrides: dict[str, Any] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Expected key=value argument, got {arg!r}")
        key, raw_value = arg.split("=", 1)
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting {key!r}")
        overrides[key] = raw_value if key in STRING_KEYS else ast.literal_eval(raw_value)
    return overrides


def _cfg(cfg: Config | None) -> Config:
    return ACTIVE_CONFIG if cfg is None else cfg


# -----------------------------------------------------------------------------
# Data generation and serialization
# -----------------------------------------------------------------------------

Example = tuple[int, int, tuple[int, ...], list[int]]


def transition(state: int, step_idx: int, letter: int, query: int, cfg: Config | None = None) -> int:
    """Gold state transition after observing letter at 1-indexed position step_idx."""
    c = _cfg(cfg)
    if c.task == "count":
        return (state + int(letter == query)) % c.mod
    if c.task == "first":
        return state if state else (step_idx if letter == query else 0)
    if c.task == "shell":
        cup_a, cup_b = SHELL_PAIRS[letter]
        if state == cup_a:
            return cup_b
        return cup_a if state == cup_b else state
    if c.task == "first_cup":
        if step_idx > 1:
            return state
        return min(SHELL_PAIRS[letter])
    if c.task == "lights":
        return state ^ (1 << letter)
    if c.task == "lights_parity":
        return state ^ 1
    raise ValueError(c.task)


def make_example(rng: random.Random, cfg: Config | None = None) -> Example:
    """Sample one (s0, query, word, states) example."""
    c = _cfg(cfg)
    while True:
        if c.p_match:
            query = rng.randrange(c.alphabet)
            word = tuple(
                query if rng.random() < c.p_match else (query + 1 + rng.randrange(c.alphabet - 1)) % c.alphabet
                for _ in range(c.word_len)
            )
        else:
            query = rng.randrange(c.alphabet)
            word = tuple(rng.randrange(c.alphabet) for _ in range(c.word_len))
        if c.task != "first" or query in word:
            break

    s0 = rng.randrange(c.mod) if c.task in ("count", "shell", "lights", "lights_parity") else 0
    states: list[int] = []
    state = s0
    for step_idx, letter in enumerate(word, start=1):
        state = transition(state, step_idx, letter, query, c)
        states.append(state)
    return s0, query, word, states


def corrupt_trace(example: Example, rng: random.Random, cfg: Config | None = None) -> list[int]:
    """Generate a corrupted trace (scatter, shift, or single-step slip)."""
    c = _cfg(cfg)
    s0, query, word, _ = example
    slip_pos = c.slip_at if (c.corrupt == "slip" and c.slip_at) else rng.randrange(1, c.word_len + 1)
    shown: list[int] = []
    state = s0
    for step_idx, letter in enumerate(word, start=1):
        gold_next = transition(state, step_idx, letter, query, c)
        if c.corrupt == "slip":
            if step_idx == slip_pos:
                state = rng.choice([v for v in range(c.mod) if v != gold_next])
            else:
                state = gold_next
        elif c.corrupt == "shift":
            state = (gold_next + 1) % c.mod
        else:
            state = rng.choice([v for v in range(c.mod) if v != gold_next])
        shown.append(state)
    return shown


def shown_traces(examples: list[Example], rng: random.Random, cfg: Config | None = None) -> list[list[int]]:
    """Build step-supervision traces nested in rho (higher rho only cleans previously corrupt rows)."""
    c = _cfg(cfg)
    if c.task != "count" or c.rho == 1.0:
        return [ex[3] for ex in examples]
    uniforms = [rng.random() for _ in examples]
    wrong_traces = [corrupt_trace(ex, rng, c) for ex in examples]
    return [ex[3] if u < c.rho else bad for ex, u, bad in zip(examples, uniforms, wrong_traces)]


def encode_prompt(example: Example, cfg: Config | None = None) -> list[int]:
    c = _cfg(cfg)
    s0, query, word, _ = example
    return [c.alphabet + s0, query, *word, c.sep_token]


def encode_row(
    example: Example,
    mode: str,
    shown: list[int] | None = None,
    cfg: Config | None = None,
) -> list[int]:
    """Serialize one training row: prompt, optional step trace, then ': s_n EOS'."""
    c = _cfg(cfg)
    s0, query, word, states = example
    tail = [c.colon_token, c.alphabet + states[-1], c.eos_token]
    if mode == "outcome":
        return encode_prompt(example, c) + tail

    values = [c.alphabet + v for v in (shown if shown is not None else states)]
    if c.every_k > 1:
        values = values[c.every_k - 1 :: c.every_k]
    if c.is_stream:
        interleaved = [tok for pair in zip(word, values) for tok in pair]
        return [c.alphabet + s0, query, *interleaved, *tail]
    return encode_prompt(example, c) + values + tail


def supervised_positions(mode: str, cfg: Config | None = None) -> list[int]:
    """Token indices in an encoded row that contribute to cross-entropy loss."""
    c = _cfg(cfg)
    n = c.word_len
    n_val = n if c.every_k == 1 else c.n_trace
    length = n + 6 if mode == "outcome" else (2 * n + 5 if c.is_stream else n + n_val + 6)
    if mode == "process" and c.is_stream:
        return [2 * t + 1 for t in range(1, n + 1)] + [2 * n + 2, 2 * n + 3, 2 * n + 4]
    return list(range(n + 3, length))


def encode_batch(
    rows: list[list[int]],
    mode: str,
    cfg: Config | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher-forced (inputs, targets) with -100 at unsupervised positions."""
    c = _cfg(cfg)
    ids = torch.tensor(rows, device=c.device)
    inputs = ids[:, :-1]
    targets = torch.full_like(ids[:, 1:], -100)
    keep = torch.tensor(supervised_positions(mode, c), device=c.device)
    targets[:, keep - 1] = ids[:, keep]
    return inputs, targets


# -----------------------------------------------------------------------------
# Shared two-block normalization-free causal Transformer
# -----------------------------------------------------------------------------


def zero_params(cfg: Config | None = None) -> dict[str, torch.Tensor]:
    c = _cfg(cfg)
    w, p_len, f_w = c.model_width, c.max_seq_len, c.mlp_width
    params: dict[str, torch.Tensor] = {
        "wte": torch.zeros(c.vocab_size, w),
        "wpe": torch.zeros(p_len, w),
        "readout": torch.zeros(c.vocab_size, w),
    }
    for block in range(c.n_blocks):
        params[f"{block}.q"] = torch.zeros(NUM_HEADS, p_len, w)
        params[f"{block}.k"] = torch.zeros(NUM_HEADS, p_len, w)
        params[f"{block}.v"] = torch.zeros(NUM_HEADS, w, w)
        params[f"{block}.mlp_in"] = torch.zeros(f_w, w)
        params[f"{block}.mlp_b"] = torch.zeros(f_w)
        params[f"{block}.mlp_out"] = torch.zeros(w, f_w)
    return params


def forward(params: dict[str, torch.Tensor], token_ids: torch.Tensor, cfg: Config | None = None) -> torch.Tensor:
    """Causal forward pass returning vocabulary logits (B, T, vocab_size)."""
    c = _cfg(cfg)
    seq_len = token_ids.shape[1]
    hidden = params["wte"][token_ids] + params["wpe"][:seq_len]
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device).triu(1)
    scale = math.sqrt(c.max_seq_len)

    for block in range(c.n_blocks):
        q = torch.einsum("btw,hdw->bhtd", hidden, params[f"{block}.q"])
        k = torch.einsum("btw,hdw->bhtd", hidden, params[f"{block}.k"])
        v = torch.einsum("btw,hvw->bhtv", hidden, params[f"{block}.v"])
        scores = (q @ k.transpose(-1, -2) / scale).masked_fill(causal_mask, -math.inf)
        hidden = hidden + (scores.softmax(dim=-1) @ v).sum(dim=1)
        hidden = hidden + F.relu(hidden @ params[f"{block}.mlp_in"].T + params[f"{block}.mlp_b"]) @ params[f"{block}.mlp_out"].T

    return hidden @ params["readout"].T


# -----------------------------------------------------------------------------
# Exact handcoded executors (Proposition 1, Appendix B)
# -----------------------------------------------------------------------------


def _slot_idx(slot: slice, offset: int = 0) -> int:
    return slot.start + offset


def _set_position_routing(
    params: dict[str, torch.Tensor],
    block: int,
    head: int,
    targets_by_pos: dict[int, list[int]],
    fallback: int,
    cfg: Config,
) -> None:
    """Configure attention head (block, head) to attend from position i to targets_by_pos[i] (or fallback)."""
    slots = cfg.slots
    params[f"{block}.k"][head, :, slots.pos] = torch.eye(cfg.max_seq_len)
    for query_pos in range(cfg.max_seq_len):
        for key_pos in targets_by_pos.get(query_pos, [fallback]):
            params[f"{block}.q"][head, key_pos, _slot_idx(slots.pos, query_pos)] = cfg.routing_scale


def _build_shared_block0(letter_positions: Any, cfg: Config) -> dict[str, torch.Tensor]:
    """Embeddings and Block 0: copy query token w* to each letter position and compute match bit [w_t == w*]."""
    slots = cfg.slots
    params = zero_params(cfg)
    params["wte"][: cfg.alphabet, slots.tok_letter] = torch.eye(cfg.alphabet)
    params["wte"][cfg.alphabet : cfg.alphabet + cfg.num_values, slots.tok_value] = torch.eye(cfg.num_values)
    params["wpe"][:, slots.pos] = torch.eye(cfg.max_seq_len)

    _set_position_routing(params, 0, 0, {pos: [1] for pos in letter_positions}, fallback=0, cfg=cfg)
    params["0.v"][0][slots.query, slots.tok_letter] = torch.eye(cfg.alphabet)

    for letter in range(cfg.alphabet):
        params["0.mlp_in"][letter, _slot_idx(slots.tok_letter, letter)] = 1.0
        params["0.mlp_in"][letter, _slot_idx(slots.query, letter)] = 1.0
        params["0.mlp_b"][letter] = -1.5
        params["0.mlp_out"][_slot_idx(slots.match), letter] = 2.0
    return params


def _set_readout(params: dict[str, torch.Tensor], colon_at: int, eos_at: int, cfg: Config) -> dict[str, torch.Tensor]:
    slots = cfg.slots
    params["readout"][cfg.alphabet : cfg.alphabet + cfg.num_values, slots.out] = 20.0 * torch.eye(cfg.num_values)
    params["readout"][cfg.colon_token, _slot_idx(slots.pos, colon_at)] = 20.0
    params["readout"][cfg.eos_token, _slot_idx(slots.pos, eos_at)] = 20.0
    return params


def build_step_executor(cfg: Config | None = None) -> dict[str, torch.Tensor]:
    """Construct exact step-by-step executor theta*_step (Proposition 1 / Appendix B)."""
    c = _cfg(cfg)
    n, num_vals, slots = c.word_len, c.num_values, c.slots

    if c.is_stream:
        letter_pos = {t: 2 * t for t in range(1, n + 1)}
        pred_step = {2 * t: t for t in range(1, n + 1)}
        prev_pos = {2 * t: (0 if t == 1 else 2 * t - 1) for t in range(1, n + 1)}
        last_state_pos = 2 * n + 1
    else:
        letter_pos = {t: 1 + t for t in range(1, n + 1)}
        pred_step = {n + 1 + t: t for t in range(1, n + 1)}
        prev_pos = {pos: (0 if t == 1 else pos) for pos, t in pred_step.items()}
        last_state_pos = 2 * n + 2

    params = _build_shared_block0(letter_pos.values(), c)
    _set_position_routing(params, 1, 0, {pos: [src] for pos, src in prev_pos.items()}, fallback=1, cfg=c)
    _set_position_routing(params, 1, 1, {pos: [letter_pos[t]] for pos, t in pred_step.items()}, fallback=0, cfg=c)
    _set_position_routing(params, 1, 2, {last_state_pos + 1: [last_state_pos]}, fallback=1, cfg=c)

    params["1.v"][0][slots.prev, slots.tok_value] = torch.eye(num_vals)
    params["1.v"][1][slots.match_token, slots.match] = torch.eye(1)
    params["1.v"][2][slots.out, slots.tok_value] = torch.eye(num_vals)

    units: list[tuple[dict[int, float], float, int]] = []
    if c.task == "count":
        for state_val in range(num_vals):
            units.append(({_slot_idx(slots.prev, state_val): 1.0, _slot_idx(slots.match_token): -1.0}, -0.5, state_val))
            units.append(
                (
                    {_slot_idx(slots.prev, state_val): 1.0, _slot_idx(slots.match_token): 1.0},
                    -1.5,
                    (state_val + 1) % c.mod,
                )
            )
    else:
        for state_val in range(1, num_vals):
            units.append(({_slot_idx(slots.prev, state_val): 1.0}, -0.5, state_val))
        units.append(({_slot_idx(slots.prev, 0): 1.0, _slot_idx(slots.match_token): -1.0}, -0.5, 0))
        for pos, step_idx in pred_step.items():
            units.append(
                (
                    {_slot_idx(slots.prev, 0): 1.0, _slot_idx(slots.match_token): 1.0, _slot_idx(slots.pos, pos): 1.0},
                    -2.5,
                    step_idx,
                )
            )

    for unit_idx, (weights, bias, out_val) in enumerate(units):
        for coord, coef in weights.items():
            params["1.mlp_in"][unit_idx, coord] = coef
        params["1.mlp_b"][unit_idx] = bias
        params["1.mlp_out"][_slot_idx(slots.out, out_val), unit_idx] = 2.0

    return _set_readout(params, colon_at=last_state_pos, eos_at=last_state_pos + 2, cfg=c)


def build_outcome_executor(cfg: Config | None = None) -> dict[str, torch.Tensor]:
    """Construct exact direct-answer executor theta*_out (Proposition 1 / Appendix B)."""
    c = _cfg(cfg)
    n, num_vals, slots = c.word_len, c.num_values, c.slots
    params = _build_shared_block0(range(2, n + 2), c)
    ans_pos = n + 3

    if c.task == "count":
        _set_position_routing(params, 1, 0, {ans_pos: list(range(2, n + 2))}, fallback=0, cfg=c)
        _set_position_routing(params, 1, 1, {ans_pos: [0]}, fallback=1, cfg=c)
        params["1.v"][0][slots.count, slots.match] = float(n)
        params["1.v"][1][slots.s0, slots.tok_value] = torch.eye(num_vals)

        gate = float(n + 2)
        unit_idx = 0
        for s0_val in range(num_vals):
            for match_count in range(n + 1):
                for offset, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                    params["1.mlp_in"][unit_idx, _slot_idx(slots.count)] = 1.0
                    params["1.mlp_in"][unit_idx, _slot_idx(slots.s0, s0_val)] = gate
                    params["1.mlp_b"][unit_idx] = offset - match_count - gate
                    params["1.mlp_out"][_slot_idx(slots.out, (s0_val + match_count) % c.mod), unit_idx] = coef
                    unit_idx += 1
    else:
        beta = c.routing_scale
        k_mat, q_mat = params["1.k"][0], params["1.q"][0]
        k_mat[0, _slot_idx(slots.match)] = 1.0
        k_mat[1, slots.pos] = -torch.arange(c.max_seq_len, dtype=torch.float)
        k_mat[2, _slot_idx(slots.pos, 1)] = 1.0

        q_mat[0, _slot_idx(slots.pos, ans_pos)] = beta * (n + 2) + c.routing_scale
        q_mat[1, _slot_idx(slots.pos, ans_pos)] = beta
        for pos in range(c.max_seq_len):
            if pos != ans_pos:
                q_mat[2, _slot_idx(slots.pos, pos)] = c.routing_scale

        for pos in range(2, n + 2):
            params["1.v"][0][_slot_idx(slots.out, pos - 1), _slot_idx(slots.pos, pos)] = 1.0
        params["1.v"][0][_slot_idx(slots.out, 0), _slot_idx(slots.pos, 0)] = 1.0

    return _set_readout(params, colon_at=n + 2, eos_at=n + 4, cfg=c)


def handcoded_params(cfg: Config | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Return exact {'process': theta*_step, 'outcome': theta*_out} executors on cfg.device."""
    c = _cfg(cfg)
    if c.task not in ("count", "first") or c.every_k != 1:
        raise ValueError("Handcoded executors are defined for task in ('count', 'first') with every_k=1")
    sols = {"process": build_step_executor(c), "outcome": build_outcome_executor(c)}
    return {mode: {k: t.to(c.device) for k, t in params.items()} for mode, params in sols.items()}


# -----------------------------------------------------------------------------
# Evaluation, corpus creation, initialization, and training
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate(
    params: dict[str, torch.Tensor],
    examples: list[Example],
    mode: str,
    cfg: Config | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Run free-running greedy continuation and return (predicted_tokens, gold_tokens)."""
    c = _cfg(cfg)

    def _trace_values(ex: Example) -> list[int]:
        vals = [c.alphabet + v for v in ex[3]]
        return vals[c.every_k - 1 :: c.every_k] if c.every_k > 1 else vals

    gold = [
        (_trace_values(ex) if mode == "process" else []) + [c.colon_token, c.alphabet + ex[3][-1], c.eos_token]
        for ex in examples
    ]

    if mode == "process" and c.is_stream:
        ids = torch.tensor([[c.alphabet + ex[0], ex[1]] for ex in examples], device=c.device)
        generated: list[torch.Tensor] = []
        for step_idx in range(c.word_len):
            letters = torch.tensor([[ex[2][step_idx]] for ex in examples], device=c.device)
            ids = torch.cat([ids, letters], dim=1)
            next_tok = forward(params, ids, c)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
            generated.append(next_tok)
        for _ in range(3):
            next_tok = forward(params, ids, c)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
            generated.append(next_tok)
        predicted = torch.cat(generated, dim=1).tolist()
    else:
        ids = torch.tensor([encode_prompt(ex, c) for ex in examples], device=c.device)
        for _ in range(len(gold[0])):
            next_tok = forward(params, ids, c)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
        predicted = ids[:, c.word_len + 3 :].tolist()

    return predicted, gold


@torch.no_grad()
def free_run(
    params: dict[str, torch.Tensor],
    examples: list[Example],
    mode: str,
    cfg: Config | None = None,
) -> tuple[float, float, float]:
    """Evaluate free-running greedy decoding: returns (exact_acc, answer_acc, step_acc)."""
    c = _cfg(cfg)
    rows: list[list[int]] = []
    gold: list[list[int]] = []
    for start in range(0, len(examples), 64):
        pred_chunk, gold_chunk = generate(params, examples[start : start + 64], mode, c)
        rows.extend(pred_chunk)
        gold.extend(gold_chunk)

    n_examples = len(gold)
    exact_acc = sum(r == g for r, g in zip(rows, gold)) / n_examples
    answer_acc = (
        sum(
            c.colon_token in r and r.index(c.colon_token) + 1 < len(r) and r[r.index(c.colon_token) + 1] == g[-2]
            for r, g in zip(rows, gold)
        )
        / n_examples
    )
    if mode == "process":
        n_val = c.word_len if c.every_k == 1 else c.n_trace
        step_acc = sum(a == b for r, g in zip(rows, gold) for a, b in zip(r[:n_val], g[:n_val])) / (n_val * n_examples)
    else:
        step_acc = float("nan")
    return exact_acc, answer_acc, step_acc


def make_corpus(cfg: Config | None = None) -> tuple[list[Example], list[list[int]], list[Example]]:
    """Generate disjoint training and held-out test corpora with independent seeded streams."""
    c = _cfg(cfg)
    rng_train = random.Random(c.seed)
    rng_corrupt = random.Random(c.seed + 10_000)
    rng_test = random.Random(c.seed + 20_000)

    train = [make_example(rng_train, c) for _ in range(c.train_size)]
    shown = shown_traces(train, rng_corrupt, c)

    seen_prompts = {tuple(encode_prompt(ex, c)) for ex in train}
    test: list[Example] = []
    tries = 0
    while len(test) < c.test_size:
        ex = make_example(rng_test, c)
        tries += 1
        if tries >= 100 * c.test_size:
            raise RuntimeError("Prompt space too small to sample disjoint held-out test set")
        if tuple(encode_prompt(ex, c)) not in seen_prompts:
            test.append(ex)

    clean_count = sum(sh == ex[3] for ex, sh in zip(train, shown))
    print(
        f"corpus  train={len(train)}  clean={clean_count}/{len(train)} ({clean_count / len(train):.3f})  "
        f"test={len(test)}  (streams: train seed, corrupt seed+10000, test seed+20000)"
    )
    if c.clean_only:
        keep = [i for i, (ex, sh) in enumerate(zip(train, shown)) if sh == ex[3]]
        train = [train[i] for i in keep]
        shown = [shown[i] for i in keep]
        print(f"clean_only: training on the {len(train)} clean traces of the rho={c.rho} corpus")
    return train, shown, test


def init_theta(cfg: Config | None = None) -> dict[str, torch.Tensor]:
    """Initialize shared parameter dictionary theta_0 (cloned identically for step and outcome training)."""
    c = _cfg(cfg)
    if c.init == "random":
        theta = {name: torch.randn_like(tensor) * c.init_std for name, tensor in zero_params(c).items()}
        if c.zero_readout:
            theta["readout"].zero_()
        return theta

    gen = torch.Generator().manual_seed(c.seed + 40_000)
    theta = {}
    for name, sol in build_step_executor(c).items():
        skip = name == "readout" or (name.endswith("mlp_b") and c.seed_bias == "zero")
        support = (sol != 0) & (not skip)
        magnitude = c.init_std * (0.5 + 1.5 * torch.rand(sol.shape, generator=gen))
        if c.seed_signs == "handcoded":
            sign = torch.sign(sol)
        else:
            sign = torch.randint(2, sol.shape, generator=gen) * 2.0 - 1.0
        theta[name] = torch.where(support, sign * magnitude, torch.zeros_like(sol))
    return theta


@torch.no_grad()
def mean_loss(
    params: dict[str, torch.Tensor],
    mode: str,
    examples: list[Example],
    shown: list[list[int]] | None = None,
    cfg: Config | None = None,
) -> float:
    """Evaluate mean token cross-entropy over examples for a fixed parameter dictionary."""
    c = _cfg(cfg)
    traces = [ex[3] for ex in examples] if shown is None else shown
    rows = [encode_row(ex, mode, sh, c) for ex, sh in zip(examples, traces)]
    inputs, targets = encode_batch(rows, mode, c)

    total_ce = 0.0
    total_tokens = 0
    for start in range(0, inputs.shape[0], c.batch_size):
        logits = forward(params, inputs[start : start + c.batch_size], c)
        tgt = targets[start : start + c.batch_size]
        total_ce += F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum").item()
        total_tokens += int((tgt != -100).sum())
    return total_ce / max(total_tokens, 1)


def _build_optimizer(params: dict[str, torch.Tensor], cfg: Config) -> torch.optim.Optimizer:
    if cfg.opt == "sgd":
        return torch.optim.SGD(params.values(), lr=cfg.lr, momentum=0.0)
    if cfg.opt == "adam":
        return torch.optim.Adam(params.values(), lr=cfg.lr, weight_decay=0.0)
    return torch.optim.AdamW(params.values(), lr=cfg.lr, weight_decay=0.0)


def train_mode(
    mode: str,
    theta_0: dict[str, torch.Tensor],
    train: list[Example],
    shown: list[list[int]],
    test: list[Example],
    *,
    keep_state: bool = False,
    disable_tqdm: bool | None = None,
    cfg: Config | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Train one clone of theta_0 under 'process' (step) or 'outcome' supervision."""
    c = _cfg(cfg)
    params = {name: tensor.clone().to(c.device).requires_grad_() for name, tensor in theta_0.items()}
    optimizer = _build_optimizer(params, c)

    rows = [encode_row(ex, mode, sh, c) for ex, sh in zip(train, shown)]
    inputs, targets = encode_batch(rows, mode, c)
    batch_gen = torch.Generator().manual_seed(c.seed)

    history: list[dict[str, Any]] = []
    eval_tick = c.eval_every if c.eval_every else max(1, c.steps // 8)
    tqdm_kwargs: dict[str, Any] = {"desc": mode, "dynamic_ncols": True}
    if disable_tqdm is not None:
        tqdm_kwargs["disable"] = disable_tqdm
    bar = _make_tqdm(range(1, c.steps + 1), **tqdm_kwargs)

    for step_idx in bar:
        batch_ids = torch.randint(len(train), (c.batch_size,), generator=batch_gen).to(c.device)
        logits = forward(params, inputs[batch_ids], c)
        loss = F.cross_entropy(logits.flatten(0, 1), targets[batch_ids].flatten(), ignore_index=-100)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()

        record: dict[str, Any] = {"step": step_idx, "loss": loss.item()}
        postfix = {"loss": f"{loss.item():.3f}"}

        if step_idx % eval_tick == 0 or step_idx == c.steps:
            exact_acc, answer_acc, step_acc = free_run(params, test, mode, c)
            record.update(exact=exact_acc, answer=answer_acc, steps_ok=step_acc)
            postfix["exact"] = f"{exact_acc:.3f}"
            line = (
                f"{mode:8s} step {step_idx:5d}  loss {loss.item():.4f}  "
                f"test exact {exact_acc:.3f}  answer {answer_acc:.3f}"
                + (f"  step acc {step_acc:.3f}" if mode == "process" else "")
            )
            _tqdm.write(line)
            if keep_state:
                record["params"] = {k: t.detach().cpu().clone() for k, t in params.items()}

        bar.set_postfix(postfix, refresh=False)
        history.append(record)

        if c.save_every and (step_idx % c.save_every == 0 or step_idx == c.steps):
            ckpt_name = f"{c.tag}_{mode}_step{step_idx}.pt" if c.tag else f"{mode}_step{step_idx}.pt"
            ckpt_path = Path(c.save_dir) / ckpt_name
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config": {
                        "task": c.task,
                        "word_len": c.word_len,
                        "alphabet": c.alphabet,
                        "mod": c.mod,
                        "n_blocks": c.n_blocks,
                        "layout": c.layout,
                        "seed": c.seed,
                        "init_std": c.init_std,
                        "every_k": c.every_k,
                        "opt": c.opt,
                        "batch_size": c.batch_size,
                    },
                    "params": {k: t.detach().cpu() for k, t in params.items()},
                    "mode": mode,
                    "step": step_idx,
                },
                ckpt_path,
            )
            _tqdm.write(f"wrote {ckpt_path}")

    return params, history


def main(argv: list[str] | None = None) -> None:
    overrides = parse_cli_overrides(sys.argv[1:] if argv is None else argv)
    cfg = configure(**overrides)
    train, shown, test = make_corpus(cfg)
    theta_0 = init_theta(cfg)

    n_params = sum(t.numel() for t in theta_0.values())
    extra = f" m={cfg.mod} rho={cfg.rho} corrupt={cfg.corrupt}" if cfg.task == "count" else ""
    every_k_str = f" every_k={cfg.every_k}" if cfg.every_k > 1 else ""
    print(
        f"task={cfg.task} layout={cfg.layout} A={cfg.alphabet} n={cfg.word_len}{extra}{every_k_str}"
        f"  L={cfg.n_blocks} opt={cfg.opt} batch={cfg.batch_size} heads={NUM_HEADS}"
        f" d_ff={cfg.mlp_width} width={cfg.model_width}  params={n_params:,}"
    )

    if cfg.task == "count" and cfg.mod == 2:
        p_q = cfg.p_match or 1.0 / cfg.alphabet
        bias = abs(1.0 - 2.0 * p_q) ** cfg.word_len
        print(f"answer bias given everything but the count: |1-2p|^n = {bias:.2e}  (p={p_q:g})")

    if cfg.task in ("count", "first") and cfg.every_k == 1:
        for mode, sol_params in handcoded_params(cfg).items():
            exact_acc, answer_acc, _ = free_run(sol_params, test, mode, cfg)
            print(f"handcoded {mode:8s}  gap {cfg.routing_gap:.2f}  exact {exact_acc:.3f}  answer {answer_acc:.3f}")
            for sigma in cfg.perturb:
                noise_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 30_000)
                worst = 1.0
                for _ in range(cfg.perturb_draws):
                    noisy = {
                        k: t + sigma * torch.randn(t.shape, generator=noise_gen).to(t.device)
                        for k, t in sol_params.items()
                    }
                    worst = min(worst, free_run(noisy, test, mode, cfg)[0])
                print(
                    f"handcoded {mode:8s}  noise std {sigma:g}  "
                    f"worst exact over {cfg.perturb_draws} draws {worst:.3f}"
                )

    if cfg.handcoded_only:
        return

    for mode in cfg.train_modes:
        train_mode(mode, theta_0, train, shown, test, cfg=cfg)


if __name__ == "__main__":
    main()
