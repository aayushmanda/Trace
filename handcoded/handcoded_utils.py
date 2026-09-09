import copy
import math
import random
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


N_BITS = 4
N_STATES = 2 ** N_BITS
DEPTH = 4
TRAIN_SIZE = 20_000
TEST_SIZE = 1_000
STEPS = 10_000
BATCH_SIZE = 128
LR = 2e-3
D_MODEL = 96
N_HEADS = 4
D_FF = 192
DATA_SEED = 123
TEST_SEED = 9_000
MODEL_SEED = 42
BATCH_SEED = 2_026


def make_checkpoints(steps=STEPS, animation_checkpoints=120):
    uniform_steps = np.linspace(0, steps, animation_checkpoints + 1).round().astype(int)
    early_steps = [1, 2, 5, 10, 20, 25, 50, 75, 100, 250, 500]
    return sorted(set(uniform_steps.tolist()) | {s for s in early_steps if s <= steps})


def bits(value, n_bits=N_BITS):
    """Decode an integer; index 0 is the leftmost bit."""
    return [int(bit) for bit in f"{value:0{n_bits}b}"]


def state_text(state):
    """Format a list of bits as a readable state such as '1000'."""
    return "".join(map(str, state))


def make_gate_names(n_bits=N_BITS):
    """List each legal gate string in a stable order."""
    gates = [f"x{i}" for i in range(n_bits)]
    for operation in ("c", "s"):
        gates.extend(f"{operation}{i}{j}" for i, j in permutations(range(n_bits), 2))
    gates.extend(f"t{i}{j}{k}" for i, j, k in permutations(range(n_bits), 3))
    return gates


def sample_gate(rng, n_bits=N_BITS):
    """Choose a family uniformly, then choose its distinct bit indices."""
    operation = rng.choice("xcst")
    arity = {"x": 1, "c": 2, "s": 2, "t": 3}[operation]
    indices = rng.sample(range(n_bits), arity)
    return operation + "".join(map(str, indices))


def apply_gate(state, gate):
    """Apply one reversible gate without modifying the input state."""
    next_state = state.copy()
    operation = gate[0]
    indices = list(map(int, gate[1:]))
    if operation == "x":
        (target,) = indices
        next_state[target] ^= 1
    elif operation == "c":
        control, target = indices
        next_state[target] ^= next_state[control]
    elif operation == "s":
        left, right = indices
        next_state[left], next_state[right] = next_state[right], next_state[left]
    elif operation == "t":
        control_a, control_b, target = indices
        next_state[target] ^= next_state[control_a] & next_state[control_b]
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return next_state


def phi(state_int, gate, n_bits=N_BITS):
    """Integer version of the exact Boolean transition rule."""
    return int(state_text(apply_gate(bits(state_int, n_bits), gate)), 2)


def sample_example(rng, depth=DEPTH, n_bits=N_BITS):
    """Return the start, gate names, and gold state after every gate."""
    n_states = 2 ** n_bits
    start = rng.randrange(n_states)
    gates = [sample_gate(rng, n_bits) for _ in range(depth)]
    states = []
    current = start
    for gate in gates:
        current = phi(current, gate, n_bits)
        states.append(current)
    return start, gates, states


GATES = make_gate_names(N_BITS)


@dataclass
class Circuit:
    start: int
    gates: list[str]
    states: list[int]

    @property
    def answer(self):
        return self.states[-1]


def make_circuits(size, seed, depth=DEPTH, n_bits=N_BITS):
    rng = random.Random(seed)
    return [Circuit(*sample_example(rng, depth, n_bits)) for _ in range(size)]


class CircuitTokenizer:
    """Map semantic states, gates, and separators to vocabulary indices."""

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


def make_tokenizer(n_bits=N_BITS):
    return CircuitTokenizer(make_gate_names(n_bits), 2 ** n_bits)


class LearnedOneLayerTransformer(nn.Module):
    """One causal attention layer and one hidden ReLU MLP, with residuals."""

    def __init__(self, vocab_size, max_length, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.readout = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, return_attention=False):
        length = ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"Input length {length} exceeds {self.max_length}")
        positions = torch.arange(length, device=ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions)[None]
        causal_mask = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
        attended, weights = self.attention(
            hidden,
            hidden,
            hidden,
            attn_mask=causal_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(hidden)
        logits = self.readout(hidden)
        return (logits, weights) if return_attention else logits


def build_random_learned_model(
    tokenizer,
    depth=DEPTH,
    *,
    seed=MODEL_SEED,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    device=None,
):
    if seed is not None:
        torch.manual_seed(seed)
    model = LearnedOneLayerTransformer(
        len(tokenizer.tokens),
        max_length=3 * depth + 5,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )
    return model.to(device) if device is not None else model


def _replace_buffers_with_random_parameters(module, *, generator=None, init_std=0.02):
    for name, buffer in list(module._buffers.items()):
        if buffer is None:
            continue
        del module._buffers[name]
        parameter = torch.empty_like(buffer)
        if parameter.is_floating_point():
            parameter.normal_(mean=0.0, std=init_std, generator=generator)
        else:
            parameter = parameter.float().normal_(mean=0.0, std=init_std, generator=generator)
        module.register_parameter(name, nn.Parameter(parameter))
    for child in module.children():
        _replace_buffers_with_random_parameters(child, generator=generator, init_std=init_std)
    return module


def make_random_trainable_copy(model, *, seed=MODEL_SEED, init_std=0.02, device=None):
    """Turn a fixed handcoded module layout into a randomly initialized trainable model."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    trainable = copy.deepcopy(model).cpu()
    _replace_buffers_with_random_parameters(trainable, generator=generator, init_std=init_std)
    return trainable.to(device) if device is not None else trainable


class FixedAttentionHead(nn.Module):
    """Ordinary causal dot-product attention with fixed projection buffers."""

    def __init__(self, width, n_positions):
        super().__init__()
        self.register_buffer("query", torch.zeros(n_positions, width))
        self.register_buffer("key", torch.zeros(n_positions, width))
        self.register_buffer("value", torch.zeros(width, width))

    def forward(self, hidden):
        query = hidden @ self.query.T
        key = hidden @ self.key.T
        value = hidden @ self.value.T
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.query.shape[0])
        length = hidden.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        return scores.masked_fill(mask, -torch.inf).softmax(dim=-1) @ value


class HandcodedProcessTransformer(nn.Module):
    """Constructive fixed-depth attention/MLP solution, encoded entirely in weights."""

    def __init__(self, tokenizer, depth=DEPTH):
        super().__init__()
        states, gates = tokenizer.n_states, len(tokenizer.gates)
        n_bits = int(math.log2(states))
        positions = 3 * depth + 4
        state = slice(0, states)
        gate = slice(state.stop, state.stop + gates)
        position = slice(gate.stop, gate.stop + positions)
        retrieved_state = slice(position.stop, position.stop + states)
        output_gate = slice(retrieved_state.stop, retrieved_state.stop + gates)
        output_state = slice(output_gate.stop, output_gate.stop + states)
        width = output_state.stop
        self.max_length = positions
        self.heads = nn.ModuleList([FixedAttentionHead(width, positions) for _ in range(4)])
        self.register_buffer("token_features", torch.zeros(len(tokenizer.tokens), width))
        self.register_buffer("position_features", torch.zeros(positions, width))
        self.register_buffer("mlp_in", torch.zeros(states * gates, width))
        self.register_buffer("mlp_bias", torch.full((states * gates,), -1.5))
        self.register_buffer("mlp_out", torch.zeros(width, states * gates))
        self.register_buffer("readout", torch.zeros(len(tokenizer.tokens), width))

        self.token_features[:states, state] = torch.eye(states)
        self.token_features[states : states + gates, gate] = torch.eye(gates)
        self.position_features[:, position] = torch.eye(positions)
        for head in self.heads:
            head.key[:, position] = torch.eye(positions)

        state_head, gate_head, final_head, _ = self.heads
        separator_position = depth + 1
        state_routes, gate_routes = {}, {}
        for step in range(depth):
            gate_position = depth + 2 + 2 * step
            state_routes[gate_position] = 0 if step == 0 else gate_position - 1
            gate_routes[gate_position - 1] = step + 1
        final_routes = {3 * depth + 2: 3 * depth + 1}
        for head, routes in zip(self.heads[:3], [state_routes, gate_routes, final_routes]):
            for destination in range(separator_position, positions):
                origin = routes.get(destination, separator_position)
                head.query[origin, position.start + destination] = 80.0
        state_head.value[retrieved_state, state] = torch.eye(states)
        gate_head.value[output_gate, gate] = torch.eye(gates)
        final_head.value[output_state, state] = torch.eye(states)

        for state_id in range(states):
            for gate_id, gate_name in enumerate(tokenizer.gates):
                unit = state_id * gates + gate_id
                self.mlp_in[unit, retrieved_state.start + state_id] = 1
                self.mlp_in[unit, gate.start + gate_id] = 1
                target = phi(state_id, gate_name, n_bits)
                self.mlp_out[output_state.start + target, unit] = 2
        self.readout[:states, output_state] = 20 * torch.eye(states)
        self.readout[states : states + gates, output_gate] = 20 * torch.eye(gates)
        self.readout[tokenizer.colon, position.start + 3 * depth + 1] = 20
        self.readout[tokenizer.eos, position.start + 3 * depth + 3] = 20

    def forward(self, ids):
        length = ids.shape[1]
        if length > self.max_length:
            raise ValueError("Prefix exceeds the hand-coded routing layout")
        features = self.token_features[ids] + self.position_features[:length][None]
        hidden = features + sum(head(features) for head in self.heads)
        hidden = hidden + F.relu(hidden @ self.mlp_in.T + self.mlp_bias) @ self.mlp_out.T
        return hidden @ self.readout.T


class FixedOutcomeBlock(nn.Module):
    """One causal attention/MLP block applies one gate in hidden activations."""

    def __init__(
        self,
        tokenizer,
        width,
        positions,
        position_features,
        token_gate,
        previous_state,
        next_state,
        retrieved_gate,
        gate_position,
        answer_position,
        initial_state=None,
    ):
        super().__init__()
        states, gates = tokenizer.n_states, len(tokenizer.gates)
        n_bits = int(math.log2(states))
        self.gate_head = FixedAttentionHead(width, positions)
        self.state_head = FixedAttentionHead(width, positions)
        for head in (self.gate_head, self.state_head):
            head.key[:, position_features] = torch.eye(positions)

        self.gate_head.query[gate_position, position_features.start + answer_position] = 80
        self.gate_head.value[retrieved_gate, token_gate] = torch.eye(gates)
        if initial_state is not None:
            self.state_head.query[0, position_features.start + answer_position] = 80
            self.state_head.value[previous_state, initial_state] = torch.eye(states)

        self.register_buffer("mlp_in", torch.zeros(states * gates, width))
        self.register_buffer("mlp_bias", torch.full((states * gates,), -2.5))
        self.register_buffer("mlp_out", torch.zeros(width, states * gates))
        for state_id in range(states):
            for gate_id, gate_name in enumerate(tokenizer.gates):
                unit = state_id * gates + gate_id
                self.mlp_in[unit, previous_state.start + state_id] = 1
                self.mlp_in[unit, retrieved_gate.start + gate_id] = 1
                self.mlp_in[unit, position_features.start + answer_position] = 1
                target = phi(state_id, gate_name, n_bits)
                self.mlp_out[next_state.start + target, unit] = 2

    def forward(self, hidden):
        hidden = hidden + self.gate_head(hidden) + self.state_head(hidden)
        activation = F.relu(hidden @ self.mlp_in.T + self.mlp_bias)
        return hidden + activation @ self.mlp_out.T


class HandcodedOutcomeTransformer(nn.Module):
    """Fixed-depth causal Transformer that emits only COLON, answer, and EOS."""

    def __init__(self, tokenizer, depth=DEPTH, positions=None):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        states, gates = tokenizer.n_states, len(tokenizer.gates)
        positions = depth + 4 if positions is None else positions
        answer_position = depth + 2
        self.max_length = positions

        token_state = slice(0, states)
        token_gate = slice(token_state.stop, token_state.stop + gates)
        position = slice(token_gate.stop, token_gate.stop + positions)
        state_slots = [
            slice(position.stop + step * states, position.stop + (step + 1) * states)
            for step in range(depth + 1)
        ]
        gate_slots = [
            slice(state_slots[-1].stop + step * gates, state_slots[-1].stop + (step + 1) * gates)
            for step in range(depth)
        ]
        width = gate_slots[-1].stop
        self.register_buffer("token_features", torch.zeros(len(tokenizer.tokens), width))
        self.register_buffer("position_features", torch.zeros(positions, width))
        self.register_buffer("readout", torch.zeros(len(tokenizer.tokens), width))
        self.token_features[:states, token_state] = torch.eye(states)
        self.token_features[states : states + gates, token_gate] = torch.eye(gates)
        self.position_features[:, position] = torch.eye(positions)

        self.blocks = nn.ModuleList(
            [
                FixedOutcomeBlock(
                    tokenizer,
                    width,
                    positions,
                    position,
                    token_gate,
                    state_slots[step],
                    state_slots[step + 1],
                    gate_slots[step],
                    gate_position=step + 1,
                    answer_position=answer_position,
                    initial_state=token_state if step == 0 else None,
                )
                for step in range(depth)
            ]
        )
        self.readout[:states, state_slots[-1]] = 20 * torch.eye(states)
        self.readout[tokenizer.colon, position.start + depth + 1] = 20
        self.readout[tokenizer.eos, position.start + depth + 3] = 20

    def forward(self, ids):
        if ids.shape[1] > self.max_length:
            raise ValueError("Prefix exceeds the fixed outcome routing layout")
        hidden = self.token_features[ids] + self.position_features[: ids.shape[1]][None]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden @ self.readout.T


def build_random_trainable_process_architecture(
    tokenizer,
    depth=DEPTH,
    *,
    seed=MODEL_SEED,
    init_std=0.02,
    device=None,
):
    model = HandcodedProcessTransformer(tokenizer, depth)
    return make_random_trainable_copy(model, seed=seed, init_std=init_std, device=device)


def build_random_trainable_outcome_architecture(
    tokenizer,
    depth=DEPTH,
    *,
    seed=MODEL_SEED,
    init_std=0.02,
    device=None,
):
    # Use the process-length position budget so this architecture can also be
    # trained/evaluated with process supervision in the 2x2 comparison.
    model = HandcodedOutcomeTransformer(tokenizer, depth, positions=3 * depth + 4)
    return make_random_trainable_copy(model, seed=seed, init_std=init_std, device=device)


@torch.no_grad()
def generate(model, prompts, max_new_tokens, eos_id):
    """Greedy autoregression; only previously generated tokens are fed back."""
    was_training = model.training
    model.eval()
    ids = prompts.clone()
    finished = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    try:
        for _ in range(max_new_tokens):
            next_ids = model(ids)[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, eos_id, next_ids)
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            finished |= next_ids == eos_id
            if finished.all():
                break
        return ids
    finally:
        model.train(was_training)


def strip_after_eos(ids, eos_id):
    result = list(map(int, ids))
    return result[: result.index(eos_id) + 1] if eos_id in result else result


@dataclass
class LanguageBatch:
    inputs: torch.Tensor
    targets: torch.Tensor

    def __len__(self):
        return len(self.inputs)

    def select(self, indices):
        return LanguageBatch(self.inputs[indices], self.targets[indices])

    def to(self, device):
        return LanguageBatch(self.inputs.to(device), self.targets.to(device))


def encode_dataset(circuits, tokenizer, mode):
    inputs, targets = [], []
    for circuit in circuits:
        prompt = tokenizer.prompt(circuit)
        full = prompt + tokenizer.continuation(circuit, mode)
        inputs.append(full[:-1])
        target = full[1:]
        target[: len(prompt) - 1] = [-100] * (len(prompt) - 1)
        targets.append(target)
    return LanguageBatch(torch.tensor(inputs), torch.tensor(targets))


def language_model_loss(model, batch):
    logits = model(batch.inputs)
    return F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100)


def make_batch_schedule(size, steps, batch_size, seed=BATCH_SEED):
    rng = random.Random(seed)
    return [[rng.randrange(size) for _ in range(batch_size)] for _ in range(steps)]


@dataclass
class GenerationEvaluation:
    prompts: torch.Tensor
    targets: dict
    answers: list[int]
    depth: int


def make_generation_evaluation(circuits, tokenizer, device):
    return GenerationEvaluation(
        prompts=torch.tensor([tokenizer.prompt(c) for c in circuits], device=device),
        targets={mode: [tokenizer.continuation(c, mode) for c in circuits] for mode in ("outcome", "process")},
        answers=[c.answer for c in circuits],
        depth=len(circuits[0].gates),
    )


def generated_answer(tokens, tokenizer):
    if tokenizer.colon not in tokens:
        return None
    index = tokens.index(tokenizer.colon) + 1
    if index < len(tokens) and tokens[index] < tokenizer.n_states:
        return tokens[index]
    return None


@torch.no_grad()
def free_run_metrics(model, evaluation, tokenizer, mode):
    budget = 3 if mode == "outcome" else 2 * evaluation.depth + 3
    output = generate(model, evaluation.prompts, budget, tokenizer.eos)
    continuations = output[:, evaluation.prompts.shape[1] :].cpu().tolist()
    predictions = [strip_after_eos(row, tokenizer.eos) for row in continuations]
    targets = evaluation.targets[mode]
    count = len(targets)
    return {
        "exact_continuation": sum(p == t for p, t in zip(predictions, targets)) / count,
        "final_answer": sum(generated_answer(p, tokenizer) == a for p, a in zip(predictions, evaluation.answers))
        / count,
    }


def make_circuit_prompts(gates, tokenizer, device):
    """Hold the gate sequence fixed and vary the initial state over all values."""
    if not gates or any(gate not in tokenizer.gates for gate in gates):
        raise ValueError("Choose a nonempty sequence of known gates")
    rows = [
        [state] + [tokenizer.ids[gate] for gate in gates] + [tokenizer.sep]
        for state in range(tokenizer.n_states)
    ]
    return torch.tensor(rows, device=device)


@torch.no_grad()
def circuit_answer_matrix(model, prompts, tokenizer, mode):
    """Read final-state probabilities after the model generates its own prefix."""
    depth = prompts.shape[1] - 2
    prefix_length = 1 if mode == "outcome" else 2 * depth + 1
    prefix = generate(model, prompts, prefix_length, tokenizer.eos)
    probabilities = model(prefix)[:, -1].softmax(dim=-1)[:, : tokenizer.n_states]
    valid = (prefix[:, -1] == tokenizer.colon) & ~(
        prefix[:, prompts.shape[1] :] == tokenizer.eos
    ).any(dim=1)
    return (probabilities * valid[:, None]).detach().cpu().numpy()


@torch.no_grad()
def inspect_fixed_circuit(model, gates, tokenizer, device):
    """Capture actual residual-stream state features after each fixed block."""
    if len(gates) != len(model.blocks):
        raise ValueError("Gate sequence must match the fixed model's depth")
    prompts = make_circuit_prompts(gates, tokenizer, device)
    prefix = generate(model, prompts, 1, tokenizer.eos)
    layer_matrices, handles = [], []

    def capture_state(block, inputs, output):
        state_rows = torch.where(block.mlp_out.abs().sum(dim=1) > 0)[0]
        layer_matrices.append(output[:, -1, state_rows].detach().cpu().numpy())

    try:
        for block in model.blocks:
            handles.append(block.register_forward_hook(capture_state))
        model(prefix)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "gates": list(gates),
        "layers": layer_matrices,
        "answer_matrix": circuit_answer_matrix(model, prompts, tokenizer, "outcome"),
    }


def train_step(model, batch, optimizer, grad_clip_norm=None):
    """Run one teacher-forced next-token update and return the scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = language_model_loss(model, batch)
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return loss.item()


def _prefix_loss_sample(data, max_examples):
    """Return a deterministic prefix sample for inexpensive checkpoint losses."""
    if data is None:
        return None
    if max_examples is None:
        return data
    if max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    return data.select(slice(0, min(max_examples, len(data))))


def _mode_data(data_by_mode, mode):
    """Accept either a mode->LanguageBatch mapping or a single LanguageBatch."""
    if data_by_mode is None:
        return None
    if isinstance(data_by_mode, dict):
        return data_by_mode.get(mode)
    return data_by_mode


@torch.no_grad()
def evaluate_checkpoint(
    model,
    mode,
    step,
    train_loss_sample,
    train_eval,
    test_eval,
    tokenizer,
    circuit_prompts,
    architecture=None,
    test_loss_sample=None,
):
    """Evaluate teacher-forced losses and free-running answer metrics."""
    train_metrics = free_run_metrics(model, train_eval, tokenizer, mode)
    test_metrics = free_run_metrics(model, test_eval, tokenizer, mode)
    row = {
        "circuit_matrix": circuit_answer_matrix(model, circuit_prompts, tokenizer, mode),
        "step": step,
        "architecture": architecture,
        "mode": mode,
        "train_loss": language_model_loss(model, train_loss_sample).item(),
        "train_answer_accuracy_sample": train_metrics["final_answer"],
        "test_answer_accuracy": test_metrics["final_answer"],
        "test_exact_continuation": test_metrics["exact_continuation"],
    }
    row["test_loss"] = (
        language_model_loss(model, test_loss_sample).item()
        if test_loss_sample is not None
        else np.nan
    )
    return row


def _format_progress_value(value, *, precision=3, percent=False):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}" if percent else f"{value:.{precision}f}"


def train_one_model(
    base,
    mode,
    data,
    batch_schedule,
    learning_rate,
    checkpoints,
    train_eval,
    test_eval,
    tokenizer,
    circuit_prompts,
    architecture=None,
    test_loss_data=None,
    loss_eval_size=256,
    progress_leave=True,
    grad_clip_norm=None,
):
    """Train a fresh copy of ``base`` and return the model plus checkpoint history.

    ``test_loss_data`` is optional and is used only for teacher-forced held-out
    loss. Free-running test accuracy is always computed from ``test_eval``.
    Set ``grad_clip_norm`` to clip gradient norm before each optimizer step.
    """
    model = copy.deepcopy(base)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Cannot train a model with no trainable parameters")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0)
    train_loss_sample = _prefix_loss_sample(data, loss_eval_size)
    test_loss_sample = _prefix_loss_sample(test_loss_data, loss_eval_size)
    checkpoint_set = set(checkpoints)
    label = f"{architecture or 'model'}/{mode}"

    history = [
        evaluate_checkpoint(
            model,
            mode,
            0,
            train_loss_sample,
            train_eval,
            test_eval,
            tokenizer,
            circuit_prompts,
            architecture=architecture,
            test_loss_sample=test_loss_sample,
        )
    ]

    progress = tqdm(
        enumerate(batch_schedule, 1),
        total=len(batch_schedule),
        desc=label,
        leave=progress_leave,
    )
    for step, indices in progress:
        train_step(model, data.select(indices), optimizer, grad_clip_norm=grad_clip_norm)
        if step not in checkpoint_set:
            continue

        row = evaluate_checkpoint(
            model,
            mode,
            step,
            train_loss_sample,
            train_eval,
            test_eval,
            tokenizer,
            circuit_prompts,
            architecture=architecture,
            test_loss_sample=test_loss_sample,
        )
        history.append(row)
        progress.set_postfix(
            train_loss=_format_progress_value(row["train_loss"]),
            test_loss=_format_progress_value(row["test_loss"]),
            train=_format_progress_value(row["train_answer_accuracy_sample"], percent=True),
            test=_format_progress_value(row["test_answer_accuracy"], percent=True),
        )

    return model, pd.DataFrame(history)


def run_experiment(
    base,
    training_data,
    batch_schedule,
    learning_rate,
    checkpoints,
    train_eval,
    test_eval,
    tokenizer,
    circuit_prompts,
    modes=("outcome", "process"),
    test_loss_data=None,
    loss_eval_size=256,
    progress_leave=True,
    grad_clip_norm=None,
):
    """Train independent random copies of one base model under each supervision mode."""
    models, histories = {}, []
    for mode in tqdm(modes, desc="supervision modes", leave=progress_leave):
        model, history = train_one_model(
            base,
            mode,
            training_data[mode],
            batch_schedule,
            learning_rate,
            checkpoints,
            train_eval,
            test_eval,
            tokenizer,
            circuit_prompts,
            test_loss_data=_mode_data(test_loss_data, mode),
            loss_eval_size=loss_eval_size,
            progress_leave=progress_leave,
            grad_clip_norm=grad_clip_norm,
        )
        models[mode] = model
        histories.append(history)
    return models, pd.concat(histories, ignore_index=True)


def run_architecture_experiment(
    bases,
    training_data,
    batch_schedule,
    learning_rate,
    checkpoints,
    train_eval,
    test_eval,
    tokenizer,
    circuit_prompts,
    modes=("outcome", "process"),
    test_loss_data=None,
    loss_eval_size=256,
    progress_leave=True,
    grad_clip_norm=None,
):
    """Train each architecture under each supervision mode.

    Returns models keyed by ``(architecture, mode)`` so the 2x2 comparison stays
    explicit: outcome architecture/process architecture crossed with
    outcome/process supervision.
    """
    models, histories = {}, []
    jobs = [(architecture, base, mode) for architecture, base in bases.items() for mode in modes]
    for architecture, base, mode in tqdm(jobs, desc="architecture/mode", leave=progress_leave):
        model, history = train_one_model(
            base,
            mode,
            training_data[mode],
            batch_schedule,
            learning_rate,
            checkpoints,
            train_eval,
            test_eval,
            tokenizer,
            circuit_prompts,
            architecture=architecture,
            test_loss_data=_mode_data(test_loss_data, mode),
            loss_eval_size=loss_eval_size,
            progress_leave=progress_leave,
            grad_clip_norm=grad_clip_norm,
        )
        models[(architecture, mode)] = model
        histories.append(history)
    return models, pd.concat(histories, ignore_index=True)
