"""Learned one-layer Transformer vs fixed process/outcome executors (weights = rules)."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from handcoded.config import DEPTH, D_FF, D_MODEL, MODEL_SEED, N_HEADS
from handcoded.gates import phi


class LearnedOneLayerTransformer(nn.Module):
    """One causal attention layer, four heads, residuals, ReLU MLP, no LayerNorm."""

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
        if ids.ndim != 2:
            raise ValueError(f"ids must be (B, T), got {tuple(ids.shape)}")
        length = ids.shape[1]
        if length > self.max_length:
            raise ValueError(f"Input length {length} exceeds {self.max_length}")
        positions = torch.arange(length, device=ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions)[None]
        causal_mask = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
        attended, weights = self.attention(
            hidden, hidden, hidden, attn_mask=causal_mask,
            need_weights=return_attention, average_attn_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(hidden)
        logits = self.readout(hidden)
        return (logits, weights) if return_attention else logits


def build_random_learned_model(
    tokenizer, depth=DEPTH, *, seed=MODEL_SEED, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, device=None,
):
    if seed is not None:
        torch.manual_seed(seed)
    model = LearnedOneLayerTransformer(
        len(tokenizer.tokens), max_length=3 * depth + 5, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
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
    """Same layout as a fixed executor, random trainable weights (reachability / 2×2)."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    trainable = copy.deepcopy(model).cpu()
    _replace_buffers_with_random_parameters(trainable, generator=generator, init_std=init_std)
    return trainable.to(device) if device is not None else trainable


class FixedAttentionHead(nn.Module):
    """Causal attention whose Q/K/V are fixed buffers (hard-wired routing)."""

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
    """Fixed weights that emit the gold gate/state trace then COLON, answer, EOS."""

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

        # Route: copy previous state onto the next gate slot; copy the next gate; copy final state.
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

        # MLP unit (s, g) fires iff retrieved state is s and current gate is g; writes φ(s,g).
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
    """One block = one gate: attend to that gate and the previous hidden state, write φ."""

    def __init__(
        self, tokenizer, width, positions, position_features, token_gate,
        previous_state, next_state, retrieved_gate, gate_position, answer_position, initial_state=None,
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
    """Depth-D blocks compute the answer internally; output is only COLON, answer, EOS."""

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
                    tokenizer, width, positions, position, token_gate,
                    state_slots[step], state_slots[step + 1], gate_slots[step],
                    gate_position=step + 1, answer_position=answer_position,
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
    tokenizer, depth=DEPTH, *, seed=MODEL_SEED, init_std=0.02, device=None,
):
    model = HandcodedProcessTransformer(tokenizer, depth)
    return make_random_trainable_copy(model, seed=seed, init_std=init_std, device=device)


def build_random_trainable_outcome_architecture(
    tokenizer, depth=DEPTH, *, seed=MODEL_SEED, init_std=0.02, device=None,
):
    # Process-length positions so this architecture can train under process supervision (2×2).
    model = HandcodedOutcomeTransformer(tokenizer, depth, positions=3 * depth + 4)
    return make_random_trainable_copy(model, seed=seed, init_std=init_std, device=device)
