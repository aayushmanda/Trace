"""Learned one-layer Transformer vs fixed process/outcome executors (weights = rules)."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from handcoded import config
from handcoded.gates import phi

_UNSET = object()   # distinguishes 'use the config default' from an explicit None


class LearnedOneLayerTransformer(nn.Module):
    """One causal attention layer, four heads, residuals, ReLU MLP, no LayerNorm."""

    def __init__(self, vocab_size, max_length, d_model=None, n_heads=None, d_ff=None):
        super().__init__()
        d_model = config.D_MODEL if d_model is None else d_model
        n_heads = config.N_HEADS if n_heads is None else n_heads
        d_ff = config.D_FF if d_ff is None else d_ff
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.readout = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, return_attention=False):
        length = ids.shape[1]
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
    tokenizer, depth=None, *, seed=_UNSET, d_model=None, n_heads=None, d_ff=None, device=None,
):
    depth = config.DEPTH if depth is None else depth
    seed = config.MODEL_SEED if seed is _UNSET else seed
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


def make_random_trainable_copy(model, *, seed=_UNSET, init_std=0.02, device=None):
    """Same layout as a fixed executor, random trainable weights (reachability / 2×2)."""
    seed = config.MODEL_SEED if seed is _UNSET else seed
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


def attach_local_heads(model, *, seed=None, device=None):
    """Linear H_t: residual stream after block t → 16 states. Jointly trained for L_out+local."""
    width = int(getattr(model, "residual_width", model.token_features.shape[1]))
    n_states = int(getattr(model, "n_states", 16))
    depth = int(getattr(model, "depth", len(model.blocks)))
    if seed is not None:
        torch.manual_seed(seed)
    heads = nn.ModuleList([nn.Linear(width, n_states) for _ in range(depth)])
    model.add_module("local_heads", heads)
    if device is not None:
        model.local_heads.to(device)
    elif next(model.parameters()).is_cuda:
        model.local_heads.to(next(model.parameters()).device)
    return model


# =============================================================================
# One architecture, two exact solutions.
#
# `UnifiedExecutor` is a D-block residual Transformer in which BOTH the process
# and the outcome executor exist as different values of the *same* buffers. It
# replaces an earlier pair of separate fixed executors that differed in blocks,
# heads, width and parameter count -- comparing those confounded supervision with
# architecture. These two differ only in weight values, so:
#
#     base = UnifiedExecutor(tok, depth)                  # all zeros
#     p    = build_fixed_process(tok, depth)              # exact, free-runs 1.0
#     o    = build_fixed_outcome(tok, depth)              # exact, free-runs 1.0
#     m    = build_random_trainable_unified(tok, depth)   # ONE theta_0 for both losses
#
# Residual slot layout (width = K + M + P + (D+1)K + DM, P = 3D+5):
#
#     token_state  [0, K)           one-hot if this token is a state
#     token_gate   [K, K+M)         one-hot if this token is a gate
#     position     [K+M, K+M+P)     one-hot position
#     state[t]     (D+1) x K        outcome: running state after block t
#                                   process: state[0] retrieved source,
#                                            state[1] computed successor
#     gate[t]      D x M            outcome: gate retrieved by block t
#                                   process: gate[0] next gate to emit
#
# Outcome uses all D blocks (2 heads each); process uses block 0 (3 heads) and
# makes blocks 1.. exact identities. Known limitation: each block owns its own
# mlp_in/mlp_out, so a randomized copy has D *independent* rule tables rather
# than one shared transition tensor. Tying them requires in-place residual
# writes instead of the fresh per-block slots used here.
# =============================================================================

HEADS_PER_BLOCK = 3   # outcome needs 2, process needs 3; both blocks carry 3


class UnifiedBlock(nn.Module):
    """Identical shape for every block: n heads, one ReLU MLP, two residual adds."""

    def __init__(self, width, positions, d_ff, n_heads=HEADS_PER_BLOCK):
        super().__init__()
        self.heads = nn.ModuleList([FixedAttentionHead(width, positions) for _ in range(n_heads)])
        self.register_buffer("mlp_in", torch.zeros(d_ff, width))
        self.register_buffer("mlp_bias", torch.zeros(d_ff))
        self.register_buffer("mlp_out", torch.zeros(width, d_ff))

    def forward(self, hidden):
        hidden = hidden + sum(head(hidden) for head in self.heads)
        return hidden + F.relu(hidden @ self.mlp_in.T + self.mlp_bias) @ self.mlp_out.T


class UnifiedExecutor(nn.Module):
    """D blocks over a shared slot layout. Zero weights until a writer fills them."""

    def __init__(self, tokenizer, depth=None, n_heads=HEADS_PER_BLOCK):
        super().__init__()
        depth = config.DEPTH if depth is None else depth
        states, gates = tokenizer.n_states, len(tokenizer.gates)
        # A full free-running process rollout is prompt (D+2) + budget (2D+3) = 3D+5
        # tokens, and `generate` appends before testing for EOS -- so 3D+4 is exactly
        # one position short and overruns position_features on the last step.
        positions = 3 * depth + 5
        self.token_state = slice(0, states)
        self.token_gate = slice(states, states + gates)
        self.position = slice(states + gates, states + gates + positions)
        base = self.position.stop
        self.state = [slice(base + t * states, base + (t + 1) * states) for t in range(depth + 1)]
        base = self.state[-1].stop
        self.gate = [slice(base + t * gates, base + (t + 1) * gates) for t in range(depth)]
        width = self.gate[-1].stop

        self.depth, self.n_states, self.n_gates = depth, states, gates
        self.max_length = positions
        self.residual_width = width
        self.answer_position = depth + 2
        self.state_slots = self.state          # eval/attach_local_heads compatibility

        self.blocks = nn.ModuleList(
            [UnifiedBlock(width, positions, states * gates, n_heads) for _ in range(depth)]
        )
        self.register_buffer("token_features", torch.zeros(len(tokenizer.tokens), width))
        self.register_buffer("position_features", torch.zeros(positions, width))
        self.register_buffer("readout", torch.zeros(len(tokenizer.tokens), width))

    def _answer_index(self, length):
        return min(self.answer_position, length - 1)

    def forward(self, ids, return_states=False):
        hidden = self.token_features[ids] + self.position_features[: ids.shape[1]][None]
        states = []
        for block in self.blocks:
            hidden = block(hidden)
            if return_states:
                states.append(hidden[:, self._answer_index(hidden.shape[1]), :])
        logits = hidden @ self.readout.T
        return (logits, states) if return_states else logits


def _unified_embed(model, tokenizer):
    """One-hot token and position features; every head keys on position."""
    states, gates = model.n_states, model.n_gates
    model.token_features[:states, model.token_state] = torch.eye(states)
    model.token_features[states : states + gates, model.token_gate] = torch.eye(gates)
    model.position_features[:, model.position] = torch.eye(model.max_length)
    for block in model.blocks:
        for head in block.heads:
            head.key[:, model.position] = torch.eye(model.max_length)


def _unified_route(head, routes, destinations, position, fallback):
    """head attends from `destination` to `routes[destination]`, else `fallback`."""
    for destination in destinations:
        head.query[routes.get(destination, fallback), position.start + destination] = 80.0


def _unified_write_phi(block, tokenizer, src, gate, dst, extra=None, n_bits=None):
    """Unit (s, g) fires iff src holds s and gate holds g; writes phi(s, g) into dst."""
    states, gates = tokenizer.n_states, len(tokenizer.gates)
    fan_in = 2 + (extra is not None)
    block.mlp_bias.fill_(-(fan_in - 0.5))
    for state_id in range(states):
        for gate_id, gate_name in enumerate(tokenizer.gates):
            unit = state_id * gates + gate_id
            block.mlp_in[unit, src.start + state_id] = 1
            block.mlp_in[unit, gate.start + gate_id] = 1
            if extra is not None:
                block.mlp_in[unit, extra] = 1
            block.mlp_out[dst.start + phi(state_id, gate_name, n_bits), unit] = 2


def _unified_identity(block):
    """Exact identity block: zero attention values and zero MLP output."""
    for head in block.heads:
        head.value.zero_()
    block.mlp_out.zero_()
    block.mlp_bias.fill_(-1.0)


def write_process_solution_(model, tokenizer):
    """Block 0 does one local transition per output position; blocks 1.. are identities."""
    _unified_embed(model, tokenizer)
    depth, states, gates = model.depth, model.n_states, model.n_gates
    n_bits = int(math.log2(states))
    block = model.blocks[0]
    src, out_gate, dst = model.state[0], model.gate[0], model.state[1]

    separator = depth + 1
    state_routes, gate_routes = {}, {}
    for step in range(depth):
        at_gate = depth + 2 + 2 * step
        state_routes[at_gate] = 0 if step == 0 else at_gate - 1
        gate_routes[at_gate - 1] = step + 1
    final_routes = {3 * depth + 2: 3 * depth + 1}

    destinations = range(separator, model.max_length)
    state_head, gate_head, final_head = block.heads[:3]
    for head, routes in zip((state_head, gate_head, final_head),
                            (state_routes, gate_routes, final_routes)):
        _unified_route(head, routes, destinations, model.position, fallback=separator)
    state_head.value[src, model.token_state] = torch.eye(states)
    gate_head.value[out_gate, model.token_gate] = torch.eye(gates)
    final_head.value[dst, model.token_state] = torch.eye(states)

    _unified_write_phi(block, tokenizer, src, model.token_gate, dst, n_bits=n_bits)
    for rest in model.blocks[1:]:
        _unified_identity(rest)

    model.readout[:states, dst] = 20 * torch.eye(states)
    model.readout[states : states + gates, out_gate] = 20 * torch.eye(gates)
    model.readout[tokenizer.colon, model.position.start + 3 * depth + 1] = 20
    model.readout[tokenizer.eos, model.position.start + 3 * depth + 3] = 20
    return model


def write_outcome_solution_(model, tokenizer):
    """Block t applies gate t inside the residual; only the last state is read out."""
    _unified_embed(model, tokenizer)
    depth, states, gates = model.depth, model.n_states, model.n_gates
    n_bits = int(math.log2(states))
    marker = model.position.start + model.answer_position

    for step, block in enumerate(model.blocks):
        gate_head, state_head = block.heads[0], block.heads[1]
        gate_head.query[step + 1, marker] = 80.0            # gate g_{t+1} sits at position t+1
        gate_head.value[model.gate[step], model.token_gate] = torch.eye(gates)
        if step == 0:
            state_head.query[0, marker] = 80.0              # s0 sits at position 0
            state_head.value[model.state[0], model.token_state] = torch.eye(states)
        _unified_write_phi(block, tokenizer, model.state[step], model.gate[step],
                           model.state[step + 1], extra=marker, n_bits=n_bits)

    model.readout[:states, model.state[-1]] = 20 * torch.eye(states)
    model.readout[tokenizer.colon, model.position.start + depth + 1] = 20
    model.readout[tokenizer.eos, model.position.start + depth + 3] = 20
    return model


def build_fixed_process(tokenizer, depth=None, device=None):
    """Exact process executor inside the unified architecture."""
    model = write_process_solution_(UnifiedExecutor(tokenizer, depth), tokenizer)
    return model.to(device) if device is not None else model


def build_fixed_outcome(tokenizer, depth=None, device=None):
    """Exact outcome executor inside the unified architecture (same shapes as above)."""
    model = write_outcome_solution_(UnifiedExecutor(tokenizer, depth), tokenizer)
    return model.to(device) if device is not None else model


def build_random_trainable_unified(tokenizer, depth=None, *, seed=_UNSET, init_std=0.02, device=None):
    """One random theta_0 in the same class as both exact solutions.

    `run_experiment` deep-copies this per supervision mode, so process and outcome
    start from identical weights -- the control the two-architecture 2x2 cannot give.
    """
    return make_random_trainable_copy(
        UnifiedExecutor(tokenizer, depth), seed=seed, init_std=init_std, device=device
    )


def unified_solution_distance(model, target):
    """||theta - theta*||_2 over matching tensors; both now live in one space."""
    have = {**dict(model.named_buffers()), **dict(model.named_parameters())}
    want = {**dict(target.named_buffers()), **dict(target.named_parameters())}
    total = 0.0
    for name, reference in want.items():
        current = have.get(name)
        if current is not None and current.shape == reference.shape:
            total += (current.detach().cpu() - reference.detach().cpu()).pow(2).sum().item()
    return math.sqrt(total)


def unified_block_grad_norms(model):
    """||grad|| per block after a backward pass: where credit actually lands."""
    norms = []
    for block in model.blocks:
        total = sum(p.grad.pow(2).sum().item() for p in block.parameters() if p.grad is not None)
        norms.append(math.sqrt(total))
    return norms
