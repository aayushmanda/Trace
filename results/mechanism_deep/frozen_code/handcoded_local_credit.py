"""Local credit on a deep outcome architecture: hidden CE, probes, patches, C_t.

The outcome model generates COLON / answer / EOS only. Intermediate states s_t are
supervised by a linear readout H_t on the residual stream after block t — not by
teacher-forced trace tokens. Do not claim an ε^{D-1} mixing-ball rate from C_t.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from handcoded.gates import phi
from handcoded.models import attach_local_heads

INTERVENTION_BACKEND = "nnsight"


def nnsight_available():
    try:
        from nnsight import NNsight  # noqa: F401
        return True
    except Exception:
        return False


def wrap_nnsight(model):
    """Wrap a generic `nn.Module` once. Does not replace loss / grad math."""
    from nnsight import NNsight
    wrapper = getattr(model, "_trace_nnsight", None)
    if wrapper is None:
        wrapper = NNsight(model)
        model._trace_nnsight = wrapper
    return wrapper


def _module_device(model, fallback="cpu"):
    try:
        return next(model.parameters()).device
    except StopIteration:
        buf = next(model.buffers(), None)
        return buf.device if buf is not None else torch.device(fallback)


def gold_states_tensor(circuits, device=None):
    tensor = torch.tensor([c.states for c in circuits], dtype=torch.long)
    return tensor.to(device) if device is not None else tensor


def compose_from(state, gates, n_bits=4):
    for gate in gates:
        state = phi(state, gate, n_bits)
    return state


def counterfactual_answer(circuit, layer, donor, n_bits=4):
    """Remaining gates of the *base* circuit: Φ_{g_D} ∘ … ∘ Φ_{g_{t+1}}(s'_t)."""
    return compose_from(int(donor), circuit.gates[layer + 1 :], n_bits)


def prompt_with_colon(circuits, tokenizer, device):
    rows = [tokenizer.prompt(c) + [tokenizer.colon] for c in circuits]
    return torch.tensor(rows, device=device)


def outcome_plus_local_loss(model, batch, gold_states, lambda_local=1.0):
    """L_out + λ (1/D) Σ_t CE(H_t(h_t), s_t). `batch` is outcome-format (no trace tokens).

    Hidden states here use `return_states` so autograd reaches θ. nnsight is for
    held-out probes and causal patches, not this loss.
    """
    if getattr(model, "local_heads", None) is None:
        attach_local_heads(model)
    logits, hiddens = model(batch.inputs, return_states=True)
    terminal = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100)
    local = logits.new_zeros(())
    depth = len(hiddens)
    for step, hidden in enumerate(hiddens):
        local = local + F.cross_entropy(model.local_heads[step](hidden), gold_states[:, step])
    local = local / depth
    return terminal + float(lambda_local) * local, terminal, local


def trained_target_ids(batch):
    return batch.targets[batch.targets != -100]


def has_gate_tokens(target_ids, tokenizer):
    lo, hi = tokenizer.n_states, tokenizer.n_states + len(tokenizer.gates)
    return bool(((target_ids >= lo) & (target_ids < hi)).any().item())


def _slot(model, layer):
    slot = model.state_slots[layer + 1]
    return slot.start, slot.stop


def oracle_slot_patch(hidden, model, layer, donors):
    """Replace the construction's next-state slot after block `layer` with one-hot s'_t."""
    start, stop = _slot(model, layer)
    pos = model._answer_index(hidden.shape[1])
    out = hidden.clone()
    out[:, pos, start:stop] = 0
    batch = torch.arange(hidden.size(0), device=hidden.device)
    out[batch, pos, start + donors] = 1.0
    return out


def _noise(shape, *, device, dtype, generator=None):
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def random_subspace_patch(hidden, model, layer, donors, generator=None):
    """Same-norm write into a random 16-d coordinate set; zeros the true state slot."""
    del donors
    start, stop = _slot(model, layer)
    pos = model._answer_index(hidden.shape[1])
    width = hidden.shape[-1]
    n_states = stop - start
    perm = torch.randperm(width, generator=generator)[:n_states].to(hidden.device)
    out = hidden.clone()
    original = out[:, pos, start:stop]
    norm = original.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    out[:, pos, start:stop] = 0
    noise = _noise((hidden.size(0), n_states), device=hidden.device, dtype=hidden.dtype, generator=generator)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8) * norm
    out[:, pos, perm] = noise
    return out


def unstructured_patch(hidden, model, layer, donors, generator=None):
    """Gaussian residual of the same L2 as the oracle slot write."""
    pos = model._answer_index(hidden.shape[1])
    oracle = oracle_slot_patch(hidden, model, layer, donors)
    delta = (oracle - hidden)[:, pos, :]
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = _noise((hidden.size(0), hidden.size(-1)), device=hidden.device, dtype=hidden.dtype, generator=generator)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8) * norm
    out = hidden.clone()
    out[:, pos, :] = hidden[:, pos, :] + noise
    return out


def probe_subspace_patch(hidden, model, layer, donors, probe, logit_scale=10.0):
    """Min-norm edit so a linear probe decodes as s'_t at the answer position."""
    pos = model._answer_index(hidden.shape[1])
    residual = hidden[:, pos, :]
    weight, bias = probe.weight, probe.bias
    current = F.linear(residual, weight, bias)
    target = torch.zeros_like(current)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    target[batch, donors] = float(logit_scale)
    delta = target - current
    inverse = torch.linalg.pinv(weight)
    out = hidden.clone()
    out[:, pos, :] = residual + delta @ inverse.T
    return out


def apply_residual_patch(hidden, model, layer, donors, method, *, generator=None, probe=None):
    """Tensor edit of the residual after block `layer`. Called *inside* an nnsight trace."""
    if method == "oracle_slot":
        return oracle_slot_patch(hidden, model, layer, donors)
    if method == "random_subspace":
        return random_subspace_patch(hidden, model, layer, donors, generator=generator)
    if method == "unstructured":
        return unstructured_patch(hidden, model, layer, donors, generator=generator)
    if method == "probe_subspace":
        if probe is None:
            raise ValueError("probe_subspace patch needs a trained probe")
        return probe_subspace_patch(hidden, model, layer, donors, probe)
    raise ValueError(f"Unknown patch method: {method}")


@torch.no_grad()
def patched_answers(model, circuits, tokenizer, layer, donors, method="oracle_slot",
                    generator=None, probe=None, wrong_layer=None, n_bits=4):
    """nnsight trace: replace block-t residual, read remaining network's answer logits."""
    del n_bits
    ids = prompt_with_colon(circuits, tokenizer, donors.device)
    applied = layer if wrong_layer is None else wrong_layer
    wrapped = wrap_nnsight(model)
    with wrapped.trace(ids):
        hidden = wrapped.blocks[applied].output
        patched = apply_residual_patch(
            hidden, model, applied, donors, method, generator=generator, probe=probe,
        )
        wrapped.blocks[applied].output = patched
        logits = wrapped.output.save()
    return logits[:, -1, : tokenizer.n_states].argmax(dim=-1)


@torch.no_grad()
def counterfactual_accuracy(
    model, circuits, tokenizer, layer, method="oracle_slot", device=None, n_bits=4,
    generator=None, probe=None, wrong_layer=None,
):
    """Fraction of (circuit, donor) pairs whose patched answer matches remaining-gate execution."""
    if device is None:
        device = _module_device(model)
    n_states = tokenizer.n_states
    correct, total = 0, 0
    model.eval()
    for donor_id in range(n_states):
        donors = torch.full((len(circuits),), donor_id, dtype=torch.long, device=device)
        predicted = patched_answers(
            model, circuits, tokenizer, layer, donors, method=method,
            generator=generator, probe=probe, wrong_layer=wrong_layer, n_bits=n_bits,
        )
        gold = torch.tensor(
            [counterfactual_answer(c, layer, donor_id, n_bits) for c in circuits],
            device=device,
        )
        correct += int((predicted == gold).sum().item())
        total += len(circuits)
    return correct / max(total, 1)


@torch.no_grad()
def collect_hiddens(model, circuits, tokenizer, device=None):
    """nnsight: answer-position residual after each block, for linear probes h_t → s_t."""
    import nnsight
    if device is None:
        device = _module_device(model)
    ids = prompt_with_colon(circuits, tokenizer, device)
    pos = model._answer_index(ids.shape[1])
    wrapped = wrap_nnsight(model)
    with wrapped.trace(ids):
        states = []
        for block in wrapped.blocks:
            states.append(block.output[:, pos, :])
        states = nnsight.save(states)
    gold = gold_states_tensor(circuits, device)
    return [tensor.detach() for tensor in states], gold


def train_linear_probes(
    train_hiddens, train_gold, eval_hiddens, eval_gold, n_states, *, steps=80, lr=0.05,
):
    """Freeze-LM linear H_t. `*_hiddens` is a list of (N, width) tensors, one per block."""
    probes, train_acc, eval_acc = [], [], []
    for step, hidden in enumerate(train_hiddens):
        hidden = hidden.detach()
        eval_hidden = eval_hiddens[step].detach()
        probe = nn.Linear(hidden.shape[-1], n_states)
        probe = probe.to(device=hidden.device, dtype=hidden.dtype)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        y_train = train_gold[:, step]
        y_eval = eval_gold[:, step]
        probe.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(hidden), y_train)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            train_acc.append(float((probe(hidden).argmax(-1) == y_train).float().mean()))
            eval_acc.append(float((probe(eval_hidden).argmax(-1) == y_eval).float().mean()))
        probes.append(probe)
    return probes, train_acc, eval_acc


def _flat(grads, params):
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            pieces.append(grad.detach().reshape(-1).float())
    return torch.cat(pieces) if pieces else torch.zeros(1)


def credit_signals(model, batch, gold_states, probes, layers=None):
    """A_t = cosine(g_term, g_local); C_t = ⟨g_term, g_local⟩ / ||g_local||² on block t.

    g_term = -∇_{θ_t} L_out, g_local = -∇_{θ_t} CE(H_t(h_t), s_t). Report C_t vs D;
    do not interpret C_t as an ε^{D-1} mixing-ball constant.
    """
    layers = list(range(len(model.blocks)) if layers is None else layers)
    logits, hiddens = model(batch.inputs, return_states=True)
    terminal = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100)
    rows = []
    for layer in layers:
        params = [p for p in model.blocks[layer].parameters() if p.requires_grad]
        if not params:
            continue
        g_term = torch.autograd.grad(terminal, params, retain_graph=True, allow_unused=True)
        readout = probes[layer] if probes is not None else model.local_heads[layer]
        local = F.cross_entropy(readout(hiddens[layer]), gold_states[:, layer])
        g_local = torch.autograd.grad(local, params, retain_graph=True, allow_unused=True)
        term_vec = -_flat(g_term, params)
        local_vec = -_flat(g_local, params)
        term_norm = float(term_vec.norm())
        local_norm = float(local_vec.norm())
        cosine = float("nan")
        c_value = float("nan")
        if term_norm > 0 and local_norm > 0:
            cosine = float(F.cosine_similarity(term_vec[None], local_vec[None]))
            c_value = float((term_vec @ local_vec) / (local_vec.square().sum() + 1e-12))
        rows.append({
            "layer": layer,
            "cosine": cosine,
            "C_t": c_value,
            "term_grad_norm": term_norm,
            "local_grad_norm": local_norm,
        })
    return rows


def chance_accuracy(n_states=16):
    return 1.0 / n_states
