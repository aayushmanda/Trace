"""NNsight-backed hidden collection and causal patching for outcome-local tests.

Collects h_t at block t, trains linear probes h_t -> s_t, and patches state at t so
downstream execution matches Phi_{g_{t+1:D}}(s'_t). Controls: oracle slot, random
subspace, unstructured noise, wrong layer. Mixed-format P_g is negative transfer,
not identification.
"""
from __future__ import annotations

import importlib

import torch

from handcoded.local_credit import (
    _patch_fn,
    counterfactual_answer,
    gold_states_tensor,
    prompt_with_colon,
    train_linear_probes,
)


def _nnsight():
    return importlib.import_module("nnsight")


def wrap_nnsight(model):
    return _nnsight().NNsight(model)


@torch.no_grad()
def collect_hiddens_nnsight(model, circuits, tokenizer, device=None):
    """Residual stream at the answer position after each block (via nnsight trace)."""
    if device is None:
        device = next(model.parameters()).device
    ids = prompt_with_colon(circuits, tokenizer, device)
    wrapped = wrap_nnsight(model)
    depth = len(model.blocks)
    answer_idx = model._answer_index(ids.shape[1])
    saved = [None] * depth
    with wrapped.trace(ids):
        for step in range(depth):
            saved[step] = wrapped.blocks[step].output[:, answer_idx, :].save()
    return [tensor.detach() for tensor in saved], gold_states_tensor(circuits, device)


@torch.no_grad()
def patched_answers_nnsight(
    model, circuits, tokenizer, layer, donors, method="oracle_slot",
    generator=None, probe=None, wrong_layer=None, n_bits=4,
):
    del n_bits
    ids = prompt_with_colon(circuits, tokenizer, donors.device)
    wrapped = wrap_nnsight(model)
    patch_fn, applied = _patch_fn(
        method, model, layer, donors, generator=generator, probe=probe, wrong_layer=wrong_layer,
    )
    with wrapped.trace(ids):
        hidden = wrapped.blocks[applied].output
        patched = patch_fn(hidden, applied)
        wrapped.blocks[applied].output[:] = patched
        logits = wrapped.output.save()
    return logits[:, -1, : tokenizer.n_states].argmax(dim=-1)


@torch.no_grad()
def counterfactual_accuracy_nnsight(
    model, circuits, tokenizer, layer, method="oracle_slot", device=None, n_bits=4,
    generator=None, probe=None, wrong_layer=None,
):
    if device is None:
        device = next(model.parameters()).device
    n_states = tokenizer.n_states
    correct, total = 0, 0
    model.eval()
    for donor_id in range(n_states):
        donors = torch.full((len(circuits),), donor_id, dtype=torch.long, device=device)
        predicted = patched_answers_nnsight(
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


def evaluate_probes_nnsight(
    model, probe_train, probe_eval, tokenizer, device, n_states, probe_steps, probe_lr,
):
    was_training = model.training
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    train_h, train_y = collect_hiddens_nnsight(model, probe_train, tokenizer, device)
    eval_h, eval_y = collect_hiddens_nnsight(model, probe_eval, tokenizer, device)
    probes, train_acc, eval_acc = train_linear_probes(
        train_h, train_y, eval_h, eval_y, n_states, steps=probe_steps, lr=probe_lr,
    )
    for param in model.parameters():
        param.requires_grad_(True)
    model.train(was_training)
    return probes, train_acc, eval_acc
