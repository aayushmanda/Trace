"""Free-run accuracy, whole-circuit answer matrices, fixed-model hidden states."""
from dataclasses import dataclass

import numpy as np
import torch

from handcoded.config import N_BITS
from handcoded.data import language_model_loss
from handcoded.gates import phi
from handcoded.generate import generate, strip_after_eos


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
    """Fix the gate sequence; vary the start state over all 16 values."""
    if not gates or any(gate not in tokenizer.gates for gate in gates):
        raise ValueError("Choose a nonempty sequence of known gates")
    rows = [
        [state] + [tokenizer.ids[gate] for gate in gates] + [tokenizer.sep]
        for state in range(tokenizer.n_states)
    ]
    return torch.tensor(rows, device=device)


@torch.no_grad()
def circuit_answer_matrix(model, prompts, tokenizer, mode):
    """P(answer state | model-generated prefix) for every start state of one circuit."""
    depth = prompts.shape[1] - 2
    prefix_length = 1 if mode == "outcome" else 2 * depth + 1
    prefix = generate(model, prompts, prefix_length, tokenizer.eos)
    probabilities = model(prefix)[:, -1].softmax(dim=-1)[:, : tokenizer.n_states]
    valid = (prefix[:, -1] == tokenizer.colon) & ~(
        prefix[:, prompts.shape[1] :] == tokenizer.eos
    ).any(dim=1)
    return (probabilities * valid[:, None]).detach().cpu().numpy()


def gold_answer_matrix(gates, n_bits=N_BITS):
    """Exact permutation: one-hot φ composition for every start state."""
    if not gates:
        raise ValueError("Choose a nonempty gate sequence")
    n_states = 2 ** n_bits
    matrix = np.zeros((n_states, n_states), dtype=np.float32)
    for start in range(n_states):
        state = start
        for gate in gates:
            state = phi(state, gate, n_bits)
        matrix[start, state] = 1.0
    return matrix


@torch.no_grad()
def inspect_fixed_circuit(model, gates, tokenizer, device):
    """Residual-stream state features after each fixed outcome block (not attention weights)."""
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


@torch.no_grad()
def evaluate_checkpoint(
    model, mode, step, train_loss_sample, train_eval, test_eval, tokenizer, circuit_prompts,
    architecture=None, test_loss_sample=None,
):
    train_metrics = free_run_metrics(model, train_eval, tokenizer, mode)
    test_metrics = free_run_metrics(model, test_eval, tokenizer, mode)
    row = {
        "circuit_matrix": circuit_answer_matrix(model, circuit_prompts, tokenizer, mode),
        "step": step, "architecture": architecture, "mode": mode,
        "train_loss": language_model_loss(model, train_loss_sample).item(),
        "train_answer_accuracy_sample": train_metrics["final_answer"],
        "test_answer_accuracy": test_metrics["final_answer"],
        "test_exact_continuation": test_metrics["exact_continuation"],
    }
    row["test_loss"] = (
        language_model_loss(model, test_loss_sample).item() if test_loss_sample is not None else np.nan
    )
    return row
