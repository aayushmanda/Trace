"""Teacher-forced training with tqdm; independent copies per (architecture, mode)."""
import copy

import pandas as pd
import torch
from tqdm.auto import tqdm

from handcoded.data import language_model_loss
from handcoded.eval import evaluate_checkpoint
from src.training.optim import make_adamw
from src.training.seed import autocast_context, configure_device, maybe_compile


def train_step(model, batch, optimizer, grad_clip_norm=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = batch.inputs.device
    with autocast_context(device):
        loss = language_model_loss(model, batch)
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return loss.item()


def _prefix_loss_sample(data, max_examples):
    if data is None:
        return None
    if max_examples is None:
        return data
    if max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    return data.select(slice(0, min(max_examples, len(data))))


def _mode_data(data_by_mode, mode):
    if data_by_mode is None:
        return None
    if isinstance(data_by_mode, dict):
        return data_by_mode.get(mode)
    return data_by_mode


def _format_progress_value(value, *, precision=3, percent=False):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}" if percent else f"{value:.{precision}f}"


def train_one_model(
    base, mode, data, batch_schedule, learning_rate, checkpoints, train_eval, test_eval,
    tokenizer, circuit_prompts, architecture=None, test_loss_data=None, loss_eval_size=256,
    progress_leave=True, grad_clip_norm=None,
):
    """Train a copy of `base` under one continuation format; history includes circuit matrices."""
    model = copy.deepcopy(base)
    device = next(model.parameters()).device
    configure_device(device)
    train_model = maybe_compile(model, device)  # eval/generate keep eager `model`
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Cannot train a model with no trainable parameters")
    optimizer = make_adamw(trainable_parameters, learning_rate, weight_decay=0, device=device)
    train_loss_sample = _prefix_loss_sample(data, loss_eval_size)
    test_loss_sample = _prefix_loss_sample(test_loss_data, loss_eval_size)
    checkpoint_set = set(checkpoints)
    label = f"{architecture or 'model'}/{mode}"
    history = [
        evaluate_checkpoint(
            model, mode, 0, train_loss_sample, train_eval, test_eval, tokenizer, circuit_prompts,
            architecture=architecture, test_loss_sample=test_loss_sample,
        )
    ]
    progress = tqdm(enumerate(batch_schedule, 1), total=len(batch_schedule), desc=label, leave=progress_leave)
    for step, indices in progress:
        train_step(train_model, data.select(indices), optimizer, grad_clip_norm=grad_clip_norm)
        if step not in checkpoint_set:
            continue
        row = evaluate_checkpoint(
            model, mode, step, train_loss_sample, train_eval, test_eval, tokenizer, circuit_prompts,
            architecture=architecture, test_loss_sample=test_loss_sample,
        )
        history.append(row)
        progress.set_postfix(
            train_loss=_format_progress_value(row["train_loss"]),
            test_loss=_format_progress_value(row["test_loss"]),
            train=_format_progress_value(row["train_answer_accuracy_sample"], percent=True),
            test=_format_progress_value(row["test_answer_accuracy"], percent=True),
        )
    frame = pd.DataFrame(history)
    if "mode" not in frame.columns:
        frame["mode"] = mode
    return model, frame


def run_experiment(
    base, training_data, batch_schedule, learning_rate, checkpoints, train_eval, test_eval,
    tokenizer, circuit_prompts, modes=("outcome", "process"), test_loss_data=None,
    loss_eval_size=256, progress_leave=True, grad_clip_norm=None,
):
    """Two learned copies of the same one-layer architecture: outcome vs process supervision."""
    models, histories = {}, []
    for mode in tqdm(modes, desc="supervision modes", leave=progress_leave):
        model, history = train_one_model(
            base, mode, training_data[mode], batch_schedule, learning_rate, checkpoints,
            train_eval, test_eval, tokenizer, circuit_prompts,
            test_loss_data=_mode_data(test_loss_data, mode), loss_eval_size=loss_eval_size,
            progress_leave=progress_leave, grad_clip_norm=grad_clip_norm,
        )
        models[mode] = model
        histories.append(history)
    return models, pd.concat(histories, ignore_index=True)


def run_architecture_experiment(
    bases, training_data, batch_schedule, learning_rate, checkpoints, train_eval, test_eval,
    tokenizer, circuit_prompts, modes=("outcome", "process"), test_loss_data=None,
    loss_eval_size=256, progress_leave=True, grad_clip_norm=None,
):
    """2×2: process vs outcome *architecture* crossed with process vs outcome *supervision*."""
    models, histories = {}, []
    jobs = [(architecture, base, mode) for architecture, base in bases.items() for mode in modes]
    for architecture, base, mode in tqdm(jobs, desc="architecture/mode", leave=progress_leave):
        model, history = train_one_model(
            base, mode, training_data[mode], batch_schedule, learning_rate, checkpoints,
            train_eval, test_eval, tokenizer, circuit_prompts, architecture=architecture,
            test_loss_data=_mode_data(test_loss_data, mode), loss_eval_size=loss_eval_size,
            progress_leave=progress_leave, grad_clip_norm=grad_clip_norm,
        )
        models[(architecture, mode)] = model
        histories.append(history)
    return models, pd.concat(histories, ignore_index=True)
