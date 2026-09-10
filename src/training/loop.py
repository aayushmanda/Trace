"""Train any nn.Module. GPT uses (x, y, mask); handcoded uses a custom loss_fn."""
import torch

from src.training.progress import progress
from src.training.seed import autocast_context


def gpt_lm_loss(model, batch, device):
    x, y, mask = batch
    x = x.to(device, dtype=torch.long, non_blocking=True)
    y = y.to(device, dtype=torch.long, non_blocking=True)
    mask = mask.to(device, dtype=torch.float32, non_blocking=True)
    _, loss = model(x, targets=y, mask=mask)
    return loss


def _maybe_autocast(device):
    return autocast_context(device)


def train_steps(model, loader, optimizer, device, n_steps, grad_clip=1.0, loss_fn=None, desc=None):
    """Run n_steps updates. loss_fn(model, batch) -> scalar; default is GPT CE."""
    iterator = iter(loader)
    model.train()
    loss = None
    step_loss = loss_fn or (lambda m, b: gpt_lm_loss(m, b, device))
    bar = progress(range(n_steps), desc=desc or "train", leave=False)
    for _ in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device):
            loss = step_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if loss is not None:
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return float(loss.detach())


def train_with_checkpoints(model, loader, optimizer, device, checkpoints, on_checkpoint,
                           grad_clip=1.0, loss_fn=None, desc=None):
    """on_checkpoint(step, model, loss) at listed steps, including 0 before updates."""
    ckpts = sorted(set(checkpoints))
    iterator = iter(loader)
    step_loss = loss_fn or (lambda m, b: gpt_lm_loss(m, b, device))
    loss = None
    bar = progress(range(0, max(ckpts) + 1), desc=desc or "train", leave=False)
    for step in bar:
        if step in ckpts:
            on_checkpoint(step, model, loss)
            model.train()
        if step == max(ckpts):
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device):
            loss = step_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if loss is not None:
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return model


def train_indexed(model, batch_fn, optimizer, device, n_steps, grad_clip=1.0,
                  start=0, checkpoints=(), on_checkpoint=None, desc=None):
    """Handcoded-style loop: batch_fn(step) -> scalar loss (already on device).

    Does not assume GPT (x, y, mask). Used by architecture_controls.
    step is the completed update count after optimizer.step (1..n_steps).
    If start==0 and 0 is in checkpoints, on_checkpoint runs before any update.
    """
    ckpts = set(checkpoints or ())
    loss = None
    if on_checkpoint and start == 0 and 0 in ckpts:
        on_checkpoint(0, model, None)
        model.train()
    bar = progress(range(start + 1, n_steps + 1), desc=desc or "train", leave=False)
    for step in bar:
        optimizer.zero_grad(set_to_none=True)
        with _maybe_autocast(device):
            loss = batch_fn(step - 1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        clip = grad_clip if grad_clip else float("inf")
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if on_checkpoint and step in ckpts:
            on_checkpoint(step, model, loss)
            model.train()
    return float(loss.detach()) if loss is not None else None
