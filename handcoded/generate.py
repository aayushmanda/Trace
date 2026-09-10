"""Greedy free-run: each new token is the model's own argmax, never gold states."""
import torch


@torch.no_grad()
def generate(model, prompts, max_new_tokens, eos_id):
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
