"""Handcoded / semantic-token LM loss. Not the GPT forward pass."""
from src.models.handcoded import h


def handcoded_lm_loss(model, batch):
    return h.language_model_loss(model, batch)
