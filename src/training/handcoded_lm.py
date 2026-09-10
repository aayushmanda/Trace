"""Handcoded / semantic-token LM loss. Not the GPT forward pass."""
import handcoded as h


def handcoded_lm_loss(model, batch):
    return h.language_model_loss(model, batch)
