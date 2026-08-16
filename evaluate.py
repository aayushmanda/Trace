"""Position-safe unbatched greedy-decoding evaluation."""

import re

import torch

from config import device
from tokenizer import NEWLINE_ID, decode, encode


@torch.no_grad()
def evaluate_accuracy(model, test_examples):
    model.eval()
    correct = 0

    for prompt, target in test_examples:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=35, stop_id=NEWLINE_ID)[0].tolist()

        gen_text = decode(out)
        pred_tail = gen_text[len(prompt) :]
        gold_answer = target.strip().split(":")[-1].strip()

        segment = pred_tail.split(":")[-1] if ":" in pred_tail else pred_tail
        nums = re.findall(r"\d+", segment)
        pred_answer = nums[-1] if nums else None
        correct += int(pred_answer == gold_answer)

    model.train()
    return correct / len(test_examples)
