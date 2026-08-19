"""
train.py
Training loop, evaluation, and the per-rho experiment runner.
"""
import random
import re
import torch

from data import build_rho_corpus, build_clean_val_set, make_batch
from model import GPTModel


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, val_examples, block_size, device, n_samples=300):
    model.eval()
    correct = 0
    sample = random.sample(val_examples, min(n_samples, len(val_examples)))

    for prompt, target in sample:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=block_size, stop_id=tokenizer.newline_id)[0].tolist()
        gen = tokenizer.decode(out)
        pred_text = gen[len(prompt):]

        gold_nums = re.findall(r'\d+', target)
        pred_nums = re.findall(r'\d+', pred_text)

        gold = gold_nums[-1] if gold_nums else None
        pred = pred_nums[-1] if pred_nums else None

        if pred == gold:
            correct += 1

    model.train()
    return correct / len(sample)


def run_experiment_for_rho(rho, tokenizer, cfg, device):
    """Trains a model instance for a specific rho value and evaluates accuracy."""
    print(f"\n==========================================")
    print(f"   STARTING RUN FOR RHO = {rho:.2f}")
    print(f"==========================================")

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]

    random.seed(train_cfg["seed_data"])
    torch.manual_seed(train_cfg["seed_model"])

    train_data = build_rho_corpus(n=data_cfg["num_samples"], rho=rho)
    clean_val_data = build_clean_val_set(n=data_cfg["val_size"])

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg["block_size"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_layer=model_cfg["n_layer"],
        dropout=model_cfg["dropout"],
        pad_id=tokenizer.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    max_iters = train_cfg["max_iters"]
    eval_every = train_cfg["eval_every"]

    for step in range(1, max_iters + 1):
        xb, yb, mb = make_batch(
            train_data, tokenizer,
            batch_size=train_cfg["batch_size"],
            block_size=model_cfg["block_size"],
            device=device,
        )
        logits, loss = model(xb, yb, mb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == max_iters:
            acc = evaluate_accuracy(
                model, tokenizer, clean_val_data, model_cfg["block_size"], device,
                n_samples=eval_cfg["eval_samples_during_training"],
            )
            print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | Clean Val Accuracy: {acc * 100:.2f}%")

    final_accuracy = evaluate_accuracy(
        model, tokenizer, clean_val_data, model_cfg["block_size"], device,
        n_samples=eval_cfg["eval_samples_final"],
    )
    return final_accuracy, model