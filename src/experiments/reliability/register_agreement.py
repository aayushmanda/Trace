"""Which rule does the register-machine GPT follow? Agreement with the true and the coherent wrong rule.

Loads models saved by `python -m src reliability --task register_machine_16_coherent --save-models DIR`.
On held-out prompts, for each step t the context is the prompt, the gold trace up to step t-1 and the
instruction a_t; the model greedily writes the four digits of (x_t, y_t), which are compared with the true
successor T and the coherent wrong successor W of the gold state. A free-running rollout from the prompt
then gives Pr[step 1 correct] and Pr[exact trace | step 1 correct].

Run from the repository root:
  uv run --offline --no-project --with 'torch==2.7.1' --with numpy --with pandas \
    python -m src.experiments.reliability.register_agreement \
    --models results/paper/register_models --output results/paper/register_agreement.csv
"""

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from src.data.registry import TASKS
from src.data.sample import generate_unique
from src.data.sequential_tasks import MODULUS, _apply_register_instruction, _coherent_register_mask, _num
from src.eval.generate import extract_generation
from src.training.optim import build_gpt


def parse_start(prompt):
    x, y, instructions = re.fullmatch(r"x(\d+);y(\d+);u([a-e]+)", prompt).groups()
    return int(x), int(y), instructions


@torch.no_grad()
def greedy(model, tokenizer, contexts, n_tokens, device):
    ids = torch.tensor([tokenizer.encode(c) for c in contexts], device=device)
    out = model.generate(ids, max_new_tokens=n_tokens, stop_id=tokenizer.newline_id, greedy=True)
    return [tokenizer.decode(row[ids.shape[1]:]) for row in out.tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="register_machine_16_coherent")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=501)
    parser.add_argument("--val-seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    task = TASKS[args.task]
    tokenizer = task.tokenizer
    train = generate_unique(task, args.train_size, args.train_seed)
    val = generate_unique(task, args.val_size, args.val_seed, {i.prompt for i in train})
    gold_steps = [inst.correct_trace.split() for inst in val]
    depth = len(gold_steps[0])
    rows = []
    for path in sorted(args.models.glob(f"{args.task}_rho*_s*.pt")):
        rho, seed = re.search(r"rho([\d.]+)_s(\d+)", path.stem).groups()
        model = build_gpt(task, SimpleNamespace(), device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        record = dict(rho=float(rho), seed=int(seed))
        for t in range(depth):
            contexts, true, wrong = [], [], []
            for inst, steps in zip(val, gold_steps):
                x, y, instructions = parse_start(inst.prompt)
                if t:
                    x, y = int(steps[t - 1][1:3]), int(steps[t - 1][3:5])
                a = instructions[t]
                tx, ty = _apply_register_instruction(x, y, a)
                dx, dy = _coherent_register_mask(a)
                true.append(f"{_num(tx)}{_num(ty)}")
                wrong.append(f"{_num((tx + dx) % MODULUS)}{_num((ty + dy) % MODULUS)}")
                contexts.append(f"{inst.prompt} {' '.join(steps[:t])}{' ' if t else ''}{a}")
            predicted = greedy(model, tokenizer, contexts, 4, device)
            record[f"agree_T_{t + 1}"] = sum(p == g for p, g in zip(predicted, true)) / len(val)
            record[f"agree_W_{t + 1}"] = sum(p == g for p, g in zip(predicted, wrong)) / len(val)
        tails = greedy(model, tokenizer, [inst.prompt for inst in val], task.max_new_tokens, device)
        first, exact, answer = [], [], []
        for tail, inst, steps in zip(tails, val, gold_steps):
            trace, predicted_answer, _ = extract_generation(tail)
            produced = [] if trace is None else trace.split()
            first.append(bool(produced) and produced[0] == steps[0])
            exact.append(trace == inst.correct_trace)
            answer.append(predicted_answer == inst.gold)
        first_correct = [e for f, e in zip(first, exact) if f]
        record.update(first_correct=sum(first) / len(val), exact=sum(exact) / len(val), answer=sum(answer) / len(val),
                      exact_given_first=sum(first_correct) / max(len(first_correct), 1))
        rows.append(record)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in record.items()
               if not k.startswith("agree") or k.endswith(("_1", "_2", f"_{depth}"))}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
