from collections import defaultdict

import torch


def split_step(task_name: str, step: str):
    if task_name.startswith("state_machine_"):
        return step[:-2], step[-2:]
    if task_name.startswith("register_machine_"):
        return step[:-4], step[-4:]
    if task_name.startswith("boolean_circuit_"):
        prefix, state = step.rsplit(">", 1)
        return prefix + ">", state
    raise ValueError(f"unsupported task {task_name!r}")


@torch.inference_mode()
def example_min_margins(model, task, instances, device, batch_size=64):
    tokenizer = task.tokenizer
    rows = []
    for index, inst in enumerate(instances):
        steps = inst.correct_trace.split()
        previous = []
        for t, step in enumerate(steps):
            before = f"{inst.prompt} " + (" ".join(previous) + " " if previous else "")
            prefix, state = split_step(task.name, step)
            rows.append((index, t, tokenizer.encode(before + prefix), tokenizer.encode(state)))
            previous.append(step)
    groups = defaultdict(list)
    for row in rows:
        groups[(len(row[2]), len(row[3]))].append(row)
    per_example = defaultdict(list)
    model.eval()
    for (context_len, target_len), batch_rows in groups.items():
        for start in range(0, len(batch_rows), batch_size):
            chunk = batch_rows[start:start + batch_size]
            context = torch.tensor([r[2] for r in chunk], dtype=torch.long, device=device)
            target = torch.tensor([r[3] for r in chunk], dtype=torch.long, device=device)
            seq = torch.cat([context, target], dim=1)
            logits, _ = model(seq[:, :-1])
            z = logits[:, context_len - 1:context_len - 1 + target_len]
            correct = z.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            wrong = z.clone()
            wrong.scatter_(-1, target.unsqueeze(-1), float("-inf"))
            state_margin = (correct - wrong.max(dim=-1).values).min(dim=1).values.cpu()
            for i, (ex_i, t, _, _) in enumerate(chunk):
                per_example[ex_i].append((t, float(state_margin[i])))
    records = []
    for i, inst in enumerate(instances):
        margins = [m for _, m in sorted(per_example[i])]
        records.append({
            "example": i, "gold": inst.gold,
            "m_min": min(margins) if margins else float("nan"),
            "n_steps": len(margins),
            "positive_path": int(all(m > 0 for m in margins)) if margins else 0,
        })
    return records
