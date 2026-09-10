import random

from src.data.dataclass import Task


def generate_unique(task: Task, size: int, seed: int, excluded=None):
    """Sample unique prompts without permanently mutating the global RNG."""
    excluded = set() if excluded is None else set(excluded)
    state = random.getstate()
    random.seed(seed)
    items, prompts = [], set()
    while len(items) < size:
        inst = task.sample()
        if inst.prompt in excluded or inst.prompt in prompts:
            continue
        items.append(inst)
        prompts.add(inst.prompt)
    random.setstate(state)
    return items


def sample_instances(sampler, n, seed, exclude=None):
    """Same uniqueness rule for a raw sampler (Boolean induced-rule path)."""
    state = random.getstate()
    random.seed(seed)
    seen = set() if exclude is None else set(exclude)
    out, tries = [], 0
    while len(out) < n:
        inst = sampler()
        tries += 1
        if tries > 200 * n:
            raise RuntimeError(f"prompt space too small for {n} unique instances")
        if inst.prompt in seen:
            continue
        seen.add(inst.prompt)
        out.append(inst)
    random.setstate(state)
    return out
