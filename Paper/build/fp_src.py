"""Fingerprint of everything in src/ that the paper's numbers depend on. Run before and after the trim."""
import hashlib
import json
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")
from src import data, shuffled  # noqa: E402
from src.train import build_gpt, make_loader, set_seed  # noqa: E402


def h(obj):
    if isinstance(obj, torch.Tensor):
        return hashlib.sha256(obj.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def insts(xs):
    return h([[i.prompt, i.correct_trace, i.wrong_trace, i.gold] for i in xs])


out = {}
for name, ntr, nva in [("boolean_circuit_8", 20000, 1000), ("register_machine_16", 20000, 500),
                       ("boolean_circuit_8", 12000, 300)]:
    task = data.TASKS[name]
    tr = data.generate_unique(task, ntr, 501)
    va = data.generate_unique(task, nva, 101, {i.prompt for i in tr})
    key = f"{name}_{ntr}"
    out[key] = dict(train=insts(tr), val=insts(va), block=task.block_size, max_new=task.max_new_tokens,
                    chance=task.chance_acc, vocab=task.tokenizer.vocab_size)
    for mode in ["outcome", "answer_first", "process"]:
        ds = data.SupervisionDataset(tr, task, mode)
        out[key][mode] = [h(ds.x), h(ds.y), h(ds.mask)]
    if hasattr(data, "RatioDataset") and ntr == 20000:
        import numpy as np
        rs = np.random.default_rng(777).random(len(tr))
        rd = data.RatioDataset(tr, task, "mixed_process", rho=1.0, ratio_scores=rs)
        out[key]["ratio_rho1_equals_process"] = [h(rd.x), h(rd.y), h(rd.mask)] == out[key]["process"]
    args = SimpleNamespace(batch_size=128, batch_seed=12345, workers=0, embedding=128, heads=4, layers=2, dropout=0.0)
    set_seed(2001)
    loader = make_loader(data.SupervisionDataset(tr, task, "process"), args, torch.device("cpu"))
    model = build_gpt(task, args, torch.device("cpu"))
    out[key]["init"] = h(torch.cat([p.detach().flatten() for p in model.state_dict().values()]))
    out[key]["batch0"] = h(next(iter(loader))[0])

lora_lines = []
for i in data.generate_unique(data.TASKS["boolean_circuit_8"], 50, 501):
    lora_lines += [f" : {i.gold}\n", f" : {i.gold} ; {i.correct_trace}\n", f" {i.correct_trace} : {i.gold}\n"]
out["lora_completions_expected"] = h(lora_lines)

for K in (3, 5, 7):
    scenes, real = shuffled.parse_bbh(K)
    held = {shuffled.prompt(e, K) for e in real}
    tr = shuffled.corpus(K, K, 12000, seed=501, exclude=held, scenes=scenes)
    te = shuffled.corpus(K, K, 300, seed=101, exclude=held | {shuffled.prompt(e, K) for e in tr}, scenes=scenes)
    import inspect
    if "every_k" in inspect.signature(shuffled.completion).parameters:  # before the trim
        comp = lambda xs, k: [shuffled.completion(e, K, shuffled.gold_states(e, K), k) for e in xs]  # noqa: E731
    else:
        comp = lambda xs, k: [shuffled.completion(e, K, bool(k)) for e in xs]  # noqa: E731
    out[f"bbh_K{K}"] = dict(real=h([[shuffled.prompt(e, K), shuffled.answer(e, K)] for e in real]),
                           train=h([shuffled.prompt(e, K) for e in tr]), test=h([shuffled.prompt(e, K) for e in te]),
                           out_completion=h(comp(tr, 0)), step_completion=h(comp(tr, 1)))

print(json.dumps(out, indent=1))
