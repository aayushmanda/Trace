"""Exact degree split of the LoRA Jacobian of Qwen2.5-0.5B on BIG-Bench Hard three-object tracking.

    python handcoded/bbh_lora_spectrum.py contexts=10 seed=2001 out=results/paper/bbh_lora_spectrum/s2001.json

A context fixes the scene, the item list, the start assignment and the queried person (drawn
as in training, src/shuffled.py --format bbh). Its three moves each take one of 6 values (3
pairs x 2 name orders, uniform, as in training), so a context has 216 prompts. For each prompt
we take the logits of the option letters at the answer position (the token after
"Answer: (") and their gradients with respect to every LoRA parameter, giving the full
Jacobian on the 6^3 grid. An orthonormal transform along the three move axes gives the exact
Efron--Stein energy at each degree, with no estimator and no fit.

At the standard LoRA initialization B = 0, so the gradient with respect to A is exactly zero
and the Jacobian is that with respect to B. The base model is run in float32.
"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shuffled import parse_bbh, sample, prompt, answer  # noqa: E402

model_name = "Qwen/Qwen2.5-0.5B"
contexts = 10
seed = 2001
out = ""
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    globals()[k] = v if k in ("model_name", "out") else ast.literal_eval(v)

K, n, V = 3, 3, 6                     # objects, moves, values per move (pair x name order)
dev = "cuda"
torch.manual_seed(seed)               # LoRA A initialization
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(dev)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                                         bias="none", target_modules="all-linear"))
model.eval()
params = [p for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name]
assert all(float(p.abs().max()) == 0.0 for p in params), "LoRA B must start at zero"
P = sum(p.numel() for p in params)
letters = [tok("\nAnswer: (" + chr(65 + j) + ")", add_special_tokens=False)["input_ids"][4] for j in range(K)]
suffix = tok("\nAnswer: (", add_special_tokens=False)["input_ids"]

# orthonormal basis of L2(uniform on V values), column 0 constant
Qb = torch.linalg.qr(torch.cat([torch.ones(V, 1), torch.randn(V, V - 1, generator=torch.Generator().manual_seed(1))], 1))[0]
Qb = (Qb * math.sqrt(V) * torch.sign(Qb[0, 0])).to(dev)
grid = np.array([(i // V ** 2, (i // V) % V, i % V) for i in range(V ** n)])
ORDER = torch.tensor((grid != 0).sum(1), device=dev)


def degree_energy(f):
    """f: (V^n, ...) on the grid -> energy per degree under the uniform law."""
    c = f.reshape((V,) * n + (-1,))
    for t in range(n):
        c = torch.tensordot(c, Qb, dims=([t], [0])) / V
        c = c.movedim(-1, t)
    e = (c ** 2).sum(-1).reshape(-1)
    return torch.zeros(n + 1, device=dev, dtype=e.dtype).index_add_(0, ORDER, e)


scenes, _ = parse_bbh(K)
rng = np.random.default_rng(seed)
results = []
for c in range(contexts):
    base = sample(K, n, rng, scenes)
    J = torch.zeros(V ** n, K, P, device=dev)
    target = torch.zeros(V ** n, K, device=dev)
    for i, (m0, m1, m2) in enumerate(grid):
        ex = dict(base, moves=[int(m // 2) for m in (m0, m1, m2)], flips=[bool(m % 2) for m in (m0, m1, m2)])
        ids = tok(prompt(ex, K), add_special_tokens=True)["input_ids"] + suffix
        z = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1, letters]
        for k in range(K):
            g = torch.autograd.grad(z[k], params, retain_graph=k < K - 1)
            J[i, k] = torch.cat([x.reshape(-1) for x in g])
        target[i, ord(answer(ex, K)[1]) - 65] = 1.0
    Wr = degree_energy(J).double().cpu().numpy()
    Ar = degree_energy(target - 1.0 / K).double().cpu().numpy()
    gphi = (torch.einsum("ikp,ik->p", J, target - 1.0 / K) / V ** n).norm().item()
    results.append(dict(W_r=Wr.tolist(), A_r=Ar.tolist(), grad_phi=gphi))
    print(f"context {c}: W_r/W = {np.round(Wr / Wr.sum(), 12).tolist()}  W={Wr.sum():.4e}  "
          f"A_r = {np.round(Ar, 6).tolist()}  |grad Phi| = {gphi:.3e}", flush=True)
    del J

Wm = np.mean([r["W_r"] for r in results], 0)
summary = dict(model=model_name, seed=seed, contexts=contexts, lora_B_params=P,
               W_r_mean=Wm.tolist(), share=(Wm / Wm.sum()).tolist(), per_context=results)
print("mean share by degree:", np.round(Wm / Wm.sum(), 12).tolist())
if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1))
