import argparse, csv, gc, json, random, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.registry import TASKS
from sweep_ratio import generate_unique

STEP_RE = re.compile(r"^([xcst]\d{1,3})>([01]{4})$")
ANSWER_RE = re.compile(r"([01]{4})")
ALL_STATES = [f"{i:04b}" for i in range(16)]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def trace_steps(trace): return trace.strip().split()


def split_step(step):
    m = STEP_RE.fullmatch(step)
    if not m: raise ValueError(f"Bad Boolean step: {step!r}")
    return m.group(1), m.group(2)


def completion_for(inst, condition, clean=None):
    if condition == "outcome": return f" : {inst.gold}\n"
    if condition == "answer_first": return f" : {inst.gold} ; {inst.correct_trace}\n"
    if condition == "process":
        trace = inst.correct_trace if clean else inst.wrong_trace
        return f" {trace} : {inst.gold}\n"
    raise ValueError(condition)


class CompletionDataset(Dataset):
    def __init__(self, instances, tokenizer, condition, rho, ratio_scores, max_length):
        self.rows, self.realized_rho = [], None
        if condition == "process":
            if rho is None: raise ValueError("process requires rho")
            clean = np.asarray(ratio_scores) < float(rho)
            self.realized_rho = float(clean.mean())
        else:
            clean = None
        for i, inst in enumerate(instances):
            completion = completion_for(inst, condition, bool(clean[i])) if condition == "process" else completion_for(inst, condition)
            pids = tokenizer(inst.prompt, add_special_tokens=True)["input_ids"]
            cids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None: cids += [tokenizer.eos_token_id]
            ids = pids + cids
            if len(ids) > max_length: raise ValueError(f"{len(ids)} tokens > --max-length={max_length}")
            self.rows.append((ids, [-100] * len(pids) + cids.copy()))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Collator:
    def __init__(self, pad): self.pad = pad
    def __call__(self, rows):
        n, w = len(rows), max(len(x) for x, _ in rows)
        ids = torch.full((n, w), self.pad, dtype=torch.long)
        labels = torch.full((n, w), -100, dtype=torch.long)
        mask = torch.zeros((n, w), dtype=torch.long)
        for i, (x, y) in enumerate(rows):
            k = len(x); ids[i, :k] = torch.tensor(x); labels[i, :k] = torch.tensor(y); mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None: raise ValueError("Tokenizer has no PAD/EOS token")
        tok.pad_token = tok.eos_token
    return tok


def choose_dtype(device):
    if device.type != "cuda": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model(args, tok, device):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=choose_dtype(device), low_cpu_mem_usage=True)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if args.full_finetune: return model.to(device)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank,
        lora_alpha=2 * args.lora_rank, lora_dropout=0.0, bias="none", target_modules="all-linear"))
    return model.to(device)


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total


def train(model, ds, args, seed, device):
    set_seed(seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.batch_seed + seed), collate_fn=Collator(args.pad_id))
    it = iter(loader); params = [p for p in model.parameters() if p.requires_grad]
    try: opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")
    except TypeError: opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    amp, amp_dtype = device.type == "cuda", choose_dtype(device)
    model.train(); recent = []
    pbar = tqdm(range(1, args.steps + 1), desc=f"train seed={seed}")
    for step in pbar:
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp): loss = model(**batch).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, args.grad_clip); opt.step()
        recent.append(float(loss.detach()))
        if len(recent) > 30: recent.pop(0)
        if step % 10 == 0: pbar.set_postfix(loss=f"{np.mean(recent):.4f}")
    return float(np.mean(recent))


@torch.inference_mode()
def generate_text(model, tok, prompts, args, device, max_new):
    old = tok.padding_side; tok.padding_side = "left"; model.eval(); outputs = []
    for start in tqdm(range(0, len(prompts), args.eval_batch_size), desc="free generation", leave=False):
        b = tok(prompts[start:start + args.eval_batch_size], return_tensors="pt", padding=True, truncation=False)
        b = {k: v.to(device) for k, v in b.items()}
        y = model.generate(**b, do_sample=False, num_beams=1, max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True)
        outputs += tok.batch_decode(y[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = old
    return outputs


def parse_generation(text, condition):
    line = text.split("\n", 1)[0].strip()
    if ":" not in line: return None, None
    if condition in {"outcome", "answer_first"}:
        m = ANSWER_RE.search(line.split(":", 1)[1]); return None, m.group(1) if m else None
    trace, ans = line.rsplit(":", 1); m = ANSWER_RE.search(ans)
    return trace.strip(), m.group(1) if m else None


def parse_trace_positionally(trace, depth):
    gates, states, steps = [None] * depth, [None] * depth, [None] * depth
    if trace is None: return gates, states, steps
    for t, token in enumerate(trace.split()[:depth]):
        m = STEP_RE.fullmatch(token)
        if m: gates[t], states[t], steps[t] = m.group(1), m.group(2), token
    return gates, states, steps


@torch.inference_mode()
def evaluate_free(model, tok, instances, condition, args, device):
    max_new = args.outcome_max_new_tokens if condition in {"outcome", "answer_first"} else args.process_max_new_tokens
    texts = generate_text(model, tok, [x.prompt for x in instances], args, device, max_new)
    ans_ok = exact_trace = exact_states = step_ok = state_ok = gate_ok = transitions = 0
    per_example, audit = [], []
    for eid, (inst, text) in enumerate(zip(instances, texts)):
        pred_trace, pred_ans = parse_generation(text, condition); aok = pred_ans == inst.gold; ans_ok += int(aok)
        record = {"example_id": eid, "answer_correct": int(aok), "exact_trace": None, "exact_state_rollout": None}
        if condition == "process":
            gold_steps = trace_steps(inst.correct_trace); depth = len(gold_steps)
            gold = [split_step(s) for s in gold_steps]; pg, ps, pst = parse_trace_positionally(pred_trace, depth)
            et = pred_trace == inst.correct_trace
            es = all(ps[t] == gold[t][1] for t in range(depth))
            exact_trace += int(et); exact_states += int(es)
            for t in range(depth):
                gate_ok += int(pg[t] == gold[t][0]); state_ok += int(ps[t] == gold[t][1]); step_ok += int(pst[t] == gold_steps[t]); transitions += 1
            record.update(exact_trace=int(et), exact_state_rollout=int(es))
        per_example.append(record)
        if len(audit) < args.audit_examples:
            audit.append({"example_id": eid, "prompt": inst.prompt, "gold_trace": inst.correct_trace,
                "wrong_trace": inst.wrong_trace, "gold_answer": inst.gold, "generated": text,
                "parsed_trace": pred_trace, "parsed_answer": pred_ans})
    n = len(instances)
    if condition != "process":
        return {"answer_accuracy": ans_ok / n, "exact_trace_accuracy": None, "exact_state_rollout_accuracy": None,
            "free_step_accuracy": None, "free_state_accuracy": None, "free_gate_accuracy": None,
            "per_example": per_example, "audit": audit}
    return {"answer_accuracy": ans_ok / n, "exact_trace_accuracy": exact_trace / n,
        "exact_state_rollout_accuracy": exact_states / n, "free_step_accuracy": step_ok / transitions,
        "free_state_accuracy": state_ok / transitions, "free_gate_accuracy": gate_ok / transitions,
        "per_example": per_example, "audit": audit}


def local_prefix(inst, t):
    """Gold prefix ending after the current gate's '>', immediately before state bits."""
    steps = trace_steps(inst.correct_trace); gate, gold_state = split_step(steps[t])
    prefix = inst.prompt + " "
    if t: prefix += " ".join(steps[:t]) + " "
    prefix += gate + ">"
    return prefix, gold_state, gate


def encode_local(tok, prefix, candidate):
    """Verify that candidate begins on a stable tokenizer boundary."""
    pids = tok(prefix, add_special_tokens=True)["input_ids"]
    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    full = tok(prefix + candidate, add_special_tokens=True)["input_ids"]
    if full != pids + cids:
        raise RuntimeError("Tokenizer boundary changed at local state. Prefix ends at '>' but encode(prefix+state) != encode(prefix)+encode(state).")
    return pids, cids


class LocalDataset(Dataset):
    def __init__(self, instances, tok):
        self.rows, self.queries = [], []
        for eid, inst in enumerate(instances):
            for t in range(len(trace_steps(inst.correct_trace))):
                prefix, gold, gate = local_prefix(inst, t); qid = len(self.queries)
                self.queries.append({"query_id": qid, "example_id": eid, "step": t + 1, "gate": gate, "gold_state": gold})
                for state in ALL_STATES:
                    pids, cids = encode_local(tok, prefix, state)
                    self.rows.append({"query_id": qid, "state": state, "ids": pids + cids, "start": len(pids)})
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class LocalCollator:
    def __init__(self, pad): self.pad = pad
    def __call__(self, rows):
        n, w = len(rows), max(len(r["ids"]) for r in rows)
        ids = torch.full((n, w), self.pad, dtype=torch.long); attn = torch.zeros((n, w), dtype=torch.long)
        cmask = torch.zeros((n, w), dtype=torch.bool); qids, states = [], []
        for i, r in enumerate(rows):
            k = len(r["ids"]); ids[i, :k] = torch.tensor(r["ids"]); attn[i, :k] = 1; cmask[i, r["start"]:k] = True
            qids.append(r["query_id"]); states.append(r["state"])
        return {"input_ids": ids, "attention_mask": attn, "candidate_mask": cmask, "qids": qids, "states": states}


@torch.inference_mode()
def score_local_states(model, tok, instances, args, device):
    ds = LocalDataset(instances, tok)
    loader = DataLoader(ds, batch_size=args.local_batch_size, shuffle=False, num_workers=0, collate_fn=LocalCollator(tok.pad_token_id))
    scores = {q["query_id"]: {} for q in ds.queries}; model.eval()
    for b in tqdm(loader, desc="16-way local margins", leave=False):
        ids, attn, cmask = b["input_ids"].to(device), b["attention_mask"].to(device), b["candidate_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1, :].float()
        targets, mask = ids[:, 1:], cmask[:, 1:]
        token_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        seq_lp = (token_lp * mask).sum(dim=1).tolist()
        for i, lp in enumerate(seq_lp): scores[b["qids"][i]][b["states"][i]] = float(lp)
    rows = []
    for q in ds.queries:
        s = scores[q["query_id"]]; gold = q["gold_state"]
        if set(s) != set(ALL_STATES): raise AssertionError("Missing one or more 16-state candidates")
        best_state = max(s, key=s.get); wrong_state = max((x for x in ALL_STATES if x != gold), key=s.get)
        vals = np.array([s[x] for x in ALL_STATES], dtype=np.float64); probs = np.exp(vals - vals.max()); probs /= probs.sum()
        rows.append({"example_id": q["example_id"], "step": q["step"], "gate": q["gate"], "gold_state": gold,
            "best_state": best_state, "best_wrong_state": wrong_state, "gold_logprob": s[gold],
            "best_wrong_logprob": s[wrong_state], "margin": s[gold] - s[wrong_state],
            "gold_prob16": float(probs[ALL_STATES.index(gold)]), "local_correct": int(best_state == gold)})
    return rows


def aggregate_local(rows, free_per_example, depth):
    groups = {}
    for r in rows: groups.setdefault(r["example_id"], []).append(r)
    exrows = []
    for eid in sorted(groups):
        rr = sorted(groups[eid], key=lambda x: x["step"])
        if len(rr) != depth: raise AssertionError(f"example {eid}: {len(rr)} rows != depth {depth}")
        exrows.append({"example_id": eid, "all_local_correct": int(all(r["local_correct"] for r in rr)),
            "min_margin": min(r["margin"] for r in rr), "mean_margin": float(np.mean([r["margin"] for r in rr])),
            "product_gold_prob16": float(np.prod([r["gold_prob16"] for r in rr])),
            "free_exact_trace": int(free_per_example[eid]["exact_trace"]),
            "free_exact_state_rollout": int(free_per_example[eid]["exact_state_rollout"])})
    local_acc = float(np.mean([r["local_correct"] for r in rows]))
    all_rate = float(np.mean([r["all_local_correct"] for r in exrows]))
    product_prob = float(np.mean([r["product_gold_prob16"] for r in exrows]))
    margins = np.array([r["min_margin"] for r in exrows], dtype=float)
    exact = np.array([r["free_exact_state_rollout"] for r in exrows], dtype=float)
    corr = float(np.corrcoef(margins, exact)[0, 1]) if len(exrows) > 1 and np.std(margins) > 0 and np.std(exact) > 0 else float("nan")
    agreement = float(np.mean([r["all_local_correct"] == r["free_exact_state_rollout"] for r in exrows]))
    return {"local_state_accuracy": local_acc, "all_local_correct_rate": all_rate,
        "mean_product_gold_prob16": product_prob, "min_margin_exact_corr": corr,
        "local_vs_free_state_agreement": agreement, "example_rows": exrows}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True); exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row));
        if not exists: w.writeheader()
        w.writerow(row)


def label_for(condition, rho): return condition if condition != "process" else f"rho_{rho:.2f}"


def save_adapter(model, tok, root, condition, rho, seed):
    path = root / label_for(condition, rho) / f"seed_{seed}"; path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path); tok.save_pretrained(path); return path


def cleanup(model):
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_one(task, tok, ds, val, condition, rho, seed, args, device):
    print("\n" + "=" * 96); print(f"condition={condition} rho={rho} seed={seed}"); print("=" * 96)
    set_seed(seed); model = build_model(args, tok, device); trainable, total, frac = trainable_summary(model)
    print(f"trainable parameters: {trainable:,}/{total:,} ({100*frac:.3f}%)")
    loss = train(model, ds, args, seed, device)
    free = evaluate_free(model, tok, val, condition, args, device)
    label = label_for(condition, rho); gen_dir = args.run_dir / "generations"; gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{label}_seed_{seed}.json").write_text(json.dumps(free["audit"], indent=2))

    local = {"local_state_accuracy": None, "all_local_correct_rate": None, "mean_product_gold_prob16": None,
        "min_margin_exact_corr": None, "local_vs_free_state_agreement": None}
    local_n = ""
    if condition == "process":
        local_n = min(args.local_eval_size, len(val)); local_val = val[:local_n]
        margin_rows = score_local_states(model, tok, local_val, args, device)
        depth = len(trace_steps(local_val[0].correct_trace))
        full = aggregate_local(margin_rows, free["per_example"][:local_n], depth)
        for k in local: local[k] = full[k]
        mdir = args.run_dir / "local_margins"
        write_rows(mdir / f"{label}_seed_{seed}_transitions.csv", margin_rows)
        write_rows(mdir / f"{label}_seed_{seed}_examples.csv", full["example_rows"])

    adapter_path = ""
    if args.save_adapters: adapter_path = str(save_adapter(model, tok, args.run_dir / "adapters", condition, rho, seed))
    row = {"model": args.model, "task": task.name, "condition": condition if condition != "process" else f"rho={rho:.2f}",
        "rho": "" if rho is None else rho, "realized_rho": "" if ds.realized_rho is None else ds.realized_rho,
        "seed": seed, "train_size": len(ds), "val_size": len(val), "local_eval_size": local_n, "steps": args.steps,
        "batch_size": args.batch_size, "full_finetune": args.full_finetune, "trainable_parameters": trainable,
        "trainable_fraction": frac, "loss": loss, "answer_accuracy": free["answer_accuracy"],
        "exact_trace_accuracy": "" if free["exact_trace_accuracy"] is None else free["exact_trace_accuracy"],
        "exact_state_rollout_accuracy": "" if free["exact_state_rollout_accuracy"] is None else free["exact_state_rollout_accuracy"],
        "free_step_accuracy": "" if free["free_step_accuracy"] is None else free["free_step_accuracy"],
        "free_state_accuracy": "" if free["free_state_accuracy"] is None else free["free_state_accuracy"],
        "free_gate_accuracy": "" if free["free_gate_accuracy"] is None else free["free_gate_accuracy"],
        "local_state_accuracy": "" if local["local_state_accuracy"] is None else local["local_state_accuracy"],
        "all_local_correct_rate": "" if local["all_local_correct_rate"] is None else local["all_local_correct_rate"],
        "mean_product_gold_prob16": "" if local["mean_product_gold_prob16"] is None else local["mean_product_gold_prob16"],
        "min_margin_exact_corr": "" if local["min_margin_exact_corr"] is None else local["min_margin_exact_corr"],
        "local_vs_free_state_agreement": "" if local["local_vs_free_state_agreement"] is None else local["local_vs_free_state_agreement"],
        "adapter_path": adapter_path}
    append_csv(args.output, row)
    print(f"loss={loss:.4f} | answer={100*free['answer_accuracy']:.2f}%")
    if condition == "process":
        print(f"free exact trace={100*free['exact_trace_accuracy']:.2f}% | exact states={100*free['exact_state_rollout_accuracy']:.2f}% | "
              f"free state={100*free['free_state_accuracy']:.2f}% | free gate={100*free['free_gate_accuracy']:.2f}%")
        print(f"16-way local state={100*local['local_state_accuracy']:.2f}% | all-local-correct={100*local['all_local_correct_rate']:.2f}% | "
              f"mean Πp16(gold)={100*local['mean_product_gold_prob16']:.2f}% | min-margin/exact corr={local['min_margin_exact_corr']:.4f} | "
              f"local/free-state agreement={100*local['local_vs_free_state_agreement']:.2f}%")
    if adapter_path: print(f"saved adapter: {adapter_path}")
    cleanup(model)


def summarize(path):
    import pandas as pd
    df = pd.read_csv(path); print("\nFINAL SUMMARY"); print("=" * 96)
    cols = ["condition", "seed", "answer_accuracy", "exact_trace_accuracy", "exact_state_rollout_accuracy",
        "free_state_accuracy", "local_state_accuracy", "all_local_correct_rate", "min_margin_exact_corr"]
    print(df[cols].to_string(index=False))
    p = df[df["rho"].notna()].copy()
    if len(p) > 1:
        pred = p["all_local_correct_rate"].astype(float).to_numpy(); obs = p["exact_state_rollout_accuracy"].astype(float).to_numpy()
        mae = float(np.mean(np.abs(pred - obs)))
        corr = float(np.corrcoef(pred, obs)[0, 1]) if np.std(pred) > 0 and np.std(obs) > 0 else float("nan")
        print(f"\nall-local-correct vs free exact-state rollout: MAE={100*mae:.2f} pp, corr={corr:.4f}")
    print("Prediction is mean_i prod_t 1[margin_it>0], NOT (mean local accuracy)^D.")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001])
    p.add_argument("--include-answer-first", action="store_true")
    p.add_argument("--train-size", type=int, default=12000)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--local-eval-size", type=int, default=300)
    p.add_argument("--train-seed", type=int, default=501); p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--ratio-seed", type=int, default=777); p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--steps", type=int, default=800); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32); p.add_argument("--local-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8); p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--max-length", type=int, default=384); p.add_argument("--process-max-new-tokens", type=int, default=128)
    p.add_argument("--outcome-max-new-tokens", type=int, default=20); p.add_argument("--audit-examples", type=int, default=8)
    p.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, default=Path("results/smollm_trace_fixed.csv"))
    p.add_argument("--run-dir", type=Path, default=Path("results/smollm_trace_fixed_artifacts"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if any(r < 0 or r > 1 for r in args.rhos): p.error("every rho must be in [0,1]")
    if args.local_eval_size < 1: p.error("--local-eval-size must be positive")
    if args.task not in TASKS: raise KeyError(f"Unknown Trace task: {args.task}")
    if not args.task.startswith("boolean_circuit_"): raise ValueError("16-way local evaluator is for boolean_circuit_* tasks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.set_float32_matmul_precision("high")
    else: print("WARNING: no CUDA GPU; this will be slow")
    task = TASKS[args.task]; tok = load_tokenizer(args.model); args.pad_id = tok.pad_token_id
    args.output.parent.mkdir(parents=True, exist_ok=True); args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
        args.output.unlink()

    train_instances = generate_unique(task, args.train_size, args.train_seed)
    train_prompts = {x.prompt for x in train_instances}
    val = generate_unique(task, args.val_size, args.val_seed, excluded=train_prompts)
    ratio_scores = np.random.default_rng(args.ratio_seed).random(len(train_instances))

    # Verify candidate-token boundary before spending time training.
    prefix, _, _ = local_prefix(val[0], 0)
    for state in ALL_STATES: encode_local(tok, prefix, state)

    cfg = vars(args).copy(); cfg["output"] = str(cfg["output"]); cfg["run_dir"] = str(cfg["run_dir"]); cfg["device"] = str(device)
    (args.run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"model={args.model} task={task.name} device={device} train={len(train_instances)} val={len(val)} prompt_overlap=0")
    print(f"Trace example:\n  prompt        = {train_instances[0].prompt}\n  correct trace = {train_instances[0].correct_trace}\n"
          f"  wrong trace   = {train_instances[0].wrong_trace}\n  gold          = {train_instances[0].gold}")

    conditions = [("outcome", None)]
    if args.include_answer_first: conditions.append(("answer_first", None))
    conditions += [("process", r) for r in sorted(set(args.rhos))]
    for condition, rho in conditions:
        ds = CompletionDataset(train_instances, tok, condition, rho, ratio_scores, args.max_length)
        if condition == "process": print(f"\nrho={rho:.3f}: realized clean fraction={ds.realized_rho:.4f}")
        for seed in args.seeds: run_one(task, tok, ds, val, condition, rho, seed, args, device)
        del ds; gc.collect()
    summarize(args.output)
    print(f"\nsaved aggregate results: {args.output}\nsaved artifacts: {args.run_dir}")


if __name__ == "__main__": main()