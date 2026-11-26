"""BIG-Bench Hard tracking shuffled objects as a pretrained-LoRA experiment (Qwen2.5-0.5B, rank 8).

K people each hold one item; K swaps follow and the question asks what one person holds at the end. Training items
are generated from the scenes and item pools of the real BBH items, which are held out for testing.

    python -m src shuffled zeroshot --out results/shuffled/zeroshot_qwen05.json
    python -m src shuffled train --K 5 --condition process --seed 2001 --out results/shuffled/bbh/bbh_K5_process.json
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch

NAMES = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude"]


def pairs(K):
    return list(itertools.combinations(range(K), 2))


def swap(state, pair):
    s = list(state)
    i, j = pair
    s[i], s[j] = s[j], s[i]
    return s

# -----------------------------------------------------------------------------
# data

# An example is a dict: state0 (item index held by each person), moves (pair indices), q (the queried person),
# items, scene and flips (pair display order). The answer is the option letter of what q holds at the end;
# options list the items in start order.

BBH_DIR = Path(__file__).resolve().parents[1] / "data" / "bbh"
BBH_FILES = {3: "three", 5: "five", 7: "seven"}
SWAP_RE = re.compile(r"(First|Then|Finally), (\w+) and (\w+)( [^.]*\.)")


def bbh_text(ex):
    K = len(ex["state0"])
    P, who = pairs(K), NAMES[:K]
    intro, rel, middle, suffix, query = ex["scene"]
    clauses = [f"{who[i]}{rel}{ex['items'][c]}" for i, c in enumerate(ex["state0"])]
    words = ["First"] + ["Then"] * (len(ex["moves"]) - 2) + ["Finally"]
    swaps = []
    for w, m, f in zip(words, ex["moves"], ex["flips"]):
        a, b = P[m][::-1] if f else P[m]
        swaps.append(f"{w}, {who[a]} and {who[b]}{suffix}")
    options = "\n".join(f"({chr(65 + j)}) {it}" for j, it in enumerate(ex["items"]))
    return (f"{intro}{', '.join(clauses[:-1])}, and {clauses[-1]}.\n{middle}{' '.join(swaps)} "
            f"{query.format(P=who[ex['q']])}\nOptions:\n{options}")


def parse_bbh(K):
    """The real BIG-Bench Hard items as examples, and their scenes: template key -> item pool.

    Every parsed item is re-rendered and must reproduce the original text and target exactly.
    """
    P, who = pairs(K), NAMES[:K]
    raw = json.loads((BBH_DIR / f"tracking_shuffled_objects_{BBH_FILES[K]}_objects.json").read_text())["examples"]
    scenes, examples = {}, []
    for item in raw:
        body, opts = item["input"].split("\nOptions:\n")
        items = [o[4:] for o in opts.split("\n")]
        head, rest = body.split("\n", 1)
        i = head.index(": " + who[0] + " ") + 2
        intro, first = head[:i], head[i:]
        rel = first[len(who[0]):first.index(items[0], len(who[0]))]
        middle = rest[:rest.index("First, ")]
        query = rest[rest.rindex(". ") + 2:]
        q = next(j for j, name in enumerate(who) if f" {name} " in f" {query} ")
        found = SWAP_RE.findall(rest[len(middle):rest.rindex(". ") + 1])
        key = (intro, rel, middle, found[0][3], query.replace(who[q], "{P}", 1))
        pool = scenes.setdefault(key, [])
        pool += [it for it in items if it not in pool]
        ex = dict(state0=list(range(K)), q=q, items=items, scene=key,
                  moves=[P.index(tuple(sorted((who.index(a), who.index(b))))) for _, a, b, _ in found],
                  flips=[who.index(a) > who.index(b) for _, a, b, _ in found])
        assert bbh_text(ex) == item["input"], item["input"]
        assert answer(ex, K) == item["target"], (answer(ex, K), item["target"])
        examples.append(ex)
    return scenes, examples


def sample(K, n, rng, scenes):
    moves = rng.integers(len(pairs(K)), size=n).tolist()
    keys = list(scenes)
    key = keys[int(rng.integers(len(keys)))]
    items = [scenes[key][j] for j in rng.choice(len(scenes[key]), K, replace=False)]
    return dict(state0=list(range(K)), moves=moves, q=int(rng.integers(K)), items=items, scene=key,
                flips=rng.integers(2, size=n).astype(bool).tolist())


def gold_states(ex, K):
    P, s, out = pairs(K), ex["state0"], []
    for m in ex["moves"]:
        s = swap(s, P[m])
        out.append(s)
    return out


def answer(ex, K):
    final = gold_states(ex, K)[-1]
    return f"({chr(65 + final[ex['q']])})"


def prompt(ex, K):
    return bbh_text(ex)


def state_line(t, s, ex):
    """Each line carries every item's option letter, so the answer is a copy from the last line."""
    held = [f"{NAMES[i]} has {ex['items'][c]} ({chr(65 + c)})" for i, c in enumerate(s)]
    return f"After swap {t}: " + ", ".join(held) + "."


def completion(ex, K, step):
    """Answer only (outcome), or the state after every swap followed by the answer (step)."""
    lines = [state_line(t, s, ex) for t, s in enumerate(gold_states(ex, K), 1)] if step else []
    return "\n" + "\n".join(lines + [f"Answer: {answer(ex, K)}"])


def corpus(K, n, size, seed, exclude, scenes):
    rng, seen, out = np.random.default_rng(seed), set(exclude), []
    for _ in range(100 * size):
        ex = sample(K, n, rng, scenes)
        key = prompt(ex, K)
        if key not in seen:
            seen.add(key)
            out.append(ex)
            if len(out) == size:
                return out
    raise RuntimeError(f"prompt space too small for {size} unique prompts at K={K} n={n}")

# -----------------------------------------------------------------------------
# zero-shot and training


def model_args(a, **kw):
    return argparse.Namespace(model=a.model, lora_rank=8, batch_size=a.batch_size, workers=0, batch_seed=12345,
                              lr=a.lr, weight_decay=0.0, steps=a.steps, grad_clip=1.0,
                              eval_batch_size=a.eval_batch_size, **kw)


@torch.inference_mode()
def score_choices(model, tok, prompts, choices, suffix, device, batch=16):
    """Index of the choice with the largest log p(choice | prompt + suffix)."""
    preds = []
    for i in range(0, len(prompts), batch):
        rows = []
        for pr in prompts[i:i + batch]:
            pids = tok(pr + suffix, add_special_tokens=True)["input_ids"]
            rows += [(pids, tok(c, add_special_tokens=False)["input_ids"]) for c in choices]
        w = max(len(p) + len(c) for p, c in rows)
        ids = torch.full((len(rows), w), tok.pad_token_id)
        mask = torch.zeros_like(ids)
        for r, (p, c) in enumerate(rows):
            ids[r, :len(p) + len(c)] = torch.tensor(p + c)
            mask[r, :len(p) + len(c)] = 1
        logp = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().log_softmax(-1)
        scores = []
        for r, (p, c) in enumerate(rows):
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=device)
            scores.append(logp[r, pos, torch.tensor(c, device=device)].sum().item())
        c = len(choices)
        preds += [int(np.argmax(scores[j:j + c])) for j in range(0, len(scores), c)]
    return preds


def zeroshot_main(argv):
    from transformers import AutoModelForCausalLM
    from src.lora import choose_dtype, load_tokenizer
    p = argparse.ArgumentParser(prog="python -m src shuffled zeroshot")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--K", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=choose_dtype(device)).to(device).eval()
    rows = []
    for K in a.K:
        test = parse_bbh(K)[1]
        choices = [chr(65 + j) for j in range(K)]
        gold = [answer(e, K)[1] for e in test]
        preds = score_choices(model, tok, [prompt(e, K) for e in test], choices, "\nAnswer: (", device)
        acc = float(np.mean([choices[i] == g for i, g in zip(preds, gold)]))
        rows.append(dict(format="bbh", K=K, n=K, zeroshot=acc, chance=1 / K))
        print(f"zero-shot {a.model} BBH K={K}  accuracy {acc:.3f}  (chance {1 / K:.3f})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))


ANSWER_RE = re.compile(r"Answer:\s*(\([A-G]\)|[A-Z][a-z]+)")


def evaluate(model, tok, test, K, step, args, device):
    """Greedy free-running generation: answer accuracy and, for step training, the trace."""
    from src.lora import generate_text
    gold_len = max(len(tok(completion(e, K, step), add_special_tokens=False)["input_ids"]) for e in test)
    texts = generate_text(model, tok, [prompt(e, K) for e in test], args, device, gold_len + 16)
    ans = trace = lines_ok = lines = 0
    for e, text in zip(test, texts):
        m = ANSWER_RE.search(text)
        ans += int(bool(m) and m.group(1) == answer(e, K))
        if step:
            want = [state_line(t, s, e) for t, s in enumerate(gold_states(e, K), 1)]
            got = [ln.strip() for ln in text.strip().split("\n") if ln.startswith("After swap")]
            trace += int(got == want)
            lines_ok += sum(g == w for g, w in zip(got, want))
            lines += len(want)
    return dict(answer_accuracy=ans / len(test), exact_trace_accuracy=trace / len(test) if step else None,
                line_accuracy=lines_ok / lines if step else None, size=len(test),
                samples=[dict(prompt=prompt(e, K), gold=answer(e, K), generated=t) for e, t in zip(test[:4], texts[:4])])


def train_main(argv):
    from src.lora import build_model, load_tokenizer, train
    p = argparse.ArgumentParser(prog="python -m src shuffled train")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--K", type=int, choices=sorted(BBH_FILES), default=5)
    p.add_argument("--condition", choices=["outcome", "process"], default="process")
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--train-size", type=int, default=12000)
    p.add_argument("--test-size", type=int, default=300)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    K = n = a.K
    scenes, real = parse_bbh(K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = {prompt(e, K) for e in real}
    train_set = corpus(K, n, a.train_size, seed=501, exclude=held, scenes=scenes)
    test = corpus(K, n, a.test_size, seed=101, exclude=held | {prompt(e, K) for e in train_set}, scenes=scenes)
    step = a.condition == "process"
    tok = load_tokenizer(a.model)
    rows = []
    for ex in train_set:
        pids = tok(prompt(ex, K), add_special_tokens=True)["input_ids"]
        cids = tok(completion(ex, K, step), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        rows.append((pids + cids, [-100] * len(pids) + cids))
    print(f"BBH K={K} {a.condition} seed={a.seed}  max tokens={max(len(r[0]) for r in rows)}\n"
          f"{prompt(train_set[0], K)}{completion(train_set[0], K, step)}")

    args = model_args(a, pad_id=tok.pad_token_id)
    model = build_model(args, tok, device)
    loss = train(model, rows, args, a.seed, device)
    res = dict(vars(a), out=str(a.out), loss=loss, chance=1 / K, **evaluate(model, tok, test, K, step, args, device))
    res["bbh"] = evaluate(model, tok, real, K, step, args, device)
    print(f"generated test: answer {res['answer_accuracy']:.3f} trace {res['exact_trace_accuracy']} | "
          f"real BBH ({len(real)}): answer {res['bbh']['answer_accuracy']:.3f} trace {res['bbh']['exact_trace_accuracy']}"
          f"  (chance {1 / K:.3f})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


def main(argv):
    cmds = {"zeroshot": zeroshot_main, "train": train_main}
    if not argv or argv[0] not in cmds:
        raise SystemExit(__doc__)
    cmds[argv[0]](argv[1:])
