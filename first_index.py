import os
import math
import random
import re
import string
import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# 1. Hyper-parameters
# ============================================================
batch_size = 256
steps_per_run = 6000
learning_rate = 3e-4
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
dropout = 0.05
label_smoothing = 0.1
max_grad_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 128
n_head = 4
n_layer = 4

DATASET_SIZE = 50000
DATASET_SEED = 100
CACHE_DIR = "./dataset_cache"

# ============================================================
# 2. Choose task
# ============================================================
TASK_NAME = "word_index"          # "word_index" | "sort_letters" | "multiply" | "count_char" | "graph_path"

# ============================================================
# 3. Task family
# ============================================================
ANSWER_SEP = " : "
TRACE_MODES = ("correct_think", "wrong_think", "no_think")


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi.get("\n", None)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


@dataclass(frozen=True)
class Instance:
    prompt: str
    correct_trace: str
    wrong_trace: str
    gold: str


@dataclass
class Task:
    name: str
    chars: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float
    ceiling_acc: float
    description: str = ""
    answer_pattern: str = r"-?\d+"
    bayes_prob: Optional[Callable[[Instance], float]] = None
    tokenizer: CharTokenizer = field(init=False)

    def __post_init__(self):
        self.tokenizer = CharTokenizer(self.chars)

    def render(self, inst: Instance, mode: str = "correct_think"):
        if mode == "correct_think":
            return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "wrong_think":
            return inst.prompt, f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "no_think":
            return inst.prompt, f" {inst.gold}"
        raise ValueError(f"unknown mode {mode}")

    def context(self, inst: Instance, mode: str) -> str:
        if mode == "free":
            return inst.prompt
        if mode == "forced_correct":
            return f"{inst.prompt} {inst.correct_trace}{ANSWER_SEP}"
        if mode == "forced_wrong":
            return f"{inst.prompt} {inst.wrong_trace}{ANSWER_SEP}"
        if mode == "direct":
            return f"{inst.prompt} "
        raise ValueError(f"unknown eval mode {mode}")

    def extract_answer(self, generated_tail: str, first: bool = False) -> Optional[str]:
        tail = generated_tail.split("\n")[0]
        segment = tail.split(":")[-1] if ":" in tail else tail
        found = re.findall(self.answer_pattern, segment)
        if not found:
            return None
        return found[0] if first else found[-1]

    def fingerprint(self, n: int = 64) -> str:
        state = random.getstate()
        random.seed(12345)
        h = hashlib.sha1()
        for _ in range(n):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                h.update(f"{prompt}{answer}\n".encode())
        random.setstate(state)
        h.update(f"{self.block_size}|{''.join(self.tokenizer.chars)}".encode())
        return h.hexdigest()[:10]


# ---------- Existing samplers ----------
def _sample_word_index() -> Instance:
    L = random.randint(4, 10)
    word = "".join(random.choice(string.ascii_lowercase) for _ in range(L))
    gold = random.randrange(L)
    prompt = f"{word};{word[gold]}"
    correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))
    fake = list(range(L))
    random.shuffle(fake)
    wrong = " ".join(f"{i}{random.choice(string.ascii_lowercase)}" for i in fake)
    return Instance(prompt, correct, wrong, str(gold))


def _word_index_bayes(inst: Instance) -> float:
    word, _, c = inst.prompt.partition(";")
    return 1.0 / max(1, word.count(c))


def _sample_sort_letters() -> Instance:
    L = random.randint(4, 7)
    letters = random.sample(string.ascii_lowercase, L)
    word = "".join(letters)
    gold = "".join(sorted(letters))
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        chosen = min(remaining)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    correct = " ".join(steps)
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        shown = remaining[:]
        random.shuffle(shown)
        chosen = random.choice(remaining)
        steps.append(f"{''.join(shown)}>{chosen}")
        remaining.remove(chosen)
    wrong = " ".join(steps)
    return Instance(word, correct, wrong, gold)


def _perturb_like(n: int) -> int:
    digits = str(n)
    d = len(digits)
    zeros = d - len(digits.rstrip("0"))
    step = 10 ** zeros
    lo = 10 ** (d - 1) if d > 1 else 0
    hi = 10 ** d - 1
    for _ in range(40):
        m = random.randint(lo, hi) // step * step
        s = str(m)
        if m != n and len(s) == d and len(s) - len(s.rstrip("0")) == zeros:
            return m
    return n


def _sample_multiply() -> Instance:
    a = random.randint(10, 99)
    tens, units = random.randint(1, 9) * 10, random.randint(1, 9)
    b = tens + units
    prompt = f"{a}*{b}"
    p_u, p_t = units * a, tens * a
    total = a * b
    correct = f"{units}*{a}={p_u} {tens}*{a}={p_t} {p_u}+{p_t}={total}"
    for _ in range(20):
        w_u, w_t = _perturb_like(p_u), _perturb_like(p_t)
        w_total = w_u + w_t
        if w_total != total:
            break
    wrong = f"{units}*{a}={w_u} {tens}*{a}={w_t} {w_u}+{w_t}={w_total}"
    return Instance(prompt, correct, wrong, str(total))


def _sample_count_char() -> Instance:
    L = random.randint(6, 14)
    alphabet = string.ascii_lowercase[:3]
    word = "".join(random.choice(alphabet) for _ in range(L))
    query = random.choice(word)
    prompt = f"{word};{query}"
    running, counts = 0, []
    for ch in word:
        running += int(ch == query)
        counts.append(running)
    correct = " ".join(f"{ch}{n}" for ch, n in zip(word, counts))
    shuffled = counts[:]
    random.shuffle(shuffled)
    wrong = " ".join(f"{random.choice(alphabet)}{n}" for n in shuffled)
    return Instance(prompt, correct, wrong, str(running))


# ---------- NEW: graph_path ----------
def _bfs_shortest(adj: Dict[int, List[int]], start: int, target: int) -> Tuple[int, List[str]]:
    """Returns (distance, list of BFS layer strings). -1 if unreachable."""
    if start == target:
        return 0, [f"{start}=0"]
    dist = {start: 0}
    q = deque([start])
    layers = [f"{start}=0"]
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                layers.append(f"{v}={dist[v]}")
                if v == target:
                    return dist[v], layers
                q.append(v)
    return -1, layers


def _sample_graph_path() -> Instance:
    n = random.randint(4, 6)                     # 4–6 nodes
    nodes = list(range(n))

    # Random sparse undirected graph
    possible = [(i, j) for i in nodes for j in nodes if i < j]
    random.shuffle(possible)
    n_edges = random.randint(n - 1, min(len(possible), n + 2))
    edges = possible[:n_edges]

    adj = {i: [] for i in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    s, t = random.sample(nodes, 2)

    # Prompt: edge list + query
    edge_str = " ".join(f"{u}-{v}" for u, v in edges)
    prompt = f"{edge_str};{s}>{t}"

    dist, correct_layers = _bfs_shortest(adj, s, t)
    gold = str(dist)
    correct = " ".join(correct_layers)

    # Wrong trace: same number of tokens, same digit statistics, destroyed logic
    # (random order + random distances in 0..n)
    fake_nodes = nodes[:]
    random.shuffle(fake_nodes)
    fake_dists = [random.randint(0, n) for _ in nodes]
    wrong = " ".join(f"{node}={d}" for node, d in zip(fake_nodes, fake_dists))

    return Instance(prompt, correct, wrong, gold)


# ---------- Registry ----------
DIGITS = string.digits
LOWER = string.ascii_lowercase

TASKS: Dict[str, Task] = {
    "word_index": Task(
        name="word_index",
        chars=LOWER + DIGITS + " ;:\n",
        block_size=64,
        max_new_tokens=35,
        sample=_sample_word_index,
        chance_acc=0.15652,
        ceiling_acc=0.89259,
        description="report the index of a queried letter",
        bayes_prob=_word_index_bayes,
    ),
    "sort_letters": Task(
        name="sort_letters",
        chars=LOWER + " :>\n",
        block_size=72,
        max_new_tokens=60,
        sample=_sample_sort_letters,
        chance_acc=0.00005,
        ceiling_acc=1.0,
        description="alphabetize a word via selection sort",
        answer_pattern=r"[a-z]+",
    ),
    "multiply": Task(
        name="multiply",
        chars=DIGITS + " :*+=\n",
        block_size=64,
        max_new_tokens=48,
        sample=_sample_multiply,
        chance_acc=0.00183,
        ceiling_acc=1.0,
        description="two-digit multiplication via partial products",
    ),
    "count_char": Task(
        name="count_char",
        chars=LOWER + DIGITS + " ;:\n",
        block_size=76,
        max_new_tokens=52,
        sample=_sample_count_char,
        chance_acc=0.23124,
        ceiling_acc=1.0,
        description="count occurrences of a letter (running tally)",
    ),
    "graph_path": Task(
        name="graph_path",
        chars=DIGITS + " -;>:=\n",
        block_size=80,
        max_new_tokens=55,
        sample=_sample_graph_path,
        chance_acc=0.18,          # approximate; re-estimate if needed
        ceiling_acc=1.0,
        description="shortest path length in a small undirected graph (BFS trace)",
        answer_pattern=r"-?\d+",
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}")
    return TASKS[name]


# ============================================================
# 4. Active task
# ============================================================
task = get_task(TASK_NAME)
tokenizer = task.tokenizer
block_size = task.block_size
max_new_tokens = task.max_new_tokens
PAD_ID = tokenizer.pad_id
NEWLINE_ID = tokenizer.newline_id
vocab_size = tokenizer.vocab_size

print(f"[TASK] {task.name} | vocab={vocab_size} | block={block_size} | "
      f"chance≈{task.chance_acc:.3f} | ceiling={task.ceiling_acc:.3f}")


# ============================================================
# 5–9. Dataset / Model / Eval / Plot / Main  (identical to previous version)
# ============================================================
def get_or_create_master_pools(n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    fp = task.fingerprint()
    correct_path = os.path.join(cache_dir, f"{task.name}_correct_{n_samples}_{fp}.pt")
    wrong_path   = os.path.join(cache_dir, f"{task.name}_wrong_{n_samples}_{fp}.pt")

    if os.path.exists(correct_path) and os.path.exists(wrong_path):
        print(f"\n[DISK] Loading cached pools for {task.name} ...")
        correct = tuple(t.to(device) for t in torch.load(correct_path, map_location=device))
        wrong   = tuple(t.to(device) for t in torch.load(wrong_path,   map_location=device))
        print("[DISK] Loaded.\n")
        return correct, wrong

    print(f"\n[GEN] Generating master pools for {task.name} ({n_samples:,} samples)...")
    rng_state = random.getstate()
    random.seed(data_seed)

    correct_xs, correct_ys, correct_masks = [], [], []
    wrong_xs,   wrong_ys,   wrong_masks   = [], [], []

    for _ in range(n_samples):
        inst = task.sample()

        for mode, xs, ys, masks in [
            ("correct_think", correct_xs, correct_ys, correct_masks),
            ("wrong_think",   wrong_xs,   wrong_ys,   wrong_masks),
        ]:
            prompt, answer = task.render(inst, mode)
            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(answer + "\n")
            full  = (p_ids + t_ids)[:block_size]
            Lp    = min(len(p_ids), len(full))
            x, y  = full[:-1], full[1:]
            mask  = [1.0 if (i + 1) >= Lp else 0.0 for i in range(len(full) - 1)]
            pad   = (block_size - 1) - len(x)
            x += [PAD_ID] * pad
            y += [PAD_ID] * pad
            mask += [0.0] * pad
            xs.append(x); ys.append(y); masks.append(mask)

    random.setstate(rng_state)

    master_correct = (
        torch.tensor(correct_xs, dtype=torch.long),
        torch.tensor(correct_ys, dtype=torch.long),
        torch.tensor(correct_masks, dtype=torch.float),
    )
    master_wrong = (
        torch.tensor(wrong_xs, dtype=torch.long),
        torch.tensor(wrong_ys, dtype=torch.long),
        torch.tensor(wrong_masks, dtype=torch.float),
    )

    print(f"[DISK] Saving pools ...")
    torch.save(master_correct, correct_path)
    torch.save(master_wrong,   wrong_path)
    print("[DISK] Saved.\n")

    return (tuple(t.to(device) for t in master_correct),
            tuple(t.to(device) for t in master_wrong))


def get_mixed_dataset_for_ratio(master_correct, master_wrong, ratio):
    n = master_correct[0].shape[0]
    n_correct = int(n * ratio)
    n_wrong   = n - n_correct
    c = [t[:n_correct] for t in master_correct]
    w = [t[:n_wrong]   for t in master_wrong]
    return tuple(torch.cat([c[i], w[i]]) for i in range(3))


def get_batch_from_fixed_dataset(dataset_tensors, batch_size):
    xs, ys, masks = dataset_tensors
    ix = torch.randint(0, xs.shape[0], (batch_size,), device=device)
    return xs[ix], ys[ix], masks[ix]


def build_validation_dataset(n_words=500, data_seed=DATASET_SEED):
    rng_state = random.getstate()
    random.seed(data_seed)
    examples = []
    seen = set()
    while len(examples) < n_words:
        inst = task.sample()
        if inst.prompt in seen:
            continue
        seen.add(inst.prompt)
        prompt, answer = task.render(inst, "correct_think")
        examples.append((prompt, answer + "\n"))
    random.setstate(rng_state)
    return examples


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss_per_token = F.cross_entropy(
            logits.view(B * T, -1), targets.view(B * T),
            reduction="none", ignore_index=PAD_ID, label_smoothing=label_smoothing
        ).view(B, T)
        loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_id is not None and idx_next.item() == stop_id:
                break
        return idx


def get_lr(step, total_steps=steps_per_run):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step > total_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


@torch.no_grad()
def evaluate_accuracy(model, test_examples):
    model.eval()
    correct = 0
    for prompt, target in test_examples:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=max_new_tokens, stop_id=NEWLINE_ID)[0].tolist()
        gen_text = tokenizer.decode(out)
        pred_tail = gen_text[len(prompt):]
        gold = target.strip().split(":")[-1].strip()
        pred = task.extract_answer(pred_tail)
        correct += int(pred == gold)
    model.train()
    return correct / len(test_examples)


def save_phase_transition_plot(results, seeds, filename=None):
    if filename is None:
        filename = f"phase_transition_{task.name}.png"
    ratios_pct = [r * 100 for r in results.keys()]
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, seed in enumerate(seeds):
        seed_accs = [results[r][i] * 100 if i < len(results[r]) else None for r in results]
        plt.scatter(ratios_pct, seed_accs, color=colors[i % len(colors)],
                    alpha=0.7, s=25, zorder=3, label=f"Seed {seed}")
    means = [(sum(accs) / len(accs)) * 100 for accs in results.values()]
    stds = [((sum(((x * 100) - m) ** 2 for x in accs) / len(accs)) ** 0.5
             if len(accs) > 1 else 0.0) for m, accs in zip(means, results.values())]
    lower = [max(0.0, m - s) for m, s in zip(means, stds)]
    upper = [min(100.0, m + s) for m, s in zip(means, stds)]
    plt.fill_between(ratios_pct, lower, upper, color="#1a73e8", alpha=0.15, label="±1 Std Dev")
    plt.plot(ratios_pct, means, color="#1a73e8", marker="o", linewidth=2.5,
             markersize=6, zorder=4, label="Mean Accuracy")
    plt.axhline(task.chance_acc * 100, color="gray", ls="--", alpha=0.6, label="Chance")
    plt.axhline(task.ceiling_acc * 100, color="green", ls="--", alpha=0.6, label="Ceiling")
    plt.title(f"Accuracy vs. Correct Trace Ratio — {task.name}", fontsize=12, fontweight="bold")
    plt.xlabel("Correct Think Trace Ratio (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xticks(ratios_pct, [f"{int(r)}%" for r in ratios_pct], rotation=45)
    plt.ylim(-2, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot saved as '{filename}'")


def run_clean_phase_experiment():
    print(f"Model Parameters: {sum(p.numel() for p in GPTModel().parameters() if p.requires_grad):,}")

    ratios_to_test = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
    seeds_to_test = [2200, 1337, 2026, 2003, 10]
    results = {}

    val_clean = build_validation_dataset(n_words=500, data_seed=DATASET_SEED)
    master_correct, master_wrong = get_or_create_master_pools()

    print("=" * 70)
    print(f"   PHASE TRANSITION EXPERIMENT — {task.name.upper()}")
    print(f"Device: {device} | Batch: {batch_size} | WD: {weight_decay} | Drop: {dropout}")
    print("=" * 70)

    for ratio in ratios_to_test:
        results[ratio] = []
        print(f"\n[Testing Ratio: {ratio*100:.1f}% Correct | {(1-ratio)*100:.1f}% Wrong]")
        train_tensors = get_mixed_dataset_for_ratio(master_correct, master_wrong, ratio)

        for seed in seeds_to_test[:1]:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = GPTModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            model.train()
            for step in range(steps_per_run):
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                xb, yb, mb = get_batch_from_fixed_dataset(train_tensors, batch_size)
                _, loss = model(xb, yb, mb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            acc = evaluate_accuracy(model, val_clean)
            results[ratio].append(acc)
            print(f"  --> Seed {seed} Accuracy: {acc*100:.2f}%")

    print("\n" + "=" * 70)
    print("        STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"{'Correct %':<12} | {'Wrong %':<12} | {'Mean Acc %':<15} | {'Std Dev %':<12}")
    print("-" * 70)
    for ratio, accs in results.items():
        mean = sum(accs) / len(accs)
        std  = (sum((x - mean)**2 for x in accs) / len(accs))**0.5 if len(accs) > 1 else 0.0
        print(f"{ratio*100:>10.1f}% | {(1-ratio)*100:>10.1f}% | {mean*100:>13.2f}% | ± {std*100:>8.2f}%")
    print("=" * 70)

    save_phase_transition_plot(results, seeds_to_test)


if __name__ == "__main__":
    run_clean_phase_experiment()