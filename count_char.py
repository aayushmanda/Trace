import os
import math
import random
import re
import string
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# Global Versioning & Hyper-parameters
# ============================================================
DATASET_VERSION = "v5_matched_strict"

batch_size = 128
steps_per_run = 4000
learning_rate = 3e-4
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
dropout = 0.05
label_smoothing = 0.1
max_grad_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 64
n_head = 4
n_layer = 1

DATASET_SIZE = 50000
DATASET_SEED = 100
VAL_SEED = 999999
CACHE_DIR = "./dataset_cache"

# Global unified vocabulary across all tasks
GLOBAL_CHARS = string.ascii_lowercase + string.digits + " ;:->+=*\n"

# ============================================================
# Choose task
# ============================================================
TASK_NAME = "count_char"  # "word_index" | "sort_letters" | "multiply" | "count_char" | "graph_path"

# ============================================================
# Task family
# ============================================================
ANSWER_SEP = " : "


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.newline_id = self.stoi.get("\n", None)

    def encode(self, s: str) -> List[int]:
        unknown = set(s) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters in input string: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


# Global Tokenizer Instance (shared vocabulary across every task)
GLOBAL_TOKENIZER = CharTokenizer(GLOBAL_CHARS)


@dataclass(frozen=True)
class Instance:
    prompt: str
    correct_trace: str
    wrong_trace: str
    gold: str


@dataclass
class Task:
    name: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float
    ceiling_acc: float
    description: str = ""
    answer_pattern: str = r"-?\d+"
    bayes_prob: Optional[Callable[[Instance], float]] = None

    @property
    def tokenizer(self) -> CharTokenizer:
        return GLOBAL_TOKENIZER

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
        h.update(DATASET_VERSION.encode())
        h.update(self.name.encode())
        h.update(f"{DATASET_SEED}|{VAL_SEED}".encode())
        for _ in range(n):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                h.update(f"{prompt}{answer}\n".encode())
        random.setstate(state)
        h.update(f"{self.block_size}|{''.join(self.tokenizer.chars)}".encode())
        return h.hexdigest()[:10]


# ============================================================
# Task Samplers
# ============================================================
def _sample_word_index() -> Instance:
    L = random.randint(4, 10)
    word = "".join(random.choice(string.ascii_lowercase) for _ in range(L))
    gold = random.randrange(L)
    prompt = f"{word};{word[gold]}"
    correct = " ".join(f"{i}{ch}" for i, ch in enumerate(word))

    # Preserve index monotonicity (0..L-1); corrupt character assignments only.
    wrong_chars = [random.choice([c for c in string.ascii_lowercase if c != ch]) for ch in word]
    wrong = " ".join(f"{i}{ch}" for i, ch in enumerate(wrong_chars))
    return Instance(prompt, correct, wrong, str(gold))


def _word_index_bayes(inst: Instance) -> float:
    word, _, c = inst.prompt.partition(";")
    return 1.0 / max(1, word.count(c))


def _sample_sort_letters() -> Instance:
    L = random.randint(7, 10)
    letters = random.sample(string.ascii_lowercase, L)
    word = "".join(letters)
    gold = "".join(sorted(letters))

    remaining, steps = list(letters), []
    while len(remaining) > 1:
        chosen = min(remaining)
        steps.append(f"{''.join(remaining)}>{chosen}")
        remaining.remove(chosen)
    correct = " ".join(steps)

    # Same state-transition structure; every selected item violates selection-sort.
    remaining, steps = list(letters), []
    while len(remaining) > 1:
        correct_choice = min(remaining)
        wrong_choices = [x for x in remaining if x != correct_choice]
        chosen = random.choice(wrong_choices)
        steps.append(f"{''.join(remaining)}>{chosen}")
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
    return n + 1


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

    # Preserve character sequence position-by-position; corrupt tally values only.
    wrong_counts = [(n + random.choice([1, 2])) % (L + 1) for n in counts]
    wrong = " ".join(f"{ch}{n}" for ch, n in zip(word, wrong_counts))
    return Instance(prompt, correct, wrong, str(running))


def _bfs_all(adj: Dict[int, List[int]], start: int) -> Dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _sample_graph_path() -> Instance:
    while True:
        n = random.randint(6, 9)
        nodes = list(range(n))
        possible = [(i, j) for i in nodes for j in nodes if i < j]
        random.shuffle(possible)
        n_edges = random.randint(n - 1, min(len(possible), n + 2))
        edges = possible[:n_edges]

        adj = {i: [] for i in nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        s, t = random.sample(nodes, 2)
        dist_map = _bfs_all(adj, s)

        if t in dist_map and dist_map[t] > 0:
            q = deque([s])
            visited = {s}
            correct_layers = [f"{s}=0"]
            found = False
            while q and not found:
                u = q.popleft()
                for v in sorted(adj[u]):
                    if v not in visited:
                        visited.add(v)
                        correct_layers.append(f"{v}={dist_map[v]}")
                        if v == t:
                            found = True
                            break
                        q.append(v)
            if found:
                break

    edge_str = " ".join(f"{u}-{v}" for u, v in edges)
    prompt = f"{edge_str};{s}>{t}"
    gold = str(dist_map[t])
    correct = " ".join(correct_layers)

    k = len(correct_layers)
    parsed_layers = [layer.split("=") for layer in correct_layers]
    wrong_pairs = [list(p) for p in parsed_layers]

    if k == 2:
        # Direct edge: only the distance value can be corrupted.
        true_d = int(parsed_layers[1][1])
        wrong_d = (true_d % n) + 1
        wrong_pairs[1][1] = str(wrong_d)
    else:
        # Swap two internal positions whose true distances differ, preserving
        # the overall node set and distance multiset exactly.
        internal = list(range(1, k - 1))
        candidate_swaps = [
            (i, j) for i in internal for j in internal
            if i < j and parsed_layers[i][1] != parsed_layers[j][1]
        ]
        if candidate_swaps:
            i, j = random.choice(candidate_swaps)
            wrong_pairs[i][1], wrong_pairs[j][1] = wrong_pairs[j][1], wrong_pairs[i][1]
        else:
            idx = random.choice(internal)
            curr_d = int(parsed_layers[idx][1])
            wrong_pairs[idx][1] = str((curr_d % n) + 1)

    wrong_layers = [f"{node}={d}" for node, d in wrong_pairs]
    wrong = " ".join(wrong_layers)

    assert any(dist_map[int(node)] != int(d) for node, d in (p.split("=") for p in wrong_layers)), \
        "Wrong trace failed graph invalidity assertion!"

    return Instance(prompt, correct, wrong, gold)


# ============================================================
# Registry
# ============================================================
TASKS: Dict[str, Task] = {
    "word_index": Task(
        name="word_index",
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
        block_size=256,
        max_new_tokens=180,
        sample=_sample_sort_letters,
        chance_acc=0.00001,
        ceiling_acc=1.0,
        description="alphabetize a word via selection sort",
        answer_pattern=r"[a-z]+",
    ),
    "multiply": Task(
        name="multiply",
        block_size=64,
        max_new_tokens=48,
        sample=_sample_multiply,
        chance_acc=0.00183,
        ceiling_acc=1.0,
        description="two-digit multiplication via partial products",
    ),
    "count_char": Task(
        name="count_char",
        block_size=76,
        max_new_tokens=52,
        sample=_sample_count_char,
        chance_acc=0.23124,
        ceiling_acc=1.0,
        description="count occurrences of a letter (running tally)",
    ),
    "graph_path": Task(
        name="graph_path",
        block_size=128,  # raised to eliminate truncation risk in the worst case
        max_new_tokens=64,
        sample=_sample_graph_path,
        chance_acc=0.18,  # approximate; re-estimate empirically if needed
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
# Active task
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
# Data Pools
# ============================================================
def get_or_create_master_pools(n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    fp = task.fingerprint()
    correct_path = os.path.join(cache_dir, f"{task.name}_correct_{n_samples}_{fp}.pt")
    wrong_path = os.path.join(cache_dir, f"{task.name}_wrong_{n_samples}_{fp}.pt")
    prompts_path = os.path.join(cache_dir, f"{task.name}_prompts_{n_samples}_{fp}.pt")

    if os.path.exists(correct_path) and os.path.exists(wrong_path) and os.path.exists(prompts_path):
        print(f"\n[DISK] Loading cached pools for {task.name} ...")
        correct = tuple(t.to(device) for t in torch.load(correct_path, map_location=device, weights_only=True))
        wrong = tuple(t.to(device) for t in torch.load(wrong_path, map_location=device, weights_only=True))
        prompts = torch.load(prompts_path, weights_only=False)
        print("[DISK] Loaded.\n")
        return correct, wrong, prompts

    print(f"\n[GEN] Generating master pools for {task.name} ({n_samples:,} samples)...")
    rng_state = random.getstate()
    random.seed(data_seed)

    correct_xs, correct_ys, correct_masks = [], [], []
    wrong_xs, wrong_ys, wrong_masks = [], [], []
    prompts = []

    for _ in range(n_samples):
        inst = task.sample()
        prompts.append(inst.prompt)

        for mode, xs, ys, masks in [
            ("correct_think", correct_xs, correct_ys, correct_masks),
            ("wrong_think", wrong_xs, wrong_ys, wrong_masks),
        ]:
            prompt, answer = task.render(inst, mode)
            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(answer + "\n")
            full = p_ids + t_ids

            if len(full) > block_size:
                raise ValueError(
                    f"Task '{task.name}' sequence length ({len(full)}) exceeds block_size ({block_size}). "
                    f"Prompt: {prompt!r}, Answer: {answer!r}"
                )

            Lp = len(p_ids)
            x, y = full[:-1], full[1:]
            mask = [1.0 if (i + 1) >= Lp else 0.0 for i in range(len(full) - 1)]
            pad = (block_size - 1) - len(x)
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

    print("[DISK] Saving pools ...")
    torch.save(master_correct, correct_path)
    torch.save(master_wrong, wrong_path)
    torch.save(prompts, prompts_path)
    print("[DISK] Saved.\n")

    return (tuple(t.to(device) for t in master_correct),
            tuple(t.to(device) for t in master_wrong),
            prompts)


def get_mixed_dataset_for_ratio(master_correct, master_wrong, ratio, seed=None):
    n = master_correct[0].shape[0]
    n_correct = int(n * ratio)

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g, device=device)
    else:
        perm = torch.randperm(n, device=device)

    correct_idx = perm[:n_correct]
    wrong_idx = perm[n_correct:]

    c = [t[correct_idx] for t in master_correct]
    w = [t[wrong_idx] for t in master_wrong]
    return tuple(torch.cat([c[i], w[i]], dim=0) for i in range(3))


def get_batch_from_fixed_dataset(dataset_tensors, batch_size):
    xs, ys, masks = dataset_tensors
    ix = torch.randint(0, xs.shape[0], (batch_size,), device=device)
    return xs[ix], ys[ix], masks[ix]


def build_validation_dataset(n_words=1000, master_prompts=None, val_seed=VAL_SEED) -> List[Instance]:
    rng_state = random.getstate()
    random.seed(val_seed)

    master_prompts_set = set(master_prompts) if master_prompts is not None else set()
    instances = []
    seen = set()

    attempts = 0
    max_attempts = n_words * 20

    while len(instances) < n_words and attempts < max_attempts:
        attempts += 1
        inst = task.sample()
        if inst.prompt in seen or inst.prompt in master_prompts_set:
            continue
        seen.add(inst.prompt)
        instances.append(inst)

    random.setstate(rng_state)

    if master_prompts_set:
        val_prompts = {inst.prompt for inst in instances}
        assert val_prompts.isdisjoint(master_prompts_set), "Validation set overlaps with master training pool!"

    return instances


# ============================================================
# Model
# ============================================================
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


# ============================================================
# Evaluation (free-generation only; forced_correct/forced_wrong removed)
# ============================================================
@torch.no_grad()
def evaluate_all_modes(model, test_instances: List[Instance]) -> Dict[str, float]:
    model.eval()
    counts = {"free": 0}

    for inst in test_instances:
        ctx_free = task.context(inst, "free")
        ids_free = torch.tensor([tokenizer.encode(ctx_free)], dtype=torch.long, device=device)
        out_free = model.generate(ids_free, max_new_tokens=max_new_tokens, stop_id=NEWLINE_ID)[0].tolist()
        pred_free = task.extract_answer(tokenizer.decode(out_free)[len(ctx_free):])
        counts["free"] += int(pred_free == inst.gold)

    model.train()
    n = len(test_instances)
    return {k: v / n for k, v in counts.items()}


def save_phase_transition_plot(results, replicates, filename=None):
    if filename is None:
        filename = f"phase_transition_{task.name}.png"
    ratios_pct = [r * 100 for r in results.keys()]
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, rep in enumerate(replicates):
        rep_accs = [results[r][i]["free"] * 100 if i < len(results[r]) else None for r in results]
        plt.scatter(ratios_pct, rep_accs, color=colors[i % len(colors)],
                    alpha=0.7, s=25, zorder=3, label=f"Replicate {rep}")

    means = [(sum(r_dict["free"] for r_dict in accs) / len(accs)) * 100 for accs in results.values()]
    stds = [((sum(((r_dict["free"] * 100) - m) ** 2 for r_dict in accs) / len(accs)) ** 0.5
             if len(accs) > 1 else 0.0) for m, accs in zip(means, results.values())]

    lower = [max(0.0, m - s) for m, s in zip(means, stds)]
    upper = [min(100.0, m + s) for m, s in zip(means, stds)]

    plt.fill_between(ratios_pct, lower, upper, color="#1a73e8", alpha=0.15, label="±1 Std Dev")
    plt.plot(ratios_pct, means, color="#1a73e8", marker="o", linewidth=2.5,
             markersize=6, zorder=4, label="Mean Free-Gen Acc")

    plt.axhline(task.chance_acc * 100, color="gray", ls="--", alpha=0.6, label="Chance")
    plt.axhline(task.ceiling_acc * 100, color="green", ls="--", alpha=0.6, label="Ceiling")

    plt.title(f"Free-Gen Accuracy vs. Correct Trace Ratio — {task.name}", fontsize=12, fontweight="bold")
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


# ============================================================
# Main experiment
# ============================================================
def run_clean_phase_experiment():
    print(f"Model Parameters: {sum(p.numel() for p in GPTModel().parameters() if p.requires_grad):,}")

    ratios_to_test = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.0]
    replicates_to_test = [2200, 1337, 2026, 2003, 10]
    results: Dict[float, List[Dict[str, float]]] = {}

    master_correct, master_wrong, master_prompts = get_or_create_master_pools()
    val_instances = build_validation_dataset(n_words=1000, master_prompts=master_prompts, val_seed=VAL_SEED)

    print("=" * 80)
    print(f"   PHASE TRANSITION EXPERIMENT — {task.name.upper()}")
    print(f"Steps: {steps_per_run} | Batch: {batch_size} | WD: {weight_decay} | Embed: {n_embd} | Layers: {n_layer}")
    print("=" * 80)

    for ratio_idx, ratio in enumerate(ratios_to_test):
        results[ratio] = []
        print(f"\n[Testing Ratio: {ratio*100:.1f}% Correct | {(1-ratio)*100:.1f}% Wrong]")

        for rep in replicates_to_test[:3]:  # all replicates, not just the first
            run_seed = rep                   # unique per (replicate, ratio) cell
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            train_tensors = get_mixed_dataset_for_ratio(
                master_correct, master_wrong, ratio, seed=run_seed
            )

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

            eval_metrics = evaluate_all_modes(model, val_instances)
            results[ratio].append(eval_metrics)
            print(f"Replicate {rep} | Free Acc: {eval_metrics['free']*100:.2f}%")

    print("\n" + "=" * 80)
    print(" STATISTICAL SUMMARY (Free-Generation Accuracy)")
    print("=" * 80)
    print(f"{'Correct %':<10} | {'Wrong %':<10} | {'Mean Free %':<13} | {'Std Dev %':<10}")
    print("-" * 50)
    for ratio, metrics_list in results.items():
        frees = [m["free"] for m in metrics_list]
        mean_free = sum(frees) / len(frees)
        std_free = (sum((x - mean_free) ** 2 for x in frees) / len(frees)) ** 0.5 if len(frees) > 1 else 0.0
        print(f"{ratio*100:>8.1f}% | {(1-ratio)*100:>8.1f}% | {mean_free*100:>11.2f}% | ± {std_free*100:>6.2f}%")
    print("=" * 80)

    save_phase_transition_plot(results, replicates_to_test)


if __name__ == "__main__":
    run_clean_phase_experiment()