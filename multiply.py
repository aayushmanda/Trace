import random
import string
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import string
import torch.nn as nn
from torch.nn import functional as F

random.seed(10)
torch.manual_seed(1300)

def make_example(think=False, corrupt=False):
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    prompt = f"{a}*{b}="

    if think:
        ones = b % 10
        tens = b // 10
        
        p1 = a * ones
        p2 = a * tens
        ans = a * b

        # Inject flawed calculation into intermediate steps or final sum
        if corrupt:
            error_type = "step1"
            noise = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            if error_type == "step1":
                p1 += noise
                p2 += noise
            else:
                ans += noise  # Fixed: changed `ans += 0` to `ans += noise`

        trace = f"{a}*{ones}={p1};{a}*{tens}={p2};{p2}0+{p1}={ans}"
        target = trace + "\n"

    else:
        ans = a * b
        
        # Inject error directly into non-thinking output
        if corrupt:
            noise = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ans += noise

        target = str(ans) + "\n"

    return prompt, target


def build_corpus(n=50000, think=False, corrupt=False):
    lines = []
    for _ in range(n):
        p, t = make_example(think=think, corrupt=corrupt)
        lines.append(p + t)
    return "".join(lines)


# Dataset Generation
think_correct_text = build_corpus(50000, think=True, corrupt=False)
think_wrong_text = build_corpus(50000, think=True, corrupt=True)
nothink_wrong_text = build_corpus(50000, think=False, corrupt=True)  # Fixed: generated missing corpus


with open("mul_think.txt", "w") as f: 
    f.write(think_correct_text)

with open("mul_wrong_think.txt", "w") as f: 
    f.write(think_wrong_text)

with open("mul_think.txt", "w") as f: f.write(think_correct_text)
with open("mul_wrong_think.txt", "w") as f: f.write(think_wrong_text)

# ------------ HYPERPARAMETERS ------------
batch_size = 64
block_size = 64
max_iters = 8000        # Iteration count per ratio run
eval_interval = 200
learning_rate = 3e-4
weight_decay = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 64
n_head = 4
n_layer = 6
dropout = 0.1
patience = 5
TOTAL_TRAIN_SAMPLES = 50000  # Total training dataset size per experiment
# -----------------------------------------

random.seed(10)
torch.manual_seed(1300)

# Tokenizer Setup
chars = sorted(list(set(string.digits + " ><*;+=\n")))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
PAD_ID = len(chars)
vocab_size = len(chars) + 1
NEWLINE_ID = stoi["\n"]

def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(ids): return ''.join(itos[i] for i in ids if i in itos)

# Robust data loading function
def load_examples(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = [l for l in f.read().split('\n') if l.strip() != '']
    examples = []
    for line in raw_lines:
        if "=" in line:
            prompt, target_part = line.split("=", 1)
            examples.append((prompt + "=", target_part + "\n"))
    return examples

# Load both datasets
correct_examples = load_examples('mul_think.txt')
wrong_examples = load_examples('mul_wrong_think.txt')

# Reserve a fixed clean validation set (100% correct) for consistent evaluation
random.shuffle(correct_examples)
val_examples = correct_examples[:1000]
train_correct_pool = correct_examples[1000:]
train_wrong_pool = wrong_examples

def make_batch(examples_list):
    batch = random.sample(examples_list, min(batch_size, len(examples_list)))
    xs, ys, masks = [], [], []
    for prompt, target in batch:
        p_ids = encode(prompt)
        t_ids = encode(target)
        full = (p_ids + t_ids)[:block_size]
        Lp = min(len(p_ids), len(full))

        x = full[:-1]
        y = full[1:]
        mask = [1 if (i + 1) >= Lp else 0 for i in range(len(full) - 1)]

        pad_len = (block_size - 1) - len(x)
        x = x + [PAD_ID] * pad_len
        y = y + [PAD_ID] * pad_len
        mask = mask + [0] * pad_len

        xs.append(x)
        ys.append(y)
        masks.append(mask)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.float, device=device))

# Network Architecture
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss_per_token = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), reduction='none', ignore_index=PAD_ID).view(B, T)
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1) if mask is not None else loss_per_token.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_id is not None and idx_next.item() == stop_id:
                break
        return idx

@torch.no_grad()
def test_accuracy(model, eval_examples, n_samples=200):
    model.eval()
    correct = 0
    sample = random.sample(eval_examples, min(n_samples, len(eval_examples)))

    for prompt, target in sample:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=block_size, stop_id=NEWLINE_ID)[0].tolist()
        gen = decode(out)
        pred = gen[len(prompt):]

        gold_nums = re.findall(r'\d+', target)
        pred_nums = re.findall(r'\d+', pred)

        gold = gold_nums[-1] if gold_nums else None
        pred = pred_nums[-1] if pred_nums else None

        if pred == gold:
            correct += 1

    acc = correct / len(sample)
    model.train()
    return acc

def run_experiment(correct_ratio):
    print(f"\n--- Running Experiment: Wrong Data Ratio = {correct_ratio * 100:.0f}% ---")
    
    # Mix training data according to target ratio
    n_correct = int(TOTAL_TRAIN_SAMPLES * correct_ratio)
    n_wrong = TOTAL_TRAIN_SAMPLES - n_correct

    train_data = random.sample(train_wrong_pool, min(n_wrong, len(train_wrong_pool))) + \
                 random.sample(train_correct_pool, min(n_correct, len(train_correct_pool)))
    random.shuffle(train_data)

    model = GPTModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(eval_iters):
                    X, Y, M = make_batch(val_examples)
                    _, loss = model(X, Y, M)
                    losses.append(loss.item())
            val_loss = sum(losses) / len(losses)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at step {iter_num}")
                    break

        xb, yb, mb = make_batch(train_data)
        logits, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    acc = test_accuracy(model, val_examples)
    print(f"Accuracy for Wrong Ratio ({correct_ratio*100:.0f}%): {acc * 100:.2f}%")
    return acc

# ------------ SWEEP EXECUTION ------------
ratios_to_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 0% wrong to 100% wrong
results = {}

for ratio in ratios_to_sweep:
    acc = run_experiment(ratio)
    results[ratio] = acc

print("\n" + "="*30)
print("       SWEEP SUMMARY       ")
print("="*30)
print("Wrong Ratio | Clean Accuracy")
print("-"*27)
for r, a in results.items():
    print(f"  {r*100:5.1f}%     |   {a*100:6.2f}%")