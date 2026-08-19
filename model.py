# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

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
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        pad_id,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.05,
    ):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.block_size = block_size
        self.pad_id = pad_id

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size={self.block_size}")

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss_per_token = F.cross_entropy(
            logits.reshape(B * T, -1),
            targets.reshape(B * T),
            reduction="none",
            ignore_index=self.pad_id,
        ).view(B, T)

        if mask is not None:
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss_per_token.mean()

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        stop_id=None,
        greedy=True,
        temperature=1.0,
        top_k=None,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                logits = logits / max(temperature, 1e-8)

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(
                        logits < values[:, [-1]],
                        float("-inf")
                    )

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if stop_id is not None and (idx_next == stop_id).all():
                break

        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 46
    pad_id = 45
    block_size = 64

    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.05

    model = GPTModel(
        vocab_size=vocab_size,
        block_size=block_size,
        pad_id=pad_id,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Parameters: {count_parameters(model) / 1e6:.3f} M")

    B, T = 2, 32

    dummy_input = torch.randint(0, vocab_size - 1, (B, T), device=device)
    dummy_targets = torch.randint(0, vocab_size - 1, (B, T), device=device)

    dummy_mask = torch.zeros(B, T, device=device)
    dummy_mask[:, 10:] = 1.0

    logits, loss = model(
        dummy_input,
        targets=dummy_targets,
        mask=dummy_mask,
    )

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    context = dummy_input[:1, :8]

    generated = model.generate(
        context,
        max_new_tokens=10,
        greedy=True,
    )

    print("Context shape:", context.shape)
    print("Generated shape:", generated.shape)