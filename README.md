# trace

Code for *Where the Credit Is: Why Running Counts Teach What Final Counts Miss*. One Transformer, two losses:
step supervision on the running count, and outcome supervision on the final count only.

```bash
uv sync
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```

| File | What it does |
|---|---|
| `handcoded/lettertrace.py` | Counting task. Writes both exact executors into one architecture, then trains step and outcome from a random initialization. |
| `handcoded/regime_both.py` | Outcome plateau (Theorem 3): exit time from the initialization ball against the horizon, inside the theorem's regime. |
| `handcoded/gate_steering.py` | Staged step learning (Theorem 4): the structured seed run through Stages A–D to the gate region and zero loss. |
| `src/` | Boolean-8 and the register machine on a character-token GPT, and rank-8 LoRA on SmolLM2 / Qwen2.5, including BIG-Bench Hard. |

Commands for every experiment are in [RUN.md](RUN.md).
