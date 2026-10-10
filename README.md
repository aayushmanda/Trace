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
| `handcoded/lettertrace_train.ipynb` | Trains step and outcome from one initialization, replays them against the two executors, and writes `handcoded/animations/lettertrace_train.gif`. |
| `handcoded/regime_both.py` | Outcome plateau (Theorem 3): exit time from the initialization ball against the horizon, inside the theorem's regime. |
| `handcoded/joint_gate.py` | Theorem 4 companion: joint gradient flow of the gate head (counting modulo 2) from randomly scaled hand-coded directions, with every factor and a full-prefix relative bias trained; logs loss, margins, gate alignment and leakage. |
| `handcoded/gate_steering.py` | Gate-head experiments in the full network: joint training (`joint_gd`, `joint_adam`) and outcome training (`outcome_all`) from a seeded initialization. |
| `src/` | Boolean-8 and the register machine on a character-token GPT, and rank-8 LoRA on SmolLM2 / Qwen2.5, including BIG-Bench Hard. |

Commands for every experiment are in [RUN.md](RUN.md).
