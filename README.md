# trace

Process supervision versus outcome supervision. Same Transformer, two losses. Process writes the running state, then the answer. Outcome writes only the answer.

![process learns the trace, outcome stays near chance](handcoded/animations/lettertrace_train.gif)

Counting mod 2, six letters. Blue is process, orange is outcome. Dashed lines are the same programs written in by hand. Process learns the trace. Outcome stays near chance: each letter the loss does not see multiplies the answer gradient by \(\lambda = 1/2\).

## Requirements

```bash
pip install -r requirements.txt
```

One GPU. A short run fits on CPU.

## Training

```bash
python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```

`handcoded/lettertrace_train.ipynb` trains process and outcome from one initialization and writes the gif.

## Layout

| Path | Contents |
|---|---|
| `handcoded/` | Counting model, spectra, figures, notebook |
| `src/` | Boolean circuits, register machine, LoRA |
| `experiments/` | Closed-form checks |
| `scripts/` | Checkpoint spectra and the \(\rho\) sweep |
| `Paper/` | LaTeX |

Other paper commands are in [RUN.md](RUN.md).
