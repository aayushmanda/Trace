# trace

Same tiny Transformer. Two losses.

Process writes the running count, then the answer. Outcome writes only the answer. The dashed lines are the same two programs, written in by hand.

![process learns the trace, outcome stays near chance](handcoded/animations/lettertrace_train.gif)

Counting mod 2, six letters. Blue is process, orange is outcome. Process reaches the trace. Outcome stays near chance. Each letter the loss does not see multiplies the answer gradient by \(\lambda = 1/2\). The step loss never pays that factor.

## run

```bash
python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```

`handcoded/lettertrace_train.ipynb` trains both from one initialization and writes the gif.

Paper runs are in [RUN.md](RUN.md).
