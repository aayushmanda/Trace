# `handcoded/` — Two-Block Transformer & Numerical Checks

PyTorch code for the two-block counting experiments, exact executors, and numerical checks in *Silent Routing*.

- `lettertrace.py` — two-block Transformer, exact executors (Prop. 1), paired step/outcome training (Fig. 1a, Tables 6–8)
- `regime_both.py` — exact float64 population GD exit times (Thm. 1, Table 5)
- `experiments/` — `sweeps.py` (`zero_readout`, `basin`, `lambda_grid`), `measure.py` (spectra, ANOVA, untied routing), `seeded_sweep.py`, shell sweeps
- `checks/` — `executor_check.py` (Prop. 1), `credit_checks.py` (App. C & G), `log_claims.py` (Tables 6–7)
- `silent_routing_demo.ipynb` — short interactive walkthrough notebook

```bash
uv run python handcoded/checks/executor_check.py
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```

See [`../RUN.md`](../RUN.md) for all commands.
