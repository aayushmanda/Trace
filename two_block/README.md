# `two_block/` — Two-Block Transformer & Numerical Checks

PyTorch code for the two-block counting experiments, exact executors, and numerical checks in *Silent Routing*.

- `counting.py` — two-block Transformer, exact executors (Prop. 1), paired step/outcome training (Fig. 1a, Tables 6–8)
- `population_gd.py` — exact float64 population GD exit times (Thm. 1, Table 5)
- `experiments/` — `sweeps.py` (`zero_readout`, `basin`, `lambda_grid`), `spectra.py` (spectra, ANOVA, untied routing), `init_scale.py`, `run_*.sh`
- `checks/` — `check_executors.py` (Prop. 1), `check_credit.py` (App. C & G), `check_logs.py` (Tables 6–7)
- `demo.ipynb` — short interactive walkthrough notebook

```bash
uv run python two_block/checks/check_executors.py
uv run python two_block/counting.py task=count word_len=6 mod=2 steps=8000
```

See [`../RUN.md`](../RUN.md) for all commands.
