"""Two-block causal Transformer counting experiments and numerical checks for Silent Routing (ICML 2026).

Core modules:
  lettertrace.py   Two-block Transformer, exact executors (Prop. 1 / App. B), step & outcome training.
  regime_both.py   Exact float64 population GD exit times inside the outcome plateau (Thm. 1 / Table 5).
  experiments/     Counting sweeps, Jacobian/ANOVA spectra, and untied routing runs (Fig. 1a, App. G).
  checks/          Independent executor, credit/degree, and training-log verification scripts.
"""
