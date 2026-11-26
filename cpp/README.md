# `cpp/` — C++20 Causal Interventions & `(Step-ODE)` Suite

Self-contained C++20 implementations of the 27-block Transformer, baseline-projection ($\Pi_0$) interventions, Stateless Block-RMS SGD, and float64 `(Step-ODE)` solver (Figures 3–4, Tables 1–2a, 9–10). Uses Apple Accelerate on macOS and `cblas_shim.h` on Linux automatically.

```bash
make all check   # build all 3 binaries and run float64 finite-difference grad check
```

- `causal_interventions.cpp` — 27-block training & rollout engine (`--mode 0..10`)
- `grad_check.cpp` — float64 reference & finite-difference gradient verifier
- `step_ode.cpp` — float64 `(Step-ODE)` solver (default suite or single-run CLI)
- `analysis/aggregate.py`, `analysis/plot_causal_figure.py` — summary stats & Figure 4 regeneration

See [`../RUN.md`](../RUN.md) for sweep commands.
