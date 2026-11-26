# `cpp/` — C++20 Causal Interventions & `(Step-ODE)` Suite

Native C++20 implementation of the 27-block Transformer, baseline-projection ($\Pi_0$) interventions, Stateless Block-RMS SGD, and float64 `(Step-ODE)` verifier (Figures 3–4, Tables 1–2a, 9–10). Uses Apple Accelerate on macOS and `cblas_shim.h` on Linux automatically.

```bash
make all check   # build all targets and run float64 finite-difference grad check
```

- `run_full_plan.cpp` — 27-block training & rollout engine (`--mode 0..10`)
- `shortcut_intervention.cpp` — float64 reference & gradient checker
- `step_ode_suite.cpp`, `ode_extra.cpp`, `ode_fixed.cpp` — float64 `(Step-ODE)` solvers
- `run_causal_fast.cpp`, `run_causal_sweep.cpp`, `aggregate_results.cpp` — fast sweep & CSV aggregator
- `analysis/aggregate.py`, `analysis/make_causal_fig.py` — summary stats & Figure 4 regeneration

See [`../RUN.md`](../RUN.md) for sweep commands.
