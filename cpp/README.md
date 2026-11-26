# C++20 Causal Shortcut-Intervention & 27-Block Verification Suite

Self-contained native C++20 implementations of the unconstrained 27-block two-block Transformer (`Definition 2` in
`papers/silent_routing/body.tex`), the causal baseline-projection interventions (`Section 5.2`, `Figure 4`, `Table 1`),
Stateless Block-RMS SGD (`Algorithm 1`, `Table 2a`), and the float64 `(Step-ODE)` population verifier. Zero Python
dependencies are required to compile and run the C++ drivers.

The `Makefile` automatically uses Apple Accelerate (`clang++ -framework Accelerate`) on macOS and portable C++20
(`g++` + `cblas_shim.h`) on Linux.

## Source Files

- `run_full_plan.cpp` — Batched `float32` multithreaded 27-block Transformer training and rollout evaluation engine
  supporting all 11 experimental modes:
  - `--mode 0`: **Natural** step training ($V(0)=0$, AdamW)
  - `--mode 1`: **Baseline suppressed** ($\Pi_0^{(p)}$ orthogonal minibatch logit projection holding $\hat c_{t,p}=0$)
  - `--mode 2`: **External logit offset** ($+c^\star x_{t-1}$ added externally with $V(0)=0$)
  - `--mode 3`: **Suppressed, then released** at `--release 600` (or `1000`)
  - `--mode 4`: **Weight-calibrated init** ($V(0)=c^\star$ along the state axis)
  - `--mode 5`: **External offset + nonzero orthogonal $V(0)$**
  - `--mode 6`: **Transient external offset** (removed at step 300)
  - `--mode 7`: **External offset + 10x readout learning rate**
  - `--mode 8`: **Stateless Block-RMS SGD** (step supervision, Algorithm 1)
  - `--mode 9`: **Stateless Block-RMS SGD** (matched outcome supervision)
  - `--mode 10`: **Outcome supervision** (AdamW)
- `shortcut_intervention.cpp` — Reference `float64` analytical forward/backward implementation with centered
  finite-difference gradient verification (`make check`, verifying `max_rel_err ~ 2.4e-7` across all 27 blocks, relative
  lag biases, and projection hooks).
- `run_causal_fast.cpp` / `run_causal_sweep.cpp` — Fast batched drivers for the five core conditions and sweeps.
- `step_ode_suite.cpp`, `ode_extra.cpp`, `ode_fixed.cpp` — Exact `float64` population `(Step-ODE)` solvers.
- `aggregate_results.cpp` — Pure C++20 multi-seed CSV aggregator.
- `cblas_shim.h` — Header-only CBLAS fallback used automatically on non-Apple platforms.
- `analysis/aggregate.py` — Aggregates `results/causal_intervention/sweep/` into solve rates, Wilson 95% CIs, and
  two-sided Fisher exact tests (`analysis/summary.json`).
- `analysis/make_causal_fig.py` — Regenerates `papers/silent_routing/figures/causal_intervention.tex` from the 20-seed
  sweep CSVs.

## Build & Reproduce

```bash
# Build all binaries and run the finite-difference gradient check
make all check

# Run the 20-seed sweep (n = 8, p = 0.25) into results/causal_intervention/sweep/
mkdir -p ../results/causal_intervention/sweep/main ../results/causal_intervention/sweep/rel600
for s in $(seq 42 61); do
  for m in 0 1 2 4 6 8 9; do
    ./run_full_plan --n 8 --p 0.25 --seed $s --mode $m --steps 2400 --outdir ../results/causal_intervention/sweep/main
  done
  ./run_full_plan --n 8 --p 0.25 --seed $s --mode 3 --release 600 --steps 2400 --outdir ../results/causal_intervention/sweep/rel600
done

# Aggregate statistics and regenerate Figure 4
python3 analysis/aggregate.py
python3 analysis/make_causal_fig.py
```
