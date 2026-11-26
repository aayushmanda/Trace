# Reproducing *Silent Routing*

Python 3.12+ (via `uv`) and C++20 (`clang++` with Apple Accelerate on macOS, or `g++` on Linux) from the repository root.
`uv sync` creates `.venv` from `uv.lock`. Training shows tqdm bars; set `TRACE_TQDM=0` to silence them and pick a GPU
with `CUDA_VISIBLE_DEVICES`. Run outputs go to `results/` (gitignored).

---

## 1. C++20 Causal Baseline-Projection, Stateless Block-RMS & `(Step-ODE)` Suite (Figures 3–4, Tables 1–2a, 9–10)

Zero Python dependencies are required to build and run the unconstrained 27-block Transformer engine and the unmasked
float64 `(Step-ODE)` solver. The `Makefile` auto-detects macOS (Apple Accelerate BLAS) vs. Linux (`g++` + `cblas_shim.h`).

```bash
# Build all binaries and run the float64 finite-difference gradient check (max_rel_err ~ 3.9e-7).
make -C cpp all check

# Reproduce the 20-seed causal intervention and Stateless Block-RMS sweeps (n = 8, p = 0.25).
mkdir -p results/causal_intervention/sweep/main results/causal_intervention/sweep/rel600
for s in $(seq 42 61); do
  for m in 0 1 2 4 6 8 9; do
    ./cpp/run_full_plan --n 8 --p 0.25 --seed $s --mode $m --steps 2400 --outdir results/causal_intervention/sweep/main
  done
  ./cpp/run_full_plan --n 8 --p 0.25 --seed $s --mode 3 --release 600 --steps 2400 --outdir results/causal_intervention/sweep/rel600
done

# Aggregate solve rates, Wilson 95% CIs, and two-sided Fisher exact tests -> cpp/analysis/summary.json
uv run python cpp/analysis/aggregate.py

# Regenerate papers/silent_routing/figures/causal_intervention.tex from the sweep CSVs
uv run python cpp/analysis/make_causal_fig.py

# Run the unmasked float64 (Step-ODE) verification suite (Appendix G.3)
./cpp/step_ode_suite
```

See [`cpp/README.md`](cpp/README.md) for all 11 modes (`0..10`).

---

## 2. Theory & Expressivity Checks (`handcoded/`, Sections 2–3, Appendices B, C, G.1, G.3)

```bash
# Exact executors (Proposition 1 / Appendix B): independent rebuild + n = 1,000 exactness test.
uv run python handcoded/checks/executor_check.py
uv run python handcoded/lettertrace.py task=count word_len=1000 mod=2 handcoded_only=True train_size=1 test_size=532

# Outcome plateau exit times (Theorem 1, Table 5 / tab:exit-times): ball B_eps (eps=0.002) and radius r=0.1 (eps=0.02).
REGIME_ZERO_READOUT=1 REGIME_STOP=ball REGIME_STEP=1e-6 REGIME_SEED=0 uv run python handcoded/regime_both.py 0.002
REGIME_ZERO_READOUT=1 uv run python handcoded/regime_both.py 0.02

# Score-degree exponents & credit checks (Appendices C, G.1, G.3; prop:score-degree).
uv run python handcoded/checks/credit_checks.py rule
uv run python handcoded/checks/credit_checks.py interaction
uv run python handcoded/checks/credit_checks.py setsums 6 0.03 0.04 0.05 > results/theorem_regime/setsums.txt
uv run python handcoded/checks/credit_checks.py structured > results/theorem_regime/structured.txt
uv run python handcoded/checks/credit_checks.py query
```

---

## 3. Counting, Spectra & Untied Routing Experiments (`handcoded/`, Figure 1a, Tables 6–8, Appendix G)

```bash
# Figure 1a & Table 8: zero-readout n = 16, ten paired seeds -> results/paper/zero_readout_n16_8k/
uv run python handcoded/experiments/sweeps.py zero_readout

# Appendix G.1: gradient decay, step-to-outcome ratio, and exact Jacobian degree spectrum
uv run python handcoded/experiments/measure.py spectrum task=count word_len=8 seed=42
uv run python handcoded/experiments/measure.py exact
bash handcoded/experiments/plateau_spectra.sh

# Table 6 (tab:gap): 20-seed horizon sweep (n in {6..16}) -> results/basin/success*.json
uv run python handcoded/experiments/sweeps.py basin success

# Appendix G.1: 20-cell horizon-alphabet lambda grid (Spearman r_S = 0.91) -> results/paper/lambda_grid/
uv run python handcoded/experiments/sweeps.py lambda_grid --all

# Table 7 (tab:controls): initialization-scale sweep (eps in {0.1, 0.5}, n in {8, 16})
bash handcoded/experiments/init_scale_sweep.sh

# Appendix G.3: single-head untied routing (17/17 step vs. 0/17 outcome within 300k updates)
uv run python handcoded/experiments/measure.py reach --format step --sigma 0.3 --lr 2 --steps 300000
uv run python handcoded/experiments/measure.py reach --format outcome --sigma 0.3 --lr 2 --steps 300000

# Table 6 & Table 7: extended 32k sweeps, deeper models, and first-occurrence control via lettertrace.py
uv run python handcoded/lettertrace.py task=count word_len=16 mod=2 steps=32000
uv run python handcoded/lettertrace.py task=first word_len=24 steps=16000
uv run python handcoded/checks/log_claims.py
```

`lettertrace.py` accepts `key=value` overrides; all defaults are documented at the top of the file. See
[`handcoded/README.md`](handcoded/README.md) for the full paper-to-code map.

---

## 4. Automata & Pretrained LLM Transfer (`src/`, Table 2b, Table 8, Appendix G.2)

```bash
# Appendix G.2: exact ANOVA degree decompositions for Boolean-8 (98.9% deg >= 2) & register mod 17 (99.9% deg >= 2)
uv run python handcoded/experiments/measure.py answers

# Boolean-8 (A_16) and modulo-17 register machine on character-token GPT-2-2L.
uv run python -m src supervision --config configs/experiments/boolean8_gpt.yaml
uv run python -m src supervision --config configs/experiments/register16_gpt.yaml

# Rank-8 LoRA on Boolean-8: SmolLM2-135M, then Qwen2.5-0.5B.
uv run python -m src lora --config configs/experiments/lora_transfer.yaml
uv run python -m src lora --config configs/experiments/lora_transfer.yaml --model Qwen/Qwen2.5-0.5B
```

---

## 5. Paper & Explainer Animations

```bash
# Compile the ICML 2026 manuscript -> papers/silent_routing/main.pdf
cd papers/silent_routing && tectonic -X compile main.tex

# Re-render the 1080p60 Manim explainer animations (see animations/README.md)
cd animations && manim -pqh silent_routing_scenes.py Scene1_OutcomePlateau Scene2_SilentRouting Scene3_StatelessBlockRMS
```
