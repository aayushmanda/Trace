# Reproducing *Silent Routing*

Python 3.12+ (via `uv`) and C++20 (`clang++` with Apple Accelerate on macOS, or `g++` on Linux) from the repository root.
`uv sync` creates `.venv` from `uv.lock`. Training shows tqdm bars; set `TRACE_TQDM=0` to silence them and pick a GPU
with `CUDA_VISIBLE_DEVICES`. Run outputs go to `results/` (gitignored).

---

## 1. C++20 Causal Baseline-Projection & Stateless Block-RMS Suite (Figures 3–4, Tables 1–2a)

Zero Python dependencies are required to build and run the unconstrained 27-block Transformer engine. The `Makefile`
auto-detects macOS (Apple Accelerate BLAS) vs. Linux (`g++` + `cblas_shim.h`).

```bash
# Build all binaries and run the float64 finite-difference gradient check (max_rel_err ~ 2.4e-7).
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
```

See [`cpp/README.md`](cpp/README.md) for all 11 modes (`0..10`) and the float64 `(Step-ODE)` solver (`step_ode_suite`).

---

## 2. Theory & ODE Checks (Sections 3–4, Appendices C–G)

```bash
# Gate flow on the full two-block network vs. the reduced three-scalar ODE (27 runs, rel. error ~2e-9).
uv run python handcoded/research/gate_network/run.py            # -> handcoded/research/gate_network/runs.json
uv run python handcoded/reduced_ode.py --K 2                    # -> results/reduced_ode/runs.json
uv run python handcoded/checks/network_vs_reduced.py            # compares the two, prints max relative error
uv run python handcoded/checks/full_readout_check.py            # unconstrained 2x2 U and W check
uv run python handcoded/checks/ode_table.py                     # ODE scaling table rows

# Outcome plateau exit time from the initialization ball; regenerate papers/silent_routing/figures/scaling.tex.
for p in 0.15 0.2 0.25; do for s in 0 1 2; do uv run python handcoded/research/lambda_collapse/run.py $p $s; done; done
uv run python handcoded/research/lambda_collapse/make_figure.py

# Exit times: ball exit and tau(0.1), zero readout -> results/theorem_regime/
REGIME_ZERO_READOUT=1 REGIME_STOP=ball REGIME_STEP=1e-6 REGIME_SEED=0 uv run python handcoded/regime_both.py 0.002
REGIME_ZERO_READOUT=1 uv run python handcoded/regime_both.py 0.002

# Credit measurements: rule credit, interaction credit, degree exponents, leading coefficient.
uv run python handcoded/checks/credit_checks.py rule
uv run python handcoded/checks/credit_checks.py interaction
uv run python handcoded/checks/credit_checks.py setsums 6 0.03 0.04 0.05 > results/theorem_regime/setsums.txt
uv run python handcoded/checks/credit_checks.py structured > results/theorem_regime/structured.txt
uv run python handcoded/checks/credit_checks.py query

# High-precision constants, derivative identities, softmax model vs. reduced flow.
uv run python handcoded/checks/theorem4_checks.py

# Unbiased parity (p = 1/2): 24 reduced-flow configurations -> results/reduced_ode/phalf_check.json
uv run python handcoded/checks/phalf_check.py
uv run python handcoded/reduced_ode.py --K 2 --p 0.5 --out results/reduced_ode/p_half.json

# Direct-proof numerical checks and exact executors (Proposition 1) at n = 1000.
uv run python handcoded/checks/direct_proofs/check_direct_proofs.py
uv run python handcoded/lettertrace.py task=count word_len=1000 mod=2 handcoded_only=True train_size=1 test_size=532
uv run python handcoded/checks/executor_check.py
```

---

## 3. Counting Experiments (Section 5, Appendix F–G)

```bash
# Figure 1a: zero-readout n = 16, ten paired seeds -> results/paper/zero_readout_n16_8k/
uv run python handcoded/experiments/sweeps.py zero_readout

# Degree spectrum and answer-dependent gradient -> results/paper/fig2/, results/paper/exact_spectra/
uv run python handcoded/experiments/measure.py spectrum task=count word_len=8 seed=42
uv run python handcoded/experiments/measure.py exact

# 20-seed horizon sweep -> results/basin/success*.json
uv run python handcoded/experiments/sweeps.py basin success

# Horizon-alphabet lambda grid -> results/paper/lambda_grid/
uv run python handcoded/experiments/sweeps.py lambda_grid --all

# Check / rewrite inline PGFPlots coordinates in papers/silent_routing/figures/
uv run python handcoded/make_inline_figures.py

# Outcome plateau checkpoints & exact Jacobian degree split -> results/paper/exact_spectra/plateau_s42/
bash handcoded/experiments/plateau_spectra.sh

# Initialization-scale sweep -> results/seeded/rnd_n{8,16}_K2_e{0.1,0.5}_s*.json
bash handcoded/experiments/init_scale_sweep.sh

# Learned routing, step vs. outcome -> results/step_reach/, results/step_reach_outcome/
uv run python handcoded/experiments/measure.py reach --format step --sigma 0.3 --lr 2 --steps 300000
uv run python handcoded/experiments/measure.py reach --format outcome --sigma 0.3 --lr 2 --steps 300000

# Unbiased p = 1/2 and two-letter alphabets -> results/p_half/
bash results/p_half/run_job.sh NAME key=value ...    # one line per job in results/p_half/jobs*.txt
uv run python results/p_half/summarize.py

# Direct lettertrace.py runs (counting mod 2/3, first-occurrence control)
uv run python handcoded/lettertrace.py task=count word_len=16 mod=2 steps=8000
uv run python handcoded/lettertrace.py task=count word_len=8 mod=3 steps=32000
uv run python handcoded/lettertrace.py task=first word_len=12 steps=8000
```

`lettertrace.py` accepts `key=value` overrides; all defaults are documented at the top of the file.

---

## 4. Automata, Controls, and Pretrained LLM Transfer (Table 2b, Appendix G.2)

```bash
# Boolean-8 (A_16) and modulo-17 register machine on character-token GPT-2-2L.
uv run python -m src supervision --config configs/experiments/boolean8_gpt.yaml
uv run python -m src supervision --config configs/experiments/register16_gpt.yaml

# Rank-8 LoRA on Boolean-8: SmolLM2-135M, then Qwen2.5-0.5B.
uv run python -m src lora --config configs/experiments/lora_transfer.yaml
uv run python -m src lora --config configs/experiments/lora_transfer.yaml --model Qwen/Qwen2.5-0.5B

# BIG-Bench Hard tracking shuffled objects, Qwen2.5-0.5B LoRA (K = 3, 5, 7; seeds 2001-2003).
uv run python -m src shuffled zeroshot --out results/shuffled/bbh/zeroshot_qwen05.json
for K in 3 5 7; do for c in outcome process; do for s in 2001 2002 2003; do
  uv run python -m src shuffled train --K $K --condition $c --seed $s --out results/shuffled/bbh/bbh_K${K}_${c}_s${s}.json
done; done; done

# BBH LoRA Jacobian degree spectrum & transfer figure (papers/silent_routing/figures/main_transfer.pdf).
uv run python handcoded/bbh_lora_spectrum.py K=5 mode=mc
uv run python handcoded/main_figures.py
```

`data/bbh/` holds the official BIG-Bench Hard items, unchanged; training items are generated from their scenes and item
pools while the official items are held out for testing.

---

## 5. Paper & Explainer Animations

```bash
# Compile the ICML 2026 manuscript -> papers/silent_routing/main.pdf
cd papers/silent_routing && tectonic -X compile main.tex

# Re-render the 1080p60 Manim explainer animations (see animations/README.md)
cd animations && manim -pqh silent_routing_scenes.py Scene1_OutcomePlateau Scene2_SilentRouting Scene3_StatelessBlockRMS
```
