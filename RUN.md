# Reproducing *Silent Routing*

Python 3.12+ (`uv sync`) and C++20 (`clang++` with Apple Accelerate on macOS, or `g++` on Linux) from the repo root.
Set `TRACE_TQDM=0` to silence progress bars and `CUDA_VISIBLE_DEVICES` to pick a GPU. Outputs go to `results/` (gitignored).

## 1. C++20 Causal Interventions, Stateless Block-RMS & `(Step-ODE)` (Figures 3–4, Tables 1–2a, 9–10)

```bash
make -C cpp all check

mkdir -p results/causal_intervention/sweep/main results/causal_intervention/sweep/rel600
for s in $(seq 42 61); do
  for m in 0 1 2 4 6 8 9; do
    ./cpp/causal_interventions --n 8 --p 0.25 --seed $s --mode $m --steps 2400 --outdir results/causal_intervention/sweep/main
  done
  ./cpp/causal_interventions --n 8 --p 0.25 --seed $s --mode 3 --release 600 --steps 2400 --outdir results/causal_intervention/sweep/rel600
done

uv run python cpp/analysis/aggregate.py
uv run python cpp/analysis/plot_causal_figure.py
./cpp/step_ode
```

## 2. Expressivity, Plateau & Credit Checks (`two_block/`, Sections 2–3, Appendices B, C, G)

```bash
# Exact executors (Proposition 1) + n = 1,000 exactness test
uv run python two_block/checks/check_executors.py
uv run python two_block/counting.py task=count word_len=1000 mod=2 handcoded_only=True train_size=1 test_size=532

# Outcome plateau exit times (Theorem 1, Table 5)
uv run python two_block/population_gd.py 0.002 --zero-readout --stop ball --step-len 1e-6 --seed 0
uv run python two_block/population_gd.py 0.02 --zero-readout --seed 0

# Score-degree exponents & credit checks (Appendices C, G.1, G.3)
uv run python two_block/checks/check_credit.py rule
uv run python two_block/checks/check_credit.py interaction
uv run python two_block/checks/check_credit.py structured
uv run python two_block/checks/check_credit.py query
uv run python two_block/checks/check_credit.py setsums 6 0.03 0.04 0.05
```

## 3. Counting, Spectra & Untied Routing (`two_block/`, Figure 1a, Tables 6–8, Appendix G)

```bash
uv run python two_block/experiments/sweeps.py zero_readout         # Fig. 1a & Table 8
uv run python two_block/experiments/sweeps.py basin success        # Table 6 (20 seeds x 6 horizons)
uv run python two_block/experiments/sweeps.py lambda_grid --all    # App. G.1 (20-cell grid, r_S = 0.91)
bash two_block/experiments/run_init_scale_sweep.sh                 # Table 7 (init scale sweep)
bash two_block/experiments/run_plateau_spectra.sh                  # App. G.1 (plateau Jacobian split)

uv run python two_block/experiments/spectra.py spectrum task=count word_len=8 seed=42
uv run python two_block/experiments/spectra.py exact
uv run python two_block/experiments/spectra.py reach --format step --sigma 0.3 --lr 2 --steps 300000
uv run python two_block/experiments/spectra.py reach --format outcome --sigma 0.3 --lr 2 --steps 300000

uv run python two_block/counting.py task=count word_len=16 mod=2 steps=32000
uv run python two_block/counting.py task=first word_len=24 steps=16000
uv run python two_block/checks/check_logs.py
```

## 4. Automata & Pretrained LLM Transfer (`src/`, Table 2b, Table 8, Appendix G.2)

```bash
uv run python two_block/experiments/spectra.py answers
uv run python -m src supervision --config configs/experiments/boolean8_gpt.yaml
uv run python -m src supervision --config configs/experiments/register16_gpt.yaml
uv run python -m src lora --config configs/experiments/lora_transfer.yaml
uv run python -m src lora --config configs/experiments/lora_transfer.yaml --model Qwen/Qwen2.5-0.5B
```
