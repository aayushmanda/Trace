# `handcoded/` — Two-Block Transformer Experiments & Numerical Checks

PyTorch implementation of the two-block causal Transformer (`Assumption A.1`), exact direct and step-by-step executors
(**Proposition 1**), outcome plateau exit-time verification (**Theorem 1**), and counting/spectrum experiments for
*Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls* (ICML 2026).

> **Note:** The 27-block causal baseline-projection interventions (`Figure 3`, `Figure 4`, `Table 1`, `Table 2a`) and
> the unmasked full-prefix `(Step-ODE)` solver (`Table 9` / `Table 10`) live in [`../cpp/`](../cpp/README.md).
> Automata and pretrained LLM transfer (`Table 2b`, `Table 8`) live in [`../src/`](../src/).

## Paper-to-Code Map

| File | Paper Reference | What it does |
|---|---|---|
| [`lettertrace.py`](lettertrace.py) | Prop. 1, Fig. 1a, Tables 2b, 6–8 (App. B, F, G) | Two-block Transformer (`ass:arch`), exact executors $\theta^\star_{\mathrm{out}}, \theta^\star_{\mathrm{step}}$, and paired step/outcome training |
| [`regime_both.py`](regime_both.py) | Thm. 1, Table 5 (`tab:exit-times`, App. G.1) | Exact float64 population GD exit times from $\mathcal B_\varepsilon$ ($\varepsilon = 0.002$) and radius $r = 0.1$ ($\varepsilon = 0.02$) |
| [`experiments/sweeps.py`](experiments/sweeps.py) | Fig. 1a, Table 6 (`tab:gap`), Table 8 (`tab:main`), App. G.1 | `zero_readout` (10 paired seeds, $n=16$, $V(0)=0$), `basin success` (20 seeds $\times$ 6 horizons), and `lambda_grid` (20-cell $(n,|\Sigma|)$ grid, $r_{\mathrm S} = 0.91$) |
| [`experiments/measure.py`](experiments/measure.py) | App. G.1–G.3 | `spectrum` & `exact` (gradient decay & Jacobian degree split), `answers` (Boolean-8 / register ANOVA), and `reach` (single-head untied routing, $17/17$ step vs. $0/17$ outcome) |
| [`experiments/seeded_sweep.py`](experiments/seeded_sweep.py) & [`init_scale_sweep.sh`](experiments/init_scale_sweep.sh) | Table 7 (`tab:controls`, App. G.1) | Initialization-scale sweep ($\varepsilon \in \{0.1, 0.5\}$ at $n \in \{8, 16\}$) |
| [`experiments/plateau_spectra.sh`](experiments/plateau_spectra.sh) | App. G.1 | Outcome plateau checkpoints ($n = 8$, seed 42) and exact Jacobian degree split at 0k–8k updates |
| [`checks/executor_check.py`](checks/executor_check.py) | Prop. 1, App. B (`app:architectures`) | Rebuilds $\theta^\star_{\mathrm{out}}$ and $\theta^\star_{\mathrm{step}}$ independently from Appendix B and verifies entry-for-entry tensor equality |
| [`checks/credit_checks.py`](checks/credit_checks.py) | App. C, G.1, G.3 (`prop:score-degree`) | Checks score-degree $\varepsilon$-exponents ($2.0, 6.1, 7.0, 11.1, 12.0$), step rule credit ($\varepsilon^6 P(\tilde\theta)$, exponent $6.00$), and outcome interaction credit |
| [`checks/log_claims.py`](checks/log_claims.py) | Tables 6–7 (`tab:gap`, `tab:controls`, App. G.1) | Audits extended 32k horizon sweeps, deeper 4–6 block models, and first-occurrence controls from `lettertrace.py` logs |
| [`silent_routing_demo.ipynb`](silent_routing_demo.ipynb) | Prop. 1, Thm. 1, Fig. 1a | Clean interactive notebook walking through exact executors, float64 exit times, and paired training |

## Quick Commands

Run from the repository root (see [`../RUN.md`](../RUN.md) for the complete list):

```bash
# Exact executor verification (Proposition 1)
uv run python handcoded/checks/executor_check.py

# Score-degree and credit scaling checks (Appendices C & G.3)
uv run python handcoded/checks/credit_checks.py rule
uv run python handcoded/checks/credit_checks.py structured

# Smoke test: paired step vs. outcome training (n = 6, mod 2)
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000
```
