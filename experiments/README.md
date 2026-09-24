# Experiments

How to run the paper stack: **[../RUN.md](../RUN.md)**.

This folder has one CLI. Implementations live under `src/experiments/`, grouped
into subfolders that match the paper's own sections:

| Folder | Paper section |
|---|---|
| `reliability/` | §2 canonical reliability sweep, App. I thresholds |
| `credit_geometry/` | §3 credit factorization/suppression, Prop. 8 frontier |
| `alignment/` | §5.2 gradient-alignment diagnostic |
| `identifiability/` | App. B identifiability |
| `circuit_match/` | §3.3 zero-init optimization-landscape test; §5.3 circuit-match gradient polarity under trace corruption |
| `plotting/` | figure builders (cross-cutting) |

Training entry points that already have `python -m src <command>`
(`supervision`, `reliability`, `lora`) stay in `src/eval/`; this CLI is for
figure builders and closed-form/numerical checks that need no training.

```bash
python experiments/run.py --help
python experiments/run.py plot-paper-figures
```

## Commands and what they support

| Command | Supports | Output |
|---|---|---|
| `python -m src supervision` | **Table 1** five-condition accuracy | `results/paper/e1_five_condition_rerun/` |
| `python -m src reliability` | **Figures 1–2**, **Table 3**, depth × reliability | `results/paper/boolean_reliability_canonical.csv`, `results/paper/depth_reliability/` |
| `python -m src lora` | Pretrained LoRA adaptation study (§3) | `results/paper/smollm/` |
| `python experiments/run.py tabular-sampling` | App. I: 1/K vs 1/2 under uniform vs family gate sampling | `results/` (see script) |
| `python experiments/run.py noise-threshold` | **Figure 3**, Tables 5–6, App. I reliability threshold | `results/noise_threshold/`, `Paper/figures/noise_threshold.pdf` |
| `python experiments/run.py family-sampling` | App. I frontier under family-first gate sampling | `results/` (see script) |
| `python experiments/run.py clean-convergence` | Corollary 5: GD trajectory converging to \(T_g\) under clean process supervision | `results/clean_convergence/` |
| `python experiments/run.py credit-suppression` | Credit identities (forward/backward decomposition, near-mixing bound, complete-mixing zero), confirmed numerically without training | `results/credit_suppression/` (also writes a `Paper/figures/credit_suppression.pdf` diagnostic plot, not currently cited by the paper) |
| `python experiments/run.py prop8-frontier` | Proposition 8: measured vs predicted reliability frontier \(\rho_c(a,D,\beta)\) | `results/prop8_frontier/`, `Paper/figures/prop8_frontier.pdf` |
| `python experiments/run.py fraction-vs-amount` | Fixed-\(\rho\), varying-\(N\) grid separating fraction-correct from corpus size | `results/fraction_vs_amount/`, `Paper/figures/fraction_vs_amount.pdf` |
| `uv run python -m src.experiments.plotting.plot_validated_paper` | Exact-corpus and concentration main figures; functional-update appendix artifacts | `Paper/figures/`, `Paper/tables/` |
| `uv run python -m src.experiments.temporal_noise.report` | Current third main figure: temporal corruption; report, appendix table and provenance for 30 new GPT models | `Paper/figures/temporal_main.pdf`, `results/novelty_extension/` |
| `uv run python -m src.experiments.plotting.plot_main_paper` | Historical and supplementary figures; constructed-parameter projection interpretation is retired | `Paper/figures/` |
| `python experiments/run.py plot-paper-figures` | Rebuilds `figures/architectures.pdf` | `Paper/figures/` |

None of the `experiments/run.py` commands train a model; each either evaluates a closed-form expression directly or reads an already-archived CSV.

Several appendix results have their own standalone scripts, run directly rather
than through this CLI (each has a `main()`/`--help`):

| Script | Supports |
|---|---|
| `python -m src.experiments.credit_geometry.checkpoint_competence_trajectory` | \(\rho_c(a,D,\beta)\) frontier vs. a real trained checkpoint |
| `python -m src.experiments.credit_geometry.credit_transmission_probe` | Theorem 1's forward × backward factorization, trained-network validation |
| `python -m src.experiments.alignment.gradient_alignment` | App. F gradient-alignment diagnostic (Figure: gradient_alignment_compact) |
| `python -m src.experiments.alignment.gradient_alignment_by_group` | Per-parameter-group breakdown of the same diagnostic |
| `python -m src.experiments.identifiability.identifiability_certificate` | Computational certificate for the fixed-depth identifiability theorem |
| `python -m src.experiments.identifiability.identifiability_probe` | Trained-network test of group-symmetry non-identifiability |
| `python -m src.experiments.credit_geometry.higher_order_credit_check` | Symbolic/autodiff checks for the higher-order credit results |
| `python -m src.experiments.circuit_match.handcoded_circuit_match` | §5.3, App. C: rule-table vs. routing gradient polarity on a matched wrong trace (`--corruption symmetric\|coherent`) |
| `python -m src.experiments.circuit_match.zero_init_softmax_table` | §3.3, App. C: process escapes `Z=0` under real AdamW, outcome does not (`--init-std` 0 for exact zero, >0 for random) |
| `python -m src.experiments.reliability.row_coverage` | §4.1: counts displayed state--operation pairs in the canonical corpora, giving $N_{\min}$ and the sample slack $\beta_N$; no training |
| `python -m src.experiments.circuit_match.gradient_polarity_threshold` | §5.3, App. C: post-hoc read of saved `grad_match.csv` giving the gradient-polarity threshold $\rho_{\rm grad}$ per parameter group; no training |

Promoted 2026-09-22 from `future_work/paper2/` (see below) into this paper's
own pipeline, since it directly answers whether the credit-geometry
mechanism shows up in a real optimizer's own parameter gradients, not only
the transition-tensor model. See `results/paper/ARTIFACT_MANIFEST.md`'s
"Circuit-match gradient polarity" entry for the exact commands, archived
CSVs, and the honest caveat about parameter-level (non-)identifiability.

Remaining precursor work toward a *different, future* paper (the empirical
escape-times experiment) lives outside this pipeline entirely, in
[`../future_work/`](../future_work/).

## Constructive executors

[`../handcoded/`](../handcoded): exact finite-parameter Transformers used by the tutorial notebook and to sanity-check the realizability construction in Appendix A (`Paper/appendix.tex`).

| File | What it is |
|---|---|
| `handcoded/` package | Semantic-token constructions (gates, tokenizer, models, train, eval, animate) |
| `handcoded/handcoded_executors.ipynb` | Builds and checks the exact executors; also the illustrated tutorial linked from the top-level README |
