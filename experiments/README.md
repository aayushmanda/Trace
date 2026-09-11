# Experiments

How to run the paper stack: **[../RUN.md](../RUN.md)**. Figure/table numbers below
match **PDF numbering** in `Paper/main.pdf` (see `Paper/EXPERIMENT_MAP.md`).

Preferred entry point is `python -m src <command>` (see `python -m src help`).
Files in this directory are either standalone scripts that only write figures
or reports (`analyze_induced.py`, `build_executor_results.py`,
`build_rule_credit_table.py`, `fit_credit_exponent.py`, `noise_threshold.py`,
`credit_suppression.py`, `recover_mechanism_deep.py`, `oracle_alignment.py`,
`gradient_transfer.py`, `summarize_alignment.py`, `plot_*.py`) or `_paths.py`,
a small `sys.path` shim they import. There is no per-command wrapper file
anymore: every `src.__main__` subcommand is run directly as
`python -m src <command>`.

```bash
python -m src help
uv run python experiments/<name>.py --help
```

---

## Paper tables and figures

| Command | Supports | Output |
|---|---|---|
| `python -m src supervision` | **Table 1** five-condition accuracy | stdout |
| `python -m src reliability` | **Figures 1–2**, **Table 3** | `results/reliability_sweeps/` |
| `python experiments/noise_threshold.py` | **Figure 3**, **Tables 5–6**, Corollary 3 identifiability (no training) | `results/noise_threshold/`, `Paper/figures/noise_threshold.pdf` |
| `python experiments/credit_suppression.py` | Thm 2 residuals, Thm 4 both halves, Cor 3 exponent (no training) | `results/credit_suppression/`, `Paper/figures/credit_suppression.pdf` |
| `python -m src executor` | **Figure 4** protocol (single run) | `results/executor_comparison/<run>/` |
| `python -m src executor-depths` | Five-seed \(D\in\{2,4,6\}\) | `results/executor_comparison/depth_replication/` |
| `python experiments/build_executor_results.py` | **Figures 4–5**, **Table 7** TeX | `Paper/figures/trained_executor_*.pdf` |
| `python experiments/recover_mechanism_deep.py` | Recovers **Tables 2–3** and escape times from crashed-run logs | `results/mechanism_deep/recovered.csv` |
| `python -m src architecture` | 10-seed 2×2 (not historical Table 6) | `results/architecture_controls_n10/` |

Completed §7 artifacts: `results/executor_comparison/depth_replication/`. Rebuild figures:

```bash
python experiments/build_executor_results.py \
  --input results/executor_comparison/depth_replication --paper Paper
```

---

## Resubmission bridge (GPT character-token; not Figure 4)

| Command | Role | Output |
|---|---|---|
| `python -m src induced` | Per-step \(\widehat P_g^{(t)}\), \(\widehat\varepsilon_{\mathrm{rule}}\) | `results/revision/induced_rule.csv` |
| `python -m src pullback` | Exact LM \(J^\top W\) vs \(-\nabla L\) | `results/revision/pullback.csv` |
| `python -m src split-verdict` | Depth table: \(D-1\), in-ball exponent, \(\delta_{\mathrm{comp}}\) | `results/revision/split_verdict.csv` |
| `python -m src escape` | Shared \(A\in\mathbb{R}^{M\times K\times K}\) **row-softmax** kernel | `results/revision/shared_kernel.csv` |
| `python -m src projected` | Projected simplex flow; **not a T2 proof** | `results/paper_revision_v2/e4_projected_kernel/` |
| `python -m src outcome-local` | Mechanism test: \(L_{\mathrm{out}}\) vs \(L_{\mathrm{out}}+\mathrm{local}\) vs process; nnsight probes/patches | `results/paper_revision_v2/causal_identification/` |
| `python experiments/analyze_induced.py` | Tables + `induced_rule.pdf` from archived CSVs | `Paper/figures/induced_rule.pdf` |
| `python -m src analyze` | `revision_*.pdf` | `Paper/figures/revision_*.pdf` |
| `python -m src length` | Train \(D=8\), eval 8/10/12/16 | `results/revision/length_generalization.csv` |
| `python -m src margins` | Gold-path \(m_{\min}\) at \(\rho=0.80\) | `results/revision/mmin_histograms.csv` |
| `run_revision_bridge.sh` | GPU 2/3 queue | Induced + length + \(m_{\min}\) + architecture plan | `results/revision/`, `logs/revision/` |
| `run_depth_sweep.sh` | `python -m src induced` grid | Exponent vs \(D-1\) | `results/induced_rule/depth.csv` |
| `run_condition_trajectory.sh` | `python -m src induced` | \(\widehat\varepsilon_{\mathrm{rule}}\) by supervision | `results/induced_rule/trajectory.csv` |
| `run_trace_fraction.sh` | `python -m src induced` | Trace-fraction vs mixing ball | `results/induced_rule/fraction.csv` |

`results/induced_rule/fast_smoke.csv` can still be folded into `analyze_induced.py`. Those exponent numbers are **not** Table 7.

---

## Constructive executors

In [`../handcoded/`](../handcoded): exact finite-parameter Transformers
(Theorem 2, Figure 3, Table 4) and the one-seed reachability notebook (Table 6).

| File | What it is |
|---|---|
| `handcoded/` package | Semantic-token constructions |
| `handcoded/handcoded_executors.ipynb` | Builds and checks the exact executors |
| `handcoded/two_model_reachability.ipynb` | Historical Table 6 (one seed) |

PDF **Table 6** is that one-seed notebook. `architecture_controls.py` is the planned 10-seed replacement (`results/architecture_controls/protocol.json` exists; do not overwrite Table 6).

Claim map vs PDF numbering: `Paper/EXPERIMENT_MAP.md`.
