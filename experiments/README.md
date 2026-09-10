# Experiments

How to run the paper stack: **[../RUN.md](../RUN.md)**. Figure/table numbers below
match **PDF numbering** in `Paper/main.pdf` (see `Paper/EXPERIMENT_MAP.md`).

Every entry point here is runnable from the repository root:

```bash
uv run python experiments/<name>.py --help
```

`_paths.py` puts the repository root, `src/` and this directory on `sys.path`, so
scripts also run from inside `experiments/`.

---

## Which experiment supports which claim

Grouped by the stage of the argument. **Status** distinguishes results the
manuscript reports from code that is present but whose numbers are not currently
cited.

### 1. The phenomenon: does a valid trace teach execution?

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `supervision_comparison.py` | The five matched conditions — outcome, answer-first, filler, process, corrupted — on state machines, register machines and Boolean circuits | **Table 1** held-out answer accuracy | stdout / CSV |
| `reliability_sweep.py` | Trains at trace reliability ρ over a grid of ρ, seeds and checkpoints; the shared machinery (`RatioDataset`, `build_model`, `evaluate`) other experiments import | **Figures 1–2**, **Table 3** | `results/reliability_sweeps/*_phase_*.csv` |
| `lora_transfer.py` | Rank-8 LoRA on SmolLM2-135M, same Boolean generator and serialization | The pretrained-model paragraph in §3 (no separate table number) | CSV |
| `copy_probe.py` | Removes the answer-bearing successor while keeping the final operation | Off this PDF as a numbered table | CSV |
| `trace_vs_step_corruption.py` | One Bernoulli per trace against one per step, at matched marginal validity | Off this PDF | `results/paper/*_aggregate.csv` |
| `trace_cleaning.py` | Removes the corrupted portion of a ρ₀ = .8 pool against removing the same count at random | Off this PDF | `calibration.csv`, `intervention.csv` |

### 2. Local competence against complete rollout

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `validate_claims.py` | Local positive-margin fraction against exact rollout; the `a_local^D` prediction and a union-bound lower bound; prefix survival across depth | Proposition 1 in the text; histograms not in this PDF | `results/claim_validation/` |
| `early_acquisition.py` | Trains each (ρ, seed) once to a long horizon and uses only an early window of clean successor-state NLL changes to predict later acquisition | Off this PDF | `early_progress.csv`, `prediction_summary.csv` |

### 3. What a corrupted trace teaches

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `competitor_support.py` | Matched competitor-support experiment, resumable, multi-seed | Off this PDF (PDF uses Prop. 1 + Fig 2) | `results/competitor_support/` |
| `loss_barrier.py` | Concentrates corrupted mass over `--m` incorrect successors (`m = 1, 3, 15` give ρ\* = 1/2, 1/4, 1/16) | Off this PDF | `results/local_loss_barrier/` |

### 4. Mechanism: where credit enters (theory fingerprints)

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `escape_time_law.py` | Population gradient flow on the K×M transition-kernel executor | Off this PDF as a figure | JSON/CSV |
| `mechanism_diagnostics.py` | Gradient-side diagnostics on trained models | Off this PDF | `results/mechanism/mechanism_{summary,steps}_*.csv` |

### 5. Paper §7 — handcoded semantic-token readout (Figs 4–5, Table 7)

This **is** the compiled paper’s trained-mechanism experiment: one-block
semantic-token Transformers (width 96), gold-prefix local tables, composition TV,
mixed-format gradient cosine. **Not** character-token GPT. **Not** the
oracle-shaped Table 4 / Table 6 nets.

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `compare_executor_rules.py` | Trains process / outcome / mixed semantic-token models; gold-prefix \(\widehat P_g\), composition TV, mixed actual vs composed-table gradient cosine | **Figure 4** protocol (single run) | `results/executor_comparison/<run>/` |
| `run_executor_depths.py` | Depths \(\{2,4,6\}\) × seeds \(\{42..46\}\), resumable | Five-seed replication | `results/executor_comparison/depth_replication/` |
| `build_executor_results.py` | Rejects incomplete coverage; writes manuscript PDFs and Table 7 TeX | **Figure 4**, **Figure 5**, **Table 7** | `Paper/figures/trained_executor_*.pdf`, `Paper/data/trained_executor_*.tex` |
| `test_executor_comparison.py` | Exact-rule recovery, composition, query isolation, split disjointness, finite-difference readout | That the readouts compute what they claim | `python -m unittest tests.test_executor_comparison` |

Completed artifacts: `results/executor_comparison/depth_replication/metrics.csv`
and per-seed `report.md`. Rebuild figures without retraining:

```bash
python experiments/build_executor_results.py \
  --input results/executor_comparison/depth_replication --paper Paper
```

### 6. Architecture × supervision (Table 6 is one seed; 10-seed not run)

PDF **Table 6** (`tab:2x2`) is the historical one-seed
`handcoded/two_model_reachability.ipynb` (not GPT; not Table 2 — Table 2 in this
PDF is attention scales \(C_\star\)). The manuscript asks for a multi-seed LR
grid; that replacement has **no numbers yet**.

| Script | What it does | Status |
|---|---|---|
| `architecture_controls.py` | Planned 10-seed LR-controlled 2×2 (`results/architecture_controls_n10/`) | `plan` only: `results/architecture_controls/protocol.json`. No calibration/confirm CSVs. **Do not overwrite Table 6.** |

### 7. Optional extra / revision-not-in-this-PDF (GPT character-token)

`app:regime-spec` on ordinary learned GPT. **Do not treat as Figure 4.**
`Paper/figures/induced_rule.pdf` and `revision_*.pdf` exist as optional assets
and are not included in the compiled `main.pdf`.

| Script | What it runs | Role | Output |
|---|---|---|---|
| `induced_rule.py` | Character-token \(\widehat P_g\), \(\widehat\varepsilon_{\mathrm{rule}}\), credit exponent | Extra | `results/revision/induced_rule.csv` (also older `results/induced_rule/*.csv`) |
| `pullback.py` | Parameter-space pullback vs \(\nabla_\theta L\) | Extra | `results/pullback.csv` |
| `analyze_induced.py` | Tables + `induced_rule.pdf` from archived CSVs | Extra figure | `Paper/figures/induced_rule.pdf` |
| `run_depth_sweep.sh` / `run_condition_trajectory.sh` / `run_trace_fraction.sh` | GPT depth / condition / trace-fraction grids | Extra | `results/induced_rule/*.csv` |
| `length_generalization.py` | Train \(D=8\), eval 10/12/16 | Extra (no PDF table) | `results/revision/length_generalization.csv` |
| `margin_histograms.py` | Gold-path \(m_{\min}\) at \(\rho=0.80\) | Extra (no PDF figure) | `results/revision/mmin_histograms.csv` |
| `run_revision_bridge.sh` | Queues the **GPT** extras + architecture n=10 | Not the §7 protocol | `results/revision/`, `logs/revision/` |
| `analyze_revision.py` | `python -m src analyze` | Extra `revision_*.pdf` | `Paper/figures/revision_*.pdf` |

`results/induced_rule/fast_smoke.csv` is an earlier exponent check that
`analyze_induced.py` can still fold in. Those exponent numbers are **not**
Table 7.

### 8. Constructive executors

In [`../handcoded/`](../handcoded): exact finite-parameter Transformers
(Theorem 2, Figure 3, Table 4) and the one-seed reachability notebook (Table 6).

| File | What it is |
|---|---|
| `handcoded/` package | Semantic-token constructions |
| `handcoded/handcoded_executors.ipynb` | Builds and checks the exact executors |
| `handcoded/two_model_reachability.ipynb` | Historical Table 6 (one seed) |
| `Paper/scripts/check_mathematics.py` | Algebra for Tables 2 / constructions |

---

## Notes

- `depth_sweep_commands.sh` is a commented record of GPT depth-sweep invocations.
- Several diagnostic scripts were recovered from git history; re-run before
  citing their archived CSVs as freshly reproduced.
- Claim map vs PDF numbering: `Paper/EXPERIMENT_MAP.md`.
