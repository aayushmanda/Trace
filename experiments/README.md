# Experiments

How to run the paper stack: **[../RUN.md](../RUN.md)**.

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
| `supervision_comparison.py` | The five matched conditions — outcome, answer-first, filler, process, corrupted — on state machines, register machines and Boolean circuits | The five-condition held-out answer accuracy table | stdout / CSV |
| `reliability_sweep.py` | Trains at trace reliability ρ over a grid of ρ, seeds and checkpoints; the shared machinery (`RatioDataset`, `build_model`, `evaluate`) other experiments import | The reliability sweeps and the local-versus-rollout gap | `results/reliability_sweeps/*_phase_*.csv` |
| `lora_transfer.py` | Rank-8 LoRA on SmolLM2-135M, same Boolean generator and serialization | The pretrained-model result: the gap survives after pretraining | CSV |
| `copy_probe.py` | Removes the answer-bearing successor while keeping the final operation | The input-copy baseline, i.e. that the modest outcome accuracy is not execution | CSV |
| `trace_vs_step_corruption.py` | One Bernoulli per trace against one per step, at matched marginal validity | That trace-level correlation alone does not reproduce the effect | `results/paper/*_aggregate.csv` |
| `trace_cleaning.py` | Removes the corrupted portion of a ρ₀ = .8 pool against removing the same count at random | The operational payoff of cleaning traces | `calibration.csv`, `intervention.csv` |

### 2. Local competence against complete rollout

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `validate_claims.py` | Local positive-margin fraction against exact rollout; the `a_local^D` prediction and a union-bound lower bound; prefix survival across depth | The greedy-rollout criterion and the dependence-free bounds | `results/claim_validation/` |
| `early_acquisition.py` | Trains each (ρ, seed) once to a long horizon and uses only an early window of clean successor-state NLL changes to predict later acquisition | Early prediction of which runs become executable | `early_progress.csv`, `prediction_summary.csv` |

### 3. What a corrupted trace teaches

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `competitor_support.py` | Matched competitor-support experiment, resumable, multi-seed | The reliability at which the attracting local margin crosses zero | `results/competitor_support/` |
| `loss_barrier.py` | Concentrates corrupted mass over `--m` incorrect successors (`m = 1, 3, 15` give ρ\* = 1/2, 1/4, 1/16) | That moving ν_max moves the measured crossing in the predicted place | `results/local_loss_barrier/` |

### 4. Mechanism: where credit enters

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `escape_time_law.py` | Population gradient flow on the K×M transition-kernel executor; escape time against ε at several depths. CPU-only, deliberately small (K=8, M=4) so the population stays cheap | The depth-graded trapping law | JSON/CSV |
| `mechanism_diagnostics.py` | Gradient-side diagnostics on trained models: alignment, projected gradient statistics, clean/corrupt cosine, teacher-forced successor NLL | The gradient-side measurements in the mechanism tables | `results/mechanism/mechanism_{summary,steps}_*.csv` |

### 5. Mechanism inside a trained Transformer (new)

These read the mechanism's own quantities out of a trained network. See
`docs/INDUCED_RULE.md` for the definitions.

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `induced_rule.py` | Reads `P̂_g[s,s'] = P_θ(s' \| s, g)` at trace-format decision positions over all `K·M = 832` pairs; splits `P̂_g − U = 1ĉ_gᵀ + F̂_g`; reports `ε̂_rule`, `γ̂`, `δ_comp`, conditional credit, rule recovery; fits the credit exponent by rescaling the model's own rule directions | The induced-rule readout and the regime check | `results/induced_rule/*.csv` |
| `pullback.py` | Corollary "parameter-space pullback": `Σ_g J_gᵀ vec(W_g)` is the parameter gradient of `Σ_g ⟨W_g, P̂_g(θ)⟩`, so one backward pass gives it. Compares against the true `∇_θ L_out` and `∇_θ L_proc`, with a norm-matched random control | Whether the outcome gradient actually routes through the induced rules | `results/pullback.csv` |
| `analyze_induced.py` | Builds the summary tables and the figure from the CSVs below (no CLI — fixed paths) | — | `Paper/figures/induced_rule.pdf` |
| `run_depth_sweep.sh` | `induced_rule.py` at depths 3–6, three seeds | The exponent against `D − 1` | `results/induced_rule/depth.csv` |
| `length_generalization.py` | Train outcome/process at circuit depth 8; greedy-eval at 8/10/12/16 with no finetuning | Composition / length extrapolation | `results/revision/length_generalization.csv` |
| `margin_histograms.py` | Gold-path \(m_{\min}\) at \(\rho=0.80\) for outcome vs process | Proposition 1: abrupt flips are bottleneck-margin crossings | `results/revision/mmin_histograms.csv` |
| `run_revision_bridge.sh` | Induced-rule probes at \(D\in\{2,4,6,8\}\) with pullback cosines; length; \(m_{\min}\); 10-seed architecture LR grid | The theory–Transformer bridge for resubmission | `results/revision/`, `results/architecture_controls_n10/` |
| `analyze_revision.py` | Figures from the revision CSVs | Paper figures | `Paper/figures/revision_*.pdf` |
| `run_condition_trajectory.sh` | Depth 4, conditions `both` / `outcome` / `process`, three seeds | `ε̂_rule` and `δ_comp` over training by supervision | `results/induced_rule/trajectory.csv` |
| `run_trace_fraction.sh` | Depth 4, trace fraction 0.02 → 1.00 | Where the model sits relative to the near-mixing ball as process supervision is withdrawn | `results/induced_rule/fraction.csv` |

`results/induced_rule/fast_smoke.csv` is an earlier, smaller exponent check
superseded by `run_trace_fraction.sh`; kept because `analyze_induced.py` folds
it into the exponent table's seed count. The published exponent
(1.920 ± 0.020 at D=3, 2.982 ± 0.007 at D=4, refit inside the theorem's
hypothesis region ε ≤ 1/2) is reproduced by running `analyze_induced.py` against
these four files as committed.

### 5b. A second, independent implementation (semantic-token executors)

`compare_executor_rules.py` re-derives the same comparison — trained-versus-exact
local rules, composition error, and gradient agreement — on a different
architecture: the semantic-token, one-layer models in the `handcoded/` package
rather than the character-token GPT `induced_rule.py`/`pullback.py` use. It is not
a dependency of the character-token results above; treat it as a second, smaller
data point built on different infrastructure.

| Script | What it runs | Supports | Output |
|---|---|---|---|
| `compare_executor_rules.py` | Trains outcome / process / mixed-format semantic-token models; reads gold-prefix local rule tables, composition TV, and (for the mixed-format model) the cosine between the actual outcome gradient and the gradient through composed extracted tables, against a true-rule direction and eight norm-matched random-rule controls | A second, architecture-independent check of local-rule recovery and gradient (non-)agreement | `results/executor_comparison/<run>/` (figures, `metrics.csv`, `report.md`; checkpoints are gitignored) |
| `test_executor_comparison.py` | Six unit tests: exact-rule recovery at every gate/state/position, composition matches direct execution, outcome/process query isolation, split disjointness, and a finite-difference check of the readout gradient | That the readouts in `compare_executor_rules.py` are computing what they claim to | — (run with `python -m unittest test_executor_comparison`, from inside `experiments/`; all 6 pass) |
| `notebooks/executor_comparison.ipynb` | Reviews the saved `pilot_seed42` comparison (figures + metrics) without retraining | A readable walkthrough of the pilot run above | — |

See [`../docs/executor_comparison.md`](../docs/executor_comparison.md) for the
figures this produces and the scope caveats already written into `report.md`.
The pilot run (`results/executor_comparison/pilot_seed42/report.md`) measured a
mixed-format full-gradient cosine of **0.079** against the true-rule direction —
independent, architecture-different corroboration of the same conclusion
`pullback.py` reaches on the character-token model: the outcome gradient does
not route through the induced rules the way the shared-executor theorems would
need it to.

### 5c. Required controls, in progress

Two scripts, uncommitted and **running as this file was written** — not part of
the map above, and not yet backing any number in the paper.

| Script | What it does | Status |
|---|---|---|
| `architecture_controls.py` | The multi-seed, learning-rate-controlled architecture × supervision comparison the manuscript's `app:controls` section calls for: same rate grid `[2e-3, 1e-3, 5e-4, 2e-4, 1e-4]`, depth 4, a single LR selected per architecture from pooled calibration (clipped, validation-only, never test), five confirmation seeds, both clipped and unclipped at the selected rate. Actions: `plan` / `calibrate` / `confirm` / `single`. Built on the semantic-token stack (`compare_executor_rules.py`'s hand-coded constructions), not the character-token GPT the paper's current Table 2 uses — treat this as the same *protocol* applied to the newer architecture family, not a rerun of that exact table. | `plan` has run (`results/architecture_controls/protocol.json` exists); calibration has not started. |
| `run_executor_depths.py` | Multi-seed depth replication of `compare_executor_rules.py`: depths `{2, 4, 6}` × seeds `{42..46}`, three parallel workers, resumable (skips a `depth{D}_seed{S}` directory once its manifest records `elapsed_seconds`) | **Running.** Started as this reorganization was underway; writing to `results/executor_comparison/depth_replication/`. |

Re-check `results/architecture_controls/` and
`results/executor_comparison/depth_replication/` once these finish, and fold
their results into section 5b / 5 above.

### 6. Constructive executors

In [`../handcoded/`](../handcoded), not here: the explicit finite-parameter
Transformer executors and the randomized-copy reachability runs.

| File | What it is |
|---|---|
| `handcoded/` package, `handcoded/fastexec.py` | Semantic-token constructions and a fast executor |
| `handcoded/handcoded_executors.ipynb` | Builds and checks the exact executors |
| `handcoded/two_model_reachability.ipynb` | Randomized copies of both oracles, trained under both objectives |
| `results/handcoded_reachability/seeds/` | Per-seed outputs of the above |
| `experiments/notebooks/verify_experiments1-5.ipynb` | Numerical checks of the stated identities |

---

## Notes

- `depth_sweep_commands.sh` is a record of the depth-sweep invocations, kept as
  documentation; it is commented out on purpose.
- `mechanism_diagnostics.py`, `escape_time_law.py`, `competitor_support.py`,
  `trace_cleaning.py`, `trace_vs_step_corruption.py`, `copy_probe.py`,
  `early_acquisition.py`, `loss_barrier.py` and `validate_claims.py` were at one
  point recovered from git history after having been deleted from the working
  tree while their outputs remained in `results/`; they carry the `_paths`
  bootstrap and import cleanly, and all 16 entry points in this directory
  import without error. Their archived CSVs in `results/` predate that
  recovery, though, so re-run before citing a number from them as freshly
  reproduced rather than archival.
- The "Supports" column maps scripts to claims in words. Confirm the mapping to
final table and figure numbers before release; the manuscript has been
renumbered during revision.

The resubmission bridge (induced-rule fingerprints in the trained Transformer,
multi-seed architecture controls, length generalization, and \(m_{\min}\)
histograms) is documented in [`../Paper/revision_bridge.md`](../Paper/revision_bridge.md)
and launched by `run_revision_bridge.sh`.
