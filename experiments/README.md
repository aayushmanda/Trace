# Experiments

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
| `analyze_induced.py` | Builds the summary tables and the figure from the two CSVs above | — | `Paper/figures/induced_rule.pdf` |
| `run_depth_sweep.sh` | `induced_rule.py` at depths 3–6, three seeds | The exponent against `D − 1` | `results/induced_rule_depth.csv` |
| `run_condition_trajectory.sh` | Depth 4, conditions `both` / `outcome` / `process`, three seeds | `ε̂_rule` and `δ_comp` over training by supervision | `results/induced_rule_traj.csv` |
| `run_trace_fraction.sh` | Depth 4, trace fraction 0.02 → 1.00 | Where the model sits relative to the near-mixing ball as process supervision is withdrawn | `results/induced_rule_fraction.csv` |

### 6. Constructive executors

In [`../handcoded/`](../handcoded), not here: the explicit finite-parameter
Transformer executors and the randomized-copy reachability runs.

| File | What it is |
|---|---|
| `handcoded/handcoded_utils.py`, `handcoded/fastexec.py` | The constructions and a fast executor |
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
  `early_acquisition.py`, `loss_barrier.py` and `validate_claims.py` were
  recovered from git history (they had been deleted from the working tree while
  their outputs remained in `results/`). They import cleanly and their CLIs are
  intact, but they have **not** been re-run since recovery — re-run before
  citing any number from them as reproduced.
- The "Supports" column maps scripts to claims in words. Confirm the mapping to
  final table and figure numbers before release; the manuscript has been
  renumbered during revision.
