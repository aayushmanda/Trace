# Artifact manifest

SHA-256 of every archived file a paper table or figure is built from, the
command that produced it, and the paper location it backs. Regenerate a hash
with `sha256sum <path>`; a mismatch means the file changed since this
manifest was written and the paper text should be re-checked against it.
Recorded 2026-09-12.

## Table 1 (`tab:supervision-comparison`, `body.tex`)

Command: `python -m src supervision --config configs/experiments/e1_five_condition_rerun.yaml`
(GPU driver log: `results/paper/gpu2_progress.log`, "finished Table 1 rerun
Sat Sep 12 11:31:37 IST 2026"). Seeds 2001-2005, 5 modes x 5 seeds = 25 rows.

| file | sha256 |
|---|---|
| `configs/experiments/e1_five_condition_rerun.yaml` | `382ba214a925031d03033c19ab5199e42162f54782cec3c83f3e0cafff4a9b10` |
| `results/paper/e1_five_condition_rerun/table1_rerun.csv` | `3c4057b113504f9bf17b0c508193c2c2ba628362e57440128f16a23bda4894cd` |
| `results/paper/e1_five_condition_rerun/table1_rerun_persist.json` | `89911617e5ac6fa5453108f1a0644e3a85da168cd426ee901ef142930a7bfe2f` |

Per-mode mean/std (%, n=5), matching `results/paper/e1_five_condition_rerun/table1_rerun_persist.json`'s
own `summary` field and the values in `body.tex`: outcome 7.72±0.34,
answer_first 7.96±0.88, filler 7.76±0.55, process 82.00±7.69, corrupted
6.12±1.06.

## Table 2 / Figure 1 (`tab:legacy-circuit`, `fig:legacy-reliability`, `body.tex`)

Source: `results/reliability_sweeps/boolean_circuit_8_phase_20260823_151153.csv`
(sha256 `d76afaac2a1c59f5d7a1e51f4545fffded57411279f4261509ae46a7e891bb42`),
41 rows, 5 seeds (2001, 2002, 2020, 3000, 2026) x 8 conditions
(outcome-only + rho in {0.30, 0.50, 0.80, 0.85, 0.90, 0.95, 1.00}), 8000
steps. Rebuilt into `figures/boolean_reliability.pdf` by
`python experiments/run.py plot-paper-figures` (`src/experiments/plot_paper_figures.py`,
which reads this exact path and no other). Per-condition mean/std recomputed
directly from this CSV and checked to match the printed table and the
`sec:sharp-jump` prose (8.39, 18.02, 61.98, 92.61 at outcome-only/0.50/0.80/1.00)
to two decimal places.

## LoRA adaptation (`sec:local-rollout`, `app:legacy-protocol`)

Source: `results/paper/smollm/smollm_trace.csv` (sha256
`1dd75b06f451f65d1c2d21409642ea3b2ab2071df0e32b7df08a1967e776a81c`), config
`configs/experiments/lora_transfer.yaml` (SmolLM2-135M, rank-8 LoRA, seeds
2001-2003, rho in {0,0.2,0.4,0.5,0.6,0.8,1.0}, 27 rows). Raw per-seed
generations, adapters, and local-margin CSVs archived under
`results/paper/smollm/artifacts/{generations,adapters,local_margins}/`.

## Closure checks (App. sections on credit identities, the reliability-frontier diagnostic, noise threshold, fraction-vs-amount)

These are closed-form/tabular numerical checks (no Transformer training);
each `summary.json` is written by the run listed and is what the
corresponding appendix figure/paragraph quotes.

| paper location | command | summary file | sha256 |
|---|---|---|---|
| App. (credit identities, `fig:credit-suppression`) | `python experiments/run.py credit-suppression` | `results/credit_suppression/summary.json` | `3f42aa330473ff4c3e0d0c6519395aa78de79cd2821e5fab1a894324e2a60b04` |
| Reliability-frontier diagnostic, App. B.7 (`fig:prop8-frontier`) | `python experiments/run.py prop8-frontier` | `results/prop8_frontier/summary.json` | `550ba80fee8493086510046a37c70f25f7b08f6254a946f8fb3b5eba863e9847` |
| App. I (noise threshold, `tab:noise-recovery`) | `python experiments/run.py noise-threshold` | `results/noise_threshold/summary.json` | `8cedae95a3d51bdd9e53bcd192fb4c212cf444ae6d2eba7dec26159249215d09` |
| App. I (gate-sampling check) | `python experiments/run.py family-sampling` | `results/noise_threshold/sampling_check.json` | `b3f07d3c19a8c648f7eb32770b4a7c369031606a79bc66b74920f707505a4590` |
| App. (fraction vs. amount) | `python experiments/run.py fraction-vs-amount` | `results/fraction_vs_amount/summary.json` | `a34b77503abe27a4d901cdd147b6e076b771bec693674dddc99f60b1517e30a4` |
| Cor. 5 (clean convergence) | `python experiments/run.py clean-convergence` | `results/clean_convergence/summary.json` | `778bac5d725174f09083e400aa1444c5aa770d59b2f50aa8d005ae5ee570e386` |

Note: `prop8_frontier/summary.json` was regenerated 2026-09-12 to pick up a
color-palette fix in `src/experiments/prop8_frontier.py` (a red/green pair in
the 5-way `COLORS` dict was not colorblind-distinguishable; `D=6`'s color
changed from `#2f7d32` to `#6b4c9a`). The regenerated file differs from the
prior hash only in ~10th-significant-digit floating-point noise in the
diagnostic `Jprime_at_rho0` field (bisection-solver jitter); every
`predicted_rho_c`, `measured_rho_c`, `bracketed`, and `abs_error` value is
byte-identical, and all numbers quoted in the paper are unaffected. The same
fix was applied to `fraction_vs_amount.py`'s `COLORS` dict (`rho=0.20`:
`#2f7d32` to `#6b4c9a`); that summary.json's hash is unchanged since its
random draws are seeded.

## Gradient-alignment diagnostic (`tab:gradient-alignment`, `sec:gradient-alignment`, `app:gradient-alignment`)

Command: `python -m src.experiments.gradient_alignment --task boolean_circuit_8
--seeds 2001 2002 2003 --steps 8000 --train-size 20000 --ref-size 256`. One
process-mode training run per seed (same architecture/optimizer as Table 1),
checkpoints at steps {0, 2000, 4000, 8000}; no model checkpoints saved to
disk (computed live via `train_with_checkpoints`'s callback). No new
hyperparameter search.

| file | sha256 |
|---|---|
| `results/gradient_alignment/gradient_alignment.csv` | `60be9bdf49d3272fb0526b43322fecca0b9b689cd8ecd20369e792f59ee7e56b` |
| `results/gradient_alignment/gradient_alignment_persist.json` | `4729cef2178b8eaf75ab86ac87d6a1190c09cb4e92ba49acdb7f42e24b0dbe7f` |

## Not in this manifest

`results/paper/depth_reliability/` (D=2/4/8 reliability sweeps) is background
data from a launched-but-descoped depth x reliability study; it backs no
table or figure in the current paper and should stay that way.
