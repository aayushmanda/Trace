# Closure TODO

What's measured, what isn't, and the exact command for each. "Closure" = a
theorem or table tested against data rather than asserted. Ordered by
return per unit effort.

## Done, in the paper

| # | Script / command | Buys | Where |
|---|---|---|---|
| 1 | `python experiments/noise_threshold.py` | Cor. "reliability threshold" (ρ_c=1/K, 1/2), ρ=0 information-completeness | §3.2, App. I (Fig. 3, Tables 5–6) |
| 2 | `python experiments/recover_mechanism_deep.py` | Table 2 provenance; escape-time table (process flat in depth, outcome censors from D=6) | §5 Table 3, App. K |
| 3 | `python experiments/credit_suppression.py` | **Thm 4 both halves** (outcome exec.-relevant part = 0.000, process = 15.0000 exactly); **Thm 2** (‖Πq‖, ‖Πb‖ trade off 3 orders of magnitude, product constant at 2.457e-2); **Cor. 2** (fitted ε-exponent = D−1 exactly, R²≥0.998, D∈{2..12}) | App. G "Numerical Confirmation of the Credit Identities" (Fig., cross-ref'd from Thm 2/4 and Cor. 2 in body.tex) |

Rerun any of these any time — all three write from archived logs/CSVs or a
closed-form no-training computation.

## Cheap reruns (minutes–hours, no GPU needed for most)

| # | Command | Buys | Note |
|---|---|---|---|
| 4 | *(nothing to run)* | ~~Cor. 2 exponent~~ | `e4_projected_kernel.yaml`'s confirmation grid was superseded by #3 above (exact ε via permutation construction vs. rejection-sampled, and now written into App. G). Don't bother running it. |
| 5 | `python -m src escape --config configs/experiments/escape_time.yaml` **after editing** `k: 16, m: 52`, transition-matrix coords (not row-softmax logits) | Escape exponent 0.973/1.979/3.927 vs D−2, in the substrate the theorem is stated in | Current config is the superseded K=8,M=4 pilot. The reported paper numbers came from a different, undiscovered script — this rebuilds them from scratch. |
| 6 | `python experiments/gradient_transfer.py` (already ran once, see `results/gradient_transfer/`) | Depth-graded transfer asymmetry: cos(g_out,g_proc) rises 0.32→0.60 with D, out→proc transfer inverts sign at D=8 | Exploratory only — likely inflated by a shared "suppress irrelevant vocabulary" component. Needs decomposition before citing. Don't write up as-is. |

## Credibility (not closure, but a reviewer will ask)

| # | Command | Buys | Cost |
|---|---|---|---|
| 7 | `python -m src supervision --config configs/experiments/e1_five_condition.yaml` **with the `confirmation:` block promoted to top level** (5 seeds, 8000 steps, 20k train) | Table 1 (five-condition comparison) with located logs, instead of recovered manuscript values | GPU-hours; this is the table with the biggest "no logs found" flag |
| 8 | `python -m src lora --config configs/experiments/lora_transfer.yaml` | Confirms provenance of the 92.11%/91.44% LoRA numbers (Table 17) | Config already matches paper scale (3 seeds, 800 steps) — check whether its logs exist before rerunning |

## Do not run (known dead ends, checked)

- **Gradient cosine against oracle θ\*−θ₀** (`oracle_alignment.py`, already ran) — permuting a block's 832 MLP units leaves the function identical, so θ\* sits in an orbit of size (832!)^D and cosine is at chance by construction. Confirmed: means below seed spread, no depth trend, 4 depths × 5 seeds. Keep as a documented negative result if anything, not a rerun target.
- **K-sweep to confirm ρ_c=1/K in the trained Transformer** — your own sweep puts the trained frontier near ρ≈0.86 against a predicted 0.0625; this would publish a prediction your own data refutes.

## New science (higher risk, do last)

| # | What | Buys |
|---|---|---|
| 9 | Preconditioner test: train the D=8 outcome cell with Shampoo/SOAP instead of AdamW | The only *causal* (not correlational) item — if the mechanism is directional suppression, reconditioning should partially restore learning |
| 10 | Coherent-vs-symmetric corruption in a **trained** model (already confirmed in the kernel: 100% vs 5.7% at ρ=0.25, `noise_threshold.py`) | Same predicted thresholds (1/K vs 1/2) as a live-training result, not just a population-optimum one |
| 11 | Verify the causal-patch pipeline (`oracle_patch_mean=1.0000` vs `random_patch_mean=0.0588` already sitting in `results/mechanism_deep/runs/*/log.txt`) against `frozen_code/mechanism_pipeline.py` before reporting it | Would close §9's "causal identification... is underway and not reported here" if the patching protocol checks out |

---

**If you only do one thing:** run #5. It's the cheapest remaining item (no
GPU, minutes) and rebuilds the escape-time exponent in the theorem's own
substrate instead of the superseded K=8,M=4 pilot config.
