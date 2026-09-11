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

## Done, verified, NOT yet in the paper (write-up only, zero new compute)

Independent rebuild of #3 above, extended to the full ρ sweep and to a causal
(not just gradient) check of the Theorem 1 construction. All CPU, all already
run; see `results/closure_handcoded/REPORT.md` for the full writeup.

| # | Script | Buys | Status |
|---|---|---|---|
| 12 | `python experiments/closure_kernel.py` (~15s) | **Thm 4, full ρ sweep** at $P_g=U$, $D=8$: process coef. matches $(K\rho-1)/(K-1)$ at $\rho\in\{0,1/K,0.25,0.5,1\}$ (e.g. 0.4659 vs predicted 0.4667 at $\rho=0.5$); outcome Rule$(G)$ = $1.08\times10^{-17}$ at every $\rho$. Reconfirms Thm 2 and Cor. 2's $D-1$ exponent (matches App. G exactly) **and** flags that $\varepsilon=0.9$ is outside the mixing-ball regime — don't quote its exponent. | Ran. `results/closure_handcoded/{thm4_mixing,thm2_residuals,corollary3,exponents}.{csv,pdf}`. Strictly extends what's in App. G (full ρ sweep instead of only $\rho=1$); worth folding the ρ-sweep table into App. G or a footnote. |
| 13 | `python experiments/closure_handcoded.py --device cpu` (~15–25 min) | **Causal check of Thm 1's construction**: nnsight-patching the known $s_t$ slot in `HandcodedOutcomeTransformer` flips the answer at every depth (`oracle_slot`=1.000, $D\in\{2,4,6\}$, chance=0.0625); random-subspace/unstructured/wrong-layer controls stay near chance. Also measures the untrained D-block net at $P_g\approx U$: outcome credit is $\sim10^3\times$ suppressed vs. the process floor, but **does not** track $\varepsilon^{D-1}$ there (the induced $\|\widehat P_g-U\|_2$ itself grows with $D$) — that exponent is a kernel-only claim (#3/#12), not a claim about the untrained net. | Ran. `results/closure_handcoded/{oracle_patches,oracle_true,init_gradient,init_induced}.{csv,pdf}`. This is the verification item #11 below was asking for, at the construction level — it validates the patch *methodology* against ground truth before spending GPU time applying it to a trained net. |

## Cheap reruns (minutes–hours, no GPU needed for most)

| # | Command | Buys | Note |
|---|---|---|---|
| 4 | *(nothing to run)* | ~~Cor. 2 exponent~~ | `e4_projected_kernel.yaml`'s confirmation grid was superseded by #3/#12 above (exact ε via permutation construction vs. rejection-sampled, and now written into App. G). Don't bother running it. |
| 5 | `python -m src escape --config configs/experiments/escape_time.yaml` **after editing** `k: 16, m: 52`, transition-matrix coords (not row-softmax logits) | Escape exponent 0.973/1.979/3.927 vs D−2, in the substrate the theorem is stated in | Current config is the superseded K=8,M=4 pilot. The reported paper numbers came from a different, undiscovered script — this rebuilds them from scratch. |
| 6 | `python experiments/gradient_transfer.py` (already ran once, see `results/gradient_transfer/`) | Depth-graded transfer asymmetry: cos(g_out,g_proc) rises 0.32→0.60 with D, out→proc transfer inverts sign at D=8 | Exploratory only — likely inflated by a shared "suppress irrelevant vocabulary" component. Needs decomposition before citing. Don't write up as-is. |

## Blocked on GPU (cuda:2/3 were at 99–100% as of the last check)

Same-architecture hygiene is already in the code (`src/eval/local_credit.py
--matched-architecture --induced-credit`; process net is now a
`copy.deepcopy` of the D-block outcome net, so it's patchable the same way).
Run in this order — do not skip to the 20-seed grid before the one-seed D=8
cell confirms non-chance.

| # | Command | Buys | Cost |
|---|---|---|---|
| 14 | `python -m src outcome-local --config configs/experiments/closure_matched_early.yaml --depth 2 --device cuda:2 --no-compile --matched-architecture --induced-credit` (repeat for `--depth 4`, `--depth 6`) | Short matched-architecture training, 400 steps each, $D\in\{2,4,6\}$ — first live-training check of Thm 4 in a trained (not just initialized) net | Minutes each, needs a free GPU |
| 15 | `python -m src outcome-local --config configs/experiments/closure_matched_d8.yaml --confirm --depth 8 --device cuda:2 --no-compile --matched-architecture --induced-credit` | One-seed $D=8$ matched confirm — gate before committing to a full seed grid | Hours |
| 16 | Trained-net causal patches on matched process vs. outcome vs. outcome_local (same nnsight pipeline as #13, applied post-training) | This is what actually closes item #11 below — a verified causal patch on a *trained* network, not just the construction | Depends on #14/#15 landing; do not pool these numbers with the §7 one-block local numbers (99.9%/TV 0.833/cosine 0.028) |

## Credibility (not closure, but a reviewer will ask)

| # | Command | Buys | Cost |
|---|---|---|---|
| 7 | `python -m src supervision --config configs/experiments/e1_five_condition.yaml` **with the `confirmation:` block promoted to top level** (5 seeds, 8000 steps, 20k train) | Table 1 (five-condition comparison) with located logs, instead of recovered manuscript values | GPU-hours; this is the table with the biggest "no logs found" flag |
| 8 | `python -m src lora --config configs/experiments/lora_transfer.yaml` | Confirms provenance of the 92.11%/91.44% LoRA numbers (Table 17) | Config already matches paper scale (3 seeds, 800 steps) — check whether its logs exist before rerunning |

## Do not run (known dead ends, checked)

- **Gradient cosine against oracle θ\*−θ₀** (`oracle_alignment.py`, already ran) — permuting a block's 832 MLP units leaves the function identical, so θ\* sits in an orbit of size (832!)^D and cosine is at chance by construction. Confirmed: means below seed spread, no depth trend, 4 depths × 5 seeds. Keep as a documented negative result if anything, not a rerun target.
- **K-sweep to confirm ρ_c=1/K in the trained Transformer** — your own sweep puts the trained frontier near ρ≈0.86 against a predicted 0.0625; this would publish a prediction your own data refutes.
- **Quoting ε=0.9 in the Cor. 3 exponent fit** (`closure_kernel.py` already ran it) — outside the mixing-ball regime, pulls the fitted exponent ~0.6 above D−1. Show it as a separate far-from-U panel if at all, never in the same fit as ε≤0.5.

## New science (higher risk, do last)

| # | What | Buys |
|---|---|---|
| 9 | Preconditioner test: train the D=8 outcome cell with Shampoo/SOAP instead of AdamW | The only *causal* (not correlational) item — if the mechanism is directional suppression, reconditioning should partially restore learning |
| 10 | Coherent-vs-symmetric corruption in a **trained** model (already confirmed in the kernel: 100% vs 5.7% at ρ=0.25, `noise_threshold.py`) | Same predicted thresholds (1/K vs 1/2) as a live-training result, not just a population-optimum one |
| 11 | Verify the causal-patch pipeline on the **trained** network (`oracle_patch_mean=1.0000` vs `random_patch_mean=0.0588` already sitting in `results/mechanism_deep/runs/*/log.txt`) against `frozen_code/mechanism_pipeline.py` before reporting it | Would close §9's "causal identification... is underway and not reported here" if the patching protocol checks out. #13 already verified the same patch methodology against the known construction (ground truth, not a trained net) — that de-risks this but does not substitute for it; #16 above is the trained-net version of this item. |

---

**If you only do one thing right now (no GPU):** run #5. It's the cheapest
open item and rebuilds the escape-time exponent in the theorem's own
substrate instead of the superseded K=8,M=4 pilot config.

**If a GPU frees up:** run #14 (all three depths), then #15. Only queue the
20-seed D=8 grid or #16's trained-net patches after #15's one-seed cell comes
back non-chance — that's the actual gate on going past item 7/8 in
`results/closure_handcoded/REPORT.md`'s own numbering.
