# Closure TODO

What's measured, what isn't, and the exact command for each. "Closure" = a
theorem or table tested against data rather than asserted.

## Scope change (2026-09-12)

The paper was rescoped to three questions only: why process supervision
reaches the correct rule while outcome-only does not (credit assignment,
§4), why training still recovers the rule under some fraction of wrong
traces (§5), and fraction-vs-amount (population threshold vs finite-sample
recovery, also §5). It is now a mechanistic/optimization-theory paper about
the shared transition-table model, not a paper about Transformer
expressivity or about testing mechanism transfer in trained networks.

**Cut from the paper, and therefore out of scope below:** Proposition 6
(constructive realizability) and the Transformer-embedding theorem — pure
representational-capacity results, not optimization; the rollout/execution
chapter (§6 as it was); and the trained-model local-rule readout study (§8
as it was). Everything in the old version of this file about matched-architecture
GPU training, causal patches on the constructive executor, escape-time
tables, or "predictive vs. mechanistic closure" for a trained Transformer's
*mechanism* belonged to that cut content and no longer applies — the paper
doesn't claim a trained-Transformer mechanism to close. Don't resume those
threads without first checking whether the section they supported still
exists (grep `body.tex` for the relevant `\section`/`\label` before trusting
an old item number).

Paper is 49 → 22 pages (12 main text + 10 appendix) after the cut.

**Script layout changed underneath this file** (not by me): individual
`experiments/*.py` scripts now live as importable modules under
`src/experiments/`, dispatched through one CLI, `python experiments/run.py
<command>` (see `experiments/run.py`'s `COMMANDS` dict and
`experiments/README.md`). All commands below already use the new names; the
paper's own script citations were updated to match.

## Done, in the paper

| # | Script / command | Buys | Where |
|---|---|---|---|
| 1 | `python experiments/run.py noise-threshold` | Reliability threshold (ρ_c=1/K, 1/2), ρ=0 information-completeness, corruption-structure dissociation | §3.2, App. A (Identifiability of the Executor from Corrupted Traces) |
| 2 | `python experiments/run.py credit-suppression` | Thm (forward-backward) both halves; near-mixing depth bound; complete-mixing population separation | App. A "Numerical Confirmation of the Credit Identities" |
| 3 | `python experiments/run.py kernel` | Reconfirms #2 and extends the reliability-threshold coefficient to the full ρ sweep; flags the mixing-ball caveat (ε=0.9 pulls the fitted exponent above the predicted rate) | App. A, cross-ref'd from `cor:noise-threshold` and `cor:near-mixing` in body.tex |
| 4 | `python experiments/run.py family-sampling` | Checks whether App. A's tabular-vs-Transformer sampling difference explains the frontier gap. It doesn't (ρ≈0.127→0.138, nowhere near the Transformer's ρ≈0.85) | §3.2 and App. A |
| 5 | `python experiments/run.py tabular-sampling` | Extends #4 to the symmetric-vs-coherent structure comparison; reproduces the existing structure table exactly under uniform sampling and shows the dissociation holds under family sampling too | App. A, "Structure against rate" paragraph |
| 8 | `python experiments/run.py clean-convergence` (new script, `src/experiments/clean_convergence.py`) | Cor. 5's convergence claim, concretely: GD from random logits on clean (ρ=1) counts reaches exact rule-cell/answer-accuracy recovery by step 1,000, while mean mass on the true successor climbs toward but never reaches 1 (0.061→0.978 by step 8,000), matching "requires unbounded logits, not attained at finite parameters" | App. A, new paragraph "Reaching the exact transition rule (Corollary 5)" + `figures/clean_convergence.pdf` |

Rerun any of these any time — all write from archived logs/CSVs or a
closed-form, no-training computation.

## Still open, still in scope

| # | Command | Buys | Cost |
|---|---|---|---|
| 6 | `python -m src supervision --config configs/experiments/e1_five_condition.yaml` **with the `confirmation:` block promoted to top level** (5 seeds, 8000 steps, 20k train) | Table 1 (five-condition comparison) with located logs, instead of recovered manuscript values | GPU-hours; this is the table with the biggest "no logs found" flag |
| 7 | `python -m src lora --config configs/experiments/lora_transfer.yaml` | Confirms provenance of the 92.11%/91.44% LoRA numbers (§3) | Config already matches paper scale (3 seeds, 800 steps) — check whether its logs exist before rerunning |

## Do not run (known dead ends, checked)

- **Gradient cosine against oracle θ\*−θ₀** (`oracle_alignment.py`) — permuting a block's MLP units leaves the function identical, so the cosine is at chance by construction. This whole line of investigation (θ\* alignment, gradient transfer between formats) was tied to the deep outcome-architecture experiments that are no longer part of the paper's scope.
- **K-sweep to confirm ρ_c=1/K in a trained Transformer** — the trained frontier sits near ρ≈0.86 against a predicted 0.0625; this would publish a prediction the paper's own data refutes. #5 above already shows this gap isn't a sampling artifact; closing it would need new theory (a finite-sample argument predicting the offset), not a rerun.
- **Quoting ε=0.9 in the near-mixing exponent fit** — outside the mixing-ball regime, pulls the fitted exponent above the predicted rate.

---

**If you only do one thing:** rerun #6. It's the table most likely to draw a
"no logs found" objection, and it's the only fully in-scope item left that
needs new compute.
