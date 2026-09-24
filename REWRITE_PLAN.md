# Rewrite plan: table-level theory → neural-parameter theory

## Current manuscript status (2026-09-23)

- `Paper/unified_executor_ode.tex` is the main neural optimization section. It
  matches the D-block `UnifiedExecutor` trained in
  `handcoded/handcoded_executors.ipynb`: one paired parameter space, masked
  token-level process/outcome losses, an exact common-class construction,
  parameter-gradient ODEs, per-block backpropagation identities, and a proof
  that exact-zero initialization stalls both token losses. The notebook uses
  clean traces; the corruption model is a separate paper intervention.
- `Paper/neural_optimization.tex` is now explicitly an auxiliary appendix
  model. Its stationary-outcome theorem, process margin velocity, tangent-Gram
  rate, and scalar depth/escape-time ODE do **not** describe the notebook's
  random-initialized Transformer or its AdamW updates. The independent-row
  transition benchmark also lives in the theory appendix.
- The manuscript compiles. Random-initialization depth dynamics for the
  UnifiedExecutor remain open. Measure per-token task-aligned gradient size
  and cancellation before promoting any auxiliary mechanism to a claim about
  the trained model.


Working notes for the paper rewrite. Three lists, frozen before any prose gets
rewritten, so a hypothesis never quietly becomes a stated result mid-edit.

## Claims we keep

- The finite-state computation setup: `s_t = F(s_{t-1}, a_t)`, process vs
  outcome supervision (`\cref{def:executor}`).
- The corrupted-process model `Q_a = ρ T_a + (1-ρ) R_a` and the margin
  `m(ρ,c) = ρ(1+c) - c`.
- The existing row-logit theory (Theorem `thm:optimization-separation`,
  `thm:high-order-flatness`, the recovery corollaries) — kept as the
  **transition-space baseline**, not as a claim about neural parameters.
- `Proposition realization` (Appendix A): exact process/outcome executors,
  with the finite-temperature margin argument already in place (this was
  never the "call softmax one-hot" version — checked, it's fine as written).
- The empirical process/outcome contrast on the character-token GPT and the
  register machine.

## Claims we replace

- Any sentence implying table-level convexity (`∇²_z L_P = Diag(p) - ppᵀ ⪰ 0`)
  makes the *neural* objective convex. It does not:
  `∇²_θ L_P = JᵀH_CE J + Σ_j(p_j - q_j)∇²_θ z_j`, and the second term is
  indefinite for a generic `z_θ`. This is now stated explicitly as a remark
  in `appendix_theory.tex` (added below), not asserted or implied anywhere
  else.
- The two-separate-architectures framing (`HandcodedProcessTransformer` vs
  `HandcodedOutcomeTransformer`, different blocks/heads/width/parameter
  count) is superseded by `UnifiedExecutor`: one D-block architecture, one
  parameter space `Θ_D`, both `θ*_P` and `θ*_O` as different weight
  assignments in it. Verified this session: both free-run at 1.0/1.0,
  identical buffer counts, at n_bits ∈ {2,3,4} and D up to 12.

## Claims not yet established — do not state as fact

- **Any specific critical depth.** The original "D=3→96%, D=4→55.7%" was an
  uncontrolled run (wrong lr, contaminated test set). The corrected,
  disjoint-test, paired-seed sweep on `UnifiedExecutor` (n_bits=3, K=8) gives
  a *smooth* decline: outcome test accuracy 81%/82% (D=3), 51%/53% (D=4),
  24%/26% (D=5), process at or near 100% throughout. This is preliminary
  (2 seeds, 1500 steps) — more seeds and D=6,7 needed before it goes in the
  paper as a reported result. **Do not embed these numbers in the main body
  yet**; they are recorded here so the final protocol can be checked against
  them, not silently drift from them.
- **The doubly-stochastic assumption for a neural parameterization.** The
  table-level flatness proof needs `E = P_a - U` to satisfy both `E·1 = 0`
  (free, holds for any row-stochastic matrix) and `1ᵀE = 0` (requires `P_a`
  column-stochastic too). Permutation matrices get this for free; a generic
  `softmax(...)` neural output does not. Any neural version of the
  high-order-flatness theorem must either impose this as an explicit
  hypothesis on the architecture, or drop the clean cancellation and derive
  whatever weaker statement actually survives.
- **The gradient-cancellation / symmetry-pairing theory** (an involution
  `ψ(τ)` with `ξ(ψ(τ);v) = -ξ(τ;v)`). Existence of such a pairing for this
  task distribution is an open question, not a fact. Do not write a theorem
  statement for this until the diagnostic below has run.
- Convergence of SGD/AdamW to any constructed `θ*` — not claimed, and
  Appendix C.14's caution about `‖θ - θ*‖` as a competence proxy still
  applies to the unified model.
- "Process gradients are always larger" — not proved for the neural case.

## Open diagnostics (run before writing the corresponding theorem)

1. Depth sweep: 4-5 seeds at D = 3,4,5,6,7, disjoint test set, on
   `UnifiedExecutor`. Decides whether the decline is `Θ(α^{D-1})`-shaped or
   saturates.
2. Cancellation-vs-vanishing check: at `θ_0`, compute per-trajectory
   `ξ(τ;v) = -⟨∇_θ ℓ_τ(θ_0), v⟩` for a batch and look at the empirical
   distribution — clustered near zero (vanishing) or spread with mean near
   zero (cancellation)? Decides whether the symmetry-pairing program above
   is worth pursuing at all.
3. Check whether `UnifiedExecutor`'s trained `P_a(θ)` is close to doubly
   stochastic in practice, even though nothing in the architecture enforces
   it — relevant to whether the flatness hypothesis above is vacuous or
   approximately satisfied empirically.

## Writing order (see conversation for the full rationale)

1. This file.
2. `appendix_theory.tex`: Gauss-Newton remark (no experiment needed).
3. `body.tex`: common-architecture setup, `Θ_D`, trajectories vs witnesses.
4. `appendix_architectures.tex`: connect `Proposition realization` to the
   verified `UnifiedExecutor` implementation.
5–9: gated behind the diagnostics above; not started.
