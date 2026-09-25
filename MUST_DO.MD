# Must-do list: closing the paper

Goal: submit a finished, consistent paper (about 8/10) instead of an ever-growing
one. Everything not on this list goes in `future_work.md`.

## Rules until submission

- **Freeze one version** of the paper and edit only that.
- **No new theory, no restructures, no new comparisons.**
- **A new idea goes in `future_work.md`**, unless it fixes something false in
  the paper.

## The one-sentence story

> Process supervision gives each step its own training signal; outcome
> supervision does not, because its signal cancels at the start of training.
> Step signal survives wrong traces as long as the true successor stays the most
> common one, which depends on how errors are spread, not just how many there are.

Every section should answer one of two questions: why process beats outcome, and
why it survives wrong traces.

---

## A. Experiments

Do these in order; item 1 comes first because everything depends on it.

1. **Fix determinism.**
   - Seed 2005 gave different results in two runs of the same configuration
     (ρ = 0.3: 66.7% vs 84.3%).
   - Test: run seed 2005 at ρ = 0.3 alone, twice.
     - If the two runs agree with each other, random state is carrying over
       between runs in one process: reseed at the start of every run.
     - If they disagree, it's GPU nondeterminism: set
       `torch.use_deterministic_algorithms(True)`,
       `torch.backends.cudnn.deterministic = True` and
       `CUBLAS_WORKSPACE_CONFIG=:4096:8`, or report means over seeds.
2. **Regenerate every table and figure from one set of result files.**
   - Settle which run is canonical for each number, including the Boolean
     baseline (7.66 vs 7.72) and the four-panel figure against Table 1.
   - Write a small script that rebuilds the tables and asserts that every
     number in the paper matches.
   - Give output files honest names (for example
     `boolean_reliability_N100k_L6.csv`, not `canonical_rerun`).
3. **Boolean 2×2 grid:** {20k, 100k} training prompts × {2, 6} layers,
   3 seeds, full ρ sweep. Use the same three seeds in every cell.
   ```bash
   for N in 20000 100000; do for L in 2 6; do
     TRACE_COMPILE=0 python -m src reliability --task boolean_circuit_8 \
       --rhos 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
       --seeds 2001 2002 2003 --checkpoints 8000 \
       --train-size $N --val-size 1000 --train-seed 501 --val-seed 101 \
       --ratio-seed 777 --batch-seed 12345 --batch-size 128 --eval-batch-size 128 \
       --include-outcome --no-compile --embedding 128 --heads 4 --layers $L \
       --dropout 0 --lr 0.0003 --weight-decay 0 --grad-clip 1 --bf16 \
       --output results/paper/boolean_reliability_N${N}_L${L}.csv --device cuda
   done; done
   ```
4. **Denser ρ sweep for Tx-sem** at λ ∈ {0, 0.5, 1}, with ρ from 0.1 to 0.9,
   3 seeds. This gives the points for the ρ figure.
5. **Measure Δ** (how well each model fits its training targets) on the
   checkpoints from items 3 and 4.
   - Tx-sem: probably reuse the Table 9 code.
   - GPT-2L: needs a new evaluation script, because character tokens require the
     probability of each full state string.
6. **Answer-token cancellation.** Recompute A and C restricted to the
   answer-state token on the existing depth-sweep checkpoints. This needs only a
   filter in the existing code.
7. **Optional but cheap: LoRA concentration run.** Scattered errors vs one
   repeated wrong rule at the same ρ, on SmolLM2-135M with the existing
   pipeline. This answers "does this only happen in tiny networks?"

## B. Analysis (no training)

- **Fit the per-step accuracy q(ρ)** for each run from exact-trace accuracy, and
  overlay the predicted step and answer curves.
- **Required ρ:** the smallest ρ where exact-trace accuracy reaches 50%,
  interpolated per seed, reported as mean ± SD.
- **The ρ figure:**
  - x-axis: error concentration c;
  - y-axis: required ρ;
  - lines: the theory floor c/(1+c) and the fitted table's sample-limited curve
    (Chernoff);
  - points: the fitted table and the networks;
  - second panel: each network's gap above the floor against √Δ.
- **Context formula:** acc ≈ p₁·q^(D−1) + (1 − p₁·q^(D−1))·π₀. On the register
  machine it predicts 15.1 / 45.2 / 85.1% against measured 16.2 / 46.9 / 85.8%.
- **Position-wise bound column** for Table 9, from existing KL values, or drop
  the column.

## C. Writing

1. **Abstract and introduction** built on the one-sentence story and the two
   questions.
2. **Open with the two-flip example:** a bit flips twice; outcome loss improves
   only at second order, process loss at first order.
3. **Add "the prompt already determines the answer."** A trace adds no
   information, so its benefit is a learning effect. Also rule out capacity with
   Proposition 1.
4. **Rewrite Section 2.1** around the learner effect: the same traces, with a
   stronger learner, need far fewer clean traces (about 0.8 vs about 0.24).
5. **A short ρ section:** define required ρ; floor from error structure, gap
   from samples and learner; the ρ figure.
6. **Context section:**
   - Proposition 3;
   - why even a Bayes reader's first step follows the pooled table;
   - the accuracy formula with its 2-point match.
7. **Novelty paragraph** against Havrilla & Iyer, Chandra et al. and Makkuva
   et al.: what they showed, and what this paper adds.
8. **Assumptions:**
   - a table saying which result needs which assumption; most need none of
     (A1)–(A3);
   - (A3) framed as the worst case;
   - a footnote that a corrupted step can coincide with the gold state.
9. **Define "correct step" as a correct transition** from the previous displayed
   state, everywhere it's used.
10. **Practical takeaway in the conclusion:** if traces come from one flawed
    teacher, their errors are concentrated; diversify error sources, because at a
    fixed error rate concentrated errors cost far more.
11. **Limitations** in three sentences, pointing to `future_work.md`.

## D. Final checks

- Main text ≤ 9 pages; total ≤ 35 pages including references.
- No `[PENDING]` markers, TODO comments or stale captions.
- Every table and figure number matches the results archive (assertion script
  passes).
- Every reference is in the bibliography, including Makkuva et al. and
  Kawaguchi.
- Reproducibility section lists the exact commands and seeds.

## Definition of done

Every item in A (1–6), B and C is finished, and D passes.

## Rough timeline

- **Week 1:** items A1–A2, then launch A3–A4 in the background; write C1–C5.
- **Week 2:** A5–A6 and the B analyses; write C6–C11; run D.

## Expected rating

| | Overall | Likely NeurIPS scores (1–6 scale) |
|---|---|---|
| Now | about 7.5 | 2, 3, 4, (5), average about 3.3 |
| After this list | about 8 | 3, 4, 4, (5), average about 4.0 |
| + optional LoRA run (A7) | about 8–8.25 | average about 4.2 |