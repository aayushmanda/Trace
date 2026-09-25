# Future work

Everything here is deliberately **not** in the current submission. Together it
makes a strong follow-up paper, and it is what would take the work from about 8
toward 9. Replaces the older `futurework.md`.

## Tier 1: toward 8.5 (Akyürek-style evidence)

1. **Which-predictor phase diagram.**
   - Measure the network's agreement with four candidate predictors: the pooled
     table, the Bayes context reader, the true rule and the one wrong rule.
   - Do it across ρ, error concentration λ and model size.
   - Goal: show which rule the network implements in each regime, and whether
     agreement with the Bayes reader rises with scale. This is the direct
     analogue of Akyürek et al.'s convergence to the Bayes estimator.
   - Partial data already exists in Tables 11–12.
2. **Clean-trace probe.**
   - A linear probe on the residual stream after step 1 that decodes "is this
     trace clean so far?"
   - Label by transition correctness or by the clean flag, not by "the displayed
     state equals the gold state".
   - This shows the context mechanism inside the network.
3. **Warm-start causal test of depth.**
   - Interpolate θ = (1−r)·θ₀ + r·θ_process, then train with the outcome loss.
   - Prediction: the smallest r at which outcome training succeeds grows with D.
     This turns the depth evidence from correlation into a causal result.

## Tier 2: toward 9 (scale and tightness)

4. **Real-model version of the main predictions.**
   - Fine-tune a pretrained model of about 1B parameters on multi-step tasks, for
     example arithmetic chains, or counting mod m in text (see item 12), with
     corrupted traces.
   - Show that concentrating errors at a fixed ρ hurts, and that outcome
     gradients cancel more on longer problems.
5. **Impossibility for one coherent wrong rule.**
   - With trace-only data, a pool with rule T at reliability ρ is distributed
     exactly like one with the wrong rule W at reliability 1−ρ, so no trace-only
     learner can recover T below ρ = 1/2.
   - This makes the threshold tight: necessary for any learner, not just
     sufficient for plurality decoding.
   - Caveat: in the paper's corruption the final answer stays correct, which
     breaks the symmetry, so the statement must be about learners that ignore
     the answer.
6. **A prediction made before the run.**
   - Use the required-ρ formula, ρ_req ≈ (c + κ√(2Δ/η))/(1+c), to predict ρ_req
     for a new model or dataset size before training, then check it.
7. **A scaling law for Δ(N, P).**
   - Fit Δ ≈ a·N^(−α) + b·P^(−β) + Δ_∞ across a grid of data sizes N and
     parameter counts P, and predict ρ_req for settings not run.
   - Test the predicted difference: required ρ grows with D for per-step
     learners but not for learners that read context.

## Tier 3: relaxing the assumptions

8. **Non-reversible rules (relax A2).**
   - Add many-to-one gates (for example AND into the target bit) to the Boolean
     circuit.
   - The contraction theorem then applies to the true rule itself, and wrong
     traces can merge back into the true trajectory.
9. **Step-level corruption (relax A3).**
   - Corrupt each step independently with probability ε instead of whole traces.
   - Prediction: survival holds at the same fraction of wrong steps, but the
     context boost shrinks, because a correct step no longer certifies a clean
     trace.
10. **Other relaxations of A3:**
    - errors that can match the true successor (the posterior after a correct
      step drops below 1);
    - wrong traces that end with wrong answers;
    - errors correlated across steps (Markov flags; grouped flags such as
      AAAABBBB vs ABABABAB).
11. **Non-uniform start states (relax A1).** Skew the start-state
    distribution: outcome supervision should get some signal toward "predict the
    answer from the operations", but still not learn the rule.

## Tier 4: other directions

12. **Counting mod m in text.**
    - A character-level task that fits (A1)–(A2) exactly: the answer depends on
      every character, it links to known parity hardness, and it has natural
      running-count traces.
    - Better than "find the first 'r' in strawberry", whose difficulty is mostly
      tokenization.
13. **Second-order type of the uniform point in Transformer parameter space.**
    - Following Makkuva et al. (ICLR 2025): at the zero-readout point, is it a
      strict saddle, or degenerate as in the table model? Does the degeneracy
      grow with the number of blocks?
14. **Theory from random initialization.** Extend the escape-time analysis to
    Gaussian starts of scale σ (Table 2 shows the empirical dependence).
15. **Why the learner falls short of the fitted table.** Separate tokenization,
    depth, training budget and samples per row as causes of the per-step
    accuracy gap.
16. **Process reward models.** Is a learned step verifier's credit local in the
    same sense, and how do verifier errors concentrated on one wrong rule affect
    policy learning?

## Extensions to the LoRA experiment (after submission)

- Process vs outcome at D = 4, 8, 16, with the cancellation measurement.
- The step-1 vs later-step split in the pretrained model.
- A larger model (SmolLM2-360M or 1.7B).