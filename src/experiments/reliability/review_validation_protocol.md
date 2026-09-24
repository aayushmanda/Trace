# Review validation protocol

These experiments are new validation runs, separate from the canonical paper artifacts.

1. Reconstruct the exact canonical Boolean training corpora using the released
   generator, seeds 501/101, and NumPy reliability scores with seed 777. Fit
   empirical local transition laws on the displayed traces, with no gold-answer
   supervision. Archive corpus hashes, counts, row margins, rule recovery, and
   greedy rollout accuracy. Repeat with two independent corpus seeds.
2. Train semantic-token process-architecture models at depth 4, 20,000 training
   circuits, 2,000 AdamW steps, seeds 42/43/44. Save early, intermediate, and final
   checkpoints. These are diagnostic architecture runs, not replications of the
   main character-token GPT.
3. At each checkpoint, use disjoint update and evaluation prompts. Separate
   clean full loss, corrupted full loss, corrupted-state-only loss, and gold
   answer loss on a corrupted prefix. Include clean-state and mixed-state
   controls. Apply gradient updates to all parameters or architectural groups,
   measure independent clean state loss, and restore parameters exactly.
   Check both the first-order prediction and finite steps. Also clone AdamW
   state and compare a candidate step against a zero-gradient momentum step.
4. Continuous corruption changes the conditional wrong-successor law between
   uniform wrong successors and one fixed wrong permutation while keeping
   clean prompts and example-level reliability scores fixed. Tabular boundaries
   are predicted before measuring recovery; neural results, if run, are
   exploratory and use the same data as their own tabular baseline.

All outcome signs and null results will be reported. Checkpoints are repeated
measurements, not independent seeds. Small probes are engineering checks only.
No manuscript figures or existing result files are overwritten.
