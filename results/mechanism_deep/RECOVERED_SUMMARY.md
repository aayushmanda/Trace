# Recovered summary: mechanism_deep study (2026-09-10)

## What happened

The background run (`python -m src outcome-local` via `run_mechanism_study.sh`,
depths {2,4,6,8} x seeds {42..46}, 20 jobs across 4 GPUs) was interrupted by
the user before it finished analysis. On inspection:

- **19/20 runs actually completed training successfully** (6000/6000 steps,
  ~5-10 min each). Each one then **crashed while writing `metrics.csv`**:
  `ValueError: dict contains fields not in fieldnames: 'local_loss',
  'terminal_loss'` in `src/training/io.write_csv` (called from
  `frozen_code/mechanism_pipeline.py`). This is a plain CSV-writer schema bug
  in the modified pipeline — new fields were added to the row dicts without
  adding them to the declared `fieldnames`. It has nothing to do with the
  shared GPU cluster or the user stopping the agent.
- Because the crash happens after `print(json.dumps(summary), ...)` for each
  checkpoint, **the actual numbers survived in `runs/*/log.txt`** as JSON
  lines (7 checkpoints per run: steps 0,1000,...,6000), even though
  `metrics.csv`, `probes.csv`, and the per-layer trained-model `patch.csv`
  breakdown were never written to disk.
- **`D8_seed46` is the one genuine failure**: killed externally at step
  ~4000/6000 (67% through), unrelated to the CSV bug. Only 4 checkpoints
  (0,1000,2000,3000) recovered for it.

Recovered by regex/JSON-parsing every `{"step": ...}` line out of the 20
`log.txt` files. Full recovered trajectories and this analysis only exist in
this file and are not yet reflected in the paper.

## Final-step answer accuracy, % (step 6000; all four D=8 cells, `process`
included, use only the 4 seeds that reached step 6000 — `D8_seed46` was
excluded rather than averaged in at its partial step 3000. SD below is
**sample** SD (n-1 denominator), matching the convention the rest of the
paper's tables use — not population SD. This matches what is now written
into the paper, `tab:mechanism-depth-sweep`.)

**Architecture note**: `outcome` and `outcome_local` train the deep
(one-block-per-transition) oracle-shaped architecture. `process` trains a
*separate, different* one-block architecture (`mechanism_pipeline.py:493`,
`build_random_trainable_process_architecture`) — it is a reference for what
the same optimizer/depth sweep reaches on an architecture already known to
succeed, not a matched-architecture cell. An earlier version of this file
(and an earlier version of the paper table) incorrectly implied all three
conditions shared the deep architecture.

**Protocol note**: this study uses 10,000 training circuits and 6,000 updates
— different from `tab:2x2`'s 20,000 circuits and 8,000 updates. The learning
rate (5e-4) was selected from a small grid for the clearest
outcome-vs-outcome+local separation, which is evidence about this selected
regime, not a learning-rate-independent claim.

| Depth | Outcome (mean ± sample sd) | Outcome+local (mean ± sample sd) | Process |
|---|---|---|---|
| 2 | 97.27 ± 0.28 (n=5) | 98.75 ± 0.51 (n=5) | 100.00 ± 0.00 (n=5) |
| 4 | 72.11 ± 2.67 (n=5) | 87.50 ± 3.94 (n=5) | 100.00 ± 0.00 (n=5) |
| 6 | 12.81 ± 1.98 (n=5) | 55.63 ± 6.33 (n=5) | 100.00 ± 0.00 (n=5) |
| 8 |  6.25 ± 0.00 (n=4) | 20.32 ± 2.30 (n=4) | 100.00 ± 0.00 (n=4) |

This is a clean, monotonic, five-seed (four at D=8) confirmation that pure
outcome supervision on the deep (one-block-per-transition) oracle-shaped
architecture degrades sharply with depth (97% -> 72% -> 13% -> 6.25%, exactly
chance for K=16 states at D=8), while the auxiliary per-block local-state loss
(`outcome_local`) substantially but incompletely rescues it at every depth
(99% -> 88% -> 56% -> 20%). `process` (different architecture, included as a
reference point, not a matched cell) stays at exact 100%. This is much
stronger evidence for the depth-suppression story than the single diverged
seed in `tab:2x2`. **This table is now written into the paper**
(`appendix_results.tex`, "A clipped, five-seed, four-depth follow-up," and a
pointer from the main-text Section 5); the causal-patch table below is not.

## Trained-model causal patch, `probe_subspace` method (coarse, averaged over layers)

Recovered from the `trained_probe_subspace_mean` field the pipeline printed
per checkpoint — this patches the **actual trained** `outcome`/`outcome_local`
models (not the oracle) at the direction a linear probe identifies for the
target state, then checks whether the output shifts to the counterfactual
answer. Chance level for K=16 states is 1/16 = 0.0625.

| Depth | Outcome | Outcome+local |
|---|---|---|
| 2 | 0.0685 (n=5) | 0.0705 (n=5) |
| 4 | 0.0651 (n=5) | 0.0685 (n=5) |
| 6 | 0.0663 (n=5) | 0.0708 (n=5) |
| 8 | 0.0642 (n=4) | 0.0660 (n=4) |

Every value sits within noise of chance (0.0625), for both conditions, at
every depth. **Caveat: the matching per-layer probe *decode* accuracy
(`probes.csv`) was lost to the same crash**, so the "highly decodable but not
causally used" contrast the pipeline code was built to test
(see `frozen_code/mechanism_pipeline.py:684`) cannot be completed from this
recovered data alone — only the causal-patch half survived. Given the
paper's separate one-block study already shows ~99% local-rule decode
accuracy, a similarly high decode accuracy here combined with chance-level
patch accuracy would be the exact pattern the theory predicts, but that
inference is not yet directly measured for this deep-architecture run.

## What full re-run would recover

The bug is a one-line fix (add `local_loss`/`terminal_loss` to the CSV
fieldnames, or drop them before writing). Each run takes 5-10 minutes; 20 runs
across 4 GPUs is under an hour. A clean re-run would additionally give:
per-layer probe decode accuracy, the full per-layer/per-method trained-model
patch breakdown (not just the `probe_subspace` scalar), and a complete
D8_seed46.
