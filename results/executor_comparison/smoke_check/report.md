# Trained versus hand-coded executor comparison

Fresh pilot: depth 4, seed 42, 2 updates, 64 unique training circuits and 16 disjoint held-out circuits.

This semantic-token one-layer Transformer run does not reconstruct the archived RegisterMachine or character-token experiments. Learned conditions share initial weights, the circuit pool, minibatches and optimizer settings. Mixed format assigns direct answers to half the examples and process traces to half, using a fixed random assignment. Equal updates do not equalize tokens or FLOPs. Hand-coded references have different capacities.

## Final held-out results

| Condition | Free answer accuracy | Local rule accuracy | Composition TV |
|---|---:|---:|---:|
| process | 6.2% | 6.6% | 1.000 |
| outcome | 6.2% | not queried | not defined |
| both | 0.0% | 6.4% | 0.700 |

Local-rule and composition measurements use separate diagnostic circuits. The displayed circuit is excluded from training and the main accuracy test. Local tables average execution positions and circuit backgrounds. Gradient readouts use first-step tables only; this distinction is recorded in the CSV.

## Figures

- [Local transition tables](transition_tables.pdf)
- [Whole-circuit comparison](circuit_comparison.pdf)
- [Checkpoint diagnostics](training_diagnostics.pdf)
- [Actual versus predicted gradients](gradient_agreement.pdf)

Final mixed-format full-gradient cosine: **0.363**; relative gradient error: **1.091**. These measurements do not by themselves establish a shared mechanism.

## Scope

- Gold-prefix rule recovery is a readout diagnostic, not a hidden-state or causal-intervention result.
- Outcome-only trace tables are omitted because those queries were not trained.
- Process composition uses the answer distribution after a greedy generated trace, not the marginal over all traces.
- Mixed-format composition and gradients condition on the direct-answer COLON branch, without a gold trace or answer in the input.
- Heatmaps retain full-vocabulary probability mass. Conditional tables and invalid-token mass are saved separately.
- One seed cannot establish robustness. Correct transitions, low composition error and gradient agreement are distinct requirements.
- No layer-to-execution-step correspondence or raw parameter similarity is assumed.

## Reproduce

```bash
python -m src executor --output results/executor_comparison/reproduction --depth 4 --seed 42 --steps 2 --train-size 64 --test-size 16 --probe-size 8 --batch-size 16 --d-model 96 --d-ff 192 --lr 0.002 --device cpu --threads 1 --backgrounds 2 --checkpoints 0 2
```
