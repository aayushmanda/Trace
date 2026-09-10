# Trained versus hand-coded executor comparison

Fresh pilot: depth 4, seed 44, 200 updates, 10,000 unique training circuits and 256 disjoint held-out circuits.

This semantic-token one-layer Transformer run does not reconstruct the archived RegisterMachine or character-token experiments. Learned conditions share initial weights, the circuit pool, minibatches and optimizer settings. Mixed format assigns direct answers to half the examples and process traces to half, using a fixed random assignment. Equal updates do not equalize tokens or FLOPs. Hand-coded references have different capacities.

## Final held-out results

| Condition | Free answer accuracy | Local rule accuracy | Composition TV |
|---|---:|---:|---:|
| process | 96.1% | 98.6% | 0.128 |
| outcome | 15.6% | not queried | not defined |
| both | 12.1% | 78.5% | 0.481 |

Local-rule and composition measurements use separate diagnostic circuits. The displayed circuit is excluded from training and the main accuracy test. Local tables average execution positions and circuit backgrounds. Gradient readouts use first-step tables only; this distinction is recorded in the CSV.

## Figures

- [Local transition tables](transition_tables.pdf)
- [Whole-circuit comparison](circuit_comparison.pdf)
- [Checkpoint diagnostics](training_diagnostics.pdf)
- [Actual versus predicted gradients](gradient_agreement.pdf)

Final mixed-format full-gradient cosine: **0.253**; relative gradient error: **1.731**. These measurements do not by themselves establish a shared mechanism.

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
python -m src executor --output results/executor_comparison/reproduction --depth 4 --seed 44 --steps 200 --train-size 10000 --test-size 256 --probe-size 64 --batch-size 128 --d-model 96 --d-ff 192 --lr 0.002 --device cuda:0 --threads 1 --backgrounds 2 --checkpoints 0 5 10 15 20 25 30 40 50 65 80 100 130 160 200
```
