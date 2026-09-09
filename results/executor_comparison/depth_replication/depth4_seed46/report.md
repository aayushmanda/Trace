# Trained versus hand-coded executor comparison

Fresh pilot: depth 4, seed 46, 2,000 updates, 10,000 unique training circuits and 256 disjoint held-out circuits.

This semantic-token one-layer Transformer run does not reconstruct the archived RegisterMachine or character-token experiments. Learned conditions share initial weights, the circuit pool, minibatches and optimizer settings. Mixed format assigns direct answers to half the examples and process traces to half, using a fixed random assignment. Equal updates do not equalize tokens or FLOPs. Hand-coded references have different capacities.

## Final held-out results

| Condition | Free answer accuracy | Local rule accuracy | Composition TV |
|---|---:|---:|---:|
| process | 100.0% | 100.0% | 0.000 |
| outcome | 14.8% | not queried | not defined |
| both | 25.4% | 100.0% | 0.855 |

Local-rule and composition measurements use separate diagnostic circuits. The displayed circuit is excluded from training and the main accuracy test. Local tables average execution positions and circuit backgrounds. Gradient readouts use first-step tables only; this distinction is recorded in the CSV.

## Figures

- [Local transition tables](transition_tables.pdf)
- [Whole-circuit comparison](circuit_comparison.pdf)
- [Checkpoint diagnostics](training_diagnostics.pdf)
- [Actual versus predicted gradients](gradient_agreement.pdf)

Final mixed-format full-gradient cosine: **-0.025**; relative gradient error: **1.007**. These measurements do not by themselves establish a shared mechanism.

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
python experiments/compare_executor_rules.py --output results/executor_comparison/reproduction --depth 4 --seed 46 --steps 2000 --train-size 10000 --test-size 256 --probe-size 64 --batch-size 128 --d-model 96 --d-ff 192 --lr 0.002 --device cpu --threads 1 --backgrounds 2 --checkpoints 0 100 500 1000 2000
```
