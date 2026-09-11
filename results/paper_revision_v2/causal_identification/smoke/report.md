# Causal identification (deep outcome architecture)

This is the **mechanism** test: do trained D-block outcome Transformers *use* a
layerwise state $s_t$ in the remaining computation $\Phi_{g_{t+1:D}}(s_t)$?
It is **not** mixed-format $\widehat P_g$ (that readout is a **limit of transfer**:
D=4 local ~99.9%, composition TV 0.833, gradient cosine 0.028).
Probes and causal patches use **nnsight** traces (`NNsight` on the `nn.Module`);
$C_t$ stays in autograd.

## What would count as success (confirmation budget, not this smoke)

- **Credit placement:** held-out answer accuracy: outcome ≈ chance (0.062) while outcome+local ≈ process ≈ 1. Same architecture/init/format
  for outcome vs outcome+local; process is the known-good token-trace comparison.
- **Decodable state:** linear probes $h_t \to s_t$ on held-out circuits are high
  for outcome+local (and, if the net actually stores $s_t$, possibly outcome too).
- **Causal use:** `probe_subspace` patch at block $t$ to donor $s_t'$ makes the
  remaining net output $\Phi_{g_{t+1:D}}(s_t')$. Oracle-slot patch on the
  **hand-built** executor stays ~1 vs random-subspace ~chance (positive control).
- **Controls:** random subspace and wrong-layer stay near chance.
- **Credit vs depth:** report $C_t = \langle g_t^{\mathrm{term}}, g_t^{\mathrm{local}}\rangle / \|g_t^{\mathrm{local}}\|^2$
  vs $D$. **Do not** claim an $\varepsilon^{D-1}$ mixing-ball rate unless mixing is checked.

## What failure means

- **All conditions ~0% held-out answer:** undertrained. Not a negative mechanism result.
  Read `train_loss` / `train_subset_answer_accuracy` for plumbing.
- **Probes high, `probe_subspace` ~chance:** $s_t$ is linearly readable but not
  causally used by later blocks (the hole this experiment is for).
- **Oracle-slot high on a *trained* net, probes low:** the construction coordinates
  moved, but that is not evidence the net computed through $\Phi$.
- Mixed $\widehat P_g$ agreement is **not** confirmation of this test.

## This run

- depth $D$ = 2, seed = 42, steps = 80, $\lambda$ = 1.0
- device = `cpu`, smoke = True
- oracle_slot mean (hand-built) = 1.000
- random_subspace mean (hand-built) = 0.094
- probe_subspace mean (hand-built) = 0.188

| condition | val answer | train-subset answer |
|---|---:|---:|
| outcome | 0.000 | 0.312 |
| outcome_local | 0.000 | 0.250 |
| process | 0.000 | 0.812 |

Held-out probe eval accuracy (final checkpoint):

- `outcome`: t=1 → 0.000, t=2 → 0.000
- `outcome_local`: t=1 → 0.000, t=2 → 0.000

Trained-net patch means (final checkpoint, average over layers):

- outcome oracle_slot = 0.094
- outcome probe_subspace = 0.031
- outcome_local oracle_slot = 0.094
- outcome_local probe_subspace = 0.125

**This dump is plumbing / smoke.** Held-out accuracies near 0 are expected at
this budget. Do not paste them into the paper. Confirmation YAML:
`configs/experiments/causal_identification.yaml` (one `--depth` at a time;
requires `--confirm`; do not launch a D-grid from `RUN.md`).

CSV: `metrics.csv`, `probes.csv`, `credit.csv`, `patch.csv`, `trained_patch.csv`,
`probe_vs_patch.csv`, `table.csv`. JSON: `persist.json`.

No GSM8K / LLM transfer is claimed.
