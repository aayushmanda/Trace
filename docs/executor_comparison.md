# Comparing trained and hand-coded computation

`src.eval.executor_comparison` produces actual forward-pass comparisons,
composition diagnostics, and an optional parameter-gradient comparison. It uses
the semantic-token models in the `handcoded/` package. It does not load the
character-token GPT checkpoints used in the broader experiments.

## Generated figures

- `transition_tables.pdf` / `.png`: four predetermined gates (NOT, CNOT, SWAP,
  Toffoli), each shown as exact rule, fixed process executor, learned process
  model, and mixed-format model if present. Rows are source states, columns are
  successor states. The color scale is 0–1 for every panel. Probabilities retain
  the mass assigned to non-state tokens rather than hiding it by renormalizing.
- `circuit_comparison.pdf` / `.png`: one predetermined circuit, evaluated at all
  16 starting states. Outcome and mixed-format models are scored on the direct
  answer branch, conditioned on COLON. Process models first generate their own
  traces. Invalid generated prefixes have zero state mass.
- `training_diagnostics.pdf` / `.png`: free-running accuracy, local-rule accuracy,
  composition error, induced rule strength, valid-state mass, and sensitivity of
  the local tables to the surrounding circuit and execution position.
- `gradient_agreement.pdf` / `.png`: actual direct-answer cross-entropy gradient
  versus the gradient through composed extracted tables. Also shows the rule
  component, gradient norms, and agreement of actual descent with true-rule and
  eight norm-matched random-rule directions. This is generated **only** for a
  model trained on both direct-answer and trace formats.

The gradient plot uses all parameters, including explicit zero entries for unused
parameters, so the vectors have the same coordinates. The table-product gradient
includes the Jacobian of the Transformer readout. Gradients are not applied to the
model. Zero-norm cosines are undefined, not reported as successful agreement.
The local matrices used for this gradient test are measured at the first execution
step; ordinary local and composition diagnostics average over all execution steps.

## Use already-trained notebook models

Run this after the existing notebook has trained its semantic-token models. Pass
only modes that the corresponding model was trained to emit. The two existing
notebooks call their models either `models['process']` / `models['outcome']` or
`trained_process` / `trained_outcome`.

```python
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src' / 'eval' / 'executor_comparison.py').exists())
sys.path.insert(0, str(ROOT))
from src.eval.executor_comparison import compare_models

rows = compare_models(
    {'process': trained_process, 'outcome': trained_outcome},
    tokenizer,
    depth=DEPTH,
    circuits=test_circuits[:128],  # held-out circuits only
    output=ROOT / 'results/executor_comparison/notebook_models',
    device=DEVICE,
)

from IPython.display import Image, display
display(Image(filename=str(ROOT / 'results/executor_comparison/notebook_models/transition_tables.png')))
```

This does not retrain or change the weights. It restores each model's train/eval
mode. Use a new output directory for each run. The single-snapshot diagnostic
plot has x=0, meaning the supplied snapshot, not a claim that its weights are
untrained. No gradient figure is produced unless a genuinely mixed-format model
is supplied under the key `both`. Do not label a process-only model as `both`.

## Reproducible fresh pilot

```bash
python -m src executor \
  --output results/executor_comparison/pilot \
  --depth 4 --seed 42 --steps 2000 \
  --train-size 10000 --test-size 256 --probe-size 64 \
  --device cpu --threads 2
```

The CLI trains three one-layer Transformers from identical initial weights:
process, outcome, and mixed-format. The mixed-format assignment is fixed per
training example with probability 0.5; this is **not** the paper's corruption
reliability rho. Both formats have correct targets. The data pool and minibatch
indices are shared. Losses are token means within each format; the mixed loss
weights the format means by their minibatch example fractions. AdamW uses lr
0.002, weight decay 0.01, and gradient clipping at norm 1 by default.

The held-out test, composition probes and displayed circuit are disjoint from the
training pool. Test and composition probe sets are also disjoint. Synthetic local
queries are independently constructed balanced state/gate queries; they are not
claimed to be disjoint from the training pool. The probe reads 52 gates × 16
source states at each position in each sampled background. The default uses two
backgrounds. Report variation across these contexts, not just their average.

The runner saves:

- Source hashes, git commit, dimensions, data seeds and environment versions in
  `manifest.json`.
- The exact train/test/probe circuits, mixed-format assignment, model weights and
  optimizer states in `checkpoints/`.
- Per-checkpoint metrics in `metrics.csv`, fixed-reference metrics in
  `fixed_metrics.csv`, and underlying matrices in compressed `.npz` files.
- Figures in both PNG and PDF, plus a measured-results report.

Saved checkpoints are ignored by Git under the repository's existing policy.
Loading one for later analysis:

```python
import torch
from src.eval.executor_comparison import h

saved = torch.load('results/executor_comparison/pilot/checkpoints/process_step02000.pt',
                   map_location='cpu', weights_only=True)
cfg = saved['config']
tokenizer = h.make_tokenizer()
model = h.build_random_learned_model(tokenizer, cfg['depth'], seed=cfg['seed'],
                                    d_model=cfg['d_model'], d_ff=cfg['d_ff'])
model.load_state_dict(saved['model'])
model.eval()
```

## Interpretation for the paper

Good local rule recovery is evidence that a trained model predicts the correct
transitions under gold prefixes. It does not prove that answer-only execution
uses that transition representation. Outcome-only trace queries would be out of
distribution, so their local heatmaps are intentionally absent.

Composition TV includes an extra non-state/invalid category. Conditional TV is
also saved, with undefined values where there is no valid-state mass. For the
process model, this compares a table product to a distribution after one greedy
trace, **not** to the exact autoregressive marginal over all possible traces.
For the mixed model it compares to the direct answer branch, conditional on COLON.

Low composition error is distinct from gradient agreement. In particular,
nearly uniform predictions can agree without identifying a mechanism. The
actual gradient uses the raw full-vocabulary answer-token cross entropy; the
surrogate uses state-conditional rule matrices, so their discrepancy also
reflects non-state mass. The gradient result is for a separately trained mixed
condition, not direct evidence about the archived outcome-only models.

No acceptance threshold, causality, near-mixing condition, layer-to-step mapping,
or shared internal computation is assumed. The pilot has one seed and fixed
updates, not matched FLOPs. A successful local heatmap alone does not validate
the proposed explanation of the supervision gap. The fixed architectures also
have different capacities from the learned one-layer models.

Suggested caption (replace run details using the actual manifest):

> Local transition distributions of exact and trained executors. Each panel
> shows all 16 source and successor states for a predetermined Boolean gate.
> Trained-model distributions are measured under valid gold prefixes and
> averaged across execution positions and surrounding circuits. Colors retain
> full-vocabulary probability mass. Outcome-only local queries are omitted
> because that model was not trained on trace prefixes. The comparison tests
> local rule prediction; composition and gradient agreement are measured
> separately. Results are from a fresh, single-seed semantic-token pilot.

## Diagnostic verification

```bash
python -m unittest tests.test_executor_comparison
```
