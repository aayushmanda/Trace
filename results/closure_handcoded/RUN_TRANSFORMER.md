# Predictive closure runs (not mechanistic)

Python: `/home/hariguru/aayus/.venv/bin/python`  
Repo: `/home/hariguru/aayus/trace`

**If GPU budget is one job: run experiment 1 confirm.** That is the discriminating result. Experiments 2 and 3 are CPU; run them tonight either way.

Closure here is **predictive**, not mechanistic. Even if all predictive cells land, score ceiling is **7.5 not 8+**: a directional split when only corruption *structure* changes, not “the frontier moved to 0.5.” Population thresholds are \(1/K=0.0625\) and \(1/2\); tabular finite-sample \(\sim 0.15\); trained Transformer (symmetric GPT sweep) \(\sim 0.85\). D.10 already concedes \(0.15\to 0.85\) unexplained. Directional agreement with that offset is decent closure.

Section 8.2 (TV 0.833, cosine 0.028) still blocks **mechanistic** closure. Sweeps cannot fix it. Title cannot promise a trained-model mechanism.

Recommended claim sentence (framing A — target distribution; trained-model results as corroboration):

> The shared-table account makes exact statements about the target executor and predicts trained-model behavior that competing accounts do not; we do not establish that the trained network implements that executor.

Framing B (discriminating prediction among explanations) sells better and is also honest. The current framing (promises a mechanism, reports it unestablished) cannot stay.

---

## Experiment 1 (PRIMARY) — Transformer, fixed ρ, two structures

Handcoded / same-architecture D-block process net. Hold ρ fixed; wrong mass uniform (`symmetric`) vs one coherent permutation (`coherent`). Same corruption amount. Rivals (sample-complexity, globality-barrier, serial-computation) predict **no split**. This account predicts a **directional SHIFT**: symmetric recovers at lower ρ than coherent.

**Do not pre-register “frontier lands on 0.5.”** Offset toward ~0.85 is expected.

What file appears if it worked:

- `results/closure_handcoded/transformer_law/confirm/accuracy.csv`
- `results/closure_handcoded/transformer_law/confirm/accuracy.pdf`
- `results/closure_handcoded/transformer_law/confirm/persist.json` (`verdict.decision`)

| Verdict | Meaning |
|---|---|
| **SHIFT** | At ρ=0.25, symmetric − coherent ≥ 0.15, **or** ρ_50(coherent) − ρ_50(symmetric) ≥ 0.10. Directional. Landing at 0.8 vs 0.9 still counts. |
| **NEGATIVE** | No split: \|Δacc\| at 0.25 and \|Δρ_50\| both < 0.10 (curves sit together, typically near ~0.85). |
| Smoke | Plumbing. Do not score SHIFT/NEGATIVE from 40 steps. |

```bash
cd /home/hariguru/aayus/trace
PY=/home/hariguru/aayus/.venv/bin/python

# CPU generator check (two laws differ at the same ρ). No GPU.
$PY experiments/run.py transformer-law --self-check

# Smoke (minutes, one GPU). Plumbing only.
$PY experiments/run.py transformer-law \
  --config configs/experiments/closure_transformer_law_smoke.yaml \
  --device cuda:2

# Confirm — the one GPU job if budget is one.
$PY experiments/run.py transformer-law \
  --config configs/experiments/closure_transformer_law_confirm.yaml \
  --confirm --device cuda:2

# Minimum discriminating cell (still directional): ρ=0.25 only, both laws.
$PY experiments/run.py transformer-law \
  --config configs/experiments/closure_transformer_law_confirm.yaml \
  --confirm --device cuda:2 --rhos 0.25
```

Title-system GPT reliability sweep (**symmetric law only**; does **not** implement 1/2). Already in the repo. Frontier ~0.85 is the D.10 offset, not this manipulation:

```bash
# Logged smoke (CPU; not a paper cell)
$PY -m src reliability --task boolean_circuit_2 --rhos 0.8 --seeds 2001 \
  --checkpoints 2 --train-size 32 --val-size 8 --batch-size 8 \
  --output results/paper_revision_v2/e5_reliability/smoke.csv --device cpu --no-compile

# Paper-scale symmetric sweep (GPU hours). Not coherent / 1/2.
$PY -m src reliability \
  --task boolean_circuit_8 \
  --rhos 0.0 0.5 0.8 1.0 \
  --seeds 2001 2002 2003 \
  --checkpoints 1000 2000 4000 \
  --train-size 20000 --val-size 1000 \
  --batch-size 128 --include-outcome \
  --output results/reliability_sweeps/boolean_circuit_8.csv \
  --device cuda:2 --no-compile
```

Do not invent a third stack. Kernel tabular fit: `python experiments/run.py noise-threshold` (proxy; 100% vs 5.7% at ρ=0.25 under **uniform-over-52-strings** sampling).

---

## Experiment 2 (SECOND) — Cor. 2 vs Table 2, pre-registered, CPU

ε = 1/2 (mixing-ball edge, stated here, not fitted to 97/72/12.8/6.25). Predict: outcome stays above chance at D=2,4; collapse from D=6; D=8 at 1/K; process high at all four depths. Then check `results/mechanism_deep/recovered.csv`.

```bash
$PY experiments/run.py table2
```

Appears: `results/closure_handcoded/table2_prediction.json` (written first) and `table2_verdict.json`.

If recovered CSV is missing: `$PY experiments/run.py recover-mechanism-deep`

---

## Experiment 3 (CHEAP EXTRA, CPU tonight) — match gate sampling

App. I samples uniformly over 52 gate strings. Sweeps sample gate **family** first (`x/c/s/t`), then an index. Rerun the tabular 1/K vs 1/2 fit under both samplings. Same ρ grid, both laws. Side-by-side ρ_50.

This removes the sampling confound. It does **not** explain the 0.15→0.85 Transformer offset.

```bash
# Full paper N (CPU, minutes). This is the one to run tonight.
$PY experiments/run.py tabular-sampling

# Plumbing only
$PY experiments/run.py tabular-sampling --quick
```

Appears: `results/closure_handcoded/tabular_sampling/frontiers.csv` (two rows: `uniform` vs `family`) and `frontiers.pdf`.

---

## Skipped — credit-norm vs failed transitions

Optional, not first-class. Consistent with this account **and** with anything that makes some transitions harder. Does not hold corruption amount fixed. Do not spend GPU on it.

---

## Mechanistic, unperformed (blocked by §8.2)

Layerwise probe + causal patch on the **trained** deep outcome model. Not a predictive sweep. Do not run from this file. Do not treat kernel files, `oracle_slot=1` on the construction, or experiment 1–3 as a substitute.

```bash
# Plumbing only (already exists)
$PY -m src outcome-local --smoke --device cpu --no-compile

# Unperformed confirmation (hours; one depth; prefer cuda:2/3 when free)
$PY -m src outcome-local \
  --config configs/experiments/causal_identification.yaml \
  --confirm --depth 2 --device cuda:2 --no-compile

# Same-architecture D-block variant (also unperformed)
$PY -m src outcome-local \
  --config configs/experiments/closure_matched_early.yaml \
  --depth 2 --device cuda:2 --no-compile --matched-architecture --induced-credit
```

Success for *mechanism* would be: probes high **and** `probe_subspace` patch on the trained net flips the remaining-gate answer; random/wrong-layer stay near chance. Construction `oracle_slot=1.0` does not close a trained mechanism.

---

## Future (not this turn) — predict the offset

Finite-sample plurality vs population (D.10, Appendix J: \(N^-<(K-1)N^+\)). If a sample-complexity statement predicted \(0.0625\to 0.15\) in the tabular fit, the remaining \(0.15\to 0.85\) becomes an optimization-budget quantity. No new theorem here. Matching sampling (exp 3) only removes the gate-family confound.

---

## What each command closes

| Command | Claim | System |
|---|---|---|
| exp 1 confirm | directional structure split at fixed ρ | trained D-block Transformer |
| `python -m src reliability` | existing symmetric reliability curve (~0.85) | character-token GPT (title system) |
| exp 2 | Cor. 2 ε^{D-1} collapse vs D | already-run D-block outcome table |
| exp 3 | tabular frontiers under sweep sampling | shared kernel, CPU |
| `python experiments/run.py noise-threshold` | App. I proxy 1/K vs 1/2 (uniform strings) | kernel, not the net |
| outcome-local confirm | mechanistic probe+patch | **unperformed** |
