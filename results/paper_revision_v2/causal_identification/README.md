# Causal identification (deep outcome architecture)

Mechanism test, not mixed-format \(\widehat P_g\). Probes and causal patches use **nnsight** (`NNsight` on the generic `nn.Module`). \(C_t\) is autograd.

## Success (confirmation budget)

- Held-out answer: outcome ≈ chance (1/16) while outcome+local ≈ process ≈ 1.
- Linear probes \(h_t \to s_t\) high on held-out circuits (outcome models).
- nnsight `probe_subspace` patch at block \(t\) to \(s_t'\) yields \(\Phi_{g_{t+1:D}}(s_t')\).
- Hand-built oracle: `oracle_slot` ≈ 1, `probe_subspace` ≈ 1, random / wrong-layer ≈ chance.
- Report \(C_t\) vs \(D\). Do **not** claim \(\varepsilon^{D-1}\) unless mixing is verified.

## Failure

- All conditions ~0% held-out answer at tiny step counts: **undertrained**, not a negative result. Read `train_loss` / `train_subset_answer_accuracy`.
- Probes high, `probe_subspace` ~chance: state is readable but not causally used.
- Mixed \(\widehat P_g\) (TV 0.833, cosine 0.028) is **not** this test.

## Commands

Smoke (plumbing; not paper numbers):

```bash
python -m src outcome-local --smoke --device cpu --no-compile
```

Confirmation (hours; one `--depth`; GPU 2/3 if free):

```bash
python -m src outcome-local --config configs/experiments/causal_identification.yaml \
  --confirm --depth 2 --device cuda:2 --no-compile
```

Do not launch a \(D\)-grid from `RUN.md`. No GSM8K / LLM transfer claim.
