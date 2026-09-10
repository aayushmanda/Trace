# Paper positioning (10 Sep 2026 revision)

Honest scores: scientific contribution ~6/10; current draft ~5/10. Path to ~7/10 is **not** a proof that ordinary Transformers implement an internal executor.

## What the paper is

**Reliability is the empirical centerpiece.** Vary trace reliability \(\rho\) while **terminal answers stay correct**. Report **local** (step) accuracy and **rollout** (full trace / answer) accuracy, with seeds, assignment logs, and persist JSON.

**Theorems 1–3 stay in the main text** as an **exact explanation in the shared-executor / kernel setting** (mixing ball, local credit vs rollout, constructive and shared \(A_g\) kernels). Scope must be stated in the title, abstract, and theorem headers. They do **not** claim a causal mechanism inside a generic Transformer.

**Trained-model readout is a limit of transfer, not confirmation.** Representative numbers: composition TV \(\approx 0.833\), pullback cosine \(\approx 0.028\). Function-level agreement can be large while parameter-space credit does not transfer. Report this as a negative/limit result.

**Architecture controls** (process vs outcome architecture \(\times\) supervision) remain necessary so the reliability story is not an artifact of one oracle-shaped net. **Table 1 and LoRA** need explicit provenance (command, YAML, output path, seed list) before they headline anything.

## What the paper is not

- Not a promise that a Transformer “contains” the Boolean-circuit executor.
- Not a requirement to prove **T2** (parameter-space transfer / causal \(\theta\)-mechanism). T2 is an **unproved candidate**.
- Not interchange interventions (roadmap **E1**, ~75 runs) as the first empirical move.

## Strongest paper

Convincing **reliability** experiment + **precise local-credit theory** (kernel model) + **honest connection** (including negative transfer). Details, IDs, and week plan: [THEORY_EMPIRICAL_REVISION_ROADMAP.md](THEORY_EMPIRICAL_REVISION_ROADMAP.md). Ledger: `Paper/theorem_ledger.md`. New artifacts: `results/paper_revision_v2/`.
