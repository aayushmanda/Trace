# Theory and empirical revision roadmap

Prepared **10 September 2026**. Horizon 6–8+ weeks. This is a work plan, not a record of new theorems or finished grids. The manuscript already has **three** central theorem results; **T2** below is a **candidate and unproved**. Completing the plan does not imply acceptance.

**Positioning (binding):** [PAPER_POSITIONING.md](PAPER_POSITIONING.md). Title/abstract must not promise a Transformer-internal executor. Empirical centerpiece = reliability \(\rho\) with correct answers. Theorems = exact **shared-executor** explanation. Trained readout (TV \(\approx 0.833\), cosine \(\approx 0.028\)) = **limit of transfer**.

---

## 1. Intended argument

1. Both **local** (one-step) and **rollout** (composed) errors matter; they come apart as depth grows.
2. In a **kernel / shared-executor** model there is an exact local-credit theory (mixing ball, exponent \(D-1\), constructive nets).
3. On **trained Transformers**, vary **trace reliability \(\rho\)** with **correct terminals** and measure local vs rollout. That is the main empirical claim.
4. Architecture \(\times\) supervision checks that the gap is not one-net folklore.
5. Induced tables and \(J^\top W\) vs \(-\nabla L\) are **diagnostics of transfer**, including **failure**. They do not certify an internal executor.

Keep statements, assumptions, intuition, and short sketches of Thm 1–3 in the main text. Appendix: full proofs, constructions, protocols.

---

## 2. Theory targets (T1–T4)

Manuscript numbering (existing) is **Thm 1–3**. Roadmap **T1–T4** are research labels.

| ID | Status | Scope | Role in the paper |
|---|---|---|---|
| **T1** | Existing (polish) | Local vs rollout credit in the mixing ball | Main-text theorem; kernel model |
| **T2** | **Unproved candidate** | Parameter-space transfer: \(\sum_t J_t^\top W_t\) explains \(-\nabla_\theta L\) for ordinary Transformers | Do **not** require a causal \(\theta\)-mechanism. Optional diagnostic; negative results allowed |
| **T3** | Existing (polish) | Exact finite-parameter / shared-kernel executor | Main-text theorem; constructive + shared \(A\in\mathbb{R}^{M\times K\times K}\) |
| **T4** | Target | Escape-time / depth law in the **shared-executor** setting | Appendix or short main-text claim if proved; not a Transformer theorem |

Novelty review is still required before advertising T2/T4 as theorems.

---

## 3. Empirical program (E1–E7)

Do **not** launch E1’s ~75-run interchange grid in week 1. **E5 is now first among E-items.**

| ID | Experiment | Claim it can support | Default output | Week-1? |
|---|---|---|---|---|
| **E5** | `python -m src reliability` (trace \(\rho\), correct answers; local step vs rollout/answer; **persist JSON**) | Reliability is the empirical centerpiece | `results/reliability_sweeps/` and `results/paper_revision_v2/e5_reliability/` | Plumbing + logged smoke; **not** the full seed\(\times\rho\) GPU grid |
| **E2** | `python -m src architecture` 10-seed 2\(\times\)2 | Architecture controls; unstable = failure in the denominator | `results/architecture_controls_n10/` | Summarize rule + unit test; **not** 80-job confirm |
| **E7** | `python -m src lora` + Table 1 `python -m src supervision` | Headline tables need **provenance** (YAML, seeds, paths) | `results/paper/smollm/`, stdout/CSV for Table 1 | Document commands; no new 7B download |
| **E3** | `python -m src induced` per-step \(\widehat P^{(t)}\) | Optional transfer diagnostic | `results/revision/induced_rule.csv` | Optional |
| **E4** | `python -m src pullback` full-vocab LM CE, \(\sum_t J_t^\top W_t\) | Optional; cosine / norm ratio / relative error, including **negative** transfer | `results/revision/pullback.csv` | Code + unit tests; **not** \(D\times\)seed GPU grid |
| **E6** | length + \(m_{\min}\) | After E5/E2; not a Fig 4 substitute | `results/revision/length_generalization.csv`, `mmin_histograms.csv` | Skip full grid |
| **E1** | Interchange / intervention (~75 runs) | Causal swap inside trained nets | — | **Skip** |

§7 semantic-token executor (`python -m src executor-depths`) is already the PDF Figs 4–5 / Table 7 study. Do not conflate it with GPT induced-rule.

---

## 4. Claim-to-evidence contract

| Claim | Allowed evidence | Not evidence |
|---|---|---|
| Local credit theory | T1/T3 proofs; shared-kernel numerics (E4-kernel / `python -m src escape`) | A Transformer cosine |
| Reliability gap (local vs rollout) as \(\rho\) varies | E5 with persist JSON, seeds, assignment scores | One cherry-picked checkpoint |
| Architecture \(\times\) format | E2 confirmation **including unstable seeds** | One-seed notebook Table 6 |
| Transfer to \(\theta\) | E4 cosine, \(\lVert\hat g\rVert/\lVert g\rVert\), \(\lVert\hat g-g\rVert/\lVert g\rVert\); on-set mass filter | \(\delta_{\mathrm{comp}}\) alone (functions can match, gradients not) |
| “Executor inside the Transformer” | **None required**; do not claim | TV \(0.833\) + cosine \(0.028\) |
| Table 1 / LoRA | Logged E7 / supervision command + files | Memory of an old CSV |

---

## 5. Week table

| Week | Work | Not this week |
|---|---|---|
| **1** | Positioning + this file; pullback Jacobian+metrics tests; architecture summarize denominator; reliability persist JSON; `RUN.md` reorder; theorem ledger stub | E1 75 runs; E5 full grid; E2 80-job confirm; T2 proof |
| **2–3** | Logged E5 replication (Boolean circuits, several \(\rho\), \(\ge 3\) seeds); provenance pass on Table 1 / LoRA | Interchange |
| **3–4** | E2 calibrate then confirm if GPUs free; report instability fraction | Overwriting PDF Table 6 |
| **4–5** | Optional E3/E4 on mixed-format checkpoints; split-verdict as **limit of transfer** | Selling T2 as proved |
| **6–8** | E6 if needed; shared-kernel T4 numerics; rewrite abstract/intro to the positioning | New architecture families |

---

## 6. First actionable sequence (week 1)

1. Keep [PAPER_POSITIONING.md](PAPER_POSITIONING.md) and this file at **repo root** (`docs/` is gitignored).
2. Reliability: persist JSON beside the CSV; smoke one \(\rho\)/seed later — **do not** start the paper grid from `RUN.md`.
3. Pullback: full-vocab masked CE; \(\sum_t J_t^\top W_t\); NaN on zero-norm cosine/ratios; unit test (step-\(t\) change and/or FD vs autodiff).
4. Architecture `summarize`: 1 success + 1 unstable = **50%**.
5. Point `RUN.md` at this roadmap, `results/paper_revision_v2/`, and **reliability first**.
6. Stub `Paper/theorem_ledger.md` (Thm 1–3 existing, T2 candidate).

Commands: [RUN.md](RUN.md).
