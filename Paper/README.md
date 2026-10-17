# Paper

> **Current state (2026-10-14).** The sources were restored to the 2026-10-11 version (40-page PDF, Theorems on
> credit, plateau and joint gate flow; appendices A–F), then revised: Lemmas 1–2 (credit), Theorem 1 (outcome plateau),
> Theorem 2 (joint gate flow), Corollary 1 (matched separation), Corollary 2 (free-running generation), a Scope paragraph,
> an experiment ladder, a related-work table, and Figure 3 (scalings; `research/lambda_collapse/`). Audit fixes: the tied
> scalar metric in Corollary D.1, and gaps in the crossing lemma. Pre-revision sources:
> `archive/backups/restored_20261011_before_reviewer_list/`; pre-restore sources: `archive/backups/before_restore_to_20261011/`.
> The layout notes below describe the earlier, later-dated revision line and are partly out of date.


Build (no pdflatex on this machine): from `Paper/`,

```
tectonic -X compile --outdir build --keep-logs main.tex
```

## Layout

| Path | Contents |
|---|---|
| `main.tex` | preamble, macros, theorem environments; inputs `body` and `appendix` |
| `body.tex` | abstract through conclusion |
| `appendix.tex` | Appendices A–F |
| `sections/` | fragments input from `body.tex`/`appendix.tex` (see below) |
| `figures/` | TikZ sources and PDF figures |
| `checklist.tex` | NeurIPS checklist (commented out in `main.tex` until submission) |
| `main.bib`, `*.sty` | bibliography and style files |
| `build/` | compiled PDF, logs, `archival_full_version.pdf` (linked from the appendix; keep next to `main.pdf`), `paper_with_supplement.zip` |
| `research/` | experiment drivers and data behind appendix figures (`focused_observables/`, `planted_perturbation/`, `numerical_audit/`), proof drafts (`drafts/`) and revision notes (`notes/`) |
| `direct_proofs/` | standalone direct-proof document and its numerical checks |
| `archive/` | earlier snapshots (`backups/before_*`) and the standalone Theorem 2 preview |

## `sections/`

| File | Where it appears |
|---|---|
| `thm2_gaussian_statement.tex` | Theorem 2(iii), main text |
| `thm2_gaussian_proof.tex` | Gaussian rule-credit proof, Appendix C |
| `aligned_transformer.tex` | minimal aligned matrix-flow proof, Appendix D |
| `full_prefix_routing.tex` | learned full-prefix routing statement and proof, Appendix D |
| `conditional_full_training.tex` | conditional full-parameter theorem (random seed + weak perturbation, H1–H7), last proof appendix; pre-change sources in `archive/backups/before_conditional_full_training/` |

The former full Theorem 4 fragments (`step_shared_setup.tex`, `clean_routing.tex`, `independent_biases.tex`, `perturbed_routing.tex`, `thm4_full_statement.tex`, and `planted_perturbation_table.tex`) remain as archived source material and are no longer included in the current manuscript. The pre-removal paper and source snapshot are in `archive/backups/before_theorem4_removal/`.

The hybrid restores the aligned structured result through `sections/aligned_transformer.tex`; it uses the matrix Frobenius metric, including the query factor of 1/K. The pre-hybrid manuscript is saved in `archive/backups/before_hybrid_aligned/`. Dependency notes and dense-gradient checks are in `research/aligned_hybrid/`.

The learned full-prefix branch was restored separately from the 64-page snapshot, with Latin feature-factor and mobility symbols. Its binary tied-family and finite-accuracy assumptions remain distinct from supplied routing. The preceding 40-page manuscript is saved in `archive/backups/before_full_prefix_restoration/`.

The audit-ready D.5 revision consolidates six full-prefix lemmas into four,
uses positive feature strength `x` and accuracy `eta`, and collects mobility
choices in dependency order. It preserves the induced metrics and quantitative
bounds and adds a conditional rollout corollary. The rebuilt PDF has 45 pages;
revision details and algebra checks are in `research/audit_ready_D5/`, with the
preceding 47-page PDF/source in `archive/backups/before_audit_ready_D5/`.

## October 2026 restructure

Main text rewritten to the Kim & Suzuki (ICLR 2025) template: claim title, three numbered results,
K=2 in the main text, proof sketch after every theorem, assumptions stated once. Main theorem is
`thm:matched` (same two-path family, step learns / outcome trapped from an open seed set); its
proof is `sections/matched_proof.tex` inside the two-path appendix `sections/two_path_appendix.tex`.
Supplied routing (old Thm 4(i)) is now `prop:supplied-routing` in the appendix; the Gaussian
rule-credit bound (old Thm 2(iii)) is `prop:gaussian-credit`. General-K definitions and a notation
table open Appendix A. The experiments figure `figures/experiments.tex` merges the old Fig. 1a and
Fig. 3a,b; the original figures are kept in the appendix. Pre-restructure source and PDF:
`archive/backups/before_matched_theorem_restructure/`. Style goal: readable and rigorous, not math-heavy.

### Theorem 4 restored (2026-10-13, at the author's request)

Main-text `thm:joint` is the old supplied-routing theorem (aligned and misaligned seeds,
O(eps^-4 log 1/eps)), proved in `sections/joint_learning_appendix.tex` (extracted from
`archive/backups/before_theorem4_removal/appendix.tex`, lines 809-1388, with
`step_shared_setup.tex`, `perturbed_routing.tex`, `thm4_full_statement.tex` and the ODE table
`j_ode_table.tex` regenerated by `handcoded/make_ode_table.py`). It is stated for its actual
trained parameters: U, W (Frobenius) and the query/key *scalars* q, k (Euclidean); for those
parameters the flow is the exact gradient flow. The matrix-metric aligned variant with the
eps^-12 bound (`sections/aligned_transformer.tex`, `prop:supplied-routing`) is no longer input.
Cut in the same pass: symmetric-seed subsection, Gaussian-law corollary, n=17 scan, "Beyond
counting", Gaussian geometry-event details; two-path diagnostics moved to Appendix F.

### Two-theorem paper (2026-10-14): "every appendix page proves Theorem 1 or 2"

Main text: Theorem 1 (separation in time; (i) = `thm:joint`, aligned seeds, K=2; (ii) = `thm:outcome-end`),
Theorem 2 (`thm:matched`), Lemma 1 (`thm:main`), Lemma 3 (`lem:binary-loss-mixture`); both theorems share the
step logit S(a + g tanh(qk) C) (`eq:gate-logit`). Deleted (not moved): step-credit lemma and its proofs
(dilution, rule-credit sites, leading coefficient, Gaussian credit), learned-routing theorem and proof,
relative position bias and time-varying projection schedules, general-K/Fourier machinery, degree/ANOVA
measurements, misaligned-seed crossing phase, remarks not used by a proof. Backups of the pre-deletion
sources are in `build/_appendix_before_delete.tex`, `build/_appendix_before_binary.tex`,
`build/_appendix_before_C_trim.tex`, `build/_joint_before_aligned_only.tex`, `build/_body_six_theorems.tex`
and `archive/backups/`. PDF: 34 pages (main text 6.5 pp).
