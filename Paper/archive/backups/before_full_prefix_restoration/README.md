# Paper

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

The former full Theorem 4 fragments (`step_shared_setup.tex`, `clean_routing.tex`, `independent_biases.tex`, `full_prefix_routing.tex`, `perturbed_routing.tex`, `thm4_full_statement.tex`, and `planted_perturbation_table.tex`) remain as archived source material and are no longer included in the current manuscript. The pre-removal paper and source snapshot are in `archive/backups/before_theorem4_removal/`.

The hybrid restores only the aligned structured result through `sections/aligned_transformer.tex`; it uses the matrix Frobenius metric, including the query factor of 1/K. The pre-hybrid manuscript is saved in `archive/backups/before_hybrid_aligned/`. Dependency notes and dense-gradient checks are in `research/aligned_hybrid/`.
