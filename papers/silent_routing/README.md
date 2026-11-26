# Silent Routing (ICML 2026)

LaTeX sources and compiled PDF for *Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls*.

## Build

Compile from this directory with [tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic -X compile main.tex
```

## Layout

| Path | Contents |
|---|---|
| `main.pdf` | Compiled 55-page manuscript (8-page main text + Appendices A–G) |
| `main.tex` | Preamble, theorem environments, macros, and appendix layout; inputs `body.tex` and `appendix.tex` |
| `body.tex` | Main paper (Abstract, Sections 1–6: Setup, Outcome Lower Bound, Silent Routing & Block-RMS, Experiments, Related Work, Conclusion) |
| `appendix.tex` | Appendices A–G (Notation & FAQ, Expressivity, Outcome Proof, Silent Routing Proof, Block-RMS Proof, Experimental Setup & Causal Interventions, Additional Results & ODE Verification) |
| `figures/` | TikZ/PGFPlots sources (`overview.tex`, `gate_head.tex`, `feature_learning.tex`, `causal_intervention.tex`, `scaling.tex`, `counting_chain.tex`, `theory_measurement.tex`, `transfer.tex`) and `main_transfer.pdf` |
| `main.bib`, `*.sty`, `icml2026.bst` | Bibliography and ICML 2026 style files |

Related code and artifacts at the repository root:
- C++20 causal intervention & 27-block verification suite: [`../../cpp/`](../../cpp/README.md)
- Python counting experiments, expressivity & credit checks: [`../../handcoded/`](../../handcoded/README.md), [`../../src/`](../../src/)
- 1080p60 Manim explainer animations: [`../../animations/`](../../animations/README.md)
