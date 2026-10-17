# ASearcher: detailed LaTeX poster

The poster is based on the supplied paper `../long_horizon_searcg.pdf`,
**Unlocking Long-Horizon Agentic Search with Large-Scale End-to-End RL**
(Gao et al., ICLR 2026). That source PDF is preserved.

- Deliverable: `long_horizon_search_poster.pdf`
- Editable source: `long_horizon_search_poster.tex`
- Preview: `build/long_horizon_search_poster.png`
- Format: one-page A0 landscape, 1189 × 841 mm

Build from the repository root:

```bash
bash Poster/build.sh
```

Requires Tectonic, Poppler (`pdftoppm`), and the Lato font family in
`/usr/share/fonts/truetype/lato/`. All diagrams, charts, equations, and text
are vector content. Font files are embedded in the PDF.

## Scientific content and provenance

The poster covers the architecture, data generation, curriculum, rollout
system, GRPO objective, main results, ablations, and emergent behavior.

1. **Agent architecture — Section 3.1.** A single QwQ-32B model handles
   reasoning, analysis, action generation, and webpage summaries. Search
   and access are the two base tools. History retains compact analyses,
   actions, and tool responses, while lengthy internal reasoning is omitted.
   Webpage chunks are at most 10k characters.
2. **Data and curriculum — Section 3.2 and Appendix C.** The paper retains
   16k QA pairs from the open-source pool and synthesizes about 25.6k more
   using source-backed fact injection and fuzzing. It verifies clarity,
   difficulty, and answer uniqueness. Stage 1 mixes difficulty levels;
   Stage 2 keeps tasks requiring at least five tool calls.
3. **Asynchronous RL — Sections 3.3–3.4.** Independent rollouts and
   decoupled training avoid a batch-generation barrier imposed by the
   longest trajectory. The diagram is schematic, not a measured timing
   chart. Training uses a 128-action turn limit, batch size 64, AdamW
   learning rate 2e-6, and about 16k H800 GPU-hours.
4. **GRPO — Equation (1).** The poster factors the objective into a clipped
   per-token term, a per-trajectory action-token average, and a group average.
   `L_i` denotes the total generated action-token count in a trajectory.
   The training reward is terminal LLM judgment; the paper does not specify
   the training judge's identity in this section. The separately reported
   evaluation judge is Qwen2.5-72B-Instruct.
5. **Benchmark evidence — Table 1.** The base agent and the zero-shot
   augmented system are explicitly separated. The latter uses DeepSeek-V3
   summaries and aggregation of 16 independent rollouts. Summary-only
   intermediate results are listed in the chart caption.
6. **Ablations — Table 2.** All variants begin from the Stage 1 checkpoint
   and train for 200 steps. “Calls” is their average training-time tool-call
   count. Equal update counts are not described as equal compute budgets.
7. **Behavior — Figures 1, 6–7 and Appendix B.** The poster summarizes the
   verification case and the reported rise in tool use from 1.67 to over 20.
   Keyword counts and a case study are presented as behavioral evidence,
   not a general guarantee of factual correctness.

## Reported values used

| System | GAIA | xBench | Frames | HLE |
| --- | ---: | ---: | ---: | ---: |
| ASearcher, base | 58.7 | 51.1 | 74.5 | 21.5 |
| + DeepSeek-V3 summary | 60.3 | 56.4 | 76.6 | 23.4 |
| + summary and K=16 aggregation | 71.8 | 75.0 | 83.4 | 24.6 |

The paper's abstract gives **58.1** for base GAIA, while its main text,
Tables 1–2, and Appendix E give **58.7**. The poster consistently uses the
table value **58.7**.

Evaluation subsets: GAIA text-only validation (103 questions), xBench
(100), Frames (824), HLE (500). Base-agent pass@1 uses four evaluation
seeds. WebSailor comparison values are taken from its official report,
as marked in the source table; the poster does not imply identical
evaluation runs for all baselines.

The supplied paper gives an anonymous release reference without a usable
repository URL. The poster therefore includes the authors' contact emails
instead of inventing a code link or QR destination.
