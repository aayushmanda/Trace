Yes. I would implement them as four closure experiments, in this order:

1. Direct test of Proposition 8 — highest priority.
   For fixed \(K=16\), choose several depths \(D\in\{2,4,8\}\), competence levels \(a\), and answer weights \(\beta\). Construct

$$
P_g(a)=U+a(T_g-U)
$$

exactly, then vary \(\rho\). For every point compute the actual combined-objective derivative

$$
J'_{\rho,\beta}(a)
$$

and check whether its sign flips exactly at your predicted

$$
\rho_c(a,D,\beta).
$$

Plot predicted boundary vs measured boundary. This directly validates the paper's new reliability-frontier result. 

2. Fixed-\(\rho\), varying-\(N\) — fraction versus amount.
   Pick several fixed reliabilities, e.g.

$$
\rho\in\{0.08,0.10,0.15,0.20,0.30\},
$$

and independently vary

$$
N\in\{500,1000,2000,5000,10000,20000,50000\}.
$$

For each pair measure rule-cell recovery and answer accuracy. At fixed \(\rho>1/K\), increasing \(N\) should approach the same population solution. This is the experiment that actually separates “fraction correct” from “number of examples.” Your current finite-sample study shows the gap but does not fully isolate these two variables. 

3. LoRA rerun — supporting evidence only.
   Let the current SmolLM2-135M sweep finish over all \(\rho\) and seeds, save the raw CSVs/logs/checkpoints, and regenerate the reported aggregates from those artifacts. Do not add new LoRA variants. Its purpose is simply:

$$
\text{Does noisy-process robustness/local-vs-rollout behavior survive pretrained adaptation?}
$$

The current manuscript still notes that the old LoRA aggregates lacked original logs. 

4. Rerun Table 1 — provenance closure.
   Run the five supervision conditions:

$$
\{\text{outcome, answer-first, filler, process, corrupted}\}
$$

with exactly the paper settings and five seeds. Save one CSV containing every seed rather than only means/stds. This is not a new scientific experiment; it simply makes Table 1 reproducible because the current manuscript says its original run logs are unavailable. 

Priority-wise:

$$
\boxed{
\text{Prop. 8 validation}
>
(\rho,N)\text{ grid}
>
\text{Table 1 rerun}
>
\text{LoRA}
}
$$

The first two improve the actual scientific claim. The last two mainly close reproducibility.

---

## Status as of 2026-09-12 (narrowed scope: theory fixes + 3 experiments, then stop)

Done directly (no training needed):
- Appendix A.1 embedding: replaced the ad hoc "bilinear table block" with a
  standard KM-unit ReLU hidden layer (reusing Theorem 9's own lookup unit
  unchanged) followed by an ordinary linear readout whose weights are a
  reshaping of A. No non-standard primitive remains.
- Proposition 8: added the explicit hypothesis `0<=beta<2/(K-1)` (proved
  sufficient for the K-beta*C(a,D)>0 denominator condition; the paper's own
  beta=1/D falls outside this simple sufficient range for small D but was
  checked numerically to satisfy the real condition regardless, with margin
  >=8.48 at K=16 for every D from 2 to 64). Strengthened the "directional/
  stalling frontier along one fixed path, not a claim about arbitrary SGD"
  framing right in the proposition statement, not just in the remark.
- Verified the O(e^-C) softmax-leakage claim numerically (fitted slope
  -1.00 against C for two competitor counts) and cited this in Appendix
  A.1's proof; did not claim this verifies the full composed bounds.
- LoRA rerun (item 5): the background SmolLM2-135M sweep from earlier this
  session had already finished (27/27 rows: outcome, answer_first, rho in
  {0,0.2,0.4,0.5,0.6,0.8,1.0}, x3 seeds). Recomputed the aggregates directly
  from `results/paper/smollm/smollm_trace.csv` and updated body.tex /
  appendix_results.tex to the final numbers (89.67% answer accuracy /
  89.78% exact rollout at rho=1.00; 82.49% local state / 23.78% exact
  rollout at rho=0.50 -- previously 92.11/91.44/83.75/24.44 from an earlier,
  less-complete run). Raw per-seed artifacts already archived under
  `results/paper/smollm/artifacts/`.
- Fixed-rho/varying-N (item 6): left untouched, as instructed.

Launched in background, not yet written into the paper (check
`results/paper/gpu2_progress.log` and `results/paper/gpu3_progress.log` for
completion; each stage also has its own `.log`):

- **Table 1 rerun** (item 4): `python -m src supervision --config
  configs/experiments/e1_five_condition_rerun.yaml` -- exactly Table 1's
  existing settings (boolean_circuit_8, 5 conditions, seeds 2001-2005,
  8000 steps, train 20000 / val 1000, batch 128, embed 128/4 heads/2
  layers), promoted unchanged from e1_five_condition.yaml's `confirmation:`
  block. Output: `results/paper/e1_five_condition_rerun/table1_rerun.csv`
  (per-seed rows) + `..._persist.json` (full config + summary). Running on
  CUDA_VISIBLE_DEVICES=2, queued before the D=2 reliability run below.
- **Depth x reliability** (item 3): `python -m src reliability --task
  boolean_circuit_{D} --rhos 0.5 0.7 0.8 0.9 1.0 --seeds 2001 2002 2003
  --checkpoints 8000` for D in {2,4,8} (same architecture/budget as the
  existing D=8 sweep behind Table 2/Figure 1: defaults train_size=20000,
  val_size=500, batch_size=128). Outputs:
  `results/paper/depth_reliability/boolean_circuit_{2,4,8}.csv`. D=2 queued
  on GPU 2 after the Table 1 rerun; D=4 then D=8 running sequentially on
  GPU 3. Rough estimate ~5-7 hours wall-clock total across both GPUs.

Next, once these finish: write the depth x reliability results into a new
short subsection/figure testing whether the measured frontier moves with D
the way `prop:outcome-rescue`'s stalling threshold predicts, and update
Table 1's caption to drop the "original run logs are unavailable" line
once `table1_rerun.csv` is in hand (reconciling against the existing
86.35/9.27/etc. numbers already flagged as an open discrepancy in body.tex).
Per the instruction that produced this batch: stop adding scope after
these are written up.

---

## Update 2026-09-12 (later the same day): scope explicitly frozen, depth x reliability write-up dropped

Overriding the "next" paragraph directly above: the depth x reliability
study (D in {2,4,8}, `results/paper/depth_reliability/boolean_circuit_{2,4,8}.csv`,
already finished per `gpu2_progress.log`/`gpu3_progress.log`) is **not**
being written into the paper. Explicit instruction: no new depths, no new
corruption laws, no new tasks, no new architectures, no new theory --
verification, compression, and submission-quality cleanup only, then
submit. Those three CSVs are intentionally left unused (see
`results/paper/ARTIFACT_MANIFEST.md`'s final section).

Table 1 rerun *was* completed and closed out: `table1_rerun.csv` reproduces
the five-condition comparison (outcome 7.72/answer-first 7.96/filler
7.76/process 82.00/corrupted 6.12, vs. chance 6.25, n=5 seeds each),
written into `body.tex`'s Table 1 and its caption, with the config/CSV/log
archived under `results/paper/e1_five_condition_rerun/` and hashed in
`results/paper/ARTIFACT_MANIFEST.md`. The `"pilot": true` field this
script used to hardcode into every persisted JSON regardless of which
config was passed (including this full-scale confirmation run) was a bug
in `src/eval/supervision.py`, now fixed; the already-written
`table1_rerun_persist.json`'s misleading `pilot`/`note` fields were
corrected in place (data rows untouched) rather than regenerated, since
regenerating would waste the ~5h GPU run for no scientific reason. Also
fixed: an unused `alpha>0` hypothesis in Corollary 12 (`cor:gradient-pullback`)
and a missing 1/K softmax-Jacobian factor in that same corollary's
threshold (it compared a row-logit gradient against a P-space coefficient
without the normalization difference between the two). Reproducibility
statement, Limitations, and the five-condition table's caption in
`main.tex`/`body.tex` updated to match. Remaining work this round: a
proof-consistency pass and a presentation/typo pass, both scoped as
verification only -- no new results.

---

## Update 2026-09-12, later still: restructuring, gradient-alignment diagnostic, Lipschitz tightening

Proof-consistency and presentation-pass agents both ran; fixes applied:
unused `alpha` in Cor. 12 removed; Prop. 11's `c2=2c1` arithmetic error
fixed to `c2=c1`; Cor. 12 given its missing stated hypotheses; a stale
"6.02" leftover from the old Table 1 (missed in the first pass) corrected
to 6.12 in `appendix_results.tex`; two figure captions fixed for
dotted-line ambiguity and (a)/(b) vs Left/Right mismatch; a red/green
colorblind-inaccessible pair fixed in `prop8_frontier.py` and
`fraction_vs_amount.py` (both regenerated, numbers unchanged).

Per explicit instruction, did a restructuring pass: Prop. 8 (the clean-answer
reliability-frontier diagnostic) demoted from the main text to a new
appendix subsection (`app:outcome-rescue-diagnostic`), leaving only a short
main-text paragraph explicitly labeled "one-dimensional diagnostic" that
points to the appendix and disclaims quantitative explanation of the
measured ~0.85 Transformer transition; Theorem 9 (realizability) was
already appropriately de-emphasized relative to the Appendix A.1 embedding
result, so left alone; added a crisp two-claim framing paragraph at the
very start of the introduction; added explicit "external validation, not
theorem verification" framing for the Transformer experiments. Fixed a
figure-title regression this caused (hardcoded "Proposition 8" in
`prop8_frontier.py`'s plot title, now numbered differently after the move;
retitled to "Reliability-frontier diagnostic" everywhere, code and docs).

Added, per explicit request (after an initial "don't add it" was reversed
mid-session): `src/experiments/gradient_alignment.py`, a gradient-alignment
diagnostic in the actual two-block GPT already used for Table 1 -- at
checkpoints (init/25%/50%/100% of an 8000-step process-mode run, seeds
2001-2003), computes an oracle local-transition-only gradient and compares
its alignment (dot product, normalized projection, cosine, plus a
per-transition-position breakdown) against the actual process-loss and
outcome-loss gradients at the same theta. No model checkpoints saved to
disk (none needed -- computed live via `train_with_checkpoints`'s
callback). Results land in `results/gradient_alignment/`; write-up still
pending as of this note (background run in progress).

Also: user shared an unrelated arXiv paper (Nair 2025, "Softmax is
1/2-Lipschitz") while gradient-alignment was training. Checked it against
our own Lemma 9 (was "Lemma 10" before the Prop. 8 move) -- no
contradiction (different norm combination: same-norm ell_p there, mixed
ell_infinity-to-ell_1 here), but it prompted rederiving our own bound from
scratch, which showed our constant of 2 was valid but not tight; the true
tight constant is 1 (verified numerically, 200k random trials, targeted
construction approaching 1.000000). Retightened Lemma 9 and propagated
through c1, c2 (Prop. 10) and c3 (Cor. 11), including recomputing the
numerical-verification percentage (0.01% -> 0.02%, exact rescaling by the
old/new bound ratio, not a guess) rather than leaving it stale.

Gradient-alignment run finished (3 seeds x 4 checkpoints, ~1h total):
clean, theory-consistent result. cos(g_local,g_proc) stays 0.80-0.88
throughout training (normalized projection stable at 0.40-0.49, low
variance); cos(g_local,g_out) starts at 0.58 at init then collapses to
~0 or slightly negative once training proceeds (normalized projection
becomes wildly unstable in sign and magnitude, -0.68 to +3.43 mean with
huge variance). Holds uniformly across all D=8 transition positions, not
concentrated at one -- rules out a global-gradient-norm artifact. Written
up as new Section 3.2 (`sec:gradient-alignment`, short version + summary
table) and new Appendix F (`app:gradient-alignment`, full protocol +
per-position table), using the exact prescribed claim language ("the
exact shared-kernel mechanism predicts... we observe the same
gradient-level asymmetry... This does not show the Transformer follows
the kernel's dynamics"). Archived under `results/gradient_alignment/`,
hashed in `results/paper/ARTIFACT_MANIFEST.md`. Per instruction: this is
the last new result -- no second new experiment after this one.

Paper now compiles cleanly at 33 pages, all cross-references resolve, no
undefined labels or citations.

---

## Update 2026-09-12, final: terminology fix + compaction pass

User caught a precise terminology error: Lemma 9 was titled "into total
variation" but TV distance is (1/2)*L1, and the proved inequality uses the
full L1 norm -- renamed to "Softmax is 1-Lipschitz from ell_infinity to
ell_1" (kept the inequality itself unchanged, math was already confirmed
correct); also retitled the nearby "contraction of total variation" phrase
in Prop. 10's proof to "contraction in ell_1" for consistency.

Then did a compaction pass per explicit request ("don't go overboard"):
across this session's edits, the same scope-hedge ("not a claim about a
trained network's trajectory" / "external validation, not verification")
had been restated in ~4-5 separate places (new intro paragraph, Prop. 8's
condensed main-text pointer, the appendix diagnostic's own closing
paragraph, the gradient-alignment main-text closing, and its appendix
Scope paragraph) -- exactly the kind of repetition the paper's own
pre-existing editorial policy ("we do not repeat this scope note at every
result") was meant to prevent. Trimmed the two most redundant standalone
paragraphs (appendix_theory.tex's "This remains a one-dimensional
diagnostic" cut from ~12 lines to 4; appendix_results.tex's
gradient-alignment Scope paragraph cut roughly in half), removed a
redundant restated sentence in body.tex's gradient-alignment section
close, and cut a meta-commentary trailing clause from the new intro
paragraph. Kept all the load-bearing, non-duplicate hedges (the
proposition statements' own inline scope clauses, the "two caveats" and
"population minimizer set" paragraphs, which each say something not said
elsewhere). Recompiled clean, 33 pages, 95 labels, zero missing
cross-references.
