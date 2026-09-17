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

---

## Update 2026-09-13: hierarchy/presentation rewrite (no new science)

Per explicit reviewer-style instruction, executed a full narrative
reorganization: no new experiment or theorem, purely structure and
wording. Changes, in the order requested:

1. **Moved the gradient-alignment section.** Cut it from Section 3
   (right after the behavioral experiment, before the reader had seen the
   exact theory) and reinserted it as its own new Section 6, "A
   Gradient-Level Test in the Actual Transformer," positioned after
   Section 5 (corruption theory) and before Related Work. New reading
   order: phenomenon (Sec. 3) -> exact mechanism (Sec. 4) -> corruption
   consequence (Sec. 5) -> actual-Transformer gradient test (Sec. 6) ->
   where the theory stops (Sec. 7 Limitations). Fixed a figure/table
   width regression from the move (an added "(counterfactual)" table-row
   label overflowed the page margin; reverted, kept the word only in
   prose/caption).
2. Rewrote the abstract: cut the reliability-frontier discussion down to
   one clause and gave the gradient-bridge result its own sentence,
   closely following the phrasing supplied.
3. Added a 4-row "what is proved / measured / open" table at the end of
   the introduction (labeled `tab:proved-measured-open`), classifying
   each of the paper's claims as exact theorem / direct measurement /
   observed-not-explained. Had to narrow the table's column widths once
   to fix an overfull hbox.
4. Fixed a second overclaim instance (the first was fixed in an earlier
   session): "no local signal to escape the uninformed starting point at
   all" -> "no first-order rule-directed signal toward T_g at that
   point," in the introduction's own paraphrase of Cor. 5.
5. Tightened the gradient-alignment section's interpretation: g_out is
   now explicitly called "counterfactual" everywhere it's introduced
   (evaluated at process-trained parameters, not parameters it ever
   optimized); added the structural-alignment caveat ("g_local supervises
   positions contained within the process objective, so positive
   process-local alignment is partly structural"); replaced "ruling out a
   global-gradient-norm artifact" with the more precise "the same
   asymmetry appears at all eight transition positions, so the aggregate
   result is not driven by a single transition," in both body.tex and the
   appendix.
6. Prop. 8/14 now consistently called a "one-dimensional stalling
   diagnostic" everywhere it's named outside its own proof (intro,
   Related Work, abstract), never given headline billing.
7. Sharpened the novelty framing in Related Work: added an explicit
   sentence naming the actual novel contributions (placement-of-supervision
   credit geometry, the corruption-structure functional, connection to
   composed execution) right where the classical-1/K-logic disclaimer
   already sat, so a reader can't summarize the paper as "rediscovered
   the noisy-label threshold."
8. Shortened the Conclusion from 4 paragraphs to 1 tight paragraph
   (mechanism -> corruption consequence -> finite-sample note -> Transformer
   gradient signature -> the quantitative-optimization gap as the literal
   final sentence), following the supplied replacement text closely.

Also fixed, mid-session, a precise terminology catch from the user on the
earlier Lipschitz tightening: Lemma 9 was titled "into total variation"
but TV distance is (1/2)*L1 and the proved inequality is the full L1 norm
-- renamed to "Softmax is 1-Lipschitz from ell_infinity to ell_1" (the
inequality itself was already correct, only the name overclaimed); also
retitled a nearby "contraction of total variation" phrase to "contraction
in ell_1" for consistency. No downstream constants affected (only the
label was imprecise).

Recompiled clean: 33 pages (unchanged), 96 labels, zero missing
cross-references, zero overfull/underfull-beyond-baseline warnings, zero
undefined citations.

---

## Update 2026-09-13, later: "analysis paper, no predictions" fixes

User settled the framing debate: pure analysis paper, don't try to close
unrestricted-Transformer-SGD, don't add predictive experiments. Worked
through the resulting punch list, all prose/math fixes, no new runs:

- **Verified, did not fix, Section 6's supposed bug.** A critique claimed
  g_out was computed with the trace still in context (same failure mode as
  the copy-task confound Appendix B.7 already describes). Read
  src/experiments/gradient_alignment.py line by line: out_target =
  f"{ANSWER_SEP}{gold}\n" and the prompt never contains the trace --
  `_encode(tokenizer, inst.prompt, out_target, ...)` produces a token
  sequence with zero trace tokens anywhere, identical in construction to
  Table 1's own OUTCOME condition. The bug does not exist; the appendix's
  own protocol prose ("built from the same underlying prompt and gold
  trace") was genuinely ambiguous and invited the misreading. Fixed that
  sentence in both appendix_results.tex and body.tex to state explicitly
  that g_out uses a wholly separate, trace-free encoding. No rerun needed.
- Added a real small proposition (`prop:outcome-nonidentifiability`, one
  paragraph proof) formalizing outcome non-identifiability under gate
  under-excitation, replacing the previous floating disclaimer sentence.
  Explicitly declined the larger ask (full L_out minimizer
  characterization) as scope creep, per the user's own list.
- Fixed the process/outcome credit normalization mismatch honestly:
  added a remark after Cor. 2 showing the per-occurrence bounds (Eq. 13,
  Eq. 14) are already comparable, and that carrying the process bound
  through L_proc's own 1/D population average does *not* introduce a
  hidden D^-1 decay (Thm. 4's f_g is a frequency, not a count, so it's
  Theta(1) whenever per-step gate probability doesn't vanish with D) --
  deliberately did NOT adopt an earlier critique's suggested "O(D^-1) vs
  O(epsilon^{D-1})" framing after checking it against f_g's own
  definition and finding it false for this paper's actual setup.
- Added one paragraph explaining why Prop. 15 (symmetric-corruption
  stalling diagnostic) does not generalize via Cor. 8's c_star-c_T
  functional: the affine path P_g(a)=U+a(T_g-U) gives every wrong
  successor equal probability by construction, so it structurally cannot
  carry the c_star information Cor. 8 needs. Declined the unification the
  first critique round wanted, per the user's own correct math objection.
- Added the Table 2/4 discrepancy's actual two-sample statement: Welch's
  t=2.89, df=5.1, p=0.033 (computed from the real per-seed numbers, not
  invented) alongside the ~7x variance-ratio caveat, replacing "reported,
  not explained" with an honest quantified statement.
- Fixed Table 3's statistics discipline: replaced the incoherent
  "normalized projection" row (which conflated near-zero cosine with
  noise, giving nonsensical mean +/- sd like -0.32 +/- 5.89) with a clean
  ||g||/||g_local|| norm-ratio row -- which revealed the outcome gradient
  is not small/suppressed but large and growing (21x-110x by seed/checkpoint),
  just uncorrelated in direction. Added a full 12-row per-seed appendix
  table so no per-seed spread is hidden behind a mean+-sd.
- Title changed to "Why Process Supervision Survives Wrong Reasoning
  Traces: Credit Geometry in a Shared Executor" (user's preferred option).
- Abstract cut to 5 sentences, Prop. 15 dropped from it entirely per
  instruction.
- Added one genre-defining sentence early in the introduction, explicitly
  naming Makkuva et al. as the precedent for the paper's own discipline
  (analyzable substrate, no claim the analysis governs real training).
- Explicitly skipped, per the user's own priority table: full L_out
  minimizer classification, trained depth sweep, new corruption laws,
  the beta=1/D tabular "budget" rerun, and the Theorem 3 Monte Carlo
  check (all flagged optional/scope-creep/not-needed).

Fixed two new table-width regressions from these edits along the way
(the proved/measured/open table's "Shared transition-table model" cell
wrapping oddly; a single-word orphan line in the "Observed, not derived"
cell) -- both cosmetic, caught via underfull-hbox warnings, not visual
inspection alone.

Recompiled clean: 34 pages, 98 labels, zero missing cross-references,
zero overfull/undefined warnings.

---

## Update 2026-09-13, final: precision pass from an independent re-review

User's re-reviewer caught two of their own earlier mistakes (good faith,
noted for the record): the Section 6 "bug" claim was wrong (we'd already
verified this), and their own "O(D^-1) vs O(epsilon^{D-1})" normalization
suggestion was wrong too (f_g absorbs the 1/D, confirmed by our own
derivation). Both self-corrections matched what we'd already independently
established. Then found real, specific issues, all fixed:

- **Arithmetic error**: the Prop 15/Cor 8 non-unification paragraph said
  the affine path places (1-a)/(K-1) mass on each wrong successor; it's
  (1-a)/K. Re-derived by hand (P_g(a) row i entries: (1-a)/K off the true
  successor, (1+(K-1)a)/K on it; row sums to 1, confirming (1-a)/K is
  right). The argument itself (all wrong entries equal) was unaffected,
  just the printed constant. Fixed, plus a stray semicolon nearby.
- **The real substantive catch**: abstract and conclusion both said "the
  same asymmetry" transfers to the trained Transformer, but Table 3's
  norm-ratio row (added last round) shows the outcome gradient is LARGE
  (21x-110x g_local's norm) and growing, not small/suppressed -- only its
  *direction* is uncorrelated. Cor. 2 proves magnitude suppression;
  Section 6 shows misdirection at large magnitude. These are different
  phenomena and conflating them under "asymmetry" overclaims what
  transferred. Fixed in the abstract, the conclusion, Table 1's caption,
  and Section 6's own closing sentence -- all four now say "directional
  asymmetry transfers, magnitude suppression does not."
- Added one sentence addressing an omission: initialization is the
  checkpoint closest to Theorem 4's uniform-table point, yet it shows the
  *mildest* measured asymmetry (cosine 0.58 vs 0.80-0.88 later) -- the
  obvious question a reviewer would ask. Explained why this isn't a
  discrepancy (Theorem 4 describes the uniform table exactly; a random
  Transformer init need not realize it; the asymmetry emerging during
  training rather than being sharpest at init is expected, not anomalous).
- Fixed Section 6's opening ("kernel credit geometry -> ? -> observed
  execution advantage. We probe that missing arrow directly") which read
  as verification language promising to close a gap the introduction
  already says isn't being closed. Replaced with "this section asks
  whether that advantage is also visible at the gradient level."
- Added a caption clause to the per-position table noting its cosines
  (0.19-0.56) aren't comparable in magnitude to the aggregate table's
  (0.80-0.88), since restricting the oracle loss to one gate's 4 bits vs.
  all 32 is a narrower/noisier slice of the same signal, not a smaller
  version of the same quantity -- worded carefully as a plausible
  explanation, not an overclaimed mechanical guarantee we haven't derived.
- Fixed the one remaining first-person-singular line ("I verified") to
  "We verified," for an anonymous submission using "we" throughout.

Recompiled clean: 34 pages, 98 labels, zero missing cross-references,
zero overfull/undefined warnings. User's assessment after this pass: 8/10,
"the theory is complete on its own terms, the proofs hold, the statistics
are defensible, and the paper says what it means." Remaining gap to 9 is
explicitly not revision-scope (substrate novelty), not something to chase
by adding more.

---

## Update 2026-09-13, readability pass (no science changed)

User asked for the paper to feel ~20-25% simpler without cutting technical
content. No new results; pure prose/structure editing, prioritized for
risk (mechanical/high-value first, since a full rewrite of a
correctness-reviewed paper risks reintroducing errors):

- **Vocabulary standardization**: consistently "shared transition-table
  model" everywhere in prose (was also "shared kernel," "shared executor,"
  "shared transition kernel," "transition-kernel model," "the kernel's...").
  Left internal LaTeX labels (`prop:shared-kernel-embedding` etc.) and the
  formal proof symbol `\ell_{\rm kernel}` untouched -- readers never see
  label strings, and renaming a symbol used across ~9 equations in one
  tight proof for zero prose-clarity benefit is pure risk. Fixed one
  genuine synonym drift ("terminal supervision" -> "outcome supervision"
  in the Conclusion) while leaving a second, textually-similar case alone
  since it's actually a distinct concept (Prop. 15's "clean terminal
  answer added on top of a corrupted trace," not standalone outcome
  supervision) -- confirmed by rereading context before touching it.
- **Title** updated to match ("Shared Executor" -> "Shared Transition-Table
  Model") -- caught this inconsistency myself during a visual check,
  since the vocabulary pass would otherwise have left the title as the
  one remaining stale term.
- **Abstract**: rewritten to 5 tighter sentences per the user's suggested
  structure, but not verbatim -- restored the "counterfactual" qualifier
  and "directional signature" phrasing (checked that "loses stable
  rule-directed alignment" doesn't reintroduce the magnitude overclaim
  fixed last round; it doesn't, so the simplification was safe to keep).
- **Intro's "What is proved" paragraph**: cut from ~7 sentences to 3,
  deferring to Table 1 (now explicitly cited as doing this job), while
  keeping the one specific technical fact Table 1 doesn't capture
  (which two results are global-convergence proofs vs. local gradient
  statements).
- **Section titles** made narrative: "Credit Assignment in a Shared
  Executor" -> "Why Outcome Credit Disappears Under Composition"; "Robust
  Rule Learning from Unreliable Traces" -> "Why Wrong Process Targets Can
  Still Teach the Right Rule"; "A Gradient-Level Test in the Actual
  Transformer" -> "Gradient Directions in the Trained Transformer".
  Split "Related Work and Limitations" into "Related Work" (Sec. 7) and
  moved the Limitations paragraph into a renamed "Discussion and
  Limitations" (Sec. 8, was "Conclusion"), matching the user's suggested
  outline; Discussion (old conclusion prose) now follows Limitations as
  the closing paragraph rather than repeating the reliability-threshold
  derivation.
- **Question -> Theorem -> Interpretation** pattern applied to Theorem 1
  as the flagship example requested: added a "Where does outcome credit
  for one transition come from?" lead-in question before the theorem, and
  split the old "Derivation and interpretation" paragraph into separate
  "Derivation." and "Interpretation." paragraphs after it. Did not
  attempt this for every theorem in the paper (diminishing returns vs.
  risk of destabilizing already-correctness-reviewed proof-adjacent
  prose); Theorem 1 is the paper's central result and the clearest
  candidate.
- **Sentence-splitting pass**: wrote a small script to score sentences by
  (dash-count + semicolon-count + while/whereas/although-count) and flag
  the worst offenders in body.tex; split the 5 highest-scoring sentences
  (up to 920 characters) into 2-4 shorter ones each, across the Table 1
  prose, Section 6's protocol paragraph and its results paragraph, and
  the Discussion. Left the five-condition definitional list (uses
  semicolons as legitimate list separators, not clause overload) alone.
- **Scope-paragraph trimming**: cut Theorem 9's Scope paragraph from ~15
  lines to ~7 (kept the two non-obvious points: doesn't cover the actual
  trained architecture, no claim about what GD reaches) and the
  fraction-vs-amount Scope paragraph from 5 lines to 2. Kept Appendix
  A.1's and Section 6's Scope paragraphs at full length, per instruction,
  since both carry a specific, non-obvious point (the parameter-submanifold
  restriction; the diagnostic-not-verification distinction) not stated
  anywhere else.

Recompiled clean: 34 pages (unchanged -- this was a redistribution/
compaction pass, not a length-reduction pass per se), 98 labels, zero
missing cross-references, zero new warnings.

## Update 2026-09-13, hard freeze round: correctness + Figure 1 + last trims

User imposed a hard freeze rule: fix correctness, fix readability, add no
new science unless a claimed result turns out to be false. Nothing here
is a new result; verified each claimed issue against the actual code/text
before touching anything (two of the three "corrections" from the prior
review round had already turned out to be wrong -- see the 2026-09-13
"precision pass" entry above -- so nothing was taken on faith this round
either):

- **Corollary 5 wording**: "so at that point there is no local signal
  toward $\bm{T}_g$ at all" overclaimed -- the corollary is a first-order
  (gradient) statement, not a claim that literally no signal of any kind
  survives. Changed to "no first-order rule-directed signal toward
  $\bm{T}_g$."
- **Section 6 / Table 3 mismatch**: prose said the protocol reports "the
  normalized projection $\langle g_{\rm local},g\rangle/\|g_{\rm
  local}\|^2$," but Table 3 has reported the norm ratio $\|g\|/\|g_{\rm
  local}\|$ since an earlier round -- the prose sentence was never updated
  to match. Fixed to say "the norm ratio ... which reflects magnitude
  rather than only angle."
- **Proposition 15 algebra**: checked -- the $(1-a)/(K-1)$ vs $(1-a)/K$
  wrong-successor-mass typo was already fixed in a prior round
  (appendix_theory.tex). No action needed.
- **Figure 1 linestyle**: the caption had already been edited in an
  earlier round to describe the outcome-only baseline and the trace-step
  curve as sharing a linestyle (both dotted, distinguished only by
  color) -- but the actual plotting code
  (`src/experiments/plot_paper_figures.py::outcome_line`) still drew the
  baseline with `ls=":"`, identical to the trace-step curve. Changed the
  baseline to dash-dot (`ls="-."`), regenerated
  `figures/boolean_reliability.{pdf,png}` via
  `python experiments/run.py plot-paper-figures` (note:
  `python -m src.experiments.plot_paper_figures` has no `__main__` guard
  and silently does nothing), and rewrote the caption to state the
  linestyles directly instead of describing the earlier ambiguity.
  Visually confirmed in the recompiled PDF (page 5): the baseline is now
  clearly dash-dot orange against the dotted blue trace-step curve.
- **Section 5.3 trim**: cut the closing sentence, which re-stated (for
  roughly the third time in the paper) that the diagnostic is not
  verification of the measured Transformer transition -- Section 6's own
  Scope paragraph and the Discussion already carry that caveat. Also
  fixed a missed vocabulary-standardization instance in the same
  subsection ("one fixed affine path in the tabular kernel" -> "... in the
  shared transition-table model").

Recompiled clean: 34 pages, 98 labels, zero undefined/multiply-defined
references, zero missing cross-references (checked programmatically).
Underfull-hbox warnings are the same pre-existing set as before this
round (appendix_architectures, appendix_theory, appendix_results,
main.bbl) -- none introduced by these edits.

## Update 2026-09-14: appendix front matter restyled after Prospect (Mehta
## et al., 2023) as a formatting reference

User shared "Distributionally Robust Optimization with Bias and Variance
Reduction" (arXiv:2310.13863) and asked for the appendix to be reformatted
in that style -- readability only, no science changed, consistent with the
still-active hard freeze rule.

- **Table of Contents**: added right after the existing "Organization"
  paragraph in main.tex, before the appendix content is `\input`. Built
  manually with `\ref`/`\nameref`/`\pageref` per entry (no new package),
  matching Prospect's own appendix ToC page. Required adding `\label`s to
  seven subsections that had none before (`app:tangent-projections`,
  `app:factorization-proof`, `app:population-stationarity`,
  `app:affine-path-proof` in appendix_theory.tex;
  `app:supervision-comparison-protocol`, `app:reliability-sweep-protocol`,
  `app:lora-protocol` in appendix_results.tex) -- pure navigation aids, no
  prose changed at those sites beyond the label itself.
- **Summary of Notation table**: added as a new, first appendix section
  (`app:notation-summary`, now Appendix A; every other appendix section
  shifts down one letter automatically via existing `\cref` labels, so nothing
  else needed updating) -- a 25-row two-column table of every symbol used
  across the three appendix files, mirroring Prospect's Appendix A / Table 1.
- **Two more missed vocabulary-standardization instances**, found while
  building the notation table's cross-references: appendix_theory.tex's own
  section title ("Proofs for the Shared Transition-Kernel Results" ->
  "...Shared Transition-Table Model") and its closing sentence ("changes
  sign in the tabular kernel" -> "...in the shared transition-table model").
  Also fixed in appendix_results.tex: a table caption said "Shared kernel,
  empirical optimum..." and "The kernel column reports..." -- both changed
  to "Shared transition-table model" for consistency with the rest of the
  paper. Left `\ell_{\rm kernel}` (the proof symbol) and
  `prop:shared-kernel-embedding`/`app:shared-kernel-embedding` (internal
  labels) untouched, per the standing rule that symbols and labels are not
  prose.

Recompiled clean: 34 pages -> 36 pages (ToC + notation table add two
pages), 105 labels (was 98; +7 new subsection labels, +1 table label), zero
undefined/multiply-defined references, zero missing cross-references
(checked programmatically), no new warnings beyond one expected underfull
hbox in the ToC block. Visually confirmed both new pages render like
Prospect's (clickable dotted-leader ToC; booktabs notation table).

## Update 2026-09-14, later: Section 6 scope sentence (no new science)

User compared the paper against Makkuva (ICLR 2025) and Prospect on an
eleven-axis scorecard; the one axis scored meaningfully lower than both
("Theory <-> experiments", 8.0 vs 9.3/9.1) restates a gap already
identified and already declined to close with new experiments earlier
this same day (see the two updates above and the preceding conversation):
the paper's theorems are proved in shared transition-table coordinates,
while Section 6 measures a full, unrestricted Transformer's parameter
gradient -- not the same object, unlike Makkuva's or Prospect's theorems,
which are proved about the exact thing their experiments measure.

Declined again to add a matched experiment in the embedded-Transformer
subspace or a Jacobian pullback of the trained model's gradient into
$P_g$-coordinates: both are new science, barred by the standing hard
freeze, and the pullback specifically has a technical problem -- the
Jacobian in `cor:gradient-pullback` is only proved valid at one fixed,
hand-built routing $\psi^\star$, not at whatever routing the actual
trained two-block GPT settles into, so pulling the trained model's
gradient through it would not obviously be more rigorous than what
Section 6 already does.

Instead, made the existing honesty explicit in prose. Section 6's opening
paragraph (body.tex) now states directly that its theorems are about
$\operatorname{Rule}(-\nabla_{P_g}L)$ in shared transition-table
coordinates, that `app:shared-kernel-embedding` shows those coordinates
are an exact Transformer subspace only at one fixed hand-constructed
routing, and that this section is therefore a transfer test of the
theorem's qualitative signature, not a same-object verification of it --
the same distinction the closing paragraph already gestured at
("not a claim that the Transformer follows the model's dynamics"), now
stated up front rather than only at the end.

Recompiled clean: zero undefined/multiply-defined references, zero
missing cross-references (checked programmatically), no new warnings
beyond the same pre-existing baseline set.

## Update 2026-09-14, later still: proofs rewritten explicit; notation
## table removed

User compared page count against Makkuva (27 pages) and observed that
despite being longer, this paper's proofs read as more "handwavy" --
compressed into `\proofstep{}`-tagged fragments rather than fully worked
derivations. Two changes, both reversing earlier compaction-pass
decisions from this same day at the user's explicit request:

- **Notation table removed** (`app:notation-summary`, added earlier
  today) along with its Table-of-Contents entry in main.tex. The ToC
  itself is kept.
- **Five proofs rewritten fully explicit**, replacing the `\proofstep{}`
  bullet-fragment style with continuous "Step 1 / Step 2 / ..." prose
  that shows every substitution, in appendix_theory.tex:
  - `app:factorization-proof` (Theorem 1 + Corollary 2): now derives
    $\nabla_{P_t}p_y$ entrywise, shows the two-sided $\Pi$-projection
    algebraically (not asserted), proves $\|\Pi e_i\|_2^2=1-1/K$ by
    direct entry-sum rather than citing it, and gives the full
    induction (base case + inductive step, with every cross term
    $UE_j$, $E_jU$, $UU$ justified) for $P_1\cdots P_r=U+E_1\cdots E_r$
    and its consequence for the forward/backward residual bound.
  - `app:population-stationarity` (Theorem 3): separates the "vanishes
    identically for $t<D$" argument from the "vanishes only after
    averaging over $s_D$" argument for $t=D$, which the compact version
    had collapsed into one step.
  - `app:affine-path-proof` (Proposition 4): full induction for the
    affine-path product formula (previously asserted directly), with
    the $D$-th-derivative claim justified from the Taylor remainder.
  - `prop:mixture-gradient`'s proof: written out via the total-expectation
    identity with $B$ as an explicit indicator, rather than described
    in prose only.
  - `prop:noisy-process-credit`'s proof: same treatment as the two
    theorems above.
  Left appendix_architectures.tex's `thm:realizability` as a "Proof
  sketch" untouched -- that file's own banner comment documents it was
  deliberately cut down from a full ~8-page construction earlier in the
  paper's history, and reversing that is a separate decision the user
  hasn't asked for yet.

No mathematical content changed -- every claim, constant, and inequality
is identical to the compact version; only the shown derivation is
longer. This directly trades away part of the earlier "make it 20-25%
simpler" compaction goal, which is an explicit, informed choice by the
user in exchange for reviewer-facing rigor, not an oversight.

Two long single-line `\[...\]` displays (both already present, now newly
overfull because the preceding prose reflow shifted the equation onto
one unbroken line each) were converted to `align*` blocks with an
explicit line break, in the depth-bound proof and in
`prop:mixture-gradient`'s proof.

Recompiled clean: 38 pages (was 36 with the notation table, 34 before
this round's proof expansion started), zero undefined/multiply-defined
references, zero missing cross-references (checked programmatically),
zero overfull/underfull warnings beyond the same pre-existing baseline
set. Visually spot-checked the rewritten `app:factorization-proof`
pages -- renders as continuous, fully-justified prose with no line
overflow.

## Update 2026-09-14, final for now: made the g_local <-> Rule(-grad_P L)
## bridge explicit in Section 6 (no new experiment)

Revisited the Makkuva-comparison discussion once more. Conclusion this
round: the cosine-similarity diagnostic already IS the right bridge for
the directional claim (process stays rule-directed, outcome does not);
no embedded-Transformer experiment or Jacobian pullback is needed for
that claim, only for the stronger, unclaimed magnitude statement
($\|\operatorname{Rule}(-\nabla L)\|\lesssim\varepsilon_{\rm rule}^{D-1}$),
which the paper is not asserting transfers.

Added one paragraph to body.tex, right before Table 3
(`tab:gradient-alignment`): "Why $g_{\rm local}$ is the right comparison
point." It states plainly that $\operatorname{Rule}(-\nabla_{P_g}L)$
isolates the component of a gradient that moves one source-dependent
transition toward the true rule; that a Transformer has no canonical
$P_g$-axis to project onto, so $g_{\rm local}$ (the oracle local-transition
gradient) is how that same question is operationalized; and that cosine
similarity with it asks about direction only, not magnitude comparability
across the table and Transformer parameterizations -- which is exactly
why the norm ratio is reported as its own row rather than folded into
the cosine. This is additive to, not a repeat of, the opening-paragraph
scope statement added earlier today (which covers the theorem-object vs
experiment-object distinction at the section level); this new paragraph
covers the narrower question of why $g_{\rm local}$ specifically was
the chosen operationalization.

Recompiled clean: 38 pages (unchanged), zero undefined/multiply-defined
references, zero missing cross-references, zero new warnings. Visually
confirmed the new paragraph renders correctly immediately before the
table.

## Update 2026-09-14, cosmetic: Figure 1 simplified, SD as shaded band,
## legend repositioned

User asked for Figure 1 (`boolean_reliability`, in
`plot_boolean_reliability_only()`,
src/experiments/plot_paper_figures.py) to be cleaner: keep only answer
accuracy, outcome-only, and chance (drop the exact-trace and
free-running trace-step series), replace per-point error bars with a
shaded standard-deviation band, and fix the legend placement, which
previously sat directly on top of the flat outcome-only/chance lines at
`loc="lower right"`.

Changes: replaced the three-series `errorbar` + `scatter_seeds` loop
with a single `ax.plot` (mean) + `ax.fill_between` (mean +/- 1 std) for
answer accuracy only; dropped the exact-trace and trace-step series and
their scatter points entirely; moved the legend to `loc="upper left"`,
which is empty space now that only one rising curve remains. Regenerated
via `python experiments/run.py plot-paper-figures` (through `uv run`,
since a bare `python`/`python3` invocation isn't on PATH in this
shell -- `uv run experiments/run.py plot-paper-figures` is the working
form). Updated Figure 1's caption in body.tex to match: no more "error
bars"/"faint points"/"step accuracy measured on generated prefixes"
sentence, since those series and markers no longer appear.

This is styling only -- the underlying CSV
(`results/reliability_sweeps/boolean_circuit_8_phase_20260823_151153.csv`)
is untouched, so its ARTIFACT_MANIFEST.md hash entry (which hashes the
source CSV, not the rendered figure) did not need updating. Table 4 in
appendix_results.tex still reports the full final-answer/exact-trace/
generated-step breakdown per condition; only the main-text figure is
now visually simplified to the three series the surrounding prose
actually discusses.

Recompiled clean: 38 pages (unchanged), zero undefined/multiply-defined
references, zero missing cross-references, no new warnings. Visually
confirmed the new figure: single orange line with a shaded band, flat
dash-dotted outcome-only line, dashed grey chance line, legend in the
open upper-left corner not overlapping any series.

## Update 2026-09-14, density pass: main text made less dense per
## paragraph (no content removed, only relocated or resequenced)

User's diagnosis this round: the paper's remaining readability problem
is density (too many jobs per paragraph), not organization or missing
structure -- and explicitly not the mathematics itself, which should
stay exactly as rigorous. Also asked to simplify presentation of some
inline math into display form where it aids reading. Applied edits
across the main text, still no new science, still no proof/theorem
content changed:

- **Abstract**: split three overloaded sentences (mechanism + suppression
  + vanishing; convergence + general principle + two instances;
  main claim + qualification) into one-claim-per-sentence form. No
  content removed.
- **Introduction**: removed the Proposition 15 / competence-threshold
  preview ("$1/K$ itself is only the threshold's value...a stalling
  condition on further progress, not a discount") from the "why training
  still succeeds under wrong traces" paragraph -- it required `a`, `beta`,
  and the affine path, none of which are defined yet at that point in the
  paper. That discussion already lives in full in Section 5.3 and
  Appendix C.7; nothing was deleted, only de-duplicated. Shrank the
  finite-sample paragraph from ~12 lines (population-vs-count restatement,
  the $\rho\approx0.15$ vs $1/K\approx0.0625$ numbers) to 2 sentences,
  pointing to Section 5.4 for the numbers instead of pre-stating them on
  page 2. Converted the two-flip losses from a dense inline sentence to
  a display equation (`eq:two-flip-losses`), mirroring the probabilities
  display right above it.
- **Section 3's supervision-comparison discussion**: cut the Welch's-test
  statistical detail (t/df/p, variance-ratio explanation) from the main
  text down to 2 sentences stating the two numbers and pointing to the
  appendix; moved the full statistical paragraph, verbatim, into
  `app:supervision-comparison-protocol` in appendix_results.tex as a new
  "The clean-process discrepancy between the two runs" paragraph. Nothing
  was cut, only relocated.
- **Section 5.3**: split the single paragraph (setup + result +
  interpretation + caveat all together) into three short paragraphs:
  motivation, the proposition's statement, then the reading/caveat. Same
  content, same crefs, same equations.
- **Section 6 opening**: restructured from one dense paragraph carrying
  the coordinate-system distinction, the embedding caveat, the diagnostic
  question, all three gradient definitions, and the full protocol at once,
  into three shorter paragraphs (what the theory studies vs. what an
  unrestricted Transformer allows testing; the three gradient definitions;
  the checkpoint/seed/architecture protocol), then the existing "why
  compare with g_local" paragraph, shortened to match the tighter
  phrasing the user asked for. Every defined quantity, caveat, and
  protocol detail from the original is still present -- reordered into
  shorter units, nothing dropped.
- **Related Work**: replaced the defensive "what we did not find already
  derived is..." framing around Proposition 15 with a direct "Our
  contribution is X, together with Y. Proposition 15 gives an additional
  pathwise consequence when Z" statement. Same claim, same crefs, calmer
  register.
- **Discussion**: merged two sentences that separately restated "does not
  imply the Transformer follows the model's dynamics" into one, removing
  one redundant clause.
- **Vocabulary standardization**: last remaining instances of "shared
  transition-table abstraction," "shared-table analysis/objectives," and
  "substrate" -> "shared transition-table model" (or "this model"),
  matching the vocabulary already standardized elsewhere in the paper.
  Caught one more miss during a visual proofread of the rendered PDF:
  "manifest qualitatively outside that abstraction" -> "...outside that
  model."

Recompiled clean: 37 pages (down from 38 -- net word-count reduction
despite the appendix statistical paragraph being added, since it's
shorter than what it replaced in the main text), zero
undefined/multiply-defined references, zero missing cross-references,
no new warnings. Visually confirmed the rewritten abstract and Section 6
opening render cleanly with the intended paragraph breaks.

## Update 2026-09-14, correctness fix: magnitude-suppression language was
## a real overclaim; plus two more density-pass edits

User's next round of feedback flagged (among five items) that
"we observe... not the magnitude suppression the mechanism predicts"
overclaims: the theorem's magnitude prediction is about
$\|\operatorname{Rule}(-\nabla_{P_g}L)\|$, a **projected** quantity in
shared transition-table coordinates, but what Section 6 measures is the
full, **unprojected** Transformer gradient norm ($\|g_{\rm out}\|$). These
are not the same object. Verified this independently before touching
anything: with $\cos(g_{\rm local},g_{\rm out})\approx0$, the
rule-aligned *component* of $g_{\rm out}$ (which is what the theorem's
magnitude claim is actually about) could be small even while the total
norm is huge -- a near-zero cosine times a huge norm does not pin down
the aligned component at all. So the paper had tested a *different*,
coordinate-mismatched quantity and reported it as if it refuted the
theorem's specific magnitude prediction. Fixed in three places (the
proved/measured/open table caption, Section 6's closing paragraph, and
Discussion): all now say the projected magnitude has no coordinate axis
to measure in an unrestricted Transformer and remains untested, rather
than implying it was tested and found not to hold. Confirmed via grep
that no other main-text instance makes the same overclaim (the
remaining "magnitude" mentions all correctly describe the measured
full-gradient norm ratio without conflating it with the theorem's
object).

Also implemented, from the same round of feedback, two low-risk items:
- **Abstract ending**: replaced "a gap we report rather than explain"
  with a synthesizing sentence ("The resulting picture separates an
  exact population mechanism...from the learner-dependent location of
  the empirical reliability frontier") -- same scope, no claim change.
- **Limitations trimmed**: removed the rerun-status/archived-logs/
  numerical-gap sentences, which duplicated the Reproducibility
  statement in main.tex verbatim; replaced with one pointer sentence.
  Limitations now reads as purely scientific (model-vs-network gap,
  synthetic tasks, frontier depends on many factors, differing
  gate-sampling), matching what a Limitations section should contain.

Declined, from the same message, without implementing:
- Changing the intro's "learning effect, not information effect"
  sentence -- re-derived the claim myself and it is a precise,
  narrowly information-theoretic statement (the answer is a
  deterministic function of the prompt, so the trace adds zero Shannon
  information about it), not a claim that credit-placement is the only
  mechanism; Setup's "changes three things at once" describes
  mechanisms and does not contradict it.
- The RegisterMachine16 figure -- already explicitly declined twice
  this session; standing by that decision.
- Removing "a statement of genre" / moving the Makkuva comparison to
  Related Work -- this framing was deliberately added earlier this same
  session specifically to preempt "why is this just Makkuva" reviewer
  pushback, after real back-and-forth; reversing it on style grounds
  alone was declined pending explicit confirmation.

Recompiled clean: 38 pages, zero undefined/multiply-defined references,
zero missing cross-references, no new warnings.

## Update 2026-09-14, investigation: reliability-sweep discrepancy

User asked to investigate (rather than immediately pick a resolution)
why the fresh 5-seed sweep (rerun today, extended to rho in
{0,0.1,0.2} plus the original {0.30,...,1.00}) shows systematically
10-16 points LOWER answer accuracy than the archived Aug-23 CSV at every
rho >= 0.80, while agreeing within noise at rho <= 0.50. A
one-directional shift across five different rho values is not what pure
seed variance looks like.

Ruled out: hyperparameter drift in the *documented* protocol -- the new
run's persisted config (train_size, val_size, seeds, checkpoints, etc.)
matches the paper's own stated protocol and the CLI defaults exactly.
Could not get a clean file-level diff against the archived run's code,
since `src/eval/reliability.py`, `src/training/loop.py`,
`src/training/optim.py`, and the task sampler were all fully rewritten
during the "Refactor code structure and optimize performance" commits
(Sep 9-13) -- there is no archived persist.json to compare against
either (only the CSV was hashed into ARTIFACT_MANIFEST.md).

Hypothesis 1, bf16 default -- RULED OUT. `--bf16` now defaults to
`True` (`add_compile_bf16_flags` in `src/training/seed.py`); tested by
rerunning 5 seeds at rho=1.00 with `--no-bf16`, everything else
identical to the fresh sweep. Result: mean 83.80% (std 2.98), 
statistically indistinguishable from the fresh bf16=True run's 84.08%
(std 5.83), and still ~9 points below the archived 92.61%. Precision is
not the cause.

Hypothesis 2, task-sampler/registry drift -- RULED OUT. Used git
archaeology to find the code that actually produced the archived CSV:
`git log --since=... --until=...` located commit `3cb5ff9` (2026-08-23
13:51:52), the commit immediately before the archived CSV's timestamp
(20260823_151153), which introduces a top-level `sweep_ratio.py` as the
then-current entry point (the modular `src/eval/reliability.py` did not
exist yet). Diffed `git show 3cb5ff9:src/boolean_circuit_tasks.py`
against current `src/data/boolean_circuit_tasks.py`, and
`git show 3cb5ff9:src/registry.py` against current
`src/data/registry.py`: only cosmetic differences (import paths,
docstrings, whitespace, combined asserts, extra unrelated depths added
to the tuple). `boolean_circuit_8`'s block_size/max_new_tokens/chance_acc
and gate-sampling logic are byte-for-byte equivalent in substance. Not
the cause.

Hypothesis 3, train_size default -- CURRENTLY TESTING, leading
candidate. The old `sweep_ratio.py` (commit `3cb5ff9`) defaults to
`--train-size 100_000 --val-size 2_000`, whereas the current CLI
(`src/__main__.py`) defaults to `--train-size 20000 --val-size 500`
-- a 5x difference in unique training examples. The fresh sweep (and
`RUN.md`'s own worked example, which explicitly passes
`--train-size 20000`) used 20,000; there is no persist.json for the
archived run to confirm what it used, but the timing (CSV timestamp is
~80 minutes after `sweep_ratio.py` was added) is consistent with a
first run made against that script's bare defaults, i.e. 100,000. More
unique training examples at fixed step budget (8000 steps x batch 128)
means less repetition and broader coverage of the depth-8 gate space,
which could plausibly buy several points of accuracy specifically at
high rho, where the model must apply gates precisely rather than
pattern-match. Launched a direct test: 5 seeds x rho in {0.80, 1.00}
with `--train-size 100000 --val-size 500`, everything else identical
to the fresh sweep, on `results/reliability_sweeps/diag_trainsize100k.csv`.
Result: CONFIRMED as the primary driver.

| rho  | diag (train=100k) | fresh (train=20k) | archived      |
|------|--------------------|--------------------|----------------|
| 0.80 | 59.88 +/- 20.25%   | 45.96 +/- 14.24%   | 61.98 +/- 21.41% |
| 1.00 | 88.96 +/- 3.06%    | 84.08 +/- 5.83%    | 92.61 +/- 2.90%  |

At rho=0.80, train_size=100k alone closes the gap entirely (59.88 vs
61.98, well inside the +/-20-point seed noise at this rho). At
rho=1.00, it closes most of the gap (moves from 8.53 points below
archived down to 3.65 points below, versus a per-condition std of
~3 points) -- the residual is small enough that it does not obviously
demand a fourth hypothesis, but is not fully pinned down either
(candidates not yet tested: val_size 500 vs whatever the archived run
actually used, or ordinary residual variance at this specific
seed/rho corner).

Conclusion so far: the archived Aug-23 numbers were almost certainly
produced with `sweep_ratio.py`'s bare defaults (train_size=100,000),
not the 20,000 documented in `RUN.md` and used by the current CLI's
default. This is a real, identified protocol drift between the
archived paper numbers and the current documented/default
reproduction recipe -- not measurement noise, not a bug in either
code path, and not evidence against any paper claim. Reported to the
user with this finding; no paper numbers touched pending their
decision on how to proceed (adopt fresh 20k-protocol numbers and
update RUN.md to describe them as canonical going forward, rerun at
train_size=100k to match archived exactly, or keep archived numbers
and add a footnote documenting the train_size discrepancy).

## Update 2026-09-15: theorem-count restructuring and operator-induction framing

Explicit user request, following a plagiarism/idea-originality check (no
issues found) and a discussion of whether to write the theory natively in
Transformer parameters (declined -- would sacrifice exactness and
generality) and a proposed parameter-space bridge (discovered already
built, appendix_architectures.tex's shared-kernel-embedding section).

Reduced the paper from 15 to 12 numbered theorem-like environments, and
switched to two-tier numbering: main text keeps a plain shared counter
(Theorem 1, Corollary 2, ...); appendix results now use
`\counterwithin{theorem}{section}` (added right after `\appendix` in
main.tex), so they read as A.1/A.2/A.3 (Appendix A) and B.1/B.2/B.3
(Appendix B) instead of continuing as "Theorem 9 ... Proposition 15."
Every cross-reference in the paper goes through `\cref`, so this required
no manual reference-number updates, only label-level changes where
environments were merged, moved, or demoted:

- `prop:outcome-nonidentifiability` moved from body.tex into
  appendix_theory.tex (app:operator-proofs), right after the
  factorization/depth-bound proof -- same label, zero other cross-refs,
  now renders as Proposition B.1. Body.tex keeps a one-sentence pointer.
- `prop:mixture-gradient` demoted: environment removed, content kept as
  an unlabeled paragraph + \cref{eq:mixture-gradient} (its one external
  reference, in the noise-threshold proof, repointed to the equation
  label instead of the retired proposition label).
- `prop:shared-kernel-embedding` + `cor:gradient-pullback` merged into one
  three-part proposition (i: embedding, ii: gradient pullback, iii:
  threshold transfer) under the surviving `prop:shared-kernel-embedding`
  label -- both proofs kept, just no longer separated by a restated
  corollary block. Now Proposition A.3.
- `cor:clean-convergence` retyped from `corollary` to `remark` (same
  label, zero reference updates needed) -- correct but secondary to the
  main corruption-recovery chain. Now Remark 5.
- `cor:noise-threshold` + `cor:general-corruption` merged into one
  `theorem` (label `cor:noise-threshold` survives, since it had ~19
  cross-references vs. `cor:general-corruption`'s ~5; the general
  threshold formula and its two named special cases now sit inside the
  same environment as the $\bm Q_g$-convergence and greedy-recovery
  claims, matching how `cor:general-corruption`'s own text already said
  it subsumed the other two numbers). Now Theorem 6. All ~5 stray
  references to the retired `cor:general-corruption` label repointed to
  `cor:noise-threshold` or, where more precise, to
  `eq:general-corruption-threshold` directly.

Also rewrote `sec:operator-credit`'s opening: $\bm P_g$ is now introduced
as the context-independent regime of a generic learner's induced
next-state conditional (Transformer as the motivating, architecturally-
verified example, not a restriction of the definition -- deliberately
worded to avoid narrowing the theory's stated architecture-agnosticism),
with a new main-text roadmap equation (`eq:chain-rule-roadmap`,
$\nabla_\theta L=\sum_g J_g(\theta)^\top\operatorname{vec}(\nabla_{P_g}L)$)
forward-pointing to the appendix bridge and to the Section 6 measurement.

Recompiled clean: 38 pages (unchanged), zero undefined/multiply-defined
references, only pre-existing underfull-hbox warnings (no new overfull
boxes). Verified via pdftotext that the numbering renders exactly as
intended: main text Theorem 1 / Corollary 2 / Theorem 3 / Proposition 4
/ Remark 5 / Theorem 6; Appendix A Theorem A.1 / Lemma A.2 / Proposition
A.3; Appendix B Proposition B.1 / B.2 / B.3.

## Update 2026-09-15: gradient-alignment diagnostic, robustness + per-group breakdown

User asked to "check the mechanistic prediction more robustly" (referring
to Section 6's gradient-alignment diagnostic, currently n=3 seeds with
some very wide per-checkpoint std, e.g. norm ratio 47.7 +/- 38.9). Backed
up the archived 3-seed CSV/persist.json to
results/gradient_alignment_archived_3seed/ first (gradient_alignment.py's
write_csv overwrites its fixed output path unconditionally, no --output
flag) -- verified the backup's sha256 matches ARTIFACT_MANIFEST.md exactly
before touching anything.

Reran the same diagnostic (same task/architecture/budget) extended to 10
seeds (2001-2010, i.e. the original 3 plus 7 new). Result: the new 10-seed
aggregate is close to the archived 3-seed one (e.g. norm ratio at step
8000: archived 79.1+/-26.5 vs new 78.7+/-22.8), so this is not a
discrepancy case like the reliability-sweep one -- it strengthens the same
claim rather than contradicting it. Computed proper statistics (saved to
results/gradient_alignment/gradient_alignment_stats_10seed.json):
paired t-test cos(local,proc) vs cos(local,out) is significant at every
checkpoint (p from 6e-11 to 8e-8), and a one-sample t-test confirms
cos(local,out) is NOT significantly different from zero at steps
2000/4000/8000 (p=0.28/0.15/0.93) after being significantly positive at
init (p<0.001) -- this makes the "collapses toward zero" claim in the
paper's prose a tested statistical fact rather than an eyeballed 3-sample
mean. Not yet incorporated into body.tex/tab:gradient-alignment; reported
to the user, pending a decision on whether to update the table to n=10
with the added significance tests.

Also acted on a separate, related request: implemented and ran a
per-parameter-group breakdown of the same diagnostic
(src/experiments/gradient_alignment_by_group.py, new file, does not touch
the archived gradient_alignment.py) -- computes the same cosine/norm-ratio
stats separately for W_Q, W_K, W_V (row-sliced from the fused
`c_attn.weight`, verified against src/models/gpt.py's actual named
parameters), W_O, W_1, W_2, W_U (lm_head), embeddings, and LayerNorm,
instead of one fully-flattened vector. Motivated by a critique proposing a
full from-scratch Transformer-parameter theory (backprop through
W_Q..W_U, an O(epsilon) initialization-scale gradient hierarchy, a W_Q=K=0
saddle, a W_U=0 symmetry-breaking cascade, and a generic D-1-exponent
analogue) -- verified each claim by hand: the backprop algebra and the
O(epsilon) hierarchy are correct and not previously in the paper, the
Q=K=0 saddle is correct but a known deep-linear-network phenomenon (not
novel), the W_U-zero-gradient claim substantially duplicates the existing
Proposition A.3 (eq:exact-grad-identity/eq:logit-gradient-at-u), the
W_U=0 cascade is correct but describes a degenerate exact-zero-init
special case not the realistic small-random-init regime, and the generic
D-1 exponent argument assumes rather than derives geometric contraction
for actual Transformer Jacobians (unlike Corollary 2, which derives it
from P_g's row/doubly-stochastic structure). Agreed plan: do not write a
new architecture-specific theorem in the main text regardless of outcome;
only consider a short subsection if the per-group experiment shows a
clean rule-directed-vs-not hierarchy (e.g. W_U/W_2 strongly aligned,
W_Q/W_K delayed/weak). 5-seed run in progress at time of writing; result
not yet known.

**Result (5 seeds, boolean_circuit_8, same checkpoints as the full-vector
diagnostic): no striking hierarchy.** Saved to
results/gradient_alignment_by_group/gradient_alignment_by_group_summary.json.
At step 8000, cos(local,proc) ranges only 0.817-0.896 across all nine
roles (embed, W_Q, W_K, W_V, W_O, W_1, W_2, W_U, layernorm) -- a spread of
0.079, small next to each role's own ~0.05-0.15 std across 5 seeds -- and
cos(local,out) similarly ranges only -0.015 to +0.071 across all roles,
uniformly near zero. The hypothesized "W_U/W_2 strongly aligned, W_Q/W_K
delayed or weak" pattern does not appear: the directional asymmetry (the
actual mechanistic prediction) shows up essentially uniformly across
every parameter group, not concentrated in the readout or MLP layers.
The norm ratio does differ substantially by role (e.g. outcome/local norm
ratio at step 8000: W_Q 60.8 vs W_U 306.7), but per the paper's own
existing framing, the norm ratio is a secondary magnitude check, not the
mechanistic claim itself (which is about direction).

Per the explicitly agreed decision rule ("only consider a subsection if
the per-group experiment shows a clean hierarchy... otherwise keep the
current P_g-space theory because it is cleaner"), this is a clean "no":
no new architecture-specific theorem or subsection, no promotion of the
W_Q..W_U parameter-gradient hierarchy to the main text. The one thing
possibly worth a single sentence (not yet added, pending user decision):
noting that the directional signature is uniform across parameter groups
rather than concentrated in one, which rules out a "readout-layer
artifact" objection to the existing full-vector diagnostic -- a minor
robustness note, not new science.

## Update 2026-09-15: readability pass, own diagnosis (not critique-derived)

User asked directly to fix readability under current scope ("taking
multiple reads"). Rather than act on the various pasted critiques'
clarity claims (several of which didn't hold up against the actual file
when checked -- e.g. "abstract qualifies before it asserts" was false
when checked against the actual text), did a fresh read of the whole
main text to find real friction points.

Found two, fixed both:
1. The intro's central mechanism paragraph ("Why does supervising the
   trace succeed...") chained six ideas (two-flip example, general-D
   forward/backward story, depth suppression, exact zero at uniform
   table, process bypass, global convergence) into one unbroken ~48-line
   block. Split into four \emph{}-labeled beats: "The mechanism.", "Why
   composition suppresses it.", "The extreme case.", "What process
   supervision changes." No content changed, no \cref removed, just
   paragraph breaks and labels.
2. Two different "how well does process supervision do" numbers (82.00%
   in tab:supervision-comparison, 92.61% in tab:legacy-circuit, a
   separate run) recur three times across the paper with different
   framing each time. The first recurrence (sec:sharp-jump) didn't
   remind the reader these were the same previously-disambiguated
   separate run; added an explicit backward pointer there.

Recompiled clean: 38 pages (unchanged), zero undefined/multiply-defined
references, no new warnings. Verified via pdftotext that the four labeled
beats render as intended.

## Update 2026-09-15: outcome-only depth/capacity sweep -- result in

User-authorized experiment (D=2,4,6,8, outcome-only condition, seeds
2001-2003, 8000 steps, train-size 20000, same architecture as everywhere
else) finished. Saved to
results/paper/depth_reliability/boolean_circuit_{D}_outcome.csv and
summarized in results/paper/depth_reliability/depth_capacity_sweep_summary.json.
Does not touch the existing archived D=2/4/8 rho>=0.5 CSVs (new filenames).

Result (chance = 6.25%):

| D | outcome mean | outcome std | process (rho=1) mean | process std |
|---|---|---|---|---|
| 2 | 88.07% | 6.99 | 97.87% | 1.72 |
| 4 | 28.87% | 6.72 | 93.47% | 3.52 |
| 6 | 11.20% | 0.92 | 90.53% | 1.79 |
| 8 |  7.80% | 1.06 | 86.93% | 6.35 |

This is the clean, striking result the capacity-confound objection
needed to be tested against, and it comes out the way the mechanism
predicts, not the way a capacity confound would. A capacity confound
("this architecture just can't express the depth-8 outcome mapping")
predicts outcome accuracy near chance at every depth, including D=2. It
doesn't: outcome-only reaches 88% at D=2, nearly matching process's
97.87% -- the same architecture clearly can learn the outcome mapping
when composition is shallow. Instead, outcome accuracy degrades smoothly
and progressively with depth (88 -> 29 -> 11 -> 7.8, converging toward
the 6.25% chance floor), while process accuracy declines only mildly
(97.87 -> 93.47 -> 90.53 -> 86.93) over the same depth range. This is
exactly the qualitative signature Corollary 2's epsilon^{D-1} outcome
suppression predicts against Theorem 6's depth-independent Theta(1)
process credit -- a progressive compositional-suppression story, not a
one-off capacity wall at D=8.

Not yet added to the paper. This is a strong candidate for a new
main-text figure/table (accuracy vs. depth, both conditions) given how
directly it answers the single most-repeated reviewer objection across
every critique in this thread -- pending the user's decision on where
and how to add it.

## Update 2026-09-16: NeurIPS 9-page restructuring (in progress), and a
## real correctness fix to the outcome-rescue diagnostic

Restructuring toward the NeurIPS 9-page main-text limit (currently 37
total pages, main text 11 of those). Verified per-page text density
(114-155 lines/page across pages 2-10, no float-driven whitespace left)
before cutting further, so remaining cuts are genuine content-density
reductions, not typesetting slack. Progress so far, all via relocating
content to its already-existing appendix home or removing genuine
duplication (not deleting any theorem, proof, number, or citation):
- Fixed 3 tables' `[H]` (forced-here) placement to `[t]`, recovering a
  full page that `[H]` was silently wasting as blank space -- the single
  biggest win, found before any content was touched.
- Compressed the introduction's mechanism walkthrough (two-flip example
  kept, the 4-beat forward/backward/depth-suppression/process-bypass
  recap cut to one paragraph deferring to Section 4, which states the
  same content formally a few pages later) and its corruption-structure
  paragraph similarly.
- Relocated Proposition 4 (order-of-improvement, prop:loss-path) and the
  clean-convergence remark (cor:clean-convergence) to sit next to their
  proofs in the appendix (already there), replacing each with a compact
  summary + pointer in the main text.
- Compressed Section 6 (gradient-alignment) opening and closing
  paragraphs, Related Work's "learning from corrupted supervision"
  paragraph, the Limitations+Discussion section (merged into one
  paragraph), and the "Finite-sample recovery" numbers paragraph +
  fig:noise-threshold's caption.
- Converted the intro's tab:proved-measured-open table to compact prose
  (tables carry padding overhead beyond their content that prose doesn't).

Result: main text 12 -> 11 pages (38 -> 37 total), body.tex 1045 -> ~890
lines. Still 2 pages short of the 9-page target; reported this honestly
to the user rather than continuing to guess, with three concrete options
(move a main-text figure/table to the appendix -- recommended,
specifically fig:noise-threshold, whose main-text payoff is already
stated in prose; trim Related Work's citation density further; or
confirm whether the actual venue limit allows 10 pages) -- awaiting the
user's choice before continuing, since further cuts start trading against
visible content rather than exposition.

Separately, verified and fixed a real correctness issue the user flagged
in app:outcome-rescue-diagnostic (prop:outcome-rescue): independently
recomputed the margin along $P_g(a)=U+a(T_g-U)$ and confirmed the true
successor exceeds every rival by exactly $a$, so greedy decoding is
already correct for any $a>0$ on this path, regardless of $\rho$. The
appendix's "full recovery under process supervision needs reliability
well above the population threshold" sentence conflated this
already-guaranteed correctness with the separate, real fact that
$\rho_c(a,D,\beta)$ governs whether gradient descent along this path
keeps making progress toward higher confidence $a$ at all (a stalling
threshold, not a correctness threshold). Rewrote the passage to state
the margin fact explicitly and reframe the $\rho$-vs-$D$ discussion as
about confidence/robustness rather than bare decodability, and added a
third caveat that the margin argument is specific to this population
table and path, not a trained Transformer's own parameters. The main-text
summary (sec:outcome-rescue) already used careful "keep making progress"
language and needed no change. Also checked the user's separate request
to "explicitly describe Appendix B.7's assumptions" (copying contributes
no derivative in $a$; corrupted prefixes are ignored when recomputing the
answer) -- both are already explicit in the existing text
(appendix_theory.tex ~871-880), so no change was needed there.
Recompiled clean, zero undefined references.

Not yet acted on: the four prioritized new experiments (matched
symmetric/coherent corruption in the trained Transformer;
process-vs-process+answer ablation; reconciling the 82%/92.6%
discrepancy with matched implementation details; gradient diagnostic at
outcome-trained checkpoints) and the optional margin-based finite-sample
theorem. These are real, substantial new experimental/theoretical work
requiring explicit authorization, not editing -- flagged to the user,
awaiting a decision on scope and ordering, and on the still-open
page-count question above.

## Update 2026-09-16: hedging-motif variation (precisely scoped, not a rewrite)

User refined the earlier "clarity=6.0 / over-hedging" claim: not that any
single caveat is wrong or that the abstract hedges before asserting
(already checked and ruled out two updates ago), but that encountering
the *same rhetorical shape* ("not a claim that X follows Y's dynamics")
five times across one read-through can read as retreat even when each
instance's content is locally justified and non-redundant. This is a
fair point the earlier per-instance check didn't address. Fixed exactly
this, nothing more: found the 5 instances (abstract, intro, sec 5.2,
sec 6 closing, discussion), left the intro's (already phrased
differently: "external validation... not verification of a quantitative
theory") alone, and reworded the other 4 so each has a distinct shape
while preserving identical scope/content:
- Abstract: "-> a claim about direction, not about whether its
  optimization follows the model's dynamics."
- Sec 5.2 (outcome-rescue): moved the caveat to *after* the finding
  instead of before it ("state the result, then scope once" -- the
  exact principle requested), one sentence: "This is a one-dimensional
  diagnostic along one fixed path, not a claim about a trained
  optimizer's trajectory."
- Sec 6 closing: "-> so what transfers here is the sign of the
  asymmetry, not the model's dynamics."
- Discussion: dropped the restatement entirely rather than rephrasing
  again, since the paragraph's own closing sentence ("predicting the
  Transformer's own quantitative optimization trajectory... remains
  open") already carries the same scope point without needing the motif
  a fifth time.
Recompiled clean: 37 pages (unchanged, as expected for a wording-only
pass), zero undefined references.

## Update 2026-09-16: three precision fixes, one of them in my own prior edit

User verified NeurIPS 2026's actual 9-page main-text limit via the
official handbook (fetched and confirmed independently rather than
trusting the pasted claim: "main text limited to nine content pages...
references, optional technical appendices and mandatory paper checklist
do not count" -- confirms the restructuring target from two updates ago
is correct) and flagged three precision issues, one of them in the
outcome-rescue fix from last update.

1. My own edit last turn overclaimed: it said "reading that frontier
   [the measured Boolean-8 accuracy curve] as measuring confident,
   robust competence" as if this were an established interpretation.
   The paper only measures binary greedy-decoding accuracy, not
   confidence or margin, so this connection is a hypothesis, not a
   fact. Fixed: reworded to "one hypothesis consistent with, but not
   established by, this diagnostic is..." and added that testing it
   would need a direct confidence/margin measurement the paper does not
   have (appendix_theory.tex, app:outcome-rescue-diagnostic).
2. Verified a real notation overclaim (pre-existing, not something
   introduced this session): cor:near-mixing's own proof explicitly
   says "this is an upper bound on projected credit; particular
   products can cancel further" (body.tex), yet two places called the
   outcome-credit decay "Theta(rulestr^{D-1})" -- a tight two-sided
   bound the paper never proves, contradicted by its own
   further-cancellation caveat. Fixed both occurrences (body.tex and
   appendix_theory.tex) to state the honest one-sided claim: process
   credit bounded below by a depth-independent constant vs. outcome
   credit bounded above by O(rulestr^{D-1}), explicitly noting the
   corollary proves a ceiling, not a matching floor.
3. Tightened one sentence in sec:operator-credit's opening that called
   the finite-attention Transformer construction realizing the
   shared-table regime "exact" without immediately distinguishing the
   parameter-space embedding (which is exact) from the computational
   equivalence (which is only approximate at finite attention score,
   already stated two sentences later as "up to a vanishing routing
   error" but not tied clearly to the word "exactly" itself). Reworded
   to state directly: approximate at any finite attention score, exact
   only in the limit.

Recompiled clean: 37 pages (unchanged), zero undefined references. The
82%/92.6% reconciliation and matched-corruption experiment remain the
open empirical-closure items, not yet started.

## Update 2026-09-16: Theorem 6 generality fix (double-stochasticity over-restricted)

A critique claimed cor:noise-threshold's blanket "assume every C_g is
doubly stochastic" hypothesis is stronger than what's actually needed --
that the row-wise recovery threshold and the convergence-to-Q_g claim
don't use double stochasticity at all, only the closed-form gradient
identity at P_g=U does. Verified this directly against the existing
proofs rather than taking it on faith:
- The row-wise argument (Q_g(i,T_g(i))=rho+(1-rho)c_T vs
  max_{j!=T_g(i)} Q_g(i,j)=(1-rho)c_star, giving rho>(c_star-c_T)/(1+c_star-c_T))
  only ever uses that C_g is row-stochastic (appendix_theory.tex's own
  proof of the general formula never invokes double stochasticity).
- The convergence claim ("gradient descent on L_proc converges to Q_g")
  is proved via row-logit convexity/separability of a sum of softmax
  cross-entropies (appendix_theory.tex, "What gradient descent reaches"
  proof block) -- fully general for any row-stochastic target law, no
  double-stochasticity or thm:complete-mixing hypotheses anywhere in it.
- Double stochasticity is used in exactly one place: simplifying
  Pi Q_g Pi into (Q_g - U) for eq:noisy-process-credit, the closed-form
  gradient-at-the-uniform-table identity specifically.
Confirmed this is a real over-restriction, not a stretch, since it's
directly checkable against proofs already in the paper -- fixing it
needed no new derivation, just correctly scoping which hypothesis
belongs to which conclusion.

Restructured cor:noise-threshold (Theorem 6, body.tex) into two tiers:
(1) the general convergence + recovery-threshold claim (including both
named special cases, symmetric rho>1/K and coherent rho>1/2) now stated
for any row-stochastic C_g, no double-stochasticity assumed; (2) the
closed-form gradient identity at P_g=U, clearly marked as needing
thm:complete-mixing's hypotheses and double stochasticity additionally.
Updated the appendix's proof opening (app:noise-threshold-proof) to
match this scoping, and the proof itself to note explicitly which step
uses which hypothesis. Recompiled clean: 38 pages (up from 37 -- this
addition works against the still-open 9-page main-text cutting target,
noted to the user), zero undefined references.

Not yet touched: the bigger proposed generalizations (approximate
shared-kernel perturbation theorem, generalizing U to an arbitrary
stationary pi) -- explicitly deferred pending the outcome of the
in-flight empirical results, per the user's own steer.

## Update 2026-09-16: proof repair for the Theorem 6 generalization

Two real gaps found in the previous update's generalization, both
verified against the actual proof text before fixing (not taken on
faith):

1. The appendix's convergence proof ("What gradient descent reaches",
   app:noise-threshold-proof) still weighted every row (g,i) by a flat
   f_g/K, which silently assumes uniform displayed-source visitation --
   itself a consequence of thm:complete-mixing's hypotheses (uniform
   s_0 + permutation T_g), the exact thing the main theorem was just
   generalized to not require. Fixed: introduced pi_{g,i} (general
   per-row visitation frequency), showed f_g/K is its special case
   under the old hypotheses, and rewrote the convexity/Hessian/step-size/
   excess-loss argument in terms of pi_{g,i}>0 (row coverage) rather
   than f_g>0 (gate coverage) -- these are genuinely different
   conditions for non-doubly-stochastic C_g, since a corruption law
   that isn't doubly stochastic can make some displayed sources at a
   given gate arbitrarily rare or unreached even when the gate itself
   fires often.
2. Q_g = rho*T_g + (1-rho)*C_g was stated in the generalized Theorem 6
   as if automatically induced by any whole-trace rho-reliable
   corruption process -- but the appendix's own existing text (already
   present before this fix) already showed this only holds exactly
   when C_g is doubly stochastic (else the displayed source's law can
   depend on whether the trace was corrupted, decoupling row-local
   rho from the global rate). Fixed: reworded the theorem to state
   this equation as the definition of the local reliability rho
   directly (matching the whole-trace rate only in the doubly-stochastic
   case), and cross-referenced the appendix's existing derivation
   explicitly instead of leaving the connection implicit.

A third claimed gap (intro conflating "converges to Q_g" with "correct
recovery") was checked and found already handled correctly in both the
abstract and the intro paragraph (each already states these as two
separate clauses, unconditional convergence vs. threshold-gated
recovery) -- no change made there.

Recompiled clean: 38 pages (unchanged), zero undefined references.

## Update 2026-09-16: two empirical-closure experiments finished

**82%/92.6% discrepancy, narrowed but not fully closed.** Four data
points now, all boolean_circuit_8/process at 8000 steps:
| config | seeds | mean |
|---|---|---|
| original Table 1 | train=20k, 2001-2005 | 82.00% |
| diag | train=100k, 2001-2005 | 85.31% |
| diag | train=100k, {2001,2002,2020,3000,2026} (archived seeds) | 88.41%+/-4.11 |
| archived Table 2 source | (2001,2002,2020,3000,2026) | 92.61%+/-2.90 |
Both train_size (100k vs 20k) and seed identity (the archived run's
particular seed draw vs the sequential 2001-2005 convention) are real,
verified contributing factors -- together they close about 6.4 of the
10.61-point gap (~60%). ~4.2 points remain unexplained; not chasing this
further without a specific new hypothesis, since val_size/eval_batch_size
differences were already reasoned to not plausibly cause a systematic
accuracy shift. Reporting as "two identified contributing factors,
partially closed" rather than "resolved," which is the honest
characterization.

**Matched symmetric-vs-coherent corruption in the trained Transformer**
(boolean_circuit_8, identical architecture/seeds/budget/train_size=20k,
rho in {0.3,0.5,0.7}, 3 seeds each; chance=6.25%):
| rho | symmetric | coherent | diff | t-test p (n=3 each) |
|---|---|---|---|---|
| 0.3 | 6.87+/-1.03 | 7.73+/-0.58 | -0.87 | 0.271 |
| 0.5 | 11.20+/-4.87 | 9.20+/-1.91 | +2.00 | 0.544 |
| 0.7 | 29.73+/-2.21 | 24.33+/-5.26 | +5.40 | 0.177 |
Direction is mostly consistent with the prediction (symmetric >= coherent
at rho=0.5 and 0.7, reversed at rho=0.3 where both sit near chance and
neither corruption law's population threshold is comfortably cleared:
symmetric's 1/K=6.25% barely is, coherent's 1/2=50% isn't). None of the
three differences reach significance at n=3 per condition -- this is
suggestive, directionally-consistent evidence, not a confirmed effect.
Unlike the tabular study's dramatic gap (100% vs 5.7% at matched rho),
the trained-Transformer effect size is modest. Would need more seeds to
say anything statistically decisive; not run yet, pending user interest.

Both results reported to the user precisely, including the honest
non-significance caveat on the corruption comparison and the ~40%
unexplained residual on the discrepancy -- not oversold as fully
resolved.

## Update 2026-09-16: completeness audit of the Theorem 6 generality fix

User asked to make the generality/claim fix across the manuscript and
appendix proofs complete, not just the theorem statement itself. Did a
systematic grep-driven audit of every "doubly stochastic" and
cor:noise-threshold reference across all four .tex files (not just a
spot check) to find any place still reflecting the old, narrower scope:

- appendix_architectures.tex's two cor:noise-threshold references
  (Proposition A.3's gradient-pullback/threshold-transfer): both already
  explicitly invoke "cor:noise-threshold's symmetric-corruption setting"
  and thm:complete-mixing's hypotheses "additionally" -- already correctly
  scoped to the doubly-stochastic branch, no change needed. Also found
  this file already defines and uses pi_{g,i}:=D^{-1} sum_t Pr(g_t=g,s_{t-1}=i)
  (line 363) for exactly the same concept my appendix_theory.tex fix
  introduced -- confirms the notation choice matches an existing
  convention rather than introducing a new one.
- appendix_results.tex's reference (numerical confirmation of "closed-form
  claims about a Rule-projected gradient at a stated P_g"): still accurate,
  since that appendix only spot-checks the doubly-stochastic closed-form
  identities, not the general convergence claim, which isn't a
  closed-form-at-one-point statement anyway.
- body.tex's Related Work discussion of cor:noise-threshold's rho>1/K
  symmetric threshold and "corruption-structure functional": still
  accurate and if anything now understates the result (functional is
  more general than stated, which is a safe direction).
- The intro's proved/measured paragraph, the abstract, and
  cor:clean-convergence (the rho=1 special case, deliberately left with
  its simpler f_g>0 hypothesis since it doesn't itself claim generality
  beyond the standard uniform-visitation model): all already consistent,
  no changes needed.

No further inconsistencies found. Recompiled clean: 38 pages, zero
undefined references; spot-checked via pdftotext that the generalized
theorem's key phrases ("any row-stochastic corruption law", "row
coverage pi_{g,i}>0") render correctly in the final PDF. Considering the
generality/claim-fix task complete as of this update.

## Update 2026-09-16: cross-task validation added (register/state machines)

User asked to show the same qualitative phenomenon holds for
register_machine and state_machine, not just boolean_circuit. Found
existing archived data for both (register_machine_16_seeds_2001-2003.csv,
several state_machine_16_phase_*.csv files) but confirmed via
ARTIFACT_MANIFEST.md that none of it is hash-tracked -- exactly the
"orphaned data" provenance problem already flagged earlier this session
as grounds for keeping register/state-machine results out of the paper.
Did not reuse it. Instead checked task structure (both are bijective/
permutation-based: state_machine uses explicit derangements, every
register_machine instruction is invertible on the joint 17x17 register
pair -- so this is external validation within the theorem's existing
assumptions, not a stress test of them, and said so plainly rather than
overclaiming) and ran fresh, documented sweeps matching the
boolean_circuit protocol exactly (architecture, seeds 2001-2003,
checkpoints 8000, train_size 20000).

Results: register_machine_16 gives a clean, smooth, low-variance
replication of the boolean-circuit curve (outcome 6.33%, rho=0 at 6.00%,
both near chance 5.88%, rising smoothly through 67.67% / 85.20% / 96.27%
to 99.87% at rho=1.0). state_machine_16 confirms the same direction in
the mean but with much higher seed-to-seed variance from rho=0.8 onward
(std up to 52.71, including one seed that fails to recover even at the
clean rho=1.0 endpoint) -- per explicit user instruction ("ignore the
seed thing"), reported this plainly without investigating the cause,
since it doesn't reverse the qualitative direction.

Added as new Appendix G (app:other-substrates, appendix_results.tex),
with a transposed accuracy table (avoided an initial 9-column layout that
caused a real 50pt overfull hbox) and a provenance table with sha256
hashes. Added one sentence to the main text's Limitations paragraph
pointing to it, and one ToC/Organization entry in main.tex -- did not
add a new main-text section or figure, respecting the still-open 9-page
main-text cutting target (main text unaffected, still ends page 11).
Updated ARTIFACT_MANIFEST.md with the new hashes and an explicit note
that the old untracked register/state-machine CSVs in
results/reliability_sweeps/ are superseded and should not be cited.
Recompiled clean: 39 pages total (up from 38, appendix-only growth),
zero undefined references, zero overfull/underfull warnings beyond
pre-existing ones.

## Update 2026-09-16: abstract/intro reframe + outcome-trained gradient mirror + discrepancy code-path isolation

Completed the two items explicitly still open from the earlier
priority-ranked list.

**Abstract/intro reframe.** Tightened the abstract's closing to
explicitly name the unifying question ("why wrong reasoning traces can
still teach the right computation") and mention the now-existing
cross-task validation (state-machine, register-machine) alongside
Boolean-circuit and pretrained-model results. Reframed the intro's
opening paragraph the same way, stating the umbrella question before the
two-part breakdown, replacing "answers three questions" (which never
matched the actual two-claims structure used throughout the rest of the
intro) with a two-part framing that does match. Length-neutral: main
text page count unaffected (still ends page 11).

**Outcome-trained gradient-alignment mirror.** Added a `--train-mode`
flag to `src/experiments/gradient_alignment.py` (default "process",
preserving the exact existing default output path/behavior for
reproducing the archived Table 3; "outcome" writes to a distinct
`_outcome_trained` suffixed path, so the archived data was never at risk
of being overwritten). Ran the mirror experiment: same architecture,
seeds, steps, train-size as the archived run, but training under
L_out instead of L_proc. Caught and corrected a real mistake before
reporting: an earlier CPU smoke-test (tiny boolean_circuit_2 config) had
written to the same output path as the real run, and I initially nearly
mistook that stale file for the finished result when the user said "grad
experiment is done" -- verified via the persist.json's own recorded
config (task/steps/train_size) that it was stale smoke-test data, said
so directly, and waited for the real background-tracked process to
actually finish before reporting anything.

Result: at outcome-trained checkpoints, the actual (no longer
counterfactual) outcome gradient's cosine with the oracle collapses to
near zero (0.010 / 0.081 / 0.121 at 25/50/100% of training, vs 0.583 at
init) and its norm shrinks toward the oracle's own (0.026 / 0.015 / 0.012
of ||g_local||, vs 0.751 at init) -- consistent with converging toward
an L_out stationary point. The counterfactual process gradient at those
same parameters stays strongly rule-directed throughout (cosine
0.889-0.919), if anything higher than at process-trained checkpoints.
This directly answers the "g_out is only ever counterfactual" criticism:
the same directional asymmetry appears from both sides, with whichever
objective is actually being trained losing rule-directed alignment and
whichever is not remaining aligned. Added as a new table + discussion in
app:gradient-alignment (tab:gradient-alignment-outcome-trained) with an
explicit caveat that the two tables' counterfactual columns are not
directly comparable checkpoint-by-checkpoint (different training
trajectories), only in their shared qualitative conclusion. Added a
2-sentence pointer in Section 6's main text. Updated ARTIFACT_MANIFEST.md
with hashes. Recompiled clean: 39 pages (unchanged), main text still 11
pages, zero undefined references.

**82%/92.6% discrepancy, code-path isolation (in progress).** Per a
concrete, well-scoped follow-up request, launched a diagnostic running
the exact matched config (train_size=100k, archived seeds
{2001,2002,2020,3000,2026}, checkpoints=8000) through the `reliability`
pipeline's rho=1 condition instead of the `supervision` pipeline's
"process" mode (which gave 88.41% for this same config) -- isolating
whether the residual ~4.2-point gap to the archived 92.61% is a
between-pipeline implementation difference or just residual seed noise.
Still running at time of writing; result not yet known.

## Update 2026-09-16: register-machine matched corruption, Appendix F wording fix, visible cross-task table, figure relocation

**Trained symmetric-vs-coherent corruption, second task family (register_machine_16).**
The critique's top scientific priority item asked for the matched
symmetric-vs-coherent corruption comparison "ideally on Boolean +
register machine" -- only Boolean-circuit-8 had been run at the time.
Added `make_register_machine_sampler_coherent` to
`src/data/sequential_tasks.py` (fixed nonzero `(dx,dy)` mod-17 offset per
instruction, seeded deterministically, verified as a genuine bijection on
the 17x17 state space that never matches the true successor, for all 5
instructions), registered as `register_machine_16_coherent` in
`src/data/registry.py`. Ran the same matched protocol as the
Boolean-circuit version (rho in {0.3,0.5,0.7}, seeds 2001-2003, 8000
steps, 20000 instances) for both symmetric and coherent conditions.

Result is much cleaner than the Boolean-circuit version: symmetric beats
coherent at all three rho (69.80 vs 13.33 at rho=0.3, 88.73 vs 47.67 at
rho=0.5, 92.27 vs 88.20 at rho=0.7), and two of the three differences
reach significance (two-sample t: p=0.003, p<0.001, p=0.302). Combined
with Boolean-circuit's directionally-consistent-but-not-significant
result, the two task families now give the same sign of effect in six of
six matched conditions, with one family reaching significance. Extended
`app:trained-corruption-structure` in appendix_results.tex with a second
table and rewritten discussion; added hashes to ARTIFACT_MANIFEST.md.

**Appendix F wording bug (user-caught, verified genuine).** The
mirror-experiment discussion in app:gradient-alignment said "whichever
objective is actually being optimized loses rule-directed alignment with
the oracle, and whichever is not remains aligned with it" -- backwards
for the process-trained case, where the actual/optimized gradient
(process) is the one that *stays* aligned. Re-verified against both
tables' actual numbers before fixing (g_proc stays high, 0.80-0.92,
regardless of which trajectory; g_out collapses to near zero regardless
of which trajectory) and rewrote to state the asymmetry tracks the
process/outcome distinction itself, not real-vs-counterfactual status.

**Visible three-task table in Section 3.** The cross-task replication
(register-machine, state-machine) previously lived only in Appendix G
plus one Limitations sentence; the critique flagged this as "partly
fixed" twice. Added `tab:cross-task-summary` directly in
sec:sharp-jump (body.tex), a compact 3-row table (chance / outcome /
process at rho=1 for Boolean-8, register-machine-16, state-machine-16),
and shortened the now-redundant Limitations sentence to point at it
instead of restating the numbers. Recompiled clean, no page-count
regression (main text still 11 pages).

**Figure relocation (page-budget attempt, did not by itself help).**
Moved `fig:noise-threshold` from body.tex into
appendix_results.tex's app:noise-threshold section (replacing the
in-place figure with a one-sentence prose description of what it shows).
This was the previously-recommended lowest-risk page-cut option.
Recompiled: main text is still exactly 11 pages -- moving one figure
did not cross a page boundary. Per-page line-density check (pdftotext,
lines per page) shows pages 2-10 essentially full (111-139 lines each)
and page 11 partially full (111 lines); closing the remaining 2-page gap
to NeurIPS's 9-content-page limit needs on the order of two more pages
of genuine prose trimming, not just float rearrangement -- flagged to
the user as a real editorial decision (which sections/detail to cut)
rather than something to do unilaterally, consistent with the standing
"no new science, and don't silently trade off quality" freeze rule.

Compiled with `/home/hariguru/.local/bin/tectonic main.tex` (no
`pdflatex`/`latexmk` on PATH in this environment) -- 41 total pages, 0
errors, only pre-existing underfull-hbox warnings (cosmetic).

## Update 2026-09-16: abstract rewrite, Scope consolidation, theorem split, discrepancy-to-footnote

Executed a 6-item editorial plan (4 of 6 items; figure and notation-table
items deferred, see below).

**Abstract cut to one closing hedge.** Rewrote to end on a single sentence
covering all three disclaimers (prove-vs-measure, direction-vs-dynamics,
population-vs-frontier) instead of three separate inline ones. Verified
every number/claim against current body.tex before adopting; adjusted one
clause ("process target supplies both signals directly") to the paper's
more precise existing phrasing ("exposes the local transition directly,
giving an exact, depth-independent gradient") since the literal
"supplies both signals" reading overstated the mechanism.

**New sec:scope subsection in Setup, one hard rule.** Added a ~20-line
Scope subsection stating what's proved vs measured, once, right after
Setup. Deleted the redundant restatements: intro's "statement of genre"
+ Makkuva-in-intro sentence (Makkuva stays only in Related Work), intro's
full "What is proved, and what is measured" paragraph (superseded by
sec:scope), Section 6's "so the answer is a transfer test..." clause, and
a third restatement of the same "no canonical Pg-axis" point in the
Discussion section. Shrank three of four appendix "Scope" paragraphs
(app:noise-threshold's, app:fraction-vs-amount's, app:gradient-alignment's)
to one-sentence pointers at sec:scope; kept A.2's Scope paragraph at full
length since it carries content (what nabla_psi does NOT cover) that
exists nowhere else.

**Split Theorem 4 (cor:noise-threshold) into five objects.** Was one
25-line theorem block mixing a definition, an unconditional convergence
claim, the general recovery threshold, two named special cases, and a
gradient identity requiring strictly stronger hypotheses (uniform s0,
doubly-stochastic C_g) -- all in one box, making the hypothesis-scoping
read as hedging rather than as the mathematical content it is. Split into
Definition 4 (local reliability), Proposition 5 (convergence, any
row-stochastic C_g), Theorem 6 (recovery threshold rho_i(C_g), row-stochastic
only), Corollary 7 (named 1/K and 1/2 cases), Proposition 8 (gradient
witness at P_g=U, the one object needing the stronger hypotheses).
Verified rendered numbering matches exactly (checked compiled PDF text).
Updated ~6 downstream references in appendix_architectures.tex and
appendix_theory.tex that specifically meant the gradient-witness
proposition or the local-reliability definition, not the umbrella
theorem (they were citing eq:noisy-process-credit/eq:symmetric-noise-credit
by way of the old single label). Left ~18 other references pointing at
cor:noise-threshold unchanged, since they genuinely mean "the recovery
threshold" umbrella result.

**Two dedup/relocation fixes.** Converted the 82%/92.6% discrepancy from
an inline paragraph interrupting Section 3's results narrative into a
footnote at the point the number first appears, keeping the full
statistical trace but out of the main flow. Cut the main-text
near-duplicate of appendix B.6's "comparison in one normalization"
paragraph (both independently derived the same
depth-independent-floor-vs-O(rulestr^{D-1})-ceiling contrast) down to one
sentence pointing at the appendix, which keeps the full derivation.

**Fixed the "six of six" arithmetic error** the critique caught: Appendix
E.1's own text said "reversed at rho=0.3" for Boolean-circuit two
sentences before claiming "six of six matched conditions" -- an internal
contradiction. Corrected to "five of six" (all three register-machine
conditions, two of three Boolean-circuit conditions), matching the
critique's exact suggested wording. Also surfaced this trained-network
corruption-structure result directly in Section 3's main text (two
sentences, per the critique), not just in the appendix.

**Page-count reality check.** After all of the above, main text
(Sections 1-8 + Reproducibility Statement) is still exactly 12 pages
(pdftotext per-page line-density check unchanged at the same boundary
as before this batch). The Scope consolidation and theorem split
improved clarity and internal consistency but were roughly a wash on
raw line count -- content was relocated/deduped, not net-deleted at the
scale needed. Real progress toward the 9-page target requires either
accepting a larger cut than editorial polish alone can deliver, or a
scope decision on which subsections lose in-text detail (not just
redundant hedging). Deferred: the mechanism figure (item 4) and notation
table (item 5) from the critique, both of which would ADD length and so
work against the still-open page target -- flagged to the user rather
than added unilaterally.

Compiled with tectonic: 41 total pages, 0 errors, only pre-existing
cosmetic underfull-hbox warnings.

## Update 2026-09-16: Corollary 2 restated as a compositional credit-transmission bound

Per a user proposal to strengthen the theory's center (not add new scope),
verified and implemented a generalization of Corollary 2 (cor:near-mixing).

**Verification first.** The user's proposed general bound was stated for
arbitrary row-stochastic P_g. Checked this directly: the clean telescoping
argument (U absorbing under left/right multiplication, cross terms
canceling) requires double stochasticity specifically -- without column-
stochasticity, P·U != U in general and the cross terms in a product don't
cancel, leaving an uncontrolled leftover term the product-of-norms bound
doesn't cover. This is the same failure mode already fixed earlier this
session in Theorem 6's proof (f_g/K vs pi_{g,i}). Reported this to the
user rather than implementing the bound as literally proposed.

**What actually changed.** Corollary 2's existing hypothesis (P_j doubly
stochastic) already matches what the general bound needs, so restated it
under the SAME hypotheses: proved ||Rule(-grad P_t l_out)||_F <=
(1-1/K)/alpha * chi_t, where chi_t = prod_{j!=t} ||Pi P_{g_j} Pi||_2 is a
newly named "credit-transmission coefficient" (exactly P_g - U for doubly
stochastic P_g, verified: Pi P Pi = P - U follows from row- and
column-stochasticity absorbing U from both sides). Today's eps_rule^{D-1}
bound now falls out as an immediate corollary (uniform bound on each
factor of chi_t), rather than being the primary stated result. Updated
the appendix proof (appendix_theory.tex) to state the exact per-gate
product before specializing, added the P-U identity as an explicit
lemma step, and added discussion connecting chi_t=0 to Theorem 3's
zero-credit result and to process supervision's depth-independent
gradient bypassing the transmission bottleneck.

Kept the same label (cor:near-mixing) and equation label
(eq:depth-credit-bound) to avoid touching ~10 downstream references
elsewhere in the paper that cite "the corollary" generically; only added
a new eq:general-credit-bound and eq:credit-transmissivity for the new
content.

**Scope discipline.** Explicitly declined two further extensions the user
also proposed (a sample-complexity/SGD heuristic argument, and a
necessary-and-sufficient characterization of when outcome credit
vanishes) since neither has a verified clean proof yet and the user's own
message flagged "only if the proof is clean, do not force it." Asked the
user directly which scope to take via AskUserQuestion; they chose the
minimal, verified option (restate Corollary 2 only).

Recompiled with tectonic: 41 total pages (no change), 0 errors, no
undefined references. Rendered numbering confirmed correct by reading
the compiled PDF text directly (Corollary 2, Equations 14-16, Theorem 3
cross-reference all render as intended).

## Update 2026-09-16: trained-network validation of Theorem 1's forward x backward factorization (in progress)

Per the user's third proposed theory-strengthening route (after declining
the tight/matching bound and Dobrushin generalization as too risky to
force), designed and launched an experiment testing the EXACT Theorem 1
identity (outcome credit = forward signal x backward signal) directly in
the trained two-block GPT, rather than testing Corollary 2's chi_t bound
(which is not expected to be pointwise tight -- see prior log entry).

**Design.** For each D in {2,4,6,8} (same architecture/optimizer/budget
as the depth-capacity sweep, outcome-only supervision, 3 seeds,
8000 steps, 20000 train instances), at the frozen final checkpoint on a
fresh held-out probe set (2000 instances):
  - forward proxy: held-out linear-probe (sklearn LogisticRegression)
    accuracy for the true intermediate state s_1 (16-way), read from the
    residual stream between the two Transformer blocks at the position
    right after gate 1.
  - backward proxy: mean L2 norm of d(outcome loss)/d(that same
    activation) across the probe batch -- a measurable analogue of "how
    much would the final answer change if this position's representation
    were different," which needs backprop through the remaining D-1
    gates' worth of the second block's attention to be nonzero.
Fixed the probed position at "right after gate 1" across all D, so
composition depth varies without moving the probed position itself: the
forward proxy should be roughly D-invariant (depends only on gate 1) while
the backward proxy is the one expected to contract with D.

New file: src/experiments/credit_transmission_probe.py.

**A real bug caught and fixed during smoke-testing.** The first version
hooked the OUTPUT of model.blocks (the full residual stream just before
ln_f/lm_head) and got EXACTLY ZERO backward gradient at every non-loss
position, including the gate-1 position. Root cause: ln_f and lm_head are
position-wise operations, so once you are at the output of the full block
stack, the loss (masked to only the answer positions) has zero direct
functional dependence on any OTHER position's row at that same tensor --
the actual cross-position mixing that lets an early position influence a
later position's loss happens INSIDE the attention layers, upstream of
that point. Fixed by hooking between the two blocks (output of block 0 /
input of block 1) instead, so gradient must still pass through block 1's
attention to reach the loss. Verified on a random-init model with a
single-position mask that this produces the expected nonzero gradient
pattern before trusting it on a real (if tiny, 30-step) training run,
where it also produced the qualitatively expected direction (backward
proxy larger at D=2 than D=4) before launching the real 8000-step version.

Launched in background (GPU 0, PID tracked, watcher confirms real
completion rather than the launcher shell). Not yet analyzed -- result
pending. Will compare each proxy's decay shape across D against the
already-measured outcome-only accuracy decay (88.07/28.87/11.20/7.80 at
D=2/4/6/8) and write up honestly regardless of outcome, including if the
product does not track the accuracy decay cleanly.

## Update 2026-09-16: closed out both experimental threads, landed compact identifiability presentation

**Both new experiments (credit-transmission probe, identifiability probe)
completed but were NOT added to the paper, per explicit user decision
after seeing the actual results.**

Credit-transmission probe results: forward proxy 0.45-0.63 across D=2,4,6,8
(well above chance, flat with depth as predicted); backward proxy x
forward proxy product: 5.09e-6 (D=2) -> 5.44e-6 (D=4) -> 2.24e-6 (D=6) ->
0.96e-6 (D=8). Compared against the measured outcome-accuracy decay
(88.07/28.87/11.20/7.80): the product completely MISSES the steepest real
transition (D=2->D=4, where accuracy drops to 33% of its value but the
product slightly INCREASES), only tracking the gentler D=4->D=8 decline.
User's diagnosis, which stands as the final word on this thread: this
shows the Transformer proxy (probe-accuracy x gradient-norm) is not a
reliable operationalization of chi_t, not that the theory is wrong --
Theorem 1's exact factorization only claims the outcome-credit NORM
factors as forward x backward in the shared-table coordinates, never that
a linear-probe/gradient-norm proxy should quantitatively predict a trained
network's final-answer accuracy. Decision: stop, do not add to paper. The
existing gradient-alignment result (Table 3, Section 6) remains the
correct and sufficient trained-network bridge for the credit-assignment
claim.

Identifiability probe results: had two real problems, not just noise.
(1) A confirmed bug: final-answer accuracy eval used a fixed
max_new_tokens=6, far too short for process mode's ~55-character target;
training loss actually converged to 0.0000 for process mode, so the
reported ~5.6% "chance-level" accuracy was simply wrong, not a real
result. (2) A deeper, non-bug problem the user identified precisely: a
hidden-state linear probe tests whether one specific residual-stream
position happens to linearly encode s_t, which is NOT what the
identifiability theorem claims (observational equivalence of two
DIFFERENT global executors on terminal maps) -- a causal Transformer is
free to defer state computation to generation time via attention rather
than computing incrementally mid-prompt, so a null probing result at a
chosen position is uninterpretable either way, independent of the
eval bug. Decision: stop, do not repair, do not replace with a different
experiment either -- the theorem's evidentiary burden does not require a
trained-network validation to be worth keeping.

**Compact main-text presentation of the identifiability theorem added**
per the user's exact 4-part recipe: (1) states the third obstruction is
coverage-independent, distinct from the vanishing-gradient/under-excitation
pair already discussed; (2) gives thm:group-symmetry-nonidentifiability's
statement inline; (3) gives the Z_16 worked example in ~3 sentences;
(4) points to app:shift-symmetry-task for the full construction. Placed
right after the existing under-excitation paragraph in Section 4 (extends
the existing "two separate obstructions" framing to three), not as a
full section-order restructure -- the "Identifiability -> Credit assignment
-> Robustness" whole-paper reframing remains a separate, larger,
not-yet-authorized decision.

**Cleaned up all now-stale experiment-dependent claims**: removed the
Related Work paragraph's promise that app:shift-symmetry-task "tests
whether a trained network's own internal representation reflects the
identifiability gap" (rewritten to say the obstruction is "addressed here
by an exact construction rather than a trained-network measurement").
Rewrote app:shift-symmetry-task itself from an experimental "Protocol"
section into an explicit "worked example, not an empirical validation"
framing, removing all mention of the linear-probe methodology and its
(unreported) results.

Left src/experiments/identifiability_probe.py and
credit_transmission_probe.py in the repo as harmless, unused exploratory
code (not referenced by the paper, no ARTIFACT_MANIFEST entries added,
since no results from them are cited).

**Net page effect**: this batch made the main-text page count slightly
WORSE, not better (Reproducibility Statement now starts on page 13, up
from page 12), since the new Section 4 paragraph costs more main-text
space than the Related Work trim and appendix rewrite saved (those are
appendix/Related-work space, appendix doesn't count toward the 9-page
limit anyway). The page-budget problem is now the single most overdue
open item, explicitly reprioritized as "highest expected value" by the
user in this same message.

## Update 2026-09-16: page-budget recovery pass

Recovered the page lost to the last two additions (Related Work
identifiability paragraph, Section 4 compact presentation): main text was
at page 13 (Reproducibility Statement boundary), now back to page 12.

Concrete cuts made:
- De-formalized Proposition 8 (gradient witness at P_g=U) from a boxed
  proposition+proof into inline prose with the same two equations kept
  (eq:noisy-process-credit, eq:symmetric-noise-credit) -- the proof was
  pure redundancy anyway, since appendix_theory.tex's app:noise-threshold-proof
  already proves both equations in full under different internal labels;
  removing the boxed environment removed real overhead (~15 lines) with
  zero content loss. Fixed the ~5 downstream \cref{prop:noise-gradient-witness}
  references in appendix_architectures.tex/appendix_theory.tex to point at
  eq:noisy-process-credit instead (caught and fixed a duplicate-\label
  amsmath error this introduced before it reached the user).
- Merged Related Work's "Representability against optimization" paragraph
  (3 sentences) into the end of the new "Identifiability" paragraph, since
  they are thematically adjacent (both about separating a structural fact
  from optimizer dynamics) -- removes one paragraph-break's worth of
  vertical spacing overhead, zero content loss.

Net: still at 12 main-text pages, NOT the 9-page target. This is the same
plateau this session has hit repeatedly -- safe, content-preserving cuts
(redundant formalization, paragraph merges, duplicate hedging) reliably
recover 1 page-equivalent when things regress, but do not by themselves
close a 3-page gap. Closing the remaining gap requires cutting real,
non-redundant content (a full subsection's discussion moved to
appendix-only, or accepting 10-11 pages), which is an editorial call, not
a mechanical one -- flagged to the user rather than done unilaterally.

## Update 2026-09-16: corrected converse identifiability theorem, verified against actual repo code

**A real error in my own earlier attempt, caught before it reached the
paper.** A prior turn sketched "trivial centralizer of <T_g> implies
{T_g} is the unique zero-loss executor" as an easy converse to the
group-symmetry theorem. Re-derived it from scratch to check: it is FALSE.
Constructed the disproof directly: the existing group-symmetry
construction's own T' only needs A^D=I for the FULL D-length composition,
not for every intermediate suffix, so a trivial centralizer of <T_g> does
not rule out this kind of position-dependent escape. Did not add the
naive theorem.

**Derived and independently verified the correct version**, using the
DIFFERENCE group H := <T_g T_h^-1 : g,h> rather than <T_g> itself.
Re-derived the full proof from first principles (not just checked the
sketch): (1) full support + zero outcome loss forces every used P_g to be
a genuine permutation matrix, via a nonnegative-matrix-with-nonnegative-
inverse-is-monomial argument; (2) using a fixed reference gate h_0 and
its (D-1)-fold power as a reference block, get two one-sided
decompositions S_g = T_g*A and S_g = B*T_g; (3) equating them shows B
commutes with every T_g T_h^-1, landing in C(H); (4) trivial C(H) forces
B=I hence S_g=T_g for every g. This is a genuinely correct theorem
(thm:fixed-depth-identifiability), and actually simpler to present than
the originally-proposed sketch (no separate "length-two prefix" lemma
needed -- the D-1-th-power trick handles general D directly).

**Computationally verified the specific claim about this paper's own
Boolean gate family, against the actual repository code, not a
reconstruction of it.** New script src/experiments/identifiability_certificate.py
imports boolean_circuit_tasks.py's real _apply_gate directly, builds all
52 gate permutations, computes H = <T_g T_h^-1> via sympy.combinatorics:
|H| = 16!/2 = 10,461,394,944,000 exactly, every generator even, so H=A_16
exactly (not just isomorphic); sympy's centralizer computation confirms
C_S16(H) = {I}. This gives a real positive result about the paper's own
primary task: the Boolean circuit's depth-8 outcome failure is NOT
attributable to an alternative exact factorization -- it's a credit
problem specifically, not a candidate-among-several explanation.

**Replaced the Z_16 toy example with the verified sink-bit Boolean-circuit
construction** in app:shift-symmetry-task (this edit was described but
not actually made in the previous turn -- caught and fixed the gap this
turn). Brute-forced the exact restricted gate family against the real
generator: 31 of 52 gate strings (4 NOT, 9 CNOT, 6 SWAP, 12 Toffoli)
commute with "flip bit 4"; corrected the stated restriction from "bit 4
only ever a target" (insufficient -- SWAP has no control/target split and
also breaks commutation when it touches bit 4) to "no gate reads or
permutes the sink bit into another coordinate." Verified computationally:
T'_g != T_g at all 16 states for every one of the 31 gates; D=8
composition matches across 20,000 random trials; D=7 mismatches in all
2,000 trials tested (confirming the parity dependence). Non-abelian
(NOT on bit 1 and CNOT controlled by bit 1 don't commute), a materially
better example than the abelian Z_16 toy case.

**Declined the proposed D=8->D=9 length-generalization Transformer
experiment** for now, per the assessment that it's not diagnostic until
the D=8 terminal fit is much closer to saturating than it currently is
(current outcome-only D=8 accuracy is near chance) -- flagged as
optional/risky ROI, not run, consistent with the standing decision to
close the experimental thread on this topic. No action taken.

**Corrected a stale claim I introduced but failed to fully clean up
last turn**: appendix_theory.tex still said app:shift-symmetry-task
"measures whether a trained network's own internal representation...
reflects this ambiguity" -- missed when body.tex and appendix_results.tex
were fixed. Now corrected everywhere.

Page-budget note: this batch added real content (a full theorem + proof,
two appendix subsections, a verification script) but almost all of it
landed in the appendix, which does not count toward the 9-page limit.
Main text grew by about one paragraph (replacing the old, less useful
Z_16 mention) and did NOT cross the page-12 boundary. Recompiled clean:
44 total pages (up 1 from appendix growth), 0 errors, main text
unchanged at 12 pages.

## Update 2026-09-17: self-inflicted file corruption during notation cleanup, full recovery

**What happened.** While renaming the tensor notation `A_g[i,j]` to `\Lambda_g[i,j]`
in appendix_architectures.tex (part of the communication/simplicity pass), a
Python regex script included a badly-escaped "no-op safeguard" line
(`re.sub(r'\\|A_g\\\|', r'', content)`) that actually matched and stripped
EVERY backslash from the entire 478-line file. This file is not git-tracked
(Paper/ is in .gitignore), so there was no automatic backup to restore from.

**Recovery approach.** Rather than guess-reconstruct the file, did a careful,
verifiable restoration: (1) rebuilt backslashes for ~80 distinct LaTeX
command tokens via word-boundary-safe regex, starting with unambiguous ones
and individually classifying every occurrence of ambiguous tokens that also
collide with English words (in, to, text, sum, log, star, paragraph, small,
subsection, centering, softmax) by reading full context for each one before
deciding whether to add a backslash back -- since several of these appear
BOTH as genuine LaTeX commands and as ordinary English prose in the same
file (e.g. "This subsection closes that gap" is prose; "\subsection{Title}"
is a command). Iteratively recompiled after each batch of fixes, using
tectonic's own error messages (Missing $ inserted, Misplaced \noalign,
Multiple \label's, There's no line here to end, Illegal unit of measure)
to surface remaining issues one at a time, plus custom Python scripts to
detect (a) "glued" multi-command sequences left over from backslash-stripped
adjacent commands (e.g. "inmathbb" -> \in\mathbb, "bmPi" -> \bm\Pi),
(b) math-mode-only commands appearing outside $...$ or display math
(a systematic check across ~50 command names), (c) commands normally
requiring a {argument} appearing without one, and (d) no-vowel suspicious
tokens. Caught and fixed a genuine false-positive class multiple times over
(commands that are ALSO common English words used as prose in this exact
file: paragraph, small, subsection x2, centering, sum x4, times x2) --
each required reverting an over-eager backslash insertion back to plain
text. Also fixed several classes of SILENT (non-compile-error) corruption
that would not have surfaced without a careful rendered-output read-through:
unescaped literal set braces (\{2,4,6,8\} had become bare {2,4,6,8}),
unescaped percent signs (6.25% needed 6.25\%), missing \\ row/line
separators in a table and two align blocks, and unescaped underscores in
a \texttt{} shell-command citation. Verified the final result both via
compile-cleanliness (0 errors) AND by reading the actual rendered PDF text
for pages 16-21 end to end against the known-correct mathematical content.

**Outcome**: full recovery confirmed. 44 total pages (matching pre-incident
state), main text still exactly 12 pages (unchanged), 0 compile errors,
no undefined references, table of contents structure intact (no bogus
subsection entries from the transient \subsection-in-prose bug).

**Process lesson**: for any future find-and-replace across this file (or
any non-git-tracked file), test regex substitutions on a small sample or
copy first, and never include an untested "safeguard" pattern in the same
batch as real fixes.

## Update 2026-09-17: continued simplicity-list cleanup (post-recovery)

Resumed the communication/simplicity list after the appendix_architectures.tex
recovery. Confirmed one earlier verification was WRONG and fixed it: found
a manually-built "Table of Contents" block in main.tex (\ref/\nameref/
\pageref/\dotfill for every appendix section, not an automatic
\tableofcontents, which is why the earlier grep for that command found
nothing) that genuinely duplicates the "Organization" paragraph's content.
Removed the ToC block (pure navigation, no explanatory content) and kept
the Organization paragraph (carries the actual "what is proved where"
content). Recompiled clean, page count unchanged (44 total, main text
still 12) -- the ToC block didn't sit at a page boundary, so no immediate
page-count win, but it is a real, confirmed duplication removed.

Checked the "6.25% appears ~8 times, consolidate to once" suggestion:
found only 2 of the 8 occurrences are in main text (body.tex), and both
serve genuine local purposes -- one is inside a figure caption (needs to
stand alone for a reader looking at just the figure) and one derives the
specific value 1/K=6.25% at the point it's first used as a threshold.
Did not act on this one; the actual distribution is more justified than
the raw repetition count suggested.

Checked and compressed appendix_theory.tex's B.5 "projection can change
sign" proofstep (app:mixture-proof): verified via reference search that
nothing \cref's it directly (only the subsection's general mixture-gradient
identity is cited elsewhere), supporting the critique's "orphaned" read.
Compressed from ~13 lines to ~6, keeping the exact same mathematical
content (the sign-change formula, the fixed-checkpoint caveat, and the
forward pointer to prop:noisy-process-credit) -- pure compression, no
content loss. Recompiled clean.

Deliberately did NOT act on the "D.1 forensics: two pages to half a page"
suggestion after re-reading it: this is the 82%/92.6% discrepancy
investigation trace (Welch's t-test + the three-factor diagnostic:
train-size, seed-identity, code-path isolation) that was built up
carefully earlier this session and is specifically what the reproducibility
score credited. Cutting it to "half a page" would remove substance, not
just reformat it, unlike the other folds -- flagged as a judgment call
for the user rather than cut unilaterally, consistent with the standing
"nothing here removes a result" framing of the original request.

Still open, unstarted: A/A_g rename downstream consistency spot-check,
rho's three-meanings subscripting, chi_t/eps_rule double-statement, em-dash
density, the mechanism figure, Figure 1/5(a) merge (needs plotting-code
changes, not just LaTeX text), table caption trims, Lemma A.2 shortening
(paper's own text already argues this specific bound isn't citable
elsewhere, so declined without literature verification), E's duplicate
restatement.

## Update 2026-09-17: the five prioritized edits (abstract, intro, split, B.3, Lambda/A)

Executed all five of the critique's prioritized edits.

1. **Abstract rewritten to end on the gradient finding.** Cut the closing
   prove/measure disclaimer (already stated in sec:scope) and ended
   instead on: "The same directional asymmetry that separates the two
   objectives in the population mechanism reappears, unprompted, in the
   actual gradients of a trained network."

2. **Intro's "Two claims are closed exactly" paragraph now names the
   identifiability separation** as a third structural fact, one sentence,
   without promoting it to an equal-weight third contribution: "whether
   terminal behavior determines the local rules at all is a different
   question from whether it supplies useful gradient information once it
   does, and this paper's own Boolean task is shown to be identifiable."

3. **Split the compressed Section 4 identifiability discussion into three
   paragraphs** matching the critique's exact outline: (a) non-identifiable
   outcome rules, from coverage or structural symmetry; (b) identifiability
   and credit are different properties; (c) the main Boolean task is
   identifiable, so its failure is a credit problem specifically.

4. **Moved thm:fixed-depth-identifiability's statement from
   appendix-only to the main text** (body.tex, within paragraph 3 above),
   matching how every other major theorem in the paper is already handled
   (statement in body.tex, full proof in the appendix's app:X-proof
   section). Left the appendix with just a one-line pointer + the proof.
   Checked all cross-references into this label from
   appendix_results.tex/appendix_architectures.tex still read correctly
   now that it lives in main text.

5. **Completed the Lambda/A notation migration in appendix_architectures.tex.**
   The tensor rename from an earlier session (A_g -> Lambda_g) had left
   the BARE, unsubscripted "A" -- used throughout as the whole-tensor
   function argument, e.g. theta(A,psi*), P_g(A) -- untouched, which is
   exactly the collision the critique flagged (A is also the centralizer
   permutation in the identifiability theorem, now sitting in the main
   text next to this appendix's own material). Found and fixed every
   remaining occurrence via a careful multi-pass search (function-argument
   forms, subscripted forms like P_{g_1}(A) and A_{g^star}, and plain
   prose references like "every A", "random A", "the transition logits A
   vary") rather than a single blind regex, verifying zero bare "A"
   tokens remain via a final automated scan before recompiling. Also
   caught two more leftover corruption-recovery artifacts surfaced during
   this pass: "A_{g}[i,\cdot]" (braced-g variant my earlier A_g->Lambda_g
   fix didn't match) and "P_bullet" (missing backslash before \bullet),
   both fixed.

Page-budget note: main text moved from 12 to 13 pages, entirely from
item 4 (the theorem statement is real, deliberately-requested content,
not accidental bloat). The user was informed. The 9-page target is
further away than before this batch, which is an explicit, known
trade-off of following this prioritized list rather than a regression.

Recompiled clean after every step: 44 total pages, 0 errors, no
undefined references. Did not touch the remaining smaller items from
this message (A.2 further compression, caption command-line trims, the
82/92.6 footnote further simplification) or the earlier list's leftovers
(rho subscripting, chi_t/eps_rule restatement, em-dash density, mechanism
figure, Figure 1/5a merge) given time already spent on high-priority items
plus the need for care after the recovery incident.
