# Proof audit

## Sources and scope

The requested timestamped filenames are absent. The matching primary source is `Paper/archive/backups/before_theorem4_removal/main.pdf` (64 pages, title *Where the Credit Is: Why Running Counts Teach What Final Counts Miss*), together with that directory's `body.tex`, `appendix.tex`, and section fragments. The secondary is the current `Paper/build/main.pdf` and current source (34-page revision described by `Paper/README.md`). The requested central result is present as primary Theorem 3, Theorem 4(i), and Corollary D.4 (`cor:matched-separation`). Source manuscripts were not edited. User modifications already present in the worktree were left intact.

The focused manuscript has one principal learning theorem and one main executor proposition. It retains the aligned supplied-routing branch, not the separate two-path architecture/trap or learned full-prefix branch. Auxiliary lemmas and architecture propositions support that theorem. The fixed binary perturbation proof is retained as a narrow appendix refinement with no additional headline theorem.

## Dependency map

| Result | Dependencies | Assumptions and audit |
|---|---|---|
| Proposition 1 | Appendix B construction | Finite alphabet, fixed finite horizon, full causal softmax, three heads/block, no LayerNorm, explicit registers/MLPs/readout. Width is linear in n at fixed K and alphabet. |
| Conditional contraction, C.2 | Independence, finite Fourier transform | Query and start conditioned on; arbitrary S subset of [n], square-integrable K-row H; exact conditional norm `(1/K) sum |rho_j|^(2(n-|S|))`. Uniform start is averaged afterward. |
| Score support, C.1 | Network estimates, C.2 proposition | Fixed local ReLU pattern, small block operator norms, fixed input-independent causal mask, fixed answer token. Score order k has at most 2k+1 input occurrences; support may be smaller. |
| Network estimates | Softmax derivative recurrence and two-block composition | Row probability weights have positive entries summing to one. Normalized kth derivatives bounded by exponential-in-k constants; tangent factor and remainder controlled. Constants depend on fixed K/operator bounds/head count; bias norm enters the scale threshold. Removed dilution statement is unused. |
| Credit summation, C.3 | Fourier + support + remainder | n>=2, 0<lambda<1, 0<epsilon, `C_att epsilon^4 <= lambda^2/2`; whole-credit bound also epsilon<=lambda. Negative exponents when 2k+1>n overbound by factors >=1, rather than truncating the exponent. |
| Bias uniformity, C.4 | Same logit recurrence | Bias is input-independent; at zero content scores softmax(beta) replaces uniform weights. No dilution lower bound is invoked. Only the broad outcome supporting argument trains bias. |
| Smoothness, C.5 | Fixed ReLU + first/second softmax derivatives | Local differentiability on enlarged ball, positive bias gap, finite task expectation; not a global ReLU-smoothness claim. |
| Outcome plateau | Local credit, KL loss identity, zero readout, energy | The local loss floor is `log K - C2 epsilon^2 lambda^n`. First-exit argument uses the enlarged neighborhood. Correct-answer probability follows separately from Pinsker/Jensen; greedy accuracy is not bounded. |
| Exact features, D.1 | Explicit active head/register embedding | Other heads and MLP weights are zero and fixed. Values are fixed. Supplied mask selects previous-state token and current letter. Scalar sigmoid is an exact attention weight, not a substitute architecture. |
| Cyclic symmetry, D.2 | Uniform start, isometric conjugation, uniqueness | Count readout stays circulant. Gate readout lies on R(I-R) by restriction. |
| Coordinate flow, D.3 | Exact features, cyclic/background symmetry | Frobenius metrics give `dot u_j=-(1/K) partial L/partial u_j`, `dot w=-(1/(2K)) partial L/partial w`; q,k use scalar Euclidean metric. Background s absent for K=2, present for K>=3. |
| Geometry, D.4 | Direct derivatives of log-sum-exp F | All K-2 background classes are included in F. Error bounds use half the background probability. |
| Levels, D.5 | Case loss and error inequalities | At L<=p_min log(2)/2, each of the two match-bit cases has P>=2^(-1/2)>1/2. Symmetry then covers every start and every recurring position. All K-1 incorrect logits are controlled. |
| Gradient bound, D.6 | 2x2 singular-value bound | Determinant tau, Frobenius norm<=sqrt(3); v^2<=1; dissipation retains the correct metric factors. |
| Drift/alignment, D.7–D.8 | Coordinate flow + asymmetric prior | Reflect p>1/2 to p_min<1/2; q0 k0(1-2p)<0; aligned seeds require no magnitude-gap condition. Barrier argument for h and monotonic Q ensures g>=0 and a remains aligned. |
| Floor/growth, D.9–D.10 | Alignment + drift + loss levels | Floor g>=epsilon^2 min(1,h*)/80 after Th. Before success, error>=p_min/17; `S>=2 sqrt(d_h)|a|/epsilon^2` yields exponential growth of |a|. Constants 4080 and 8010 preserved. |
| Descent/persistence, D.11 | Gradient bound, levels | A fixed nonzero aligned a floor suffices; success loss cannot increase. Bound `dot L <= -epsilon^12 L^2/(196608 K d_h)` proves L->0. The slower epsilon^-12 bound is auxiliary, not the headline learning time. |
| Matching, Appendix E | Seed verification + norm comparison | Same seed, same mask, same trainable family, same scalar/Frobenius metric. Outcome time reduced by fixed factor K; only objective and teacher-forced format change. |
| Overlap/asymptotics | Exact time formulas | epsilon=c_* n^-1/2, N_b=Theta(sqrt(n)), d_h=Theta(n), fixed K,p; sufficient small fixed c_* and large n. Finite-horizon overlap requires the exact inequality. |
| Fixed perturbations, Appendix F | Statewise gate computation + invariant region + floor + descent | K=2, p<1/2, aligned, epsilon<=p, fixed Delta_Q/Delta_K, key Frobenius orthogonality, operator norms <= initial scalar magnitude/8. No arbitrary/trained/value/general-K perturbation claim. |

## Repairs and source inconsistencies

1. **Tied metric in the matched corollary.** The original corollary calls `(U,W,q,k)` a constant orthogonal block projection. In the stated scalar Euclidean metric this is inaccurate: `||Pi_Q||_F^2=K`, whereas `||Pi_K||_F^2=1`. Appendix E explicitly embeds the scalar flow, proves `||delta vartheta||^2 <= ||A delta vartheta||^2 <= K ||delta vartheta||^2`, and applies its energy identity to the ambient local loss floor. The resulting certified time is `r0^2 lambda^-n/(2 K C2)`. This preserves the asymptotic result at fixed K without silently changing the positive training metric.
2. **Executor copy norm.** The source writes an unqualified one-hot copy error <= leakage. In Euclidean norm the constant can be sqrt(2). The focused proof states the correct coordinate infinity norm and uses it only in coordinate preactivation estimates.
3. **Executor proof completeness.** Added the actual position source sets, token/position embeddings, fallback routes, and format readout coordinates from the independent implementation. Accounted for head-3 leakage at intermediate states and proved the summed outcome tent bound rather than applying a scalar Lipschitz statement to a vector of many tents. Finite margins remain >19 at the constructed settings.
4. **Embedding operator bound.** Repeated identical non-query embeddings may make the complete token embedding block's operator norm grow with the fixed alphabet. Use `a0=max(4 sqrt(K),sqrt(2|Sigma|))` instead of asserting 4 sqrt(K) for every alphabet. The alphabet is fixed in the asymptotic comparison, so constants remain n-independent. This makes an existing assumption explicit rather than asserting an alphabet-uniform guarantee.
5. **Observable error.** `results/paper/zero_readout_n16_8k/summary.json` gives 93.11% complete-continuation accuracy and 95.45% final-answer accuracy for process, with 49.82% outcome answer accuracy. The source prose calls 93.1% answer accuracy. The focused paper corrects this and retains both observables.
6. **Count-label interpretation.** Removed the secondary prose's assertion that the advantage comes only from label format. The sparse-label comparison retains gold teacher-forced input states and does not match information. The shorter inconclusive control is reported as well.
7. **Robustness proof notation.** Replaced the signed-scalar condition `v<=1` in a dissipation inequality by `v^2<=1`, which is what the seed actually implies and what the inequality uses. No assumption on the sign of v is added.

## Verification performed

- Exhaustive finite executors: n=2 through 6; alphabet sizes 2 and 3; K=2,3,4; all starts, queries and words: **31,635 prompts, 363,654 supervised predictions**. Both parameter settings, all format/count margins, and full greedy continuations passed. Minimum observed margin exceeds 19.9995. See `evidence/executor_verification.log` and JSON.
- Source mathematical checker: **0 failures**, including threshold/constant grids and dense autograd agreement of the general-K Transformer-induced coordinate dynamics. See `evidence/theorem4_checks.log`. Numerical checks do not certify arbitrary-size analytic inequalities.
- Controlled figure regenerated from six original run files, distinguishing theorem-radius from larger-radius displacement. No new training.
- Boolean-8 main, answer-first and 6400-update controls checked against all three original CSVs each; BBH checked against all 18 official per-seed entries. Sparse control summaries checked against their per-seed values.
- Routing success summaries recomputed from the original 17 pairs; medians agree to the reported rounding. See `evidence/routing_rows.json`.
- Tectonic/BibTeX build passes; no missing citations, undefined or duplicate labels, overfull boxes or missing figures. All 28 rendered pages inspected in contact sheets; theorem and figure pages also checked at page scale.

## Remaining qualifications

There is no unresolved central mathematical blocker identified in this audit. This is an analytic source-proof audit, not independent formal certification: `lean/Trace/Analysis/ScoreCredit.lean` contains `sorry`, so Lean does not certify the score-expansion/plateau chain. Full pretrained or counting model training was not rerun; the manuscript reports existing logged results and regenerated summaries. The initialization constants in the source are existential local-bound constants, not numerical calibration constants. The empirical small-initialization runs cannot verify their precise numerical threshold.

Architecture-wide executor expressivity does not imply capacity or reachability in the restricted trainable family. The principal step result does not prove the first transition, unrestricted parameter learning, random routing discovery, or free-running rollout accuracy. The near-chance outcome probability statement is not a greedy-accuracy theorem and is not a permanent-failure result.
