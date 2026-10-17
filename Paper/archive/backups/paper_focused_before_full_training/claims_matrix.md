# Claims matrix

| Major claim | Manuscript location | Proof/evidence | Scope |
|---|---|---|---|
| Same seed and trained blocks, only supervision objective changes | Theorem 1; Appendix E | Equations (seed, metric, objectives); norm-comparison energy proof | Separate trajectories and teacher-forced formats; fixed values, features, routing |
| Process develops persistent positive margins | Theorem 1(ii), Appendix D | Exact head embedding; symmetry; induced metric ODE; floor/growth/descent | Recurring transitions t>=2; all K-1 wrong counts; aligned seed |
| Process time O(n^(5/2) log n) | Theorem 1(iii), Appendix E | Explicit 4080/8010 time bound; epsilon=Theta(n^-1/2), d_h=Theta(n) | Fixed K,p,alphabet; sufficiently small scale |
| Outcome initial plateau Omega(lambda^-n) | Theorem 1(i), Appendices C/E | Fourier conditioning, score support/remainder, loss floor, energy | Initial finite interval; not arbitrary-accuracy lower bound |
| Expected correct-answer probability near 1/K | Theorem 1(i) | KL identity, Pinsker/Jensen | Expected assigned probability; no greedy accuracy guarantee |
| Broader ambient full-joint outcome flow allowed | Appendix C discussion/proof | Input-independent biased attention and orthogonal-block energy | Does not broaden the positive theorem |
| Both exact complete programs exist | Proposition 1, Appendix B | Explicit full-causal two-block softmax/ReLU tensors; finite leakage margins | Overall architecture, not restricted trained subspace |
| Finite implementation matches executors | Appendix H | `scripts/verify_executors.py`; `evidence/executor_verification.json` | 31,635 prompts; 363,654 supervised predictions; finite sizes only |
| Transformer coordinate gradients correct | Appendix D/H | Analytic Frobenius/scalar derivation; `verification/theorem4_checks.py` | Numerical checks supplementary, 0 failures |
| Fixed off-direction QK refinement | Appendix F | Complete statewise gate/invariant/floor/descent proof from primary | Binary aligned, fixed perturbations <=1/8, key orthogonality; values fixed |
| Outcome displacement grows with horizon | Figure 1, Appendix G.1 | `evidence/controlled_rows.json`; six original logs in SHA manifest | Three seeds; finite-update approximate flow; two radii |
| Fast step exit initially fits prior | Figure 1 caption, Appendix G.1 | Source objective, readout dynamics, displacement logs | Not a claim of rule completion |
| Counting step solves 16/17; outcome 0/17 | Section 5, Appendix G.2 | `evidence/routing_rows.json`; 17 hashed source pairs | Finite GD, unstructured QK matrices, outside principal theorem |
| n=16 step answer 95.45%, continuation 93.11%; outcome answer 49.82% | Section 5 | `evidence/counting_summary.json`, `counting_per_seed.csv` | Ten paired AdamW seeds; separate observables |
| Qwen Boolean-8 step 98.3%, outcome 11.1% | Table 1, Appendix G.3 | `evidence/qwen_rows.json` and all original per-seed CSVs | Pretrained synthetic benchmark, outside theorem |
| Outcome 8x budget 10.0% +/-0.9 | Table 1 | Three 6400-update CSVs | Existing runs, not newly trained |
| Answer-first Qwen 10.1% +/-1.0 | Table 1 | Three answer-first CSVs | Answer precedes trace |
| BBH three-object step 98.4–100%, outcome 35.2% | Table 1, Appendix G.4 | `evidence/bbh_rows.json`, nested official metrics | 250 official test items; training template-generated |
| BBH five-object reversal | Table 1, Appendix G.4 | Step seeds 100,100,46%; outcome 94.4,83.6,80% | Negative seed retained; process mean 82%, outcome mean 86% |
| BBH seven-object step 96.4–100%, outcome 15.2–24.4% | Table 1 | Official metrics for all seeds 2001–2003 | No counting-style lower bound for Qwen |
| One-count-label process advantage | Section 5, Appendix G.5 | `evidence/labels_v2_30k_summary.json`, all per-seed entries | Equal label count, unequal teacher-forced prefix information |
| Shorter label control inconclusive | Appendix G.5 | `evidence/labels_summary.json` | 8k updates, three seeds at n=8 |
| Accurate relation to parity/Markov/automata work | Section 6, bibliography | Primary Kim–Suzuki ICLR/arXiv; Makkuva arXiv; Liu ICLR entry | Distinct architecture/optimizer/task assumptions; no subsumption claim |

The evidence manifest contains source paths and SHA-256 hashes. Model training was not newly reproduced; plot/table extraction and finite proof/executor checks were rerun.
