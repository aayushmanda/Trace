# Numerical checks and detailed measurements

Detailed measurements behind the paper's appendix. The paper keeps one summary table per section; this file keeps the
per-draw and per-seed numbers, with the scripts that produce them (arguments are in each script's docstring).
All letter-task numbers use counting with four letters unless stated; exact computations are summed over all words in
float64.

## Theory checks (appendix C)

Scripts: `handcoded/theory_checks.py`, `handcoded/degree_setsums.py`, `handcoded/two_block_spectrum.py`,
`handcoded/rule_credit.py`, `handcoded/rule_credit_query.py`, `handcoded/rule_credit_structured.py`,
`handcoded/construction_from_appendix.py`.

- **Row 2 (exponents by degree).** Two blocks, width 16, n = 6, exact over all 4^6 words. Exponent of
  sum_{|S|=r} ||(J z)_S|| in sigma, fitted on [0.005, 0.05]: 2.0, 6.1, 7.0, 11.1, 12.0 for r = 1..5. The proved exponents are
  attained at r = 1, 2, the degrees the theorems use.
- **Rows 4 and 5.** Over 20 exact full-batch updates (n = 4, 6, 8; sigma in {1e-2, 3e-3}) the step loss falls by
  1.5–2.2 sigma^2 and the outcome loss by 7e-9 to 7e-6 sigma^2.
- **Row 7 (rule credit, fixed query).** |Sigma| = 4 uniform letters, p = 1/4, lambda = 1/2, sigma = 0.02; exponent in sigma 6.00
  throughout. n * rule credit / sigma^6, draws 1, 2:

  | n | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|
  | draws 1, 2 | 309, 133 | 308, 130 | 309, 130 | 324, 131 | 304, 134 |

## Inside the theorems' regime (appendix D.2)

Scripts: `handcoded/regime_both.py`, `handcoded/rule_credit.py`, `handcoded/interaction_credit.py`,
`handcoded/curvature.py`. Two blocks, K = 2, width 16, three heads of dimension 8, MLP width 16, two letters with match
probability 1/4.

Exit time tau(0.1), zero readout:

| | n=4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| outcome, sigma=0.002, draw 1 | 9.3e3 | 4.6e4 | 2.3e5 | 1.0e6 | 4.7e6 | 2.2e7 |
| outcome, sigma=0.002, draw 2 | 1.2e4 | 6.1e4 | 2.9e5 | 1.3e6 | 6.2e6 | 2.8e7 |
| outcome, sigma=0.002, draw 3 | 1.5e4 | 7.7e4 | 3.8e5 | 1.8e6 | 7.8e6 | 3.8e7 |
| outcome, sigma=0.02, draw 1 | 7.8e2 | 3.8e3 | 1.9e4 | 9.4e4 | 2.1e6 | 3.0e7 |
| step, sigma=0.002, draws 1–3 | 4.18, 4.33, 5.55 at every n | | | | | |
| step, sigma=0.02, draw 1 | 4.17 | 4.18 | 4.19 | 4.19 | 4.20 | 4.20 |

Curvature: the top Hessian eigenvalue of the answer-dependent part Phi_out at theta_0 falls from 2.2e-3 at n = 4 to
1.0e-6 at n = 14, while that of the step loss stays at 0.25–0.26 (sigma = 0.02, Lanczos).

Credits at initialization, sigma = 0.02, three draws (exponents in sigma between 0.005 and 0.02: 1.00 prior, 6.00 rule,
2.1 total outcome credit, 7.0 interaction credit under a uniform start, 6.0 with the start fixed):

| | n=4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| prior credit, draw 1 | 3.3e-2 | 3.3e-2 | 3.3e-2 | 3.3e-2 | 3.3e-2 | 3.3e-2 |
| rule credit, draw 1 | 3.3e-9 | 2.5e-9 | 1.8e-9 | 1.3e-9 | 1.1e-9 | 8.8e-10 |
| rule credit * n / sigma^6, draws 1–3 | 207, 195, 120 | 231, 194, 112 | 227, 175, 97 | 208, 144, 122 | 211, 141, 145 | 192, 147, 169 |
| total outcome credit, draw 1 | 1.6e-4 | 3.2e-5 | 6.6e-6 | 1.4e-6 | 3.2e-7 | 7.1e-8 |
| interaction credit, uniform start, draw 1 | 2.8e-11 | 9.2e-12 | 2.6e-12 | 6.7e-13 | 1.7e-13 | 4.2e-14 |
| rule / interaction, draws 1–3 | 119, 64, 80 | 268, 129, 150 | 709, 315, 351 | 2,000, 803, 1,350 | 6,730, 2,600, 5,310 | 21,100, 9,410, 21,300 |

With the start state fixed, the rule credit exceeds the interaction credit by 1.3–2.5x at n = 4 and 30–70x at n = 12.

## Spectra (appendix D.3)

Scripts: `handcoded/spectrum.py`, `handcoded/exact_spectrum.py`, `handcoded/bbh_lora_spectrum.py`.

Answer spectra by task (share of answer variance by degree):

| Task | lambda | share at r <= 1 | share at r >= 2 | mean degree |
|---|---|---|---|---|
| count mod K, four letters | abs(3/4 + e^{2 pi i/K}/4) | <= (1+n) lambda^{2(n-1)} | – | linear in n |
| Boolean-8 (n = 8) | 0.820 | 1.1% | 98.9% | 6.3 |
| register machine (n = 16) | 0.953 | 0.08% | 99.9% | 12.3 |
| first occurrence | – | >= p(1-p) at degree 1 | – | – |

Boolean-8: 4 bits (K = 16), NOT, CNOT, SWAP and Toffoli gates on uniformly random wires (52 distinct gates, n = 8).
Register machine: (x, y) in Z_17^2 (K = 289), five instructions chosen uniformly, n = 16, answer x.

Qwen2.5-0.5B, rank-8 LoRA, three-object BIG-Bench Hard (answer pure degree 3); share of LoRA-gradient energy by degree,
mean over three LoRA initializations, ten contexts each: degree 0: 99.60 ± 0.02%, degree 1: 0.256 ± 0.011%,
degree 2: 0.118 ± 0.007%, degree 3: 0.022 ± 0.002%. The measured answer-dependent gradient is 2–6% of the bound of
the gradient-split lemma.

Gradient decay, two-block Transformer at initialization, K = 2, four letters, seed 42 (log-linear fit 0.508 per letter,
R^2 = 0.995; seeds 43, 44: 0.513, 0.480):

| n | 2 | 4 | 6 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| full answer-token gradient | 0.159 | 0.146 | 0.165 | 0.183 | 0.143 | 0.168 |
| J_0 | 0.111 | 0.111 | 0.148 | 0.143 | 0.120 | 0.127 |
| sum_{r>=1} J_r | 2.1e-4 | 3.0e-4 | 4.2e-4 | 4.6e-4 | 4.4e-4 | 4.0e-4 |

## Outcome accuracy over n and lambda (appendix D.6)

Script: `handcoded/lambda_grid.py`, figure `handcoded/lambda_grid_figure.py`. Counting mod 2, alphabet A in
{4, 6, 10, 20} (lambda = 1 - 2/A), 8,000 AdamW updates, seeds 42–44, mean held-out exact accuracy:

| n | out 0.5 | out 0.67 | out 0.8 | out 0.9 | step 0.5 | step 0.67 | step 0.8 | step 0.9 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.50 | 0.57 | 0.70 | 0.79 | 0.64 | 0.79 | 0.73 | 0.84 |
| 8 | 0.51 | 0.51 | 0.62 | 0.71 | 0.74 | 0.60 | 0.81 | 0.75 |
| 10 | 0.50 | 0.51 | 0.55 | 0.71 | 0.79 | 0.46 | 0.73 | 0.70 |
| 12 | 0.50 | 0.52 | 0.52 | 0.64 | 1.00 | 0.16 | 0.30 | 0.69 |
| 16 | 0.50 | 0.50 | 0.50 | 0.57 | 0.23 | 0.71 | 0.21 | 0.43 |

No outcome run reaches 90%; step runs reach it in 17 of 60.

## BIG-Bench Hard per run (appendix D.7)

Script: `handcoded/run.py` (job queues). Qwen2.5-0.5B, rank-8 LoRA, accuracy (%) on the 750 real items:

| objects (swaps) | degree-<=1 share | outcome, three runs | step, three runs | chance |
|---|---|---|---|---|
| 3 (3) | 0 | 35.2, 35.2, 35.2 | 100, 98.4, 100 | 33 |
| 5 (5) | 1.56% | 94.4, 83.6, 80.0 | 100, 100, 46.0 | 20 |
| 7 (7) | 3.34% | 24.4, 16.0, 15.2 | 96.4, 100, 100 | 14 |

At three objects, outcome runs select the most common option, "(B)", whose base rate is 35.2% on real items and 28.7%
on generated test items. The failed five-object step run also scores 45% on generated items. Zero-shot accuracy is 33.6,
16.4 and 11.2% for 3, 5 and 7 objects.

## Step reachability and routing (appendix E)

Scripts: `handcoded/step_reach_twoblock.py` (restricted two-block training, with `--armijo` for the basin ratio),
`handcoded/routing_sim.py` (isolated local-head routing), `handcoded/route_theorem_sim.py` (order parameters of the
Theorem 3 proof plan). Sweep scripts and JSON outputs: `results/step_reach/`, `results/route/`.

## Figure provenance and summary conventions

Regenerate the manuscript figures with `python3 handcoded/main_figures.py --result k2-zero`.
The plotting scripts need NumPy and Matplotlib, and do not rerun model training.
`handcoded/figure_style.py` defines the shared serif typography and semantic palette.
All figures are vector PDFs at the paper's 5.5-inch text width; PNGs are previews.
The complete list of input records and plotted values is saved in
`results/paper/main_figure_revision/figure_sources.json`.

- Figure 1: ten paired seeds at n=16, K=2, zero readout. The learning curves show pointwise medians and 25th--75th seed percentiles. Eight step seeds exceed 90% at 8k updates; no outcome seed does. The spectrum is a separate n=8, K=2, sigma=0.02 measurement with nonzero readout.
- Figure 2(a): medians and interquartile bands over three initialization draws. The green lambda^(-n) guide is anchored at n=4, not fitted.
- Figure 2(b,c): means and full seed ranges for each of 20 horizon--alphabet cells, three seeds per cell. Exact stored contraction values are used, including 2/3 for alphabet size 6; contraction values are not rounded before exponentiation. Spearman correlation on cell means is 0.911278 with lambda^n and -0.361752 with n. This corrects the previously quoted -0.19 horizon correlation, which is not reproduced by these records. The green state-only prior curve is a reference, not a fitted accuracy law.
- Figure 3: median answer accuracy, thick interquartile intervals, thin full seed ranges. GPT-2L Boolean-8 has five seeds; the register, Qwen Boolean-8 and BBH groups have three. BBH uses the nested `bbh` evaluation fields (250 official items per size), rather than the generated 300-item test set. These intervals describe seed variation; they are not confidence intervals. The exact numeric summary table remains in the appendix.

The standalone predictor comparison can be regenerated with
`python3 handcoded/lambda_grid_figure.py`; it uses the same data and summary code as Figure 2.

## Staged-routing constants check

`python3 handcoded/routing_horizon_check.py` enumerates all prompts at n=2 and n=4,
for p in {0.1, 0.25, 0.5, 0.9}, head width in {1, 8}, sigma in {0.02, 0.1} and both gate
signs. All 64 cases pass the Stage B coefficient, Stage C leakage, separator-margin and
Stage D burn-in bounds, the explicit threshold of Lemma R-scale where it applies, and the
"after T'" inequality of Lemma R-D on a logarithmic time grid from T' to e^200 T'. It uses
the exact prior for the numerical B/C calculation and separately checks the scalar Stage A
curvature bound p(1-p). It does not simulate the full Stage D flow or replace the proof.

The routing theorem guarantees positive margins at every time t >= T, with T the
polynomial burn-in; the leakage tolerance covers [T_D, T'], and a second argument (loss
upper bound ~ log^2 t / t against the lower bound ~ t^(-1/B) forced by a nonpositive
margin) covers t >= T'. The earlier duration parameter H is gone.

Theorem 2(iii) uses one a_delta depending only on delta,p,K, then fixes n and the
widths before choosing a deterministic sigma_delta(n,widths). Its probability
event is simultaneous over scales below that threshold for that fixed
architecture and horizon. It does not assert a single event for all horizons
or a scale threshold uniform in n at fixed widths. Residual width is at least
max(K,3), as required by the embedding rank and the six-entry monomial proof.
