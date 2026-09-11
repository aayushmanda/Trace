# Closure: handcoded constructions + same-architecture D-block

No new theorems. Not §7 mixed one-block transfer. Not GSM8K / 7B.
Kernel quantities are the theorems' objects (numpy, no training). Network
measurements are the D-block outcome architecture (`build_random_trainable_outcome_architecture`)
and the exact `HandcodedOutcomeTransformer` construction.

Reproduce:

```bash
conda activate aayus   # or: /home/hariguru/aayus/.venv/bin/python
cd /home/hariguru/aayus/trace

# 1–3. Kernel (CPU, ~15s)
python experiments/run.py kernel

# 4, 6. Oracle tables + nnsight patches + init Def 20 (CPU, ~15–25 min)
python experiments/run.py handcoded --device cpu

# 5, 7. Short matched-architecture train (prefer cuda:2/3 when free)
python -m src outcome-local \
  --config configs/experiments/closure_matched_early.yaml \
  --depth 2 --device cuda:2 --no-compile --matched-architecture --induced-credit

# Week-scale D=8 one-seed (hours; do not launch a 20-seed grid from this file)
python -m src outcome-local \
  --config configs/experiments/closure_matched_d8.yaml \
  --confirm --depth 8 --device cuda:2 --no-compile \
  --matched-architecture --induced-credit
```

---

## Theorem 4, both halves (closes the 0 / 2 scores)

Figure: `thm4_mixing.pdf`. Table: `thm4_mixing.csv`.
Same Appendix I process path (`corrupted_counts` / `projected_gradient_at_mixing`).

At \(P_g = U\), \(D=8\), \(K=16\):

| \(\rho\) | process coef along \(T_g-U\) | predicted \((K\rho-1)/(K-1)\) | cosine | outcome \(\|\mathrm{Rule}(G)\|_F\) | outcome product \(\|\bar q\|\|\bar b\|/p_y\) |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.0667 | −0.0667 | −0.690 | \(1.08\times 10^{-17}\) | 0 |
| \(1/K=0.0625\) | +0.0047 | 0 | 0.068 | \(1.08\times 10^{-17}\) | 0 |
| 0.25 | +0.2055 | 0.200 | 0.947 | \(1.08\times 10^{-17}\) | 0 |
| **0.5** | **+0.4659** | **0.4667** | 0.989 | **\(1.08\times 10^{-17}\)** | **0** |
| 1 | +1.0000 | 1.000 | 0.998 | \(1.08\times 10^{-17}\) | 0 |

Reviewer request was “process 0.4659, outcome \(3\times 10^{-16}\)”. Measured:
process **0.4659** (identical to App I), outcome Rule **\(1.08\times 10^{-17}\)**,
per-occurrence product **exactly 0**.

Row-stochastic \(G\Pi\) retains a last-occurrence remainder \(\approx 0.008\)
(\(\Pi b_D \neq 0\)). The theorem's projector is \(\mathrm{Rule}=\Pi G\Pi\),
the same operator used for the process half.

---

## Theorem 2 (factorization, not just the product)

Figure: `thm2_residuals.pdf`. Table: `residuals.csv`.
\(\varepsilon=0.4\), \(D=8\), 400 circuits, same samples for both residuals.

| \(t\) | \(\|\bar q_{t-1}\|\) | \(\|\bar b_t\|\) | product \(/p_y\) | process floor |
|---:|---:|---:|---:|---:|
| 1 | \(9.68\times 10^{-1}\) | \(1.59\times 10^{-3}\) | \(2.46\times 10^{-2}\) | 23.6 |
| 4 | \(6.20\times 10^{-2}\) | \(2.48\times 10^{-2}\) | \(2.46\times 10^{-2}\) | 23.8 |
| 8 | \(1.59\times 10^{-3}\) | \(9.68\times 10^{-1}\) | \(2.46\times 10^{-2}\) | 23.8 |

Forward residual decays, backward residual grows, product is constant.
Process floor stays \(\Theta(1)\) (~24), about \(10^3\times\) the outcome product.

Exact executor \(P_g=T_g\) (not a mixing ball): `oracle_true.csv`.
\(\|\bar q\|=\|\bar b\|=\sqrt{1-1/K}=0.9682\), credit = process floor = \(1-1/K=0.9375\)
at every \(D\in\{2,\ldots,12\}\). That is Theorems 1–2 at the construction,
and why Corollary 3 needs \(P\) near \(U\).

---

## Corollary 3 (the figure a reviewer flags)

Figure: `corollary3.pdf` (credit vs \(D\)); `exponents.pdf` (fitted exponent of \(\varepsilon\)).
\(P_g=U+\varepsilon(T_{\mathrm{rand}}-U)\) so \(\|P_g-U\|_2=\varepsilon\) for \(\varepsilon\in\{0.1,0.3,0.5,0.9\}\).

Selected cells \(\|\mathrm{Rule}(-\nabla_{P_t}\ell_{\mathrm{out}})\|_F\):

| \(\varepsilon\) | \(D=2\) | \(D=6\) | \(D=12\) | process floor |
|---:|---:|---:|---:|---:|
| 0.1 | 1.50 | \(1.50\times 10^{-4}\) | \(1.50\times 10^{-10}\) | ~16 |
| 0.3 | 4.67 | \(3.64\times 10^{-2}\) | \(2.66\times 10^{-5}\) | ~20 |
| 0.5 | 9.54 | 0.470 | \(7.32\times 10^{-3}\) | ~28 |
| 0.9 | 66.9 | 17.6 | 6.33 | ~140 |

Process floor does not decay with \(D\). Outcome credit at \(\varepsilon=0.1\) is
\(\varepsilon^{D-1}\) to three digits. \(\varepsilon=0.9\) is **not** a mixing ball;
the bound is one-sided and the decay is slower. Do not quote \(\varepsilon=0.9\) as
the rate.

Log-log slope vs \(\varepsilon\), mixing-ball regime \(\varepsilon\le 0.5\)
(`exponents_mixing_ball.csv`):

| \(D\) | slope | predicted \(D-1\) | \(R^2\) |
|---:|---:|---:|---:|
| 4 | 3.020 | 3 | 0.99997 |
| 6 | 5.001 | 5 | 1.000 |
| 8 | 7.000 | 7 | 1.000 |
| 12 | 11.000 | 11 | 1.000 |

Including \(\varepsilon=0.9\) pulls the fitted exponent ~0.6 above \(D-1\)
(`exponents.csv`). Caption the mixing-ball fit, show \(\varepsilon=0.9\) as the
far-from-\(U\) panel.

---

## Handcoded oracle patches (Theorem 1 construction)

nnsight, 8 circuits, donors \(\{0,5,10,15\}\), CPU. `oracle_patches.csv`.

| \(D\) | oracle_slot | random_subspace | unstructured | wrong_layer | probe_subspace |
|---:|---:|---:|---:|---:|---:|
| 2 | **1.000** | 0.125 | 0.016 | 0.188 | 0.266 |
| 4 | **1.000** | 0.125 | 0.055 | 0.320 | 0.039 |
| 6 | **1.000** | 0.156 | 0.094 | 0.344 | 0.052 |

Chance is \(1/16=0.0625\). Patching the known \(s_t\) slot flips the remaining-gate
answer at every depth. Random subspace stays near chance. Linear `probe_subspace`
on the construction is **not** the slot (the coordinates are already one-hot);
that gap is expected and is why trained-net identification uses the probe, not
the construction coordinates.

---

## Init / Def 20 on the untrained D-block net

Gold-prefix readout (`measure_induced_credit`), 3 seeds, CPU. `init_induced.csv`.
Theory lives at \(P_g\approx U\). Random init of
`build_random_trainable_outcome_architecture` is already there:

| \(D\) | mean \(\|\widehat P_g-U\|_2\) | outcome Rule credit | process floor |
|---:|---:|---:|---:|
| 2 | 0.008 | \(1.68\times 10^{-2}\) | 15.00 |
| 4 | 0.015 | \(1.44\times 10^{-2}\) | 15.00 |
| 6 | 0.021 | \(1.37\times 10^{-2}\) | 15.01 |
| 8 | 0.055 | \(2.67\times 10^{-2}\) | 15.09 |

Process floor at complete mixing is \((1-1/K)/(1/K)=15\). Outcome credit is
\(\sim 10^3\times\) smaller. **Do not** claim \(\varepsilon^{D-1}\) from this
readout: mixing radius of the *averaged* table is small, but credit does not
track \(\varepsilon^{D-1}\) (endpoint / \(p_y\) terms on gold prefixes). The
exponent lives in `corollary3.pdf`. This panel is: at the parameter where the
theory holds, the net's induced tables give suppressed outcome credit and an
un-suppressed process floor.

---

## Same-architecture hygiene (item 7)

`src/eval/local_credit.py` now takes `--matched-architecture`. Process is
`copy.deepcopy` of the D-block outcome net, so it has `state_slots` and is
patched. Default (no flag) is still the historical one-block process.

`--induced-credit` writes `induced_credit.csv` at each checkpoint (Def 20
gold-prefix, mixing radius, outcome Rule, process floor).

Configs:

- `configs/experiments/closure_matched_early.yaml` — 400 steps, \(D\) overridden, matched + induced.
- `configs/experiments/closure_matched_d8.yaml` — 6000 steps, \(D=8\), one seed, `do_not_launch` / `--confirm`.

**Not started this session:** GPUs 0–3 were all at 99–100% utilization
(GPU 2: 34 GB / 80 GB). A 6000-step D=8 job was not launched on leftover
memory. Short early training (\(D\in\{2,4,6\}\), 400 steps) and the 20-seed
D=8 confirm remain for when cuda:2/3 is free.

---

## What each figure closes

| File | Theorem | Status |
|---|---|---|
| `corollary3.pdf` | Cor. 3 | **run** |
| `exponents.pdf` / `exponents_mixing_ball.csv` | Cor. 3 exponent \(D-1\) | **run** |
| `thm2_residuals.pdf` | Thm. 2 | **run** |
| `thm4_mixing.pdf` / `thm4_mixing.csv` | Thm. 4 both halves | **run** |
| `oracle_true.csv` / `oracle_true_credit.pdf` | Thm. 1–2 at \(T_g\) | **run** |
| `oracle_patches.pdf` | Thm. 1 causal slot | **run** |
| `init_gradient.pdf` | Cor. 3 / Thm. 4 at init \(P\approx U\) | **run** (init only) |
| early `induced_credit.csv` + trained patches | Thm. 4 in the net + causal | **not run** (GPU busy) |
| matched D=8 20-seed | week-scale confirm | **not run** |

---

## Week-scale remainder

1. When cuda:2/3 is free: `closure_matched_early.yaml` at \(D=2,4,6\), then
   `closure_matched_d8.yaml --confirm` (one seed). Then 20 seeds only if the
   one-seed D=8 cell is not chance.
2. Trained-net patches on the matched process (same D-block) vs outcome vs
   outcome_local — this is what takes causal closure past 8.
3. Do not pool those numbers with §7 one-block local 99.9% / TV 0.833 / cosine 0.028.
4. Do not plot `oracle_alignment.py` cosines as a mechanism figure.
