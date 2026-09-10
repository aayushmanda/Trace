# Theorem ledger

Manuscript folder is often gitignored. This file tracks what may be stated as a theorem.
It is **not** a proof. Roadmap IDs **T1–T4** are research labels; PDF numbering differs
(see `Paper/EXPERIMENT_MAP.md`).

Positioning: [../PAPER_POSITIONING.md](../PAPER_POSITIONING.md),
[../THEORY_EMPIRICAL_REVISION_ROADMAP.md](../THEORY_EMPIRICAL_REVISION_ROADMAP.md).
T2 candidate (unproved): [T2_CANDIDATE.md](T2_CANDIDATE.md).

| Label | Status | Setting | Notes |
|---|---|---|---|
| **Thm 1** (manuscript `thm:realizability`) | Existing | Exact finite-parameter Transformer executors | Capacity, not a trained-GPT existence theorem |
| **Thm 2** (manuscript `thm:forward-backward`) | Existing | Shared-kernel local vs rollout credit | Mixing ball; process vs outcome factorization |
| **Thm 3** (manuscript `thm:complete-mixing`) | Existing | Shared \(P_g=U\): outcome credit 0, process recovers \(T_g-U\) | Kernel model; reversibility used for the zero-gradient claim |
| **T1** (roadmap) | Existing (polish) | Same as manuscript local-credit / mixing-ball theorems | Keep in main text; state assumptions |
| **T2** (roadmap) | **Candidate, unproved** | Projected simplex flow of the shared kernel; optional \(\theta\)-transfer | See statement below. **Do not promote to a theorem.** |
| **T3** (roadmap) | Existing (polish) | Constructive + shared \(A\in\mathbb{R}^{M\times K\times K}\) | Kernel / executor setting |
| **T4** (roadmap) | Target, unproved | Escape-time / depth law in the shared-executor setting | Numerics: row-softmax `python -m src escape` vs projected `python -m src projected` |

Roadmap T1/T3 align with manuscript Thm 2–3 / 1 polish, not with PDF “Theorem 1” numbering.

---

## T1 (existing) — local vs rollout credit in the mixing ball

**Setting.** Shared executor tables \(P_t\in\mathbb{R}^{K\times K}\) (row-stochastic),
forward state \(q_{t-1}\), backward answer residual \(b_t\), answer probability \(p_y>0\).
\(\mathrm{Rule}(\cdot)\) is the executor-relevant (centered) part of a matrix.

**Statement (manuscript `thm:forward-backward`, informal).**
Outcome CE credit at step \(t\) factorizes as
\(\mathrm{Rule}(-\nabla_{P_t}\ell_{\mathrm{out}})=\bar q_{t-1}\bar b_t^\top/p_y\).
A local process target needs neither inference and has a depth-independent
Frobenius lower bound. If tables are doubly stochastic and sit within rule
strength \(\varepsilon_{\mathrm{rule}}\) of uniform, and \(p_y\ge\alpha>0\), then
\[
\bigl\|\mathrm{Rule}(-\nabla_{P_t}\ell_{\mathrm{out}})\bigr\|_F
\le \frac{1-1/K}{\alpha}\,\varepsilon_{\mathrm{rule}}^{\,D-1}
\qquad\text{for every }t.
\]
Process supervision pays no \(D-1\) factor.

**Not claimed.** This is not a theorem about ordinary Transformer parameters \(\theta\).

---

## T2 (unproved candidate) — projected flow of the mixing radius

**Do not cite as a theorem.** Informal candidate only. Assumptions, what is not
proved, and the \(\theta\)-transfer diagnostic: [T2_CANDIDATE.md](T2_CANDIDATE.md).

**Candidate statement.** On the shared-kernel simplex, let \(r(t)\) be a mixing
radius (e.g. \(\max_g\|P_g(t)-U\|_2\)). Under **projected** Euclidean gradient
flow of population outcome loss,
\[
D^+ r(t) \;\le\; C\, r(t)^{D-1}
\]
for some \(C<\infty\) (upper-right Dini derivative). This is a dynamical
restatement of “outcome credit is \(O(r^{D-1})\) near uniform,” not a proved ODE
theorem and not a Transformer theorem.

**Related unproved transfer (optional diagnostic, not required).**
\(\sum_t J_t^\top W_t \approx -\nabla_\theta L\) for ordinary Transformers.
Empirical transfer may be **negative** (e.g. cosine \(\approx 0.028\)).

Numerics that **do not prove** T2: `python -m src projected` (simplex projection)
vs `python -m src escape` (row-softmax logits \(P_g=\mathrm{softmax}(A_g)\)).
