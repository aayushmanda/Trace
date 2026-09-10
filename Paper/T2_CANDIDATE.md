# T2 candidate (unproved)

**Status: not a theorem.** Do not advertise this in the title, abstract, or
theorem headers. Completing the week-2 numerics does not prove T2.

Roadmap label **T2** is a research candidate. Manuscript Theorems 1–3 (realizability,
local vs rollout credit, complete mixing) stay in the kernel / shared-executor
setting. Positioning: [../PAPER_POSITIONING.md](../PAPER_POSITIONING.md).

---

## Candidate statement

On a **shared executor** with row-stochastic tables \(P_g(t)\in\Delta^{K-1}\)
(one table per gate, reused across depth), let \(U=\mathbf{1}\mathbf{1}^\top/K\)
and let the mixing radius be
\[
r(t)\;=\;\max_{g\in[M]}\bigl\|P_g(t)-U\bigr\|_2.
\]
Consider **projected Euclidean gradient flow** of the population outcome loss
\(L_{\mathrm{out}}\) on the product of simplices (equivalently: Euclidean
projection of a gradient step onto the row-simplex after a tangent-space
centering so each row sums to zero).

**Candidate inequality (unproved):** there exists \(C<\infty\), depending on
\((K,M,D)\) and the loss Lipschitz geometry but **not** claimed to be
constructed here, such that
\[
D^+ r(t) \;\le\; C\, r(t)^{D-1},
\]
where \(D^+\) is the upper-right Dini derivative
\(D^+ r(t)=\limsup_{h\downarrow 0}\bigl(r(t+h)-r(t)\bigr)/h\).

**Intuition (not a proof).** T1 says outcome local credit is
\(O(r^{D-1})\) in the mixing ball. If the vector field that moves \(P\) is
controlled by that credit, \(r\) cannot grow faster than order \(r^{D-1}\)
near complete mixing. Process loss is first-order in \(r\) and is **not**
the candidate.

---

## Assumptions (required for the candidate even to be well-posed)

1. Shared kernel: the same \(P_g\) (or \(A_g\)) is used at every occurrence of
   gate \(g\). Not a generic Transformer residual stream.
2. Tables remain row-stochastic; the flow is the **Euclidean projection** onto
   that constraint. Row-softmax logits \(P_g=\mathrm{softmax}(A_g)\) are a
   **different geometry** (`python -m src escape`). They are not this candidate.
3. Loss is population (or large-batch) **outcome** composition CE, not process
   local CE and not serialized LM CE on a Transformer.
4. \(r(t)\) is a mixing-ball radius in an operator norm compatible with T1
   (implementation uses the matrix 2-norm). Other radii are not automatically
   equivalent.
5. Gates / true rules are permutations if one wants the complete-mixing
   zero-gradient fact (manuscript Thm 3). The candidate inequality is about
   *rates near mixing*, not about identifying \(T_g\).

---

## What is **not** proved

- The inequality \(D^+ r\le C r^{D-1}\) itself. No Lyapunov argument, no
  comparison ODE, no explicit \(C\) is supplied. Discrete Euler steps with a
  numerical Dini ratio are **diagnostics**, not a proof.
- That projected flow is the gradient flow of training in any neural net.
- Transfer to Transformer parameters: \(\sum_t J_t^\top W_t \approx -\nabla_\theta L\).
  That is a **separate** unproved readout (Corollary 34 / pullback). Cosine
  \(\approx 0.028\) with composition TV \(\approx 0.833\) is allowed as a
  **negative / limit-of-transfer** result, not a confirmation of T2.
- Existence of an internal Boolean-circuit executor in a trained GPT.
- Escape-time / depth laws for row-softmax SGD (roadmap **T4**).
- Interchange / causal interventions (roadmap interchange grid; not this file).

---

## Allowed evidence vs non-evidence

| Allowed | Not evidence |
|---|---|
| T1 kernel algebra (already a theorem) as *motivation* | A Transformer cosine |
| Shared-kernel projected Euler smoke (`python -m src projected`) | Pilot CSV from week 2 |
| Honest negative pullback numbers | “T2 proved numerically” |

Runner: `src/eval/projected_kernel.py`. Protocol:
`configs/experiments/e4_projected_kernel.yaml` and
`results/paper_revision_v2/protocols/e4_projected_kernel.yaml`.
