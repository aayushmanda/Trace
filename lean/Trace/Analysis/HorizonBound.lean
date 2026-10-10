import Trace.Model.Prelude
import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
import Mathlib.NumberTheory.LegendreSymbol.AddCharacter
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# The conditional outcome signal decays like `λ^{n−|S|}` (eq:horizon)

`horizonBound` : for every finite set `S` of letter positions and every freezing `w` of the
letters in `S`, the conditional signal `condSignal T S w = E[y − u | F_S](w)` satisfies
`∑ₐ (condSignal T S w a)² ≤ (λ^{n−|S|})²`, for `0 ≤ p ≤ 1`.

The proof is the paper's (Paper/appendix.tex:276-282): the Fourier coefficient at mode `j` of
the conditional signal is `α^{j s₀} ∏_{t∈S} α^{j b_t} · ρ_j^{n−|S|}` for `j ≠ 0` and `0` for
`j = 0`; Parseval on `ZMod K` (`parseval_zmod`) converts this into the squared norm, and
`|ρ_j| ≤ λ` (`mode_le_mixrate`) gives the bound. Characters are `ZMod.stdAddChar`.
-/

open scoped BigOperators ComplexConjugate

namespace Trace.Proof.HorizonBound
set_option linter.unusedSectionVars false
variable {K : ℕ} {Sym : Type u} [Fintype Sym] [DecidableEq Sym]

/-- An additive character turns a finite sum into a product.
INTERNAL: a one-line induction absent from Mathlib under this name. -/
theorem addChar_map_sum {A M ι : Type*} [AddCommMonoid A] [CommMonoid M] (ψ : AddChar A M)
    (s : Finset ι) (f : ι → A) : ψ (∑ i ∈ s, f i) = ∏ i ∈ s, ψ (f i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih => rw [Finset.sum_insert ha, Finset.prod_insert ha, AddChar.map_add_eq_mul, ih]

/-- The conditional outcome signal `E[G | F_S](w)`: the letters in `S` are frozen at `w`, the
others are redrawn as a fresh i.i.d. sample `w'`.
INTERNAL: concrete rendering of `E[G | F_S]` on the finite product space.
TEXLINE: Paper/appendix.tex:264-270 -/
noncomputable def condSignal (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n))
    (w : Fin n → Sym) (a : ZMod K) : ℝ :=
  ∑ w', T.letterWeight w' * T.targetSignal (S.piecewise w w') a

/-- Fourier coefficient of `G = y − u` at mode `j`: `α^{j s_n} − [j = 0]`.
INTERNAL: Fourier transform of the one-hot target minus uniform.
TEXLINE: Paper/appendix.tex:276-281 -/
theorem sum_targetSignal_char [NeZero K] (T : Trace.Task K Sym) {n : ℕ} (v : Fin n → Sym)
    (j : ZMod K) :
    ∑ a, (T.targetSignal v a : ℂ) * ZMod.stdAddChar (j * a) =
      ZMod.stdAddChar (j * T.answer v) - if j = 0 then 1 else 0 := by
  have h := AddChar.sum_mulShift j (ZMod.isPrimitive_stdAddChar K)
  have e : ∀ a, (T.targetSignal v a : ℂ) * ZMod.stdAddChar (j * a) =
      (if a = T.answer v then ZMod.stdAddChar (j * a) else 0) -
        (1 / (K : ℂ)) * ZMod.stdAddChar (a * j) := by
    intro a
    simp only [Trace.Task.targetSignal, mul_comm a j]
    split_ifs <;> push_cast <;> ring
  simp_rw [e, Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_univ, if_true,
    ← Finset.mul_sum, h, ZMod.card]
  split_ifs <;> simp [NeZero.ne K]

/-- Independence: the conditional character of the answer factors over letters.
INTERNAL: the product step of the proof of lem:target-contraction.
TEXLINE: Paper/appendix.tex:278-281 -/
theorem ex_char_answer [NeZero K] (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n))
    (w : Fin n → Sym) (j : ZMod K) :
    ∑ w', (T.letterWeight w' : ℂ) * ZMod.stdAddChar (j * T.answer (S.piecewise w w')) =
      ZMod.stdAddChar (j * T.s0) * ∏ t, ∑ x, ((T.μ x).toReal : ℂ) *
        ZMod.stdAddChar (j * if (if t ∈ S then w t else x) = T.wstar then 1 else 0) := by
  rw [Fintype.prod_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun w' _ => ?_
  simp only [Trace.Task.answer, Trace.Task.letterWeight, mul_add, Finset.mul_sum,
    AddChar.map_add_eq_mul, addChar_map_sum, Finset.piecewise, Complex.ofReal_prod]
  rw [mul_left_comm, Finset.prod_mul_distrib]
  rfl

/-- A PMF on a finite type sums to one.
INTERNAL: bookkeeping. -/
theorem sum_mu_toReal (μ : PMF Sym) : ∑ a, (μ a).toReal = 1 := by
  rw [← ENNReal.toReal_sum (fun a _ => PMF.apply_ne_top μ a)]
  have h : ∑ a, μ a = 1 := by rw [← tsum_fintype (L := .unconditional _)]; exact μ.tsum_coe
  simp [h]

/-- Averaging a constant against `μ` returns it.
INTERNAL: bookkeeping. -/
theorem sum_mu_char_const [NeZero K] (μ : PMF Sym) (z : ℂ) :
    ∑ x, ((μ x).toReal : ℂ) * z = z := by
  rw [← Finset.sum_mul, ← Complex.ofReal_sum, sum_mu_toReal, Complex.ofReal_one, one_mul]

/-- One unseen letter contributes the multiplier `ρ_j = 1 − p + p α^j`.
INTERNAL: the multiplier computation.
TEXLINE: Paper/appendix.tex:253-254 -/
theorem sum_mu_char_mode [NeZero K] (T : Trace.Task K Sym) (j : ZMod K) :
    ∑ x, ((T.μ x).toReal : ℂ) * ZMod.stdAddChar (j * if x = T.wstar then 1 else 0) =
      ((1 - T.p : ℝ) : ℂ) + T.p * ZMod.stdAddChar j := by
  have e : ∀ x, ((T.μ x).toReal : ℂ) * ZMod.stdAddChar (j * if x = T.wstar then 1 else 0) =
      ((T.μ x).toReal : ℂ) * 1 +
        (if x = T.wstar then ((T.μ x).toReal : ℂ) * (ZMod.stdAddChar j - 1) else 0) := by
    intro x
    split_ifs <;> simp [AddChar.map_zero_eq_one]; ring
  simp_rw [e, Finset.sum_add_distrib, sum_mu_char_const (K := K), Finset.sum_ite_eq',
    Finset.mem_univ, if_true, Trace.Task.p]
  push_cast; ring

/-- `|1 − p + p e^{iθ}|² = 1 − 2p(1 − p)(1 − cos θ)`.
INTERNAL: elementary trigonometry.
TEXLINE: Paper/appendix.tex:253 -/
theorem norm_sq_mode (p θ : ℝ) :
    ‖((1 - p : ℝ) : ℂ) + p * Complex.exp (θ * Complex.I)‖ ^ 2 = 1 - 2 * p * (1 - p) * (1 - Real.cos θ) := by
  rw [Complex.sq_norm, Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im]
  have := Real.sin_sq_add_cos_sq θ
  linear_combination p ^ 2 * this

/-- `cos(2πk/K) ≤ cos(2π/K)` for `1 ≤ k < K`.
INTERNAL: elementary trigonometry behind `λ = max_{j≠0} |ρ_j|`.
TEXLINE: Paper/appendix.tex:254 -/
theorem cos_mode_le {K k : ℕ} (hk1 : 1 ≤ k) (hkK : k < K) :
    Real.cos (2 * Real.pi * k / K) ≤ Real.cos (2 * Real.pi / K) := by
  have hK : (0 : ℝ) < K := by exact_mod_cast (show 0 < K by omega)
  have hpi := Real.pi_pos
  rcases le_or_gt (2 * k) K with h | h
  · apply Real.cos_le_cos_of_nonneg_of_le_pi (by positivity)
    · rw [div_le_iff₀ hK]
      have : (2 * k : ℝ) ≤ K := by exact_mod_cast h
      nlinarith
    · apply div_le_div_of_nonneg_right _ hK.le
      have : (1 : ℝ) ≤ k := by exact_mod_cast hk1
      nlinarith
  · have hrw : 2 * Real.pi * k / K = 2 * Real.pi - 2 * Real.pi * ((K - k : ℕ) : ℝ) / K := by
      rw [Nat.cast_sub hkK.le]; field_simp; ring
    rw [hrw, Real.cos_two_pi_sub]
    apply Real.cos_le_cos_of_nonneg_of_le_pi (by positivity)
    · rw [div_le_iff₀ hK]
      have : (2 * ((K - k : ℕ) : ℝ)) ≤ K := by
        rw [Nat.cast_sub hkK.le]; have : (K : ℝ) < 2 * k := by exact_mod_cast h
        linarith
      nlinarith
    · apply div_le_div_of_nonneg_right _ hK.le
      have : (1 : ℝ) ≤ ((K - k : ℕ) : ℝ) := by exact_mod_cast (show 1 ≤ K - k by omega)
      nlinarith

/-- `|ρ_j| ≤ λ` for every nonconstant mode `j`.
PAPER: Paper/appendix.tex:253-254 (`λ = max_{j≠0}|ρ_j|`, the inequality direction used in eq:horizon) -/
theorem mode_le_mixrate {K : ℕ} [NeZero K] {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (j : ZMod K) (hj : j ≠ 0) :
    ‖((1 - p : ℝ) : ℂ) + p * ZMod.stdAddChar j‖ ≤ Trace.mixrate K p := by
  have hK : (0 : ℝ) < K := by exact_mod_cast (NeZero.pos K)
  have hval1 : 1 ≤ j.val := Nat.one_le_iff_ne_zero.mpr ((ZMod.val_ne_zero j).mpr hj)
  have hvalK : j.val < K := ZMod.val_lt j
  have e1 : (ZMod.stdAddChar j : ℂ) = Complex.exp (((2 * Real.pi * j.val / K : ℝ) : ℂ) * Complex.I) := by
    rw [ZMod.stdAddChar_apply, ZMod.toCircle_apply]; push_cast; ring_nf
  have e2 : Trace.mixrate K p =
      ‖((1 - p : ℝ) : ℂ) + p * Complex.exp (((2 * Real.pi / K : ℝ) : ℂ) * Complex.I)‖ := by
    rw [Trace.mixrate]; push_cast; ring_nf
  rw [e1, e2, ← sq_le_sq₀ (norm_nonneg _) (norm_nonneg _), norm_sq_mode, norm_sq_mode]
  have := cos_mode_le (K := K) hval1 hvalK
  have hpp : 0 ≤ 2 * p * (1 - p) := by nlinarith
  nlinarith


/-- The conditional character of the answer has modulus at most `λ^{n−|S|}` at a mode `j ≠ 0`.
INTERNAL: the modulus statement in the proof of lem:target-contraction, with `|ρ_j| ≤ λ` applied.
TEXLINE: Paper/appendix.tex:278-282 -/
theorem norm_ex_char_answer_le [NeZero K] (T : Trace.Task K Sym) (hp0 : 0 ≤ T.p) (hp1 : T.p ≤ 1)
    {n : ℕ} (S : Finset (Fin n)) (w : Fin n → Sym) {j : ZMod K} (hj : j ≠ 0) :
    ‖∑ w', (T.letterWeight w' : ℂ) * ZMod.stdAddChar (j * T.answer (S.piecewise w w'))‖ ≤
      Trace.mixrate K T.p ^ (n - S.card) := by
  rw [ex_char_answer, norm_mul, AddChar.norm_apply, one_mul, norm_prod]
  calc ∏ t, ‖∑ x, ((T.μ x).toReal : ℂ) *
        ZMod.stdAddChar (j * if (if t ∈ S then w t else x) = T.wstar then 1 else 0)‖
      ≤ ∏ t, (if t ∈ S then (1 : ℝ) else Trace.mixrate K T.p) := by
        refine Finset.prod_le_prod (fun _ _ => norm_nonneg _) fun t _ => ?_
        by_cases ht : t ∈ S
        · simp only [ht, if_true]
          rw [sum_mu_char_const (K := K), AddChar.norm_apply]
        · simp only [ht, if_false]
          rw [sum_mu_char_mode]
          exact mode_le_mixrate hp0 hp1 j hj
    _ = Trace.mixrate K T.p ^ (n - S.card) := by
        rw [Finset.prod_ite]
        simp [Finset.filter_not, Finset.card_sdiff]

/-- Parseval on `ZMod K` for the standard characters.
INTERNAL: Mathlib has character orthogonality (`AddChar.sum_mulShift`) but no Parseval in this
form. -/
theorem parseval_zmod {K : ℕ} [NeZero K] (g : ZMod K → ℝ) :
    ((K : ℝ) * ∑ a, g a ^ 2 : ℝ) =
      ∑ j : ZMod K, ‖∑ a, (g a : ℂ) * ZMod.stdAddChar (j * a)‖ ^ 2 := by
  have hψ := ZMod.isPrimitive_stdAddChar K
  apply Complex.ofReal_injective
  have key : ∀ j : ZMod K, ((‖∑ a, (g a : ℂ) * ZMod.stdAddChar (j * a)‖ ^ 2 : ℝ) : ℂ) =
      ∑ a, ∑ b, (g a : ℂ) * g b * ZMod.stdAddChar (j * (a - b)) := by
    intro j
    rw [Complex.sq_norm, ← Complex.mul_conj, map_sum, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    rw [map_mul (starRingEnd ℂ), Complex.conj_ofReal, ← AddChar.map_neg_eq_conj, mul_sub, sub_eq_add_neg,
      AddChar.map_add_eq_mul]
    ring
  rw [Complex.ofReal_sum]
  simp_rw [key]
  push_cast
  rw [Finset.sum_comm]
  have h2 : ∀ a : ZMod K, ∑ j : ZMod K, ∑ b, (g a : ℂ) * g b * ZMod.stdAddChar (j * (a - b)) =
      K * (g a : ℂ) ^ 2 := by
    intro a
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, AddChar.sum_mulShift _ hψ, sub_eq_zero]
    simp [ZMod.card, sq]
    ring
  simp_rw [h2, Finset.mul_sum]


/-- The letter weights sum to one, in `ℂ`.
INTERNAL: bookkeeping. -/
theorem sum_letterWeight_complex (T : Trace.Task K Sym) (n : ℕ) :
    ∑ w : Fin n → Sym, (T.letterWeight w : ℂ) = 1 := by
  rw [← Complex.ofReal_sum]
  simp only [Trace.Task.letterWeight]
  rw [← Fintype.prod_sum (fun (_ : Fin n) a => (T.μ a).toReal)]
  simp [sum_mu_toReal]

/-- Fourier coefficient of the conditional signal.
INTERNAL: linearity of conditioning and `sum_targetSignal_char`.
TEXLINE: Paper/appendix.tex:276-282 -/
theorem transform_condSignal [NeZero K] (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n))
    (w : Fin n → Sym) (j : ZMod K) :
    ∑ a, (condSignal T S w a : ℂ) * ZMod.stdAddChar (j * a) =
      (∑ w', (T.letterWeight w' : ℂ) * ZMod.stdAddChar (j * T.answer (S.piecewise w w'))) -
        if j = 0 then 1 else 0 := by
  simp only [condSignal, Complex.ofReal_sum, Complex.ofReal_mul, Finset.sum_mul]
  rw [Finset.sum_comm]
  simp_rw [mul_assoc, ← Finset.mul_sum, sum_targetSignal_char, mul_sub, Finset.sum_sub_distrib,
    ← Finset.sum_mul, sum_letterWeight_complex, one_mul]

/-- **eq:horizon, in the form the contraction uses**: `‖E[G | F_S]‖² ≤ λ^{2(n−|S|)}`
pointwise in the frozen letters. The paper's sharper identity
`= K⁻¹ ∑_{j≠0} |ρ_j|^{2(n−|S|)} ≤ (1 − K⁻¹) λ^{2(n−|S|)}` is not needed downstream: the `j = 0`
term vanishes and every other term is bounded by `λ^{2(n−|S|)}`, so only the factor
`1 − K⁻¹` is dropped.
PAPER: Paper/appendix.tex:267-270 (eq:horizon, weakened by the factor `1 − 1/K`) -/
theorem horizonBound [NeZero K] (T : Trace.Task K Sym) (hp0 : 0 ≤ T.p) (hp1 : T.p ≤ 1)
    {n : ℕ} (S : Finset (Fin n)) (w : Fin n → Sym) :
    ∑ a, condSignal T S w a ^ 2 ≤ (Trace.mixrate K T.p ^ (n - S.card)) ^ 2 := by
  have hK : (0 : ℝ) < K := by exact_mod_cast (NeZero.pos K)
  have hb : ∀ j : ZMod K, ‖∑ a, (condSignal T S w a : ℂ) * ZMod.stdAddChar (j * a)‖ ^ 2 ≤
      (Trace.mixrate K T.p ^ (n - S.card)) ^ 2 := by
    intro j
    rw [transform_condSignal]
    by_cases hj : j = 0
    · subst hj
      simp only [zero_mul, AddChar.map_zero_eq_one, mul_one, sum_letterWeight_complex, if_true,
        sub_self, norm_zero]
      simpa using sq_nonneg _
    · simp only [hj, if_false, sub_zero]
      exact pow_le_pow_left₀ (norm_nonneg _) (norm_ex_char_answer_le T hp0 hp1 S w hj) 2
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hb j
  rw [← parseval_zmod, Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul] at hsum
  exact le_of_mul_le_mul_left hsum hK

end Trace.Proof.HorizonBound

/-! ### Run record
Newest first. History, not instruction — what this file claims is above.

* r1 · proved · `horizonBound` and all helpers, fully closed
-/
