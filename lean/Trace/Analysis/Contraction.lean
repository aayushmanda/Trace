import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Meta.ModelClosure
import Mathlib.Logic.Function.DependsOn
import Trace.Analysis.HorizonBound

/-!
# Conditional contraction (lem:target-contraction, part (i) of thm:main)

`Trace.Proof.unseenLettersContraction`: for every feature `H` with `K` rows that depends only on
the letters in `S`, `‖E[Hᵀ(y − u)]‖ ≤ λ^{n−|S|} ‖H‖_{L²}`.

The proof follows Paper/appendix.tex:276-289:
* `ex_mul_eq_ex_mul_condExp` is the tower property, proved on the finite product space by the
  involution `swapOff` that exchanges the letters outside `S` between two independent samples;
* `Trace.Proof.HorizonBound.horizonBound` (in `Trace/Analysis/HorizonBound.lean`) is eq:horizon,
  `‖E[G | F_S]‖ ≤ λ^{n−|S|}` pointwise;
* `featureCredit_le_of_tower` is the Cauchy–Schwarz step.

`hprior`, `hK` and the strict bounds `0 < p < 1` are not used: the bound holds for every
`p ∈ [0, 1]` and every `K ≥ 1`.
-/

namespace Trace.Proof.Contraction
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix
variable {K : ℕ} {Sym : Type u} [Fintype Sym] [DecidableEq Sym]

/-- The letter weights sum to one.
INTERNAL: bookkeeping. -/
theorem sum_letterWeight (T : Trace.Task K Sym) (n : ℕ) :
    ∑ w : Fin n → Sym, T.letterWeight w = 1 := by
  simp only [Trace.Task.letterWeight]
  rw [← Fintype.prod_sum (fun (_ : Fin n) a => (T.μ a).toReal)]
  simp [Trace.Proof.HorizonBound.sum_mu_toReal]

/-- INTERNAL: letter weights are nonnegative. -/
theorem letterWeight_nonneg (T : Trace.Task K Sym) {n : ℕ} (w : Fin n → Sym) :
    0 ≤ T.letterWeight w :=
  Finset.prod_nonneg fun _ _ => ENNReal.toReal_nonneg

/-- Swap the coordinates of a pair of letter sequences outside `S`.
INTERNAL: the involution behind the tower property. -/
def swapOff {n : ℕ} (S : Finset (Fin n)) (q : (Fin n → Sym) × (Fin n → Sym)) :
    (Fin n → Sym) × (Fin n → Sym) :=
  (S.piecewise q.1 q.2, S.piecewise q.2 q.1)

/-- INTERNAL: `swapOff` is an involution. -/
theorem swapOff_involutive {n : ℕ} (S : Finset (Fin n)) :
    Function.Involutive (swapOff (Sym := Sym) S) := by
  intro q
  ext i <;> by_cases h : i ∈ S <;> simp [swapOff, Finset.piecewise, h]

/-- INTERNAL: `swapOff` preserves the product weight of the pair (letters are i.i.d.). -/
theorem letterWeight_swapOff (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n))
    (q : (Fin n → Sym) × (Fin n → Sym)) :
    T.letterWeight (swapOff S q).1 * T.letterWeight (swapOff S q).2 =
      T.letterWeight q.1 * T.letterWeight q.2 := by
  simp only [Trace.Task.letterWeight, ← Finset.prod_mul_distrib]
  refine Finset.prod_congr rfl fun i _ => ?_
  by_cases h : i ∈ S <;> simp [swapOff, Finset.piecewise, h, mul_comm]

/-- **Tower property.** For `f` depending only on the letters in `S`,
`E[f g] = E[f · E[g | F_S]]`, with `E[g | F_S](w) = E_{w'} g(w on S, w' off S)`.
INTERNAL: the step labelled "tower property" in the proof of lem:target-contraction.
TEXLINE: Paper/appendix.tex:283-286 -/
theorem ex_mul_eq_ex_mul_condExp (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n))
    (f g : (Fin n → Sym) → ℝ) (hf : DependsOn f (S : Set (Fin n))) :
    ∑ w, T.letterWeight w * (f w * g w) =
      ∑ w, T.letterWeight w * (f w * ∑ w', T.letterWeight w' * g (S.piecewise w w')) := by
  set F : (Fin n → Sym) × (Fin n → Sym) → ℝ :=
    fun q => T.letterWeight q.1 * T.letterWeight q.2 * (f q.1 * g q.1) with hF
  have h1 : ∑ q, F q = ∑ w, T.letterWeight w * (f w * g w) := by
    rw [Fintype.sum_prod_type]
    simp only [hF]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [← Finset.sum_mul, ← Finset.mul_sum, sum_letterWeight, mul_one]
  have h2 : ∑ q, F (swapOff S q) =
      ∑ w, T.letterWeight w * (f w * ∑ w', T.letterWeight w' * g (S.piecewise w w')) := by
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun w' _ => ?_
    have hfw : f (S.piecewise w w') = f w :=
      hf fun i hi => by simp [Finset.piecewise, Finset.mem_coe.mp hi]
    simp only [hF]
    rw [letterWeight_swapOff]
    simp only [swapOff, hfw]
    ring
  rw [← h1, ← h2]
  exact (Equiv.sum_comp (Function.Involutive.toPerm _ (swapOff_involutive S)) F).symm

/-- Cauchy–Schwarz half of lem:target-contraction: once the tower property holds for some
`Gb` with `‖Gb w‖ ≤ lam` pointwise, the credit of an `S`-measurable `H` is at most
`lam · ‖H‖_{L²}`.
INTERNAL: the last two lines of the alignat display in the proof of lem:target-contraction,
with the supremum over unit `v` replaced by testing against `v = E[Hᵀ G]` itself.
TEXLINE: Paper/appendix.tex:283-289 -/
theorem featureCredit_le_of_tower [NeZero K] (T : Trace.Task K Sym) {n : ℕ} (S : Finset (Fin n)) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) (hH : DependsOn H (S : Set (Fin n)))
    (Gb : (Fin n → Sym) → ZMod K → ℝ) (lam : ℝ) (hlam : 0 ≤ lam)
    (hP : ∀ w : Fin n → Sym, 0 ≤ T.letterWeight w) (hP1 : ∑ w : Fin n → Sym, T.letterWeight w = 1)
    (tower : ∀ f : (Fin n → Sym) → ℝ, DependsOn f (S : Set (Fin n)) → ∀ a,
      ∑ w, T.letterWeight w * (f w * T.targetSignal w a) =
        ∑ w, T.letterWeight w * (f w * Gb w a))
    (hGb : ∀ w, ∑ a, Gb w a ^ 2 ≤ lam ^ 2) :
    Trace.featureCredit T H ≤ lam * T.l2Norm H := by
  set x : ι → ℝ := T.Ex fun w => (H w)ᵀ *ᵥ T.targetSignal w with hx
  have hxi : ∀ i, x i = ∑ w, T.letterWeight w * ∑ a, H w a i * T.targetSignal w a := by
    intro i
    simp [hx, Trace.Task.Ex, Finset.sum_apply, Matrix.mulVec, dotProduct]
  have hN : Trace.featureCredit T H = √(∑ i, x i ^ 2) := by
    rw [Trace.featureCredit, EuclideanSpace.norm_eq]
    simp [hx]
  set N := √(∑ i, x i ^ 2) with hNdef
  set f : ZMod K → (Fin n → Sym) → ℝ := fun a w => ∑ i, H w a i * x i with hf
  have hfdep : ∀ a, DependsOn (f a) (S : Set (Fin n)) := by
    intro a w w' h
    simp only [hf, hH h]
  set F : (Fin n → Sym) → ℝ := fun w => ∑ a, ∑ i, H w a i ^ 2 with hF
  have hX : ∑ i, x i ^ 2 = ∑ w, T.letterWeight w * ∑ a, f a w * Gb w a := by
    calc ∑ i, x i ^ 2 = ∑ a, ∑ w, T.letterWeight w * (f a w * T.targetSignal w a) := by
          have e : ∀ i, x i ^ 2 = (∑ w, T.letterWeight w * ∑ a, H w a i * T.targetSignal w a) * x i := by
            intro i; rw [sq]; nth_rewrite 1 [hxi i]; rfl
          simp only [e, hf, Finset.mul_sum, Finset.sum_mul]
          rw [Finset.sum_comm, Finset.sum_comm (s := Finset.univ (α := ZMod K))]
          refine Finset.sum_congr rfl fun w _ => ?_
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun i _ => ?_
          ring
      _ = ∑ a, ∑ w, T.letterWeight w * (f a w * Gb w a) :=
          Finset.sum_congr rfl fun a _ => tower (f a) (hfdep a) a
      _ = _ := by
          rw [Finset.sum_comm]; simp only [Finset.mul_sum]
  have hfa : ∀ w, √(∑ a, f a w ^ 2) ≤ √(F w) * N := by
    intro w
    have hF0 : 0 ≤ F w := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
    rw [hNdef, ← Real.sqrt_mul hF0]
    apply Real.sqrt_le_sqrt
    show ∑ a, f a w ^ 2 ≤ (∑ a, ∑ i, H w a i ^ 2) * ∑ i, x i ^ 2
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun a _ => Finset.sum_mul_sq_le_sq_mul_sq _ (fun i => H w a i) x
  have hL : ∑ w, T.letterWeight w * √(F w) ≤ T.l2Norm H := by
    have := Real.sum_mul_le_sqrt_mul_sqrt Finset.univ (fun w => √(T.letterWeight w))
      (fun w => √(T.letterWeight w) * √(F w))
    convert this using 1
    · refine Finset.sum_congr rfl fun w _ => ?_
      rw [← mul_assoc, Real.mul_self_sqrt (hP w)]
    · have hF0 : ∀ w, 0 ≤ F w := fun w =>
        Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
      simp only [mul_pow, Real.sq_sqrt (hP _), Real.sq_sqrt (hF0 _), hP1, Real.sqrt_one, one_mul]
      simp [Trace.Task.l2Norm, Trace.Task.Ex, hF]
  have hXle : ∑ i, x i ^ 2 ≤ lam * T.l2Norm H * N := by
    rw [hX]
    calc ∑ w, T.letterWeight w * ∑ a, f a w * Gb w a
        ≤ ∑ w, T.letterWeight w * (√(F w) * N * lam) := by
          refine Finset.sum_le_sum fun w _ => mul_le_mul_of_nonneg_left ?_ (hP w)
          refine (Real.sum_mul_le_sqrt_mul_sqrt _ _ _).trans ?_
          refine mul_le_mul (hfa w) ?_ (Real.sqrt_nonneg _) (by positivity)
          exact Real.sqrt_le_sqrt (hGb w) |>.trans_eq (Real.sqrt_sq hlam)
      _ = (∑ w, T.letterWeight w * √(F w)) * (lam * N) := by
          rw [Finset.sum_mul]; refine Finset.sum_congr rfl fun w _ => ?_; ring
      _ ≤ T.l2Norm H * (lam * N) := mul_le_mul_of_nonneg_right hL (by positivity)
      _ = _ := by ring
  rw [hN]
  have hN0 : 0 ≤ N := Real.sqrt_nonneg _
  have hNsq : N ^ 2 = ∑ i, x i ^ 2 := Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hL0 : 0 ≤ T.l2Norm H := Real.sqrt_nonneg _
  rcases hN0.eq_or_lt with h0 | hpos
  · rw [← h0]; positivity
  · nlinarith

end Trace.Proof.Contraction

section
open scoped Matrix

/-- Proof-side owner for `Trace.unseen_letters_contraction`; its statement is fixed by the proof charter.
PAPER: Paper/appendix.tex:264-290 (lem:target-contraction; eq:credit-degree, Paper/body.tex:178) -/
theorem Trace.Proof.unseenLettersContraction (hprior : Trace.Prior)
    {K : ℕ} [NeZero K] (hK : 2 ≤ K)
    {Sym : Type u} [Fintype Sym] [DecidableEq Sym] (T : Trace.Task K Sym)
    (hp0 : 0 < T.p) (hp1 : T.p < 1)
    (n : ℕ) (S : Finset (Fin n)) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) (hH : DependsOn H (S : Set (Fin n))) :
    Trace.featureCredit T H ≤ Trace.mixrate K T.p ^ (n - S.card) * T.l2Norm H := by
  refine Trace.Proof.Contraction.featureCredit_le_of_tower T S H hH
    (Trace.Proof.HorizonBound.condSignal T S) _ (pow_nonneg (norm_nonneg _) _)
    (Trace.Proof.Contraction.letterWeight_nonneg T) (Trace.Proof.Contraction.sum_letterWeight T n)
    (fun f hf a => Trace.Proof.Contraction.ex_mul_eq_ex_mul_condExp T S f
      (fun w => T.targetSignal w a) hf)
    (Trace.Proof.HorizonBound.horizonBound T hp0.le hp1.le S)

end

/-! ### Run record
Newest first. History, not instruction — what this file claims is above.

* r1 · proved · `unseenLettersContraction` via tower (`swapOff`) + `horizonBound` + Cauchy–Schwarz
-/
