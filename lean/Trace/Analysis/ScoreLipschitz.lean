import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Analysis.Contraction
import Trace.Analysis.ScoreLocality

/-!
# From locality costs to a Lipschitz bound on a credit functional

INTERNAL: the summation step of `lem:score-sum` (Paper/appendix.tex:507-521). If the family
`F θ w` has decompositions on `B` of cost close to `M` at weight `κ = λ` and scale `r`, then
`θ ↦ E⟨F θ, y − u⟩` is `λ^n M / r`-Lipschitz on `B`: the piece reading the letters in `S`
moves by at most `l_S ‖θ − θ'‖`, and part (i) (`unseenLettersContraction`) charges it
`λ^{n−|S|} = λ^n κ^{−|S|}`.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.SLip

open Trace Trace.Proof.SL Matrix

variable {K : ℕ} [NeZero K] {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {n : ℕ}

/-- INTERNAL: part (i) for a vector-valued feature reading the letters in `S`, with the
`L²` norm bounded by the sup norm (as `piece_credit` in `LetterDecomposition.lean`). -/
theorem pieceCredit (hprior : Prior) (hK : 2 ≤ K) (T : Task K Sym) (hp0 : 0 < T.p)
    (hp1 : T.p < 1) (S : Finset (Fin n)) (h : (Fin n → Sym) → EuclideanSpace ℝ (ZMod K))
    (hS : DependsOn h (S : Set (Fin n))) {b : ℝ} (hb0 : 0 ≤ b) (hb : ∀ w, ‖h w‖ ≤ b) :
    |T.Ex (fun w => (h w).ofLp ⬝ᵥ T.targetSignal w)| ≤ mixrate K T.p ^ (n - S.card) * b := by
  set H : (Fin n → Sym) → Matrix (ZMod K) Unit ℝ := fun w a _ => h w a with hHdef
  have hH : DependsOn H (S : Set (Fin n)) := by
    intro x y hxy
    show (fun a _ => h x a) = fun a _ => h y a
    rw [hS hxy]
  have key := unseenLettersContraction hprior hK T hp0 hp1 n S H hH
  have h1 : featureCredit T H = |T.Ex (fun w => (h w).ofLp ⬝ᵥ T.targetSignal w)| := by
    unfold featureCredit
    rw [EuclideanSpace.norm_eq, Fintype.sum_unique, Real.norm_eq_abs, sq_abs, Real.sqrt_sq_eq_abs]
    congr 1
    simp only [Task.Ex, hHdef, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    rfl
  have h2 : T.l2Norm H ≤ b := by
    unfold Task.l2Norm
    rw [Real.sqrt_le_left hb0]
    calc T.Ex (fun w => ∑ a, ∑ i, H w a i ^ 2) = ∑ w, T.letterWeight w * ‖h w‖ ^ 2 := by
          simp only [Task.Ex, hHdef, Fintype.sum_unique, smul_eq_mul, EuclideanSpace.norm_eq,
            Real.norm_eq_abs, sq_abs]
          congr 1; ext w
          rw [Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)]
      _ ≤ ∑ w, T.letterWeight w * b ^ 2 := by
          gcongr with w
          · exact Trace.Proof.Contraction.letterWeight_nonneg T w
          · exact hb w
      _ = b ^ 2 := by rw [← Finset.sum_mul, Trace.Proof.Contraction.sum_letterWeight, one_mul]
  rw [← h1]
  exact key.trans (mul_le_mul_of_nonneg_left h2 (pow_nonneg (norm_nonneg _) _))

variable {P : Type*} [PseudoMetricSpace P] {B : Set P}

/-- INTERNAL: the credit of one decomposition: `|Φ θ − Φ θ'| ≤ λ^n (cost / r) dist θ θ'`. -/
theorem credit_lip_dec (hprior : Prior) (hK : 2 ≤ K) (T : Task K Sym) (hp0 : 0 < T.p)
    (hp1 : T.p < 1) (hlam0 : 0 < mixrate K T.p) {r : ℝ} (hr : 0 < r)
    {F : P → (Fin n → Sym) → EuclideanSpace ℝ (ZMod K)} (d : Dec B F) {θ θ' : P}
    (hθ : θ ∈ B) (hθ' : θ' ∈ B) :
    |T.Ex (fun w => (F θ w).ofLp ⬝ᵥ T.targetSignal w) -
        T.Ex (fun w => (F θ' w).ofLp ⬝ᵥ T.targetSignal w)| ≤
      mixrate K T.p ^ n * (d.cost (mixrate K T.p) r / r) * dist θ θ' := by
  set lam := mixrate K T.p with hlam
  have hsplit : T.Ex (fun w => (F θ w).ofLp ⬝ᵥ T.targetSignal w) -
      T.Ex (fun w => (F θ' w).ofLp ⬝ᵥ T.targetSignal w) =
      ∑ S, T.Ex (fun w => (d.piece S θ w - d.piece S θ' w).ofLp ⬝ᵥ T.targetSignal w) := by
    simp only [Task.Ex, ← Finset.sum_sub_distrib, ← smul_sub]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [d.eq θ hθ w, d.eq θ' hθ' w, ← Finset.smul_sum]
    congr 1
    simp only [WithLp.ofLp_sum, WithLp.ofLp_sub, sum_dotProduct, sub_dotProduct,
      Finset.sum_sub_distrib]
  rw [hsplit]
  have hD := dist_nonneg (x := θ) (y := θ')
  have hpiece : ∀ S : Finset (Fin n), |T.Ex (fun w => (d.piece S θ w - d.piece S θ' w).ofLp ⬝ᵥ
      T.targetSignal w)| ≤ lam ^ n * (lam⁻¹ ^ S.card * d.l S) * dist θ θ' := by
    intro S
    have hdep : DependsOn (fun w => d.piece S θ w - d.piece S θ' w) (S : Set (Fin n)) :=
      fun x y hxy => by simp only; rw [d.dep S θ hxy, d.dep S θ' hxy]
    have := pieceCredit hprior hK T hp0 hp1 S _ hdep (mul_nonneg (d.l_nonneg S) hD)
      (fun w => d.lip S θ hθ θ' hθ' w)
    refine this.trans (le_of_eq ?_)
    have hcard : S.card ≤ n := by
      simpa using Finset.card_le_univ S
    have e : lam ^ n = lam ^ (n - S.card) * lam ^ S.card := by
      rw [← pow_add, Nat.sub_add_cancel hcard]
    rw [e, inv_pow, ← hlam]
    field_simp
  refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun S _ => hpiece S).trans ?_)
  rw [← Finset.sum_mul, ← Finset.mul_sum]
  apply mul_le_mul_of_nonneg_right _ hD
  apply mul_le_mul_of_nonneg_left _ (pow_nonneg hlam0.le _)
  rw [Dec.cost, Finset.sum_div]
  refine Finset.sum_le_sum fun S _ => ?_
  have hk : 0 ≤ lam⁻¹ ^ S.card := pow_nonneg (inv_nonneg.2 hlam0.le) _
  have hs := d.s_nonneg S
  have hl := d.l_nonneg S
  rw [le_div_iff₀ hr]
  nlinarith [mul_nonneg hk hs]

/-- INTERNAL: **summation of the credits** (Paper/appendix.tex:507-521): a family of cost `M`
at weight `λ` and scale `r` gives a `λ^n M / r`-Lipschitz credit functional on `B`. -/
theorem credit_lip (hprior : Prior) (hK : 2 ≤ K) (T : Task K Sym) (hp0 : 0 < T.p)
    (hp1 : T.p < 1) (hlam0 : 0 < mixrate K T.p) {r M : ℝ} (hr : 0 < r)
    {F : P → (Fin n → Sym) → EuclideanSpace ℝ (ZMod K)} (hF : Bd B (mixrate K T.p) r F M)
    {θ θ' : P} (hθ : θ ∈ B) (hθ' : θ' ∈ B) :
    |T.Ex (fun w => (F θ w).ofLp ⬝ᵥ T.targetSignal w) -
        T.Ex (fun w => (F θ' w).ofLp ⬝ᵥ T.targetSignal w)| ≤
      mixrate K T.p ^ n * (M / r) * dist θ θ' := by
  set lam := mixrate K T.p with hlam
  have hD := dist_nonneg (x := θ) (y := θ')
  refine le_of_forall_pos_le_add fun η hη => ?_
  have hc : 0 < lam ^ n * dist θ θ' + 1 := by positivity
  obtain ⟨d, hd⟩ := hF (η * r / (lam ^ n * dist θ θ' + 1)) (by positivity)
  refine (credit_lip_dec hprior hK T hp0 hp1 hlam0 hr d hθ hθ').trans ?_
  have h1 : d.cost lam r / r ≤ M / r + η / (lam ^ n * dist θ θ' + 1) := by
    rw [div_add_div _ _ hr.ne' hc.ne', le_div_iff₀ (by positivity)]
    calc d.cost lam r / r * (r * (lam ^ n * dist θ θ' + 1))
        = d.cost lam r * (lam ^ n * dist θ θ' + 1) := by field_simp
      _ ≤ (M + η * r / (lam ^ n * dist θ θ' + 1)) * (lam ^ n * dist θ θ' + 1) :=
          mul_le_mul_of_nonneg_right hd hc.le
      _ = M * (lam ^ n * dist θ θ' + 1) + r * η := by field_simp
  have hpow : 0 ≤ lam ^ n := pow_nonneg hlam0.le _
  calc lam ^ n * (d.cost lam r / r) * dist θ θ'
      ≤ lam ^ n * (M / r + η / (lam ^ n * dist θ θ' + 1)) * dist θ θ' := by gcongr
    _ = lam ^ n * (M / r) * dist θ θ' +
          η * (lam ^ n * dist θ θ' / (lam ^ n * dist θ θ' + 1)) := by ring
    _ ≤ lam ^ n * (M / r) * dist θ θ' + η * 1 := by
        gcongr
        rw [div_le_one hc]; linarith
    _ = _ := by ring

end Trace.Proof.SLip
