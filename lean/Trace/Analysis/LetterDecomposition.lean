import Trace.Model.Prelude
import Mathlib.Analysis.Normed.Operator.Basic
import Trace.Model.Prior
import Trace.Analysis.Contraction

/-!
# One-letter decompositions and their Lipschitz tracking

A function `f` of the letters has a *one-letter decomposition* of size `(α, β)` when
`f w = f₀ + ∑_t g_t(w_t)` with `‖f₀‖ ≤ α` and `∑_t sup_a ‖g_t(a)‖ ≤ β`. This is the bookkeeping
behind "(i): 0 or 1 letter" in the proof of `lem:score-sum` (Paper/appendix.tex:510-512): the
letter-free part is charged `λ^n`, each one-letter part `λ^{n-1}`.

`Track U θc f α₀ β₀ α₁ β₁` says that `f θc` has size `(α₀, β₀)` and that, for `θ ∈ U`, the
difference `f θ − f θc` has size `(α₁ ‖θ − θc‖, β₁ ‖θ − θc‖)`. Closure under sums and under
parameter-dependent linear maps is what propagates it through a uniform-attention network.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof

open scoped BigOperators

section Dec

variable {Sym : Type u} {n : ℕ} {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- One-letter decomposition of size `(α, β)`. INTERNAL: bookkeeping for "0 or 1 letter".
TEXLINE: Paper/appendix.tex:510-512 -/
def LetterDec (f : (Fin n → Sym) → E) (α β : ℝ) : Prop :=
  ∃ (f0 : E) (g : Fin n → Sym → E) (b : Fin n → ℝ),
    (∀ w, f w = f0 + ∑ t, g t (w t)) ∧ ‖f0‖ ≤ α ∧ (∀ t, 0 ≤ b t) ∧
      (∀ t a, ‖g t a‖ ≤ b t) ∧ ∑ t, b t ≤ β

namespace LetterDec

variable {f g : (Fin n → Sym) → E} {α β α' β' : ℝ}

/-- INTERNAL: sizes are nonnegative. -/
theorem nonneg (h : LetterDec f α β) : 0 ≤ α ∧ 0 ≤ β := by
  obtain ⟨f0, g, b, -, h0, hb, -, hs⟩ := h
  exact ⟨(norm_nonneg _).trans h0, (Finset.sum_nonneg fun t _ => hb t).trans hs⟩

/-- INTERNAL: monotonicity in the sizes. -/
theorem mono (h : LetterDec f α β) (hα : α ≤ α') (hβ : β ≤ β') : LetterDec f α' β' := by
  obtain ⟨f0, g, b, hf, h0, hb, hg, hs⟩ := h
  exact ⟨f0, g, b, hf, h0.trans hα, hb, hg, hs.trans hβ⟩

/-- INTERNAL: a constant is letter-free. -/
theorem const (c : E) : LetterDec (fun _ : Fin n → Sym => c) ‖c‖ 0 :=
  ⟨c, fun _ _ => 0, fun _ => 0, fun w => by simp, le_rfl, fun _ => le_rfl,
    fun _ _ => by simp, by simp⟩

/-- INTERNAL: a function of one letter. -/
theorem letter (t : Fin n) (h : Sym → E) {B : ℝ} (hB : 0 ≤ B) (hh : ∀ a, ‖h a‖ ≤ B) :
    LetterDec (fun w : Fin n → Sym => h (w t)) 0 B := by
  classical
  refine ⟨0, fun t' a => if t' = t then h a else 0, fun t' => if t' = t then B else 0,
    fun w => by simp, by simp, fun t' => by dsimp only; split_ifs <;> simp [hB], fun t' a => ?_,
    by simp⟩
  dsimp only; split_ifs <;> simp [hh]

/-- INTERNAL: sums. -/
theorem add (hf : LetterDec f α β) (hg : LetterDec g α' β') :
    LetterDec (fun w => f w + g w) (α + α') (β + β') := by
  obtain ⟨f0, g1, b1, hf, h0, hb1, hg1, hs1⟩ := hf
  obtain ⟨f0', g2, b2, hf', h0', hb2, hg2, hs2⟩ := hg
  refine ⟨f0 + f0', fun t a => g1 t a + g2 t a, fun t => b1 t + b2 t, fun w => ?_,
    (norm_add_le _ _).trans (add_le_add h0 h0'), fun t => add_nonneg (hb1 t) (hb2 t),
    fun t a => (norm_add_le _ _).trans (add_le_add (hg1 t a) (hg2 t a)), ?_⟩
  · dsimp only; rw [hf, hf', Finset.sum_add_distrib]; abel
  · rw [Finset.sum_add_distrib]; exact add_le_add hs1 hs2

/-- INTERNAL: a bounded linear map. -/
theorem map (L : E →L[ℝ] F) (hf : LetterDec f α β) :
    LetterDec (fun w => L (f w)) (‖L‖ * α) (‖L‖ * β) := by
  obtain ⟨f0, g, b, hf, h0, hb, hg, hs⟩ := hf
  refine ⟨L f0, fun t a => L (g t a), fun t => ‖L‖ * b t, fun w => by dsimp only; rw [hf, map_add, map_sum],
    (L.le_opNorm _).trans (mul_le_mul_of_nonneg_left h0 (norm_nonneg _)),
    fun t => mul_nonneg (norm_nonneg _) (hb t),
    fun t a => (L.le_opNorm _).trans (mul_le_mul_of_nonneg_left (hg t a) (norm_nonneg _)), ?_⟩
  rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left hs (norm_nonneg _)

/-- INTERNAL: a linear map of norm at most `m`. -/
theorem map_le (L : E →L[ℝ] F) {m : ℝ} (hL : ‖L‖ ≤ m) (hf : LetterDec f α β) :
    LetterDec (fun w => L (f w)) (m * α) (m * β) :=
  (hf.map L).mono (mul_le_mul_of_nonneg_right hL hf.nonneg.1)
    (mul_le_mul_of_nonneg_right hL hf.nonneg.2)

/-- INTERNAL: scalar multiples. -/
theorem smul (c : ℝ) (hf : LetterDec f α β) :
    LetterDec (fun w => c • f w) (|c| * α) (|c| * β) := by
  obtain ⟨f0, g, b, hf, h0, hb, hg, hs⟩ := hf
  refine ⟨c • f0, fun t a => c • g t a, fun t => |c| * b t,
    fun w => by dsimp only; rw [hf, smul_add, Finset.smul_sum], ?_,
    fun t => mul_nonneg (abs_nonneg _) (hb t), fun t a => ?_, ?_⟩
  · rw [norm_smul, Real.norm_eq_abs]; exact mul_le_mul_of_nonneg_left h0 (abs_nonneg _)
  · rw [norm_smul, Real.norm_eq_abs]; exact mul_le_mul_of_nonneg_left (hg t a) (abs_nonneg _)
  · rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left hs (abs_nonneg _)

/-- INTERNAL: differences. -/
theorem sub (hf : LetterDec f α β) (hg : LetterDec g α' β') :
    LetterDec (fun w => f w - g w) (α + α') (β + β') := by
  have := hf.add (hg.smul (-1))
  simp only [abs_neg, abs_one, one_mul, neg_one_smul, ← sub_eq_add_neg] at this
  exact this

/-- INTERNAL: finite sums. -/
theorem sum {ι : Type*} (s : Finset ι) {f : ι → (Fin n → Sym) → E} {α β : ι → ℝ}
    (h : ∀ j ∈ s, LetterDec (f j) (α j) (β j)) :
    LetterDec (fun w => ∑ j ∈ s, f j w) (∑ j ∈ s, α j) (∑ j ∈ s, β j) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using (const (n := n) (Sym := Sym) (0 : E))
  | insert j s hj ih =>
    simp only [Finset.sum_insert hj]
    exact (h j (Finset.mem_insert_self _ _)).add (ih fun k hk => h k (Finset.mem_insert_of_mem hk))

/-- INTERNAL: a pointwise bound. -/
theorem norm_le (h : LetterDec f α β) (w : Fin n → Sym) : ‖f w‖ ≤ α + β := by
  obtain ⟨f0, g, b, hf, h0, hb, hg, hs⟩ := h
  rw [hf]
  refine (norm_add_le _ _).trans (add_le_add h0 ?_)
  exact (norm_sum_le _ _).trans ((Finset.sum_le_sum fun t _ => hg t (w t)).trans hs)

/-- INTERNAL: congruence. -/
theorem congr (h : LetterDec f α β) (hfg : ∀ w, f w = g w) : LetterDec g α β := by
  rwa [show g = f from funext fun w => (hfg w).symm]

/-- INTERNAL: congruence, with the sizes rewritten too. -/
theorem congr' (h : LetterDec f α β) (hfg : ∀ w, f w = g w) (ha : α = α') (hb : β = β') :
    LetterDec g α' β' := by
  subst ha hb; exact h.congr hfg

end LetterDec

/-- Lipschitz tracking of one-letter decompositions around a centre `θc`.
INTERNAL: bookkeeping for the Lipschitz route to the gradient bound.
TEXLINE: Paper/appendix.tex:510-512 -/
def Track {P : Type*} [NormedAddCommGroup P] (U : Set P) (θc : P)
    (f : P → (Fin n → Sym) → E) (α₀ β₀ α₁ β₁ : ℝ) : Prop :=
  LetterDec (f θc) α₀ β₀ ∧
    ∀ θ ∈ U, LetterDec (fun w => f θ w - f θc w) (α₁ * ‖θ - θc‖) (β₁ * ‖θ - θc‖)

namespace Track

variable {P : Type*} [NormedAddCommGroup P] {U : Set P} {θc : P}
  {f g : P → (Fin n → Sym) → E} {α₀ β₀ α₁ β₁ α₀' β₀' α₁' β₁' : ℝ}

/-- INTERNAL: monotonicity. -/
theorem mono (h : Track U θc f α₀ β₀ α₁ β₁) (h0 : α₀ ≤ α₀') (h1 : β₀ ≤ β₀') (h2 : α₁ ≤ α₁')
    (h3 : β₁ ≤ β₁') : Track U θc f α₀' β₀' α₁' β₁' :=
  ⟨h.1.mono h0 h1, fun θ hθ => (h.2 θ hθ).mono (mul_le_mul_of_nonneg_right h2 (norm_nonneg _))
    (mul_le_mul_of_nonneg_right h3 (norm_nonneg _))⟩

/-- INTERNAL: sums. -/
theorem add (hf : Track U θc f α₀ β₀ α₁ β₁) (hg : Track U θc g α₀' β₀' α₁' β₁') :
    Track U θc (fun θ w => f θ w + g θ w) (α₀ + α₀') (β₀ + β₀') (α₁ + α₁') (β₁ + β₁') := by
  refine ⟨hf.1.add hg.1, fun θ hθ => ((hf.2 θ hθ).add (hg.2 θ hθ)).congr' ?_ ?_ ?_⟩
  all_goals first | (intro w; dsimp only; abel) | ring

/-- INTERNAL: constant scalar multiples. -/
theorem smul (c : ℝ) (hf : Track U θc f α₀ β₀ α₁ β₁) :
    Track U θc (fun θ w => c • f θ w) (|c| * α₀) (|c| * β₀) (|c| * α₁) (|c| * β₁) := by
  refine ⟨hf.1.smul c, fun θ hθ => ?_⟩
  have := (hf.2 θ hθ).smul c
  simp only [smul_sub, ← mul_assoc] at this
  exact this

/-- INTERNAL: finite sums. -/
theorem sum {ι : Type*} (s : Finset ι) {f : ι → P → (Fin n → Sym) → E}
    {α₀ β₀ α₁ β₁ : ι → ℝ} (h : ∀ j ∈ s, Track U θc (f j) (α₀ j) (β₀ j) (α₁ j) (β₁ j)) :
    Track U θc (fun θ w => ∑ j ∈ s, f j θ w) (∑ j ∈ s, α₀ j) (∑ j ∈ s, β₀ j)
      (∑ j ∈ s, α₁ j) (∑ j ∈ s, β₁ j) := by
  refine ⟨LetterDec.sum s fun j hj => (h j hj).1, fun θ hθ => ?_⟩
  have := LetterDec.sum s fun j hj => (h j hj).2 θ hθ
  simp only [← Finset.sum_mul] at this
  exact this.congr fun w => Finset.sum_sub_distrib _ _

/-- INTERNAL: a letter-free, parameter-dependent vector. -/
theorem const (v : P → E) (h0 : ‖v θc‖ ≤ α₀) (h1 : ∀ θ ∈ U, ‖v θ - v θc‖ ≤ α₁ * ‖θ - θc‖) :
    Track U θc (fun θ (_ : Fin n → Sym) => v θ) α₀ 0 α₁ 0 :=
  ⟨(LetterDec.const _).mono h0 le_rfl, fun θ hθ =>
    (LetterDec.const _).mono (h1 θ hθ) (by simp)⟩

/-- INTERNAL: a parameter-dependent function of one letter. -/
theorem letter (t : Fin n) (h : P → Sym → E) (hβ₀ : 0 ≤ β₀) (hβ₁ : 0 ≤ β₁)
    (h0 : ∀ a, ‖h θc a‖ ≤ β₀) (h1 : ∀ θ ∈ U, ∀ a, ‖h θ a - h θc a‖ ≤ β₁ * ‖θ - θc‖) :
    Track U θc (fun θ (w : Fin n → Sym) => h θ (w t)) 0 β₀ 0 β₁ :=
  ⟨LetterDec.letter t _ hβ₀ h0, fun θ hθ =>
    (LetterDec.letter t (fun a => h θ a - h θc a) (mul_nonneg hβ₁ (norm_nonneg _))
      (h1 θ hθ)).mono (by simp) le_rfl⟩

/-- INTERNAL: a parameter-dependent bounded linear map, the product rule
`M'g' − Mg = M'(g' − g) + (M' − M)g`. -/
theorem clm (M : P → E →L[ℝ] F) {m₀ m₁ : ℝ} (hc : θc ∈ U) (hM : ∀ θ ∈ U, ‖M θ‖ ≤ m₀)
    (hML : ∀ θ ∈ U, ‖M θ - M θc‖ ≤ m₁ * ‖θ - θc‖) (hf : Track U θc f α₀ β₀ α₁ β₁) :
    Track U θc (fun θ w => M θ (f θ w)) (m₀ * α₀) (m₀ * β₀) (m₀ * α₁ + m₁ * α₀)
      (m₀ * β₁ + m₁ * β₀) := by
  refine ⟨hf.1.map_le _ (hM θc hc), fun θ hθ => ?_⟩
  have h1 := (hf.2 θ hθ).map_le (M θ) (hM θ hθ)
  have h2 := hf.1.map_le (M θ - M θc) (hML θ hθ)
  refine (h1.add h2).congr' (fun w => ?_) (by ring) (by ring)
  simp only [_root_.sub_apply, map_sub]; abel

/-- INTERNAL: a pointwise bound away from the centre. -/
theorem norm_le (h : Track U θc f α₀ β₀ α₁ β₁) {θ : P} (hθ : θ ∈ U) (w : Fin n → Sym) :
    ‖f θ w‖ ≤ α₀ + β₀ + (α₁ + β₁) * ‖θ - θc‖ := by
  have h1 := h.1.norm_le w
  have h2 := (h.2 θ hθ).norm_le w
  calc ‖f θ w‖ = ‖f θc w + (f θ w - f θc w)‖ := by abel_nf
    _ ≤ ‖f θc w‖ + ‖f θ w - f θc w‖ := norm_add_le _ _
    _ ≤ _ := by linarith

/-- INTERNAL: congruence on `U`. -/
theorem congr (h : Track U θc f α₀ β₀ α₁ β₁) (hc : θc ∈ U) (hfg : ∀ θ ∈ U, ∀ w, f θ w = g θ w) :
    Track U θc g α₀ β₀ α₁ β₁ :=
  ⟨h.1.congr (hfg θc hc), fun θ hθ => (h.2 θ hθ).congr fun w => by rw [hfg θ hθ, hfg θc hc]⟩

end Track

end Dec

open Matrix

section Credit
variable {K : ℕ} [NeZero K] {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {n : ℕ}

theorem letterWeight_nonneg (T : Task K Sym) (w : Fin n → Sym) : 0 ≤ T.letterWeight w :=
  Finset.prod_nonneg fun _ _ => ENNReal.toReal_nonneg

theorem sum_letterWeight (T : Task K Sym) : ∑ w : Fin n → Sym, T.letterWeight w = 1 := by
  unfold Task.letterWeight
  have : ∑ a : Sym, (T.μ a).toReal = 1 := by
    have h := T.μ.tsum_coe
    rw [tsum_fintype] at h
    rw [← ENNReal.toReal_sum (fun a _ => PMF.apply_ne_top _ _), h]
    simp
  have h := Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset Sym))
    (fun _ a => (T.μ a).toReal)
  rw [Fintype.piFinset_univ] at h
  rw [← h]
  simp [this]

theorem piece_credit (hprior : Prior) (hK : 2 ≤ K) (T : Task K Sym) (hp0 : 0 < T.p)
    (hp1 : T.p < 1) (S : Finset (Fin n)) (h : (Fin n → Sym) → EuclideanSpace ℝ (ZMod K))
    (hS : DependsOn h (S : Set (Fin n))) {B : ℝ} (hB0 : 0 ≤ B) (hB : ∀ w, ‖h w‖ ≤ B) :
    |T.Ex (fun w => (h w).ofLp ⬝ᵥ T.targetSignal w)| ≤ mixrate K T.p ^ (n - S.card) * B := by
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
  have h2 : T.l2Norm H ≤ B := by
    unfold Task.l2Norm
    rw [Real.sqrt_le_left hB0]
    calc T.Ex (fun w => ∑ a, ∑ i, H w a i ^ 2) = ∑ w, T.letterWeight w * ‖h w‖ ^ 2 := by
          simp only [Task.Ex, hHdef, Fintype.sum_unique, smul_eq_mul, EuclideanSpace.norm_eq,
            Real.norm_eq_abs, sq_abs]
          congr 1; ext w
          rw [Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)]
      _ ≤ ∑ w, T.letterWeight w * B ^ 2 := by
          gcongr with w
          · exact letterWeight_nonneg T w
          · exact hB w
      _ = B ^ 2 := by rw [← Finset.sum_mul, sum_letterWeight, one_mul]
  rw [← h1]
  exact key.trans (mul_le_mul_of_nonneg_left h2 (pow_nonneg (norm_nonneg _) _))

/-- **Credit of a one-letter decomposition.** The letter-free part is charged `λ^n`, each
one-letter part `λ^{n-1}` (part (i) with `|S| = 0` or `1`).
INTERNAL: "(i): 0 or 1 letter" in the proof of lem:score-sum.
TEXLINE: Paper/appendix.tex:510-512 -/
theorem LetterDec.credit_le (hprior : Prior) (hK : 2 ≤ K) (T : Task K Sym) (hp0 : 0 < T.p)
    (hp1 : T.p < 1) {h : (Fin n → Sym) → EuclideanSpace ℝ (ZMod K)} {α β : ℝ}
    (hd : LetterDec h α β) :
    |T.Ex (fun w => (h w).ofLp ⬝ᵥ T.targetSignal w)| ≤
      mixrate K T.p ^ n * α + mixrate K T.p ^ (n - 1) * β := by
  obtain ⟨f0, g, b, hf, h0, hb, hg, hs⟩ := hd
  have hlam : 0 ≤ mixrate K T.p := norm_nonneg _
  have expand : T.Ex (fun w => (h w).ofLp ⬝ᵥ T.targetSignal w) =
      T.Ex (fun w : Fin n → Sym => f0.ofLp ⬝ᵥ T.targetSignal w) +
        ∑ t, T.Ex (fun w => (g t (w t)).ofLp ⬝ᵥ T.targetSignal w) := by
    simp only [Task.Ex, hf, WithLp.ofLp_add, WithLp.ofLp_sum, add_dotProduct, sum_dotProduct,
      smul_add, Finset.smul_sum, Finset.sum_add_distrib]
    rw [Finset.sum_comm]
  have e0 := piece_credit hprior hK T hp0 hp1 (∅ : Finset (Fin n)) (fun _ => f0) (fun _ _ _ => rfl)
    (norm_nonneg f0) (fun _ => le_rfl)
  have et : ∀ t, |T.Ex (fun w => (g t (w t)).ofLp ⬝ᵥ T.targetSignal w)| ≤
      mixrate K T.p ^ (n - 1) * b t := by
    intro t
    have := piece_credit hprior hK T hp0 hp1 {t} (fun w => g t (w t))
      (fun x y hxy => by
        simp only at hxy ⊢
        rw [hxy t (by simp)]) (hb t) (fun w => hg t (w t))
    simpa using this
  rw [expand]
  calc _ ≤ |T.Ex (fun w : Fin n → Sym => f0.ofLp ⬝ᵥ T.targetSignal w)| +
        ∑ t, |T.Ex (fun w => (g t (w t)).ofLp ⬝ᵥ T.targetSignal w)| :=
        (abs_add_le _ _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _))
    _ ≤ mixrate K T.p ^ n * α + ∑ t, mixrate K T.p ^ (n - 1) * b t := by
        refine add_le_add ?_ (Finset.sum_le_sum fun t _ => et t)
        rw [Finset.card_empty, Nat.sub_zero] at e0
        exact e0.trans (mul_le_mul_of_nonneg_left h0 (pow_nonneg hlam _))
    _ ≤ _ := by
        rw [← Finset.mul_sum]
        exact add_le_add le_rfl (mul_le_mul_of_nonneg_left hs (pow_nonneg hlam _))

end Credit

end Trace.Proof
