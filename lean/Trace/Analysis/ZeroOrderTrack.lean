import Trace.Analysis.LetterDecomposition
import Trace.Analysis.ParamNorm
import Trace.Analysis.ZeroOrderPattern

/-!
# Power counting for the zero-order network

Along the frozen-pattern network `patLogits` (uniform attention, ReLU replaced by a `0/1`
pattern) we track, position by position, a one-letter decomposition of the residual stream and
its Lipschitz dependence on `θ` (`Track`). On a parameter set `U` on which every weight matrix
has operator norm at most `a ε` and every MLP bias vector norm at most `B`:

* every position carries sizes `(Rε, Rε, R, R)` (`GoodStream`, first part);
* the answer position, which holds no letter in the prompt, carries `(Rε, Rε², R, Rε)`: a letter
  reaches it only through an attention value map `W_V`, which costs a factor `ε`;
* the readout costs one more `ε`, so the logits' Lipschitz constants are `C ε` for the
  letter-free part and `C ε²` for the one-letter part — the two terms
  `ε λ^n + ε² λ^{n-1}` of the zero-order credit (Paper/appendix.tex:510-512).

`relu_frozen` shows that, for `ε` small against `a` and `R`, the bias dominates the
pre-activation, so the true ReLU acts by the bias sign pattern. This is the "fixed ReLU pattern
on `B_ε`" of prop:score-degree (Paper/appendix.tex:325-489).
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.ZeroOrder

open Matrix ParamNorm

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K] {A : Arch nH n}

/-- INTERNAL: the size hypotheses on the parameter set `U`.
TEXLINE: Paper/appendix.tex:43-52 -/
structure Hyp (U : Set (Params Sym K A)) (θc : Params Sym K A) (ε a B : ℝ) : Prop where
  mem : θc ∈ U
  wV : ∀ θ ∈ U, ∀ l h, opNorm (θ.wV l h) ≤ a * ε
  wIn : ∀ θ ∈ U, ∀ l, opNorm (θ.wIn l) ≤ a * ε
  wOut : ∀ θ ∈ U, ∀ l, opNorm (θ.wOut l) ≤ a * ε
  readout : ∀ θ ∈ U, opNorm θ.readout ≤ a * ε
  tokCol : ∀ θ ∈ U, ∀ v,
    ‖(WithLp.toLp 2 (fun d => θ.tokEmb d v) : EuclideanSpace ℝ (Fin A.dres))‖ ≤ a * ε
  posCol : ∀ θ ∈ U, ∀ i,
    ‖(WithLp.toLp 2 (fun d => θ.posEmb d i) : EuclideanSpace ℝ (Fin A.dres))‖ ≤ a * ε
  bIn : ∀ θ ∈ U, ∀ l, ‖(WithLp.toLp 2 (θ.bIn l) : EuclideanSpace ℝ (Fin A.dmlp))‖ ≤ B

/-- INTERNAL: position-wise sizes of a residual stream. -/
def GoodStream (U : Set (Params Sym K A)) (θc : Params Sym K A) (ε R : ℝ)
    (x : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ) : Prop :=
  (∀ i, Track U θc (fun θ w => (WithLp.toLp 2 (x θ w i) : EuclideanSpace ℝ (Fin A.dres)))
      (R * ε) (R * ε) R R) ∧
    Track U θc (fun θ w => (WithLp.toLp 2 (x θ w (Fin.last (n + 3))) :
      EuclideanSpace ℝ (Fin A.dres))) (R * ε) (R * ε ^ 2) R (R * ε)

/-- INTERNAL: a `0/1` diagonal has operator norm at most `1`. -/
theorem norm_mat_diagonal_le {m : Type*} [Fintype m] [DecidableEq m] (D : m → ℝ)
    (hD : ∀ i, D i = 0 ∨ D i = 1) : ‖mat (Matrix.diagonal D)‖ ≤ 1 := by
  refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun v => ?_
  rw [one_mul]
  refine le_of_sq_le_sq ?_ (norm_nonneg _)
  have hv : mat (Matrix.diagonal D) v = WithLp.toLp 2 (Matrix.diagonal D *ᵥ v.ofLp) := rfl
  rw [hv, EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [WithLp.ofLp_toLp, Matrix.mulVec_diagonal]
  rcases hD i with h | h <;> simp [h, sq_nonneg]

/-- INTERNAL: averaging tracked functions over a nonempty set preserves the sizes. -/
theorem track_avg {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {U : Set (Params Sym K A)} {θc : Params Sym K A} {ι : Type*} (V : Finset ι)
    (hV : V.Nonempty) {f : ι → Params Sym K A → (Fin n → Sym) → E} {α₀ β₀ α₁ β₁ : ℝ}
    (hf : ∀ j ∈ V, Track U θc (f j) α₀ β₀ α₁ β₁) :
    Track U θc (fun θ w => ∑ j ∈ V, ((V.card : ℝ)⁻¹) • f j θ w) α₀ β₀ α₁ β₁ := by
  have h := Track.sum V fun j hj => (hf j hj).smul ((V.card : ℝ)⁻¹)
  have hc : (V.card : ℝ) ≠ 0 := by exact_mod_cast hV.card_pos.ne'
  have key : ∀ s : ℝ, ∑ _j ∈ V, |((V.card : ℝ)⁻¹)| * s = s := by
    intro s
    rw [Finset.sum_const, nsmul_eq_mul, abs_inv, Nat.abs_cast, ← mul_assoc,
      mul_inv_cancel₀ hc, one_mul]
  simpa only [key] using h

/-- INTERNAL: the `θ`-dependent matrix map `θ ↦ mat (F θ)` of a parameter block, bounded on
`U`, is a valid factor for `Track.clm`. -/
theorem track_block {m k : Type*} [Fintype m] [Fintype k] [DecidableEq k]
    {U : Set (Params Sym K A)} {θc : Params Sym K A} {F : Params Sym K A → Matrix m k ℝ}
    (hF : IsBlock F) {m₀ : ℝ} (hc : θc ∈ U) (hM : ∀ θ ∈ U, opNorm (F θ) ≤ m₀)
    {f : Params Sym K A → (Fin n → Sym) → EuclideanSpace ℝ k} {α₀ β₀ α₁ β₁ : ℝ}
    (hf : Track U θc f α₀ β₀ α₁ β₁) :
    Track U θc (fun θ w => mat (F θ) (f θ w)) (m₀ * α₀) (m₀ * β₀) (m₀ * α₁ + 1 * α₀)
      (m₀ * β₁ + 1 * β₀) :=
  Track.clm (fun θ => mat (F θ)) hc hM (fun θ _ => by rw [one_mul]; exact hF.lip θ θc) hf

/-- INTERNAL: the uniform-attention update in `mat` form. -/
theorem toLp_attnRes (θ : Params Sym K A) (l : Fin 2) (x : Fin (n + 4) → Fin A.dres → ℝ)
    (i : Fin (n + 4)) :
    (WithLp.toLp 2 (attnRes θ l x i) : EuclideanSpace ℝ (Fin A.dres)) =
      WithLp.toLp 2 (x i) + ∑ h, ∑ j ∈ A.vis l h i,
        ((A.vis l h i).card : ℝ)⁻¹ • mat (θ.wV l h) (WithLp.toLp 2 (x j)) := by
  simp only [attnRes, WithLp.toLp_add, WithLp.toLp_sum, WithLp.toLp_smul, mat_toLp]

/-- INTERNAL: arithmetic for the attention step. -/
theorem attn_arith {R ε a N : ℝ} (hR : 0 ≤ R) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (ha : 0 ≤ a)
    (hN : 0 ≤ N) :
    R * ε + N * (a * ε * (R * ε)) ≤ R * (1 + N * (a + 1)) * ε ∧
    R + N * (a * ε * R + 1 * (R * ε)) ≤ R * (1 + N * (a + 1)) ∧
    R * ε ^ 2 + N * (a * ε * (R * ε)) ≤ R * (1 + N * (a + 1)) * ε ^ 2 ∧
    R * ε + N * (a * ε * R + 1 * (R * ε)) ≤ R * (1 + N * (a + 1)) * ε := by
  have h1 : a * ε ≤ a + 1 := by nlinarith
  have hRe : 0 ≤ R * ε := mul_nonneg hR hε
  have hRe1 : R * ε ≤ R := by nlinarith
  refine ⟨?_, ?_, ?_, ?_⟩
  · nlinarith [mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right h1 hRe) hN]
  · have h2 : a * ε * R ≤ a * R := by nlinarith [mul_nonneg ha hR]
    nlinarith [mul_le_mul_of_nonneg_left (add_le_add h2 hRe1) hN]
  · have h3 : a * ε * (R * ε) ≤ (a + 1) * (R * ε ^ 2) := by
      have : a * ε * (R * ε) = a * (R * ε ^ 2) := by ring
      rw [this]; nlinarith [mul_nonneg hR (sq_nonneg ε)]
    nlinarith [mul_le_mul_of_nonneg_left h3 hN]
  · nlinarith

/-- INTERNAL: **attention step.** The uniform-attention update keeps the shape of the sizes:
`R ↦ R (1 + nH (a + 1))`. A letter enters the answer position only through `W_V`.
TEXLINE: Paper/appendix.tex:95-96 -/
theorem track_attnRes {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a B R : ℝ}
    (hH : Hyp U θc ε a B) (ha : 0 ≤ a) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hR : 0 ≤ R) (l : Fin 2)
    {x : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    (hx : GoodStream U θc ε R x) :
    GoodStream U θc ε (R * (1 + nH * (a + 1))) (fun θ w => attnRes θ l (x θ w)) := by
  have hatt : ∀ i, Track U θc (fun θ w => ∑ h, ∑ j ∈ A.vis l h i,
      ((A.vis l h i).card : ℝ)⁻¹ • mat (θ.wV l h) (WithLp.toLp 2 (x θ w j) :
        EuclideanSpace ℝ (Fin A.dres)))
      (nH * (a * ε * (R * ε))) (nH * (a * ε * (R * ε))) (nH * (a * ε * R + 1 * (R * ε)))
      (nH * (a * ε * R + 1 * (R * ε))) := by
    intro i
    have := Track.sum Finset.univ fun h _ => track_avg (A.vis l h i) (A.vis_nonempty l h i)
      fun j _ => track_block (isBlock_wV l h) hH.mem (fun θ hθ => hH.wV θ hθ l h) (hx.1 j)
    simpa only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using this
  have e : ∀ i, (fun θ w => (WithLp.toLp 2 (attnRes θ l (x θ w) i) :
      EuclideanSpace ℝ (Fin A.dres))) = fun θ w => (WithLp.toLp 2 (x θ w i) :
        EuclideanSpace ℝ (Fin A.dres)) + ∑ h, ∑ j ∈ A.vis l h i,
          ((A.vis l h i).card : ℝ)⁻¹ • mat (θ.wV l h) (WithLp.toLp 2 (x θ w j)) := by
    intro i; funext θ w; exact toLp_attnRes θ l (x θ w) i
  obtain ⟨q1, q2, q3, q4⟩ := attn_arith hR hε hε1 ha (Nat.cast_nonneg (α := ℝ) nH)
  refine ⟨fun i => ?_, ?_⟩
  · rw [e i]
    exact ((hx.1 i).add (hatt i)).mono (by linarith) (by linarith) q2 q2
  · rw [e]
    exact (hx.2.add (hatt _)).mono (by linarith) q3 q2 q4

/-- INTERNAL: the pattern MLP in `mat` form: `y + W_out D W_in y + W_out D b`. -/
theorem toLp_mlpPat (D : Fin A.dmlp → ℝ) (θ : Params Sym K A) (l : Fin 2)
    (y : Fin (n + 4) → Fin A.dres → ℝ) (i : Fin (n + 4)) :
    (WithLp.toLp 2 (mlpPat D θ l y i) : EuclideanSpace ℝ (Fin A.dres)) =
      WithLp.toLp 2 (y i) +
        ((mat (θ.wOut l)).comp ((mat (Matrix.diagonal D)).comp (mat (θ.wIn l))))
          (WithLp.toLp 2 (y i)) +
        ((mat (θ.wOut l)).comp (mat (Matrix.diagonal D))) (WithLp.toLp 2 (θ.bIn l)) := by
  have hsplit : (fun m => D m * ((θ.wIn l *ᵥ y i) m + θ.bIn l m)) =
      Matrix.diagonal D *ᵥ (θ.wIn l *ᵥ y i) + Matrix.diagonal D *ᵥ θ.bIn l := by
    funext m; simp only [Pi.add_apply, Matrix.mulVec_diagonal, mul_add]
  simp only [mlpPat, ContinuousLinearMap.comp_apply, mat_toLp, hsplit, Matrix.mulVec_add,
    WithLp.toLp_add, add_assoc]

/-- INTERNAL: arithmetic for the MLP step. -/
theorem mlp_arith {R ε a B : ℝ} (hR : 0 ≤ R) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (ha : 0 ≤ a)
    (hB : 0 ≤ B) :
    R * ε + a * ε * (1 * (a * ε)) * (R * ε) + a * ε * (1 * B) ≤
        ((a + 1) ^ 2 * R + (a + 1) * (B + 1)) * ε ∧
    R * ε + a * ε * (1 * (a * ε)) * (R * ε) + 0 ≤ ((a + 1) ^ 2 * R + (a + 1) * (B + 1)) * ε ∧
    R + (a * ε * (1 * (a * ε)) * R + 2 * a * ε * (R * ε)) + (B + a * ε) ≤
        (a + 1) ^ 2 * R + (a + 1) * (B + 1) ∧
    R + (a * ε * (1 * (a * ε)) * R + 2 * a * ε * (R * ε)) + 0 ≤
        (a + 1) ^ 2 * R + (a + 1) * (B + 1) ∧
    R * ε ^ 2 + a * ε * (1 * (a * ε)) * (R * ε ^ 2) + 0 ≤
        ((a + 1) ^ 2 * R + (a + 1) * (B + 1)) * ε ^ 2 ∧
    R * ε + (a * ε * (1 * (a * ε)) * (R * ε) + 2 * a * ε * (R * ε ^ 2)) + 0 ≤
        ((a + 1) ^ 2 * R + (a + 1) * (B + 1)) * ε := by
  have hae : a * ε ≤ a := by nlinarith
  have hae0 : 0 ≤ a * ε := mul_nonneg ha hε
  have ha2 : a * ε * (a * ε) ≤ a * a := mul_le_mul hae hae hae0 ha
  have hRe : 0 ≤ R * ε := mul_nonneg hR hε
  have hRe2 : 0 ≤ R * ε ^ 2 := mul_nonneg hR (sq_nonneg ε)
  have hRe1 : R * ε ≤ R := by nlinarith
  have hRe21 : R * ε ^ 2 ≤ R * ε := by nlinarith
  have hBa : 0 ≤ a * B := mul_nonneg ha hB
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hRe, mul_le_mul_of_nonneg_left hε hBa,
      mul_nonneg ha hRe, mul_nonneg hB hε, mul_nonneg ha hε]
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hRe, mul_nonneg ha hRe, mul_nonneg hB hε,
      mul_nonneg ha hε, mul_nonneg hBa hε]
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hR, mul_le_mul_of_nonneg_left hRe1 hae0,
      mul_le_mul_of_nonneg_right hae hR]
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hR, mul_le_mul_of_nonneg_left hRe1 hae0,
      mul_le_mul_of_nonneg_right hae hR, mul_nonneg ha hR]
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hRe2, mul_nonneg ha hRe2,
      mul_nonneg (mul_nonneg ha hB) (sq_nonneg ε), mul_nonneg hB (sq_nonneg ε),
      mul_nonneg ha (sq_nonneg ε), sq_nonneg ε]
  · nlinarith [mul_le_mul_of_nonneg_right ha2 hRe, mul_le_mul_of_nonneg_left hRe21 hae0,
      mul_le_mul_of_nonneg_right hae hRe, mul_nonneg ha hRe, mul_nonneg hB hε,
      mul_nonneg ha hε, mul_nonneg hBa hε]

/-- INTERNAL: **MLP step.** With a `0/1` pattern the MLP is affine in the stream:
`R ↦ (a + 1)² R + (a + 1)(B + 1)`. The bias term is letter-free.
TEXLINE: Paper/appendix.tex:325-489 -/
theorem track_mlpPat {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a B R : ℝ}
    (hH : Hyp U θc ε a B) (ha : 0 ≤ a) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hR : 0 ≤ R) (l : Fin 2)
    (D : Fin A.dmlp → ℝ) (hD : ∀ m, D m = 0 ∨ D m = 1)
    {y : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    (hy : GoodStream U θc ε R y) :
    GoodStream U θc ε ((a + 1) ^ 2 * R + (a + 1) * (B + 1))
      (fun θ w => mlpPat D θ l (y θ w)) := by
  have hB : 0 ≤ B := (norm_nonneg _).trans (hH.bIn θc hH.mem l)
  have hDg := norm_mat_diagonal_le D hD
  set Dg := mat (Matrix.diagonal D) with hDgdef
  set M : Params Sym K A → EuclideanSpace ℝ (Fin A.dres) →L[ℝ] EuclideanSpace ℝ (Fin A.dres) :=
    fun θ => (mat (θ.wOut l)).comp (Dg.comp (mat (θ.wIn l))) with hMdef
  set c : Params Sym K A → EuclideanSpace ℝ (Fin A.dres) :=
    fun θ => ((mat (θ.wOut l)).comp Dg) (WithLp.toLp 2 (θ.bIn l)) with hcdef
  have hWo : ∀ θ ∈ U, ‖mat (θ.wOut l)‖ ≤ a * ε := fun θ hθ => hH.wOut θ hθ l
  have hWi : ∀ θ ∈ U, ‖mat (θ.wIn l)‖ ≤ a * ε := fun θ hθ => hH.wIn θ hθ l
  have hM : ∀ θ ∈ U, ‖M θ‖ ≤ a * ε * (1 * (a * ε)) := by
    intro θ hθ
    refine ((mat (θ.wOut l)).opNorm_comp_le _).trans (mul_le_mul (hWo θ hθ) ?_
      (norm_nonneg _) (by nlinarith))
    exact (Dg.opNorm_comp_le _).trans (mul_le_mul hDg (hWi θ hθ) (norm_nonneg _) zero_le_one)
  have hML : ∀ θ ∈ U, ‖M θ - M θc‖ ≤ 2 * a * ε * ‖θ - θc‖ := by
    intro θ hθ
    have e : M θ - M θc = (mat (θ.wOut l) - mat (θc.wOut l)).comp (Dg.comp (mat (θ.wIn l))) +
        (mat (θc.wOut l)).comp (Dg.comp (mat (θ.wIn l) - mat (θc.wIn l))) := by
      ext1 v; simp [hMdef]
    rw [e]
    refine (norm_add_le _ _).trans ?_
    have h1 : ‖(mat (θ.wOut l) - mat (θc.wOut l)).comp (Dg.comp (mat (θ.wIn l)))‖ ≤
        ‖θ - θc‖ * (1 * (a * ε)) :=
      (ContinuousLinearMap.opNorm_comp_le _ _).trans (mul_le_mul ((isBlock_wOut l).lip θ θc)
        ((Dg.opNorm_comp_le _).trans (mul_le_mul hDg (hWi θ hθ) (norm_nonneg _) zero_le_one))
        (norm_nonneg _) (norm_nonneg _))
    have h2 : ‖(mat (θc.wOut l)).comp (Dg.comp (mat (θ.wIn l) - mat (θc.wIn l)))‖ ≤
        a * ε * (1 * ‖θ - θc‖) :=
      (ContinuousLinearMap.opNorm_comp_le _ _).trans (mul_le_mul (hWo θc hH.mem)
        ((Dg.opNorm_comp_le _).trans (mul_le_mul hDg ((isBlock_wIn l).lip θ θc) (norm_nonneg _)
          zero_le_one)) (norm_nonneg _) (by nlinarith))
    nlinarith
  have hc0 : ‖c θc‖ ≤ a * ε * (1 * B) := by
    have h := (ContinuousLinearMap.le_opNorm ((mat (θc.wOut l)).comp Dg)
      (WithLp.toLp 2 (θc.bIn l))).trans (mul_le_mul (((mat (θc.wOut l)).opNorm_comp_le _).trans
        (mul_le_mul (hWo θc hH.mem) hDg (norm_nonneg _) (by nlinarith))) (hH.bIn θc hH.mem l)
        (norm_nonneg _) (by nlinarith))
    calc ‖c θc‖ ≤ a * ε * 1 * B := h
      _ = a * ε * (1 * B) := by ring
  have hc1 : ∀ θ ∈ U, ‖c θ - c θc‖ ≤ (B + a * ε) * ‖θ - θc‖ := by
    intro θ hθ
    have e : c θ - c θc = (mat (θ.wOut l) - mat (θc.wOut l)) (Dg (WithLp.toLp 2 (θ.bIn l))) +
        mat (θc.wOut l) (Dg (WithLp.toLp 2 (θ.bIn l) - WithLp.toLp 2 (θc.bIn l))) := by
      simp [hcdef, map_sub]
    rw [e]
    refine (norm_add_le _ _).trans ?_
    have h1 : ‖(mat (θ.wOut l) - mat (θc.wOut l)) (Dg (WithLp.toLp 2 (θ.bIn l)))‖ ≤
        ‖θ - θc‖ * (1 * B) :=
      ((ContinuousLinearMap.le_opNorm _ _)).trans (mul_le_mul ((isBlock_wOut l).lip θ θc)
        ((Dg.le_opNorm _).trans (mul_le_mul hDg (hH.bIn θ hθ l) (norm_nonneg _) zero_le_one))
        (norm_nonneg _) (norm_nonneg _))
    have h2 : ‖mat (θc.wOut l) (Dg (WithLp.toLp 2 (θ.bIn l) - WithLp.toLp 2 (θc.bIn l)))‖ ≤
        a * ε * (1 * ‖θ - θc‖) :=
      ((ContinuousLinearMap.le_opNorm _ _)).trans (mul_le_mul (hWo θc hH.mem)
        ((Dg.le_opNorm _).trans (mul_le_mul hDg ((isVBlock_bIn l).lip θ θc) (norm_nonneg _)
          zero_le_one)) (norm_nonneg _) (by nlinarith))
    nlinarith
  have hcT := Track.const (n := n) (Sym := Sym) c hc0 hc1
  have e : ∀ i, (fun θ w => (WithLp.toLp 2 (mlpPat D θ l (y θ w) i) :
      EuclideanSpace ℝ (Fin A.dres))) = fun θ w => (WithLp.toLp 2 (y θ w i) :
        EuclideanSpace ℝ (Fin A.dres)) + M θ (WithLp.toLp 2 (y θ w i)) + c θ := by
    intro i; funext θ w; exact toLp_mlpPat D θ l (y θ w) i
  obtain ⟨q1, q2, q3, q4, q5, q6⟩ := mlp_arith hR hε hε1 ha hB
  refine ⟨fun i => ?_, ?_⟩
  · rw [e i]
    exact (((hy.1 i).add (Track.clm M hH.mem hM hML (hy.1 i))).add hcT).mono q1 q2 q3 q4
  · rw [e]
    exact ((hy.2.add (Track.clm M hH.mem hM hML hy.2)).add hcT).mono q1 q5 q3 q6

/-- INTERNAL: each prompt position holds either one letter or a fixed token. -/
theorem prompt_cases (T : Task K Sym) (i : Fin (n + 4)) :
    (∃ t : Fin n, ∀ w : Fin n → Sym, T.outcomePrompt w i = Token.letter (w t)) ∨
      ∃ v : Token Sym K, ∀ w : Fin n → Sym, T.outcomePrompt w i = v := by
  unfold Task.outcomePrompt
  by_cases h0 : i.val = 0
  · exact Or.inr ⟨Token.value T.s0, fun w => by simp [h0]⟩
  by_cases h1 : i.val = 1
  · exact Or.inr ⟨Token.letter T.wstar, fun w => by simp [h1]⟩
  by_cases h2 : i.val - 2 < n
  · exact Or.inl ⟨⟨i.val - 2, h2⟩, fun w => by simp [h0, h1, h2]⟩
  · exact Or.inr ⟨if i.val = n + 2 then Token.sep else Token.colon, fun w => by
      simp [h0, h1, h2]⟩

/-- INTERNAL: the answer position holds the fixed token `":"`. -/
theorem prompt_last (T : Task K Sym) :
    ∃ v : Token Sym K, ∀ w : Fin n → Sym, T.outcomePrompt w (Fin.last (n + 3)) = v :=
  ⟨Token.colon, fun w => by
    unfold Task.outcomePrompt
    simp only [Fin.val_last]
    rw [if_neg (by omega), if_neg (by omega), dif_neg (by omega), if_neg (by omega)]⟩

/-- INTERNAL: **embedding step.** Sizes `R = 2a`; the answer position is letter-free.
TEXLINE: Paper/appendix.tex:107-115 -/
theorem track_embed {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a B : ℝ}
    (hH : Hyp U θc ε a B) (ha : 1 ≤ a) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (T : Task K Sym) :
    GoodStream U θc ε (2 * a) (fun θ w => embed T θ w) := by
  have hpos : ∀ i, Track U θc (fun θ (_ : Fin n → Sym) =>
      (WithLp.toLp 2 (fun d => θ.posEmb d i) : EuclideanSpace ℝ (Fin A.dres))) (a * ε) 0 1 0 :=
    fun i => Track.const _ (hH.posCol θc hH.mem i)
      (fun θ _ => by rw [one_mul]; exact (isVBlock_posCol i).lip θ θc)
  have htokc : ∀ v : Token Sym K, Track U θc (fun θ (_ : Fin n → Sym) =>
      (WithLp.toLp 2 (fun d => θ.tokEmb d v) : EuclideanSpace ℝ (Fin A.dres))) (a * ε) 0 1 0 :=
    fun v => Track.const _ (hH.tokCol θc hH.mem v)
      (fun θ _ => by rw [one_mul]; exact (isVBlock_tokCol v).lip θ θc)
  have hae : 0 ≤ a * ε := by nlinarith
  have hae1 : a * ε ≤ a := by nlinarith
  have hconst : ∀ i (v : Token Sym K), (∀ w : Fin n → Sym, T.outcomePrompt w i = v) →
      Track U θc (fun θ w => (WithLp.toLp 2 (embed T θ w i) : EuclideanSpace ℝ (Fin A.dres)))
        (a * ε + a * ε) (0 + 0) (1 + 1) (0 + 0) := by
    intro i v hv
    have e : (fun θ w => (WithLp.toLp 2 (embed T θ w i) : EuclideanSpace ℝ (Fin A.dres))) =
        fun θ (_ : Fin n → Sym) => (WithLp.toLp 2 (fun d => θ.tokEmb d v) :
          EuclideanSpace ℝ (Fin A.dres)) + WithLp.toLp 2 (fun d => θ.posEmb d i) := by
      funext θ w
      rw [← WithLp.toLp_add]
      congr 1
      funext d
      simp [embed, hv w]
    rw [e]
    exact (htokc v).add (hpos i)
  refine ⟨fun i => ?_, ?_⟩
  · rcases prompt_cases T i with ⟨t, ht⟩ | ⟨v, hv⟩
    · have hlet := Track.letter (U := U) (θc := θc) t
        (fun θ (b : Sym) => (WithLp.toLp 2 (fun d => θ.tokEmb d (Token.letter b)) :
          EuclideanSpace ℝ (Fin A.dres))) hae zero_le_one
        (fun b => hH.tokCol θc hH.mem _)
        (fun θ _ b => by rw [one_mul]; exact (isVBlock_tokCol _).lip θ θc)
      have e : (fun θ w => (WithLp.toLp 2 (embed T θ w i) :
          EuclideanSpace ℝ (Fin A.dres))) = fun θ w => (WithLp.toLp 2
            (fun d => θ.tokEmb d (Token.letter (w t))) : EuclideanSpace ℝ (Fin A.dres)) +
            WithLp.toLp 2 (fun d => θ.posEmb d i) := by
        funext θ w
        rw [← WithLp.toLp_add]
        congr 1
        funext d
        simp [embed, ht w]
      rw [e]
      exact (hlet.add (hpos i)).mono (by linarith) (by linarith) (by linarith) (by linarith)
    · exact (hconst i v hv).mono (by linarith) (by nlinarith) (by linarith) (by linarith)
  · obtain ⟨v, hv⟩ := prompt_last (n := n) T
    exact (hconst _ v hv).mono (by linarith) (by have := sq_nonneg ε; nlinarith) (by linarith)
      (by nlinarith)

/-- INTERNAL: **readout step.** One more factor `ε`.
TEXLINE: Paper/appendix.tex:95-96 -/
theorem track_readout {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a B R : ℝ}
    (hH : Hyp U θc ε a B)
    {y : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    (hy : GoodStream U θc ε R y) :
    Track U θc (fun θ w => (WithLp.toLp 2 (θ.readout *ᵥ y θ w (Fin.last (n + 3))) :
      EuclideanSpace ℝ (ZMod K)))
      (a * ε * (R * ε)) (a * ε * (R * ε ^ 2)) (a * ε * R + 1 * (R * ε))
      (a * ε * (R * ε) + 1 * (R * ε ^ 2)) :=
  track_block isBlock_readout hH.mem (fun θ hθ => hH.readout θ hθ) hy.2

/-- INTERNAL: size after block-0 attention. -/
noncomputable def r1 (a N : ℝ) : ℝ := 2 * a * (1 + N * (a + 1))

/-- INTERNAL: size after block 0. -/
noncomputable def r2 (a N : ℝ) : ℝ := (a + 1) ^ 2 * r1 a N + (a + 1) * (2 + 1)

/-- INTERNAL: size after block-1 attention. -/
noncomputable def r3 (a N : ℝ) : ℝ := r2 a N * (1 + N * (a + 1))

/-- INTERNAL: size after block 1. -/
noncomputable def r4 (a N : ℝ) : ℝ := (a + 1) ^ 2 * r3 a N + (a + 1) * (2 + 1)

/-- INTERNAL: the sizes are nonnegative. -/
theorem r_nonneg {a N : ℝ} (ha : 0 ≤ a) (hN : 0 ≤ N) :
    0 ≤ r1 a N ∧ 0 ≤ r2 a N ∧ 0 ≤ r3 a N ∧ 0 ≤ r4 a N := by
  have h1 : 0 ≤ r1 a N := by unfold r1; positivity
  have h2 : 0 ≤ r2 a N := by unfold r2; positivity
  have h3 : 0 ≤ r3 a N := by unfold r3; positivity
  exact ⟨h1, h2, h3, by unfold r4; positivity⟩

/-- INTERNAL: **the zero-order network, tracked.** The two pre-MLP streams (needed to freeze
the ReLU pattern) and the answer logits, whose Lipschitz sizes are `(a+1) r₄ ε` (letter-free)
and `(a+1) r₄ ε²` (one letter). TEXLINE: Paper/appendix.tex:510-512 -/
theorem track_patLogits {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a : ℝ}
    (hH : Hyp U θc ε a 2) (ha : 1 ≤ a) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (T : Task K Sym)
    (D : Fin 2 → Fin A.dmlp → ℝ) (hD : ∀ l m, D l m = 0 ∨ D l m = 1) :
    GoodStream U θc ε (r1 a nH) (fun θ w => attnRes θ 0 (embed T θ w)) ∧
    GoodStream U θc ε (r3 a nH)
      (fun θ w => attnRes θ 1 (mlpPat (D 0) θ 0 (attnRes θ 0 (embed T θ w)))) ∧
    Track U θc (fun θ w => (WithLp.toLp 2 (patLogits T A D θ w) : EuclideanSpace ℝ (ZMod K)))
      (a * ε * (r4 a nH * ε)) (a * ε * (r4 a nH * ε ^ 2)) ((a + 1) * r4 a nH * ε)
      ((a + 1) * r4 a nH * ε ^ 2) := by
  have ha0 : 0 ≤ a := by linarith
  obtain ⟨h1, h2, h3, h4⟩ := r_nonneg ha0 (Nat.cast_nonneg (α := ℝ) nH)
  have s0 := track_embed hH ha hε hε1 T
  have s1 := track_attnRes hH ha0 hε hε1 (by positivity) 0 s0
  have s2 := track_mlpPat hH ha0 hε hε1 h1 0 (D 0) (hD 0) s1
  have s3 := track_attnRes hH ha0 hε hε1 h2 1 s2
  have s4 := track_mlpPat hH ha0 hε hε1 h3 1 (D 1) (hD 1) s3
  have s5 := track_readout hH s4
  refine ⟨s1, s3, ?_⟩
  refine s5.mono le_rfl le_rfl (le_of_eq (by unfold r4; ring)) (le_of_eq (by unfold r4; ring))

/-- INTERNAL: **frozen ReLU pattern.** If the bias has margin `3ε/4` with the sign recorded by
`D`, and the stream is small enough that `‖W_in y‖ ≤ ε/2`, the ReLU acts as `D`.
TEXLINE: Paper/appendix.tex:325-489 -/
theorem relu_frozen {U : Set (Params Sym K A)} {θc : Params Sym K A} {ε a B R : ℝ}
    (hH : Hyp U θc ε a B) (ha : 0 ≤ a) (hε : 0 < ε) (hR : 0 ≤ R)
    (hdiam : ∀ θ ∈ U, ‖θ - θc‖ ≤ ε) (hsmall : 8 * a * R * ε ≤ 1) (l : Fin 2)
    (D : Fin A.dmlp → ℝ)
    {y : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    (hy : GoodStream U θc ε R y)
    (hb : ∀ θ ∈ U, ∀ m, (D m = 1 ∧ 3 * ε / 4 ≤ θ.bIn l m) ∨ (D m = 0 ∧ θ.bIn l m ≤ -(3 * ε / 4))) :
    ∀ θ ∈ U, ∀ w i m, max ((θ.wIn l *ᵥ y θ w i) m + θ.bIn l m) 0 =
      D m * ((θ.wIn l *ᵥ y θ w i) m + θ.bIn l m) := by
  intro θ hθ w i m
  have hy0 : ‖(WithLp.toLp 2 (y θ w i) : EuclideanSpace ℝ (Fin A.dres))‖ ≤ 4 * R * ε := by
    have := (hy.1 i).norm_le hθ w
    have hd := hdiam θ hθ
    nlinarith
  have hWy : |(θ.wIn l *ᵥ y θ w i) m| ≤ ε / 2 := by
    have h1 : |(θ.wIn l *ᵥ y θ w i) m| ≤
        ‖(WithLp.toLp 2 (θ.wIn l *ᵥ y θ w i) : EuclideanSpace ℝ (Fin A.dmlp))‖ := by
      have := PiLp.norm_apply_le (WithLp.toLp 2 (θ.wIn l *ᵥ y θ w i) :
        EuclideanSpace ℝ (Fin A.dmlp)) m
      simpa [Real.norm_eq_abs] using this
    have h2 : ‖(WithLp.toLp 2 (θ.wIn l *ᵥ y θ w i) : EuclideanSpace ℝ (Fin A.dmlp))‖ ≤
        a * ε * (4 * R * ε) := by
      rw [← mat_toLp]
      exact ((mat (θ.wIn l)).le_opNorm _).trans
        (mul_le_mul (hH.wIn θ hθ l) hy0 (norm_nonneg _) (by positivity))
    have h3 : a * ε * (4 * R * ε) ≤ ε / 2 := by nlinarith
    linarith
  rw [abs_le] at hWy
  rcases hb θ hθ m with ⟨hD, hbm⟩ | ⟨hD, hbm⟩
  · rw [hD, one_mul, max_eq_left (by linarith)]
  · rw [hD, zero_mul, max_eq_right (by linarith)]

end Trace.Proof.ZeroOrder
