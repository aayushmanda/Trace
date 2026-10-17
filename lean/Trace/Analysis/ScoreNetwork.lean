import Trace.Model.Prelude
import Trace.Analysis.ScoreLocality

/-!
# Locality costs of the two-block forward pass

INTERNAL: the power counting of `prop:score-degree` (Paper/appendix.tex:308-319, 325-489),
carried out with the locality costs of `Trace/Analysis/ScoreLocality.lean` instead of a
`ζ`-Taylor expansion. On the doubled ball around `ε θ̃` (Paper/appendix.tex:227-230) every
parameter block costs `O(ε)`, an input costs `O(ε/κ)` (it reads at most one letter), an
attention score costs `O(ε⁴/κ²)`, and the score-dependent part `z_θ − z^{[0]}_θ` of the answer
logits costs `O(ε⁷/κ³)`. With `κ = λ` this is the bound `C ε⁶ λ^{n−3}` of `lem:score-sum`
after the conditional contraction.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.SN

open Trace Trace.Proof.SL Matrix

/-! ## Matrices as maps between Euclidean spaces -/

section Euclid

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq k]

/-- INTERNAL: a real matrix as a continuous linear map between Euclidean spaces. -/
noncomputable def toE (M : Matrix m k ℝ) : EuclideanSpace ℝ k →L[ℝ] EuclideanSpace ℝ m :=
  LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin M)

lemma norm_toE (M : Matrix m k ℝ) : ‖toE M‖ = opNorm M := rfl

lemma toE_toLp (M : Matrix m k ℝ) (v : k → ℝ) :
    toE M (WithLp.toLp 2 v) = WithLp.toLp 2 (M *ᵥ v) := by
  simp [toE, Matrix.toLin'_apply]

lemma toE_apply (M : Matrix m k ℝ) (v : EuclideanSpace ℝ k) :
    toE M v = WithLp.toLp 2 (M *ᵥ v.ofLp) := by
  rw [← toE_toLp]

lemma toE_add (M N : Matrix m k ℝ) : toE (M + N) = toE M + toE N := by
  ext1 v; simp [toE_apply, Matrix.add_mulVec]

lemma toE_sub (M N : Matrix m k ℝ) : toE (M - N) = toE M - toE N := by
  ext1 v; simp [toE_apply, Matrix.sub_mulVec]

lemma toE_smul (c : ℝ) (M : Matrix m k ℝ) : toE (c • M) = c • toE M := by
  ext1 v; simp [toE_apply, Matrix.smul_mulVec]

/-- INTERNAL: the `ℓ²` operator norm is at most the Frobenius norm. -/
lemma norm_toE_le (M : Matrix m k ℝ) {c : ℝ} (hc : 0 ≤ c)
    (h : ∑ i, ∑ j, M i j ^ 2 ≤ c ^ 2) : ‖toE M‖ ≤ c := by
  refine ContinuousLinearMap.opNorm_le_bound _ hc fun v => ?_
  rw [toE_apply]
  have hv : 0 ≤ ‖v‖ := norm_nonneg _
  refine le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_
  rw [EuclideanSpace.norm_sq_eq, mul_pow, EuclideanSpace.norm_sq_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  calc ∑ i, (M *ᵥ v.ofLp) i ^ 2 ≤ ∑ i, (∑ j, M i j ^ 2) * ∑ j, v.ofLp j ^ 2 :=
        Finset.sum_le_sum fun i _ => Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ = (∑ i, ∑ j, M i j ^ 2) * ∑ j, v.ofLp j ^ 2 := by rw [Finset.sum_mul]
    _ ≤ c ^ 2 * ∑ j, v.ofLp j ^ 2 :=
        mul_le_mul_of_nonneg_right h (Finset.sum_nonneg fun _ _ => sq_nonneg _)

/-- INTERNAL: a unit coordinate vector. -/
noncomputable def unit (c : k) : EuclideanSpace ℝ k := WithLp.toLp 2 (Pi.single c 1)

lemma norm_unit (c : k) : ‖unit c‖ = 1 := by
  rw [unit, EuclideanSpace.norm_eq]
  simp [Pi.single_apply, apply_ite]

lemma toE_unit (M : Matrix m k ℝ) (c : k) :
    toE M (unit c) = WithLp.toLp 2 (fun d => M d c) := by
  rw [unit, toE_toLp]
  congr 1; ext d
  simp [Matrix.mulVec, dotProduct, Pi.single_apply]

end Euclid

/-! ## Parameter blocks -/

section Blocks

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n}

/-- INTERNAL: the sum of squares of distinct coordinates is at most the squared norm. -/
lemma sumsq_le_of_inj {R : Type*} [Fintype R] (θ : Params Sym K A)
    (e : R → ParamIdx Sym K A) (he : Function.Injective e) :
    ∑ p, θ (e p) ^ 2 ≤ ‖θ‖ ^ 2 := by
  classical
  rw [EuclideanSpace.norm_sq_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  rw [← Finset.sum_image (f := fun i => θ.ofLp i ^ 2) (s := Finset.univ) he.injOn]
  exact Finset.sum_le_univ_sum_of_nonneg fun _ => sq_nonneg _

/-- INTERNAL: a matrix block read off injectively from the parameter vector has operator
norm at most the norm of the parameters. -/
lemma norm_toE_blk {R C : Type*} [Fintype R] [Fintype C] [DecidableEq C]
    (W : Params Sym K A → Matrix R C ℝ) (e : R × C → ParamIdx Sym K A)
    (he : Function.Injective e) (hW : ∀ θ i j, W θ i j = θ (e (i, j))) (θ : Params Sym K A) :
    ‖toE (W θ)‖ ≤ ‖θ‖ := by
  refine norm_toE_le _ (norm_nonneg _) ?_
  simp_rw [hW]
  rw [← Fintype.sum_prod_type' (fun i j => θ (e (i, j)) ^ 2)]
  exact sumsq_le_of_inj θ e he

lemma blk_linear {R C : Type*} (W : Params Sym K A → Matrix R C ℝ)
    (e : R × C → ParamIdx Sym K A) (hW : ∀ θ i j, W θ i j = θ (e (i, j)))
    (c : ℝ) (θ θ' : Params Sym K A) : W (c • θ + θ') = c • W θ + W θ' := by
  ext i j; simp [hW]

lemma blk_sub {R C : Type*} (W : Params Sym K A → Matrix R C ℝ)
    (e : R × C → ParamIdx Sym K A) (hW : ∀ θ i j, W θ i j = θ (e (i, j)))
    (θ θ' : Params Sym K A) : W (θ - θ') = W θ - W θ' := by
  ext i j; simp [hW]

end Blocks

section Ball

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n} {B : Set (Params Sym K A)} {θt : Params Sym K A} {ε κ : ℝ}

/-- INTERNAL: a parameter block on the ball `‖θ − ε θ̃‖ ≤ ε/4` costs `(|a0| + 2) ε` at scale
`r = ε`: it is bounded by `(a0 + 1/4) ε` and `1`-Lipschitz. -/
lemma bd_blk {R C : Type*} [Fintype R] [Fintype C] [DecidableEq C]
    (W : Params Sym K A → Matrix R C ℝ) (e : R × C → ParamIdx Sym K A)
    (he : Function.Injective e) (hW : ∀ θ i j, W θ i j = θ (e (i, j))) {a0 : ℝ}
    (ht : opNorm (W θt) ≤ a0) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (W θ)) ((|a0| + 2) * ε) := by
  have hsup : ∀ θ ∈ B, ‖toE (W θ)‖ ≤ ε * |a0| + ε / 4 := by
    intro θ hθ
    have e1 : W θ = ε • W θt + W (θ - ε • θt) := by
      rw [← blk_linear W e hW]; congr 1; abel
    rw [e1, toE_add, toE_smul]
    refine (norm_add_le _ _).trans (add_le_add ?_ ((norm_toE_blk W e he hW _).trans (hB θ hθ)))
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hε, norm_toE]
    exact mul_le_mul_of_nonneg_left (ht.trans (le_abs_self _)) hε
  have hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ‖toE (W θ) - toE (W θ')‖ ≤ 1 * dist θ θ' := by
    intro θ _ θ' _
    rw [← toE_sub, ← blk_sub W e hW, one_mul, dist_eq_norm]
    exact norm_toE_blk W e he hW _
  refine (Bd.param (ι := Fin n) (α := Sym) (B := B) (κ := κ) (r := ε)
    (by positivity) zero_le_one hsup hlip).mono ?_
  nlinarith [abs_nonneg a0]

/-- INTERNAL: the bias vector `b_in` as a Euclidean vector. -/
noncomputable def biasE (θ : Params Sym K A) (l : Fin 2) : EuclideanSpace ℝ (Fin A.dmlp) :=
  WithLp.toLp 2 (θ.bIn l)

lemma norm_biasE_le (θ : Params Sym K A) (l : Fin 2) : ‖biasE θ l‖ ≤ ‖θ‖ := by
  refine le_of_pow_le_pow_left₀ two_ne_zero (norm_nonneg _) ?_
  rw [biasE, EuclideanSpace.norm_sq_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  exact sumsq_le_of_inj θ (fun m : Fin A.dmlp =>
    Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m))))))))
    (fun p q h => by simpa using h)

lemma biasE_linear (c : ℝ) (θ θ' : Params Sym K A) (l : Fin 2) :
    biasE (c • θ + θ') l = c • biasE θ l + biasE θ' l := by
  ext m; simp [biasE, Params.bIn]

/-- INTERNAL: the bias costs `ε N_b + ε/4 + ε`. -/
lemma bd_bias {a0 cemb Nb : ℝ} (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε)
    (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) (l : Fin 2) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => biasE θ l) (ε * Nb + ε / 4 + ε) := by
  have hsup : ∀ θ ∈ B, ‖biasE θ l‖ ≤ ε * Nb + ε / 4 := by
    intro θ hθ
    have e1 : biasE θ l = ε • biasE θt l + biasE (θ - ε • θt) l := by
      rw [← biasE_linear]; congr 1; abel
    rw [e1]
    refine (norm_add_le _ _).trans (add_le_add ?_ ((norm_biasE_le _ l).trans (hB θ hθ)))
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hε]
    exact mul_le_mul_of_nonneg_left (hI.bIn_norm_le l) hε
  have hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ‖biasE θ l - biasE θ' l‖ ≤ 1 * dist θ θ' := by
    intro θ _ θ' _
    have : biasE θ l - biasE θ' l = biasE (θ - θ') l := by
      have := biasE_linear (-1) θ' θ l
      rw [neg_one_smul, neg_add_eq_sub, neg_one_smul, neg_add_eq_sub] at this
      rw [this]
    rw [this, one_mul, dist_eq_norm]
    exact norm_biasE_le _ l
  have hNb : 0 ≤ Nb := (norm_nonneg _).trans (hI.bIn_norm_le l)
  have h0 : 0 ≤ ε * Nb + ε / 4 := by positivity
  simpa using (Bd.param (ι := Fin n) (α := Sym) (B := B) (κ := κ) (r := ε) h0 zero_le_one
    hsup hlip)

variable {a0 cemb Nb : ℝ}

lemma bd_tokEmb (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE θ.tokEmb) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.tokEmb) (fun p => Sum.inl p) Sum.inl_injective (fun _ _ _ => rfl)
    hI.tokEmb_le hε hB

lemma bd_posEmb (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE θ.posEmb) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.posEmb) (fun p => Sum.inr (Sum.inl p))
    (fun p q h => by simpa using h) (fun _ _ _ => rfl) hI.posEmb_le hε hB

lemma bd_wQ (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (l : Fin 2) (h : Fin nH) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wQ l h)) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.wQ l h) (fun p => Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2))))
    (fun p q hpq => by simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at hpq
                       exact Prod.ext hpq.2.2.1 hpq.2.2.2)
    (fun _ _ _ => rfl) (hI.wQ_le l h) hε hB

lemma bd_wK (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (l : Fin 2) (h : Fin nH) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wK l h)) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.wK l h)
    (fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2)))))
    (fun p q hpq => by simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at hpq
                       exact Prod.ext hpq.2.2.1 hpq.2.2.2)
    (fun _ _ _ => rfl) (hI.wK_le l h) hε hB

lemma bd_wV (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (l : Fin 2) (h : Fin nH) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wV l h)) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.wV l h)
    (fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2))))))
    (fun p q hpq => by simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at hpq
                       exact Prod.ext hpq.2.2.1 hpq.2.2.2)
    (fun _ _ _ => rfl) (hI.wV_le l h) hε hB

lemma bd_wIn (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (l : Fin 2) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wIn l)) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.wIn l)
    (fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, p.1, p.2)))))))
    (fun p q hpq => by simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at hpq
                       exact Prod.ext hpq.2.1 hpq.2.2)
    (fun _ _ _ => rfl) (hI.wIn_le l) hε hB

lemma bd_wOut (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (l : Fin 2) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wOut l)) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.wOut l)
    (fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr
      (Sum.inl (l, p.1, p.2)))))))))
    (fun p q hpq => by simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at hpq
                       exact Prod.ext hpq.2.1 hpq.2.2)
    (fun _ _ _ => rfl) (hI.wOut_le l) hε hB

lemma bd_readout (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε)
    (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    Bd B κ ε (fun θ (_ : Fin n → Sym) => toE θ.readout) ((|a0| + 2) * ε) :=
  bd_blk (fun θ => θ.readout)
    (fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr p))))))))
    (fun p q hpq => by simpa using hpq)
    (fun _ _ _ => rfl) hI.readout_le hε hB

/-- INTERNAL: on the ball, every MLP bias keeps the sign of `b̃_m` with modulus `≥ 3ε/4`. -/
lemma bias_sign (hI : InitHyp θt a0 cemb Nb) (hε : 0 ≤ ε) {θ : Params Sym K A}
    (hθ : ‖θ - ε • θt‖ ≤ ε / 4) (l : Fin 2) (m : Fin A.dmlp) :
    (0 < θt.bIn l m → 3 * ε / 4 ≤ θ.bIn l m) ∧ (¬ 0 < θt.bIn l m → θ.bIn l m ≤ -(3 * ε / 4)) := by
  have hc : |(θ - ε • θt).bIn l m| ≤ ε / 4 := by
    have := PiLp.norm_apply_le (θ - ε • θt)
      (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m))))))))
    rw [Real.norm_eq_abs] at this
    exact this.trans hθ
  have e1 : θ.bIn l m = ε * θt.bIn l m + (θ - ε • θt).bIn l m := by
    simp [Params.bIn]
  have h1 := hI.bIn_ge_one l m
  rw [abs_le] at hc
  constructor
  · intro hp
    rw [abs_of_pos hp] at h1
    rw [e1]; nlinarith
  · intro hp
    rw [abs_of_nonpos (not_lt.1 hp)] at h1
    rw [e1]; nlinarith

end Ball

/-! ## The forward pass in Euclidean form -/

section Forward

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n}

/-- INTERNAL: the residual after the attention sublayer of block `l` (the `let` in
`Trace.block`). -/
noncomputable def attnOut (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i : Fin (n + 4)) : Fin A.dres → ℝ :=
  x i + ∑ h, ∑ j ∈ A.vis l h i, attnWeight ζ θ l h x i j • (θ.wV l h *ᵥ x j)

lemma block_eq (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2) (x : Fin (n + 4) → Fin A.dres → ℝ)
    (i : Fin (n + 4)) : block ζ θ l x i = attnOut ζ θ l x i +
      θ.wOut l *ᵥ fun m => max ((θ.wIn l *ᵥ attnOut ζ θ l x i) m + θ.bIn l m) 0 := rfl

lemma vec_attnOut (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2) (x : Fin (n + 4) → Fin A.dres → ℝ)
    (i : Fin (n + 4)) : WithLp.toLp 2 (attnOut ζ θ l x i) = WithLp.toLp 2 (x i) +
      ∑ h, ∑ j ∈ A.vis l h i, attnWeight ζ θ l h x i j • toE (θ.wV l h) (WithLp.toLp 2 (x j)) := by
  simp [attnOut, toE_toLp, WithLp.toLp_sum]

lemma score_eq (θ : Params Sym K A) (l : Fin 2) (h : Fin nH) (x : Fin (n + 4) → Fin A.dres → ℝ)
    (i j : Fin (n + 4)) : score θ l h x i j = (Real.sqrt A.dh)⁻¹ *
      inner ℝ (toE (θ.wQ l h) (WithLp.toLp 2 (x i))) (toE (θ.wK l h) (WithLp.toLp 2 (x j))) := by
  rw [score, toE_toLp, toE_toLp, EuclideanSpace.inner_eq_star_dotProduct]
  simp [dotProduct_comm]

lemma card_vis_pos (l : Fin 2) (h : Fin nH) (i : Fin (n + 4)) : 0 < ((A.vis l h i).card : ℝ) := by
  exact_mod_cast (A.vis_nonempty l h i).card_pos

lemma attnWeight_zero (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) :
    attnWeight 0 θ l h x i j = 1 / ((A.vis l h i).card : ℝ) := by
  simp [attnWeight]

/-- INTERNAL: the softmax weight as `(1/m)(1 + δ)` with `δ = e^ψ (1 + u)⁻¹ − 1` and
`u = m⁻¹ ∑_k (e^{ψ_k} − 1)`. -/
lemma attnWeight_one (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) :
    attnWeight 1 θ l h x i j = 1 / ((A.vis l h i).card : ℝ) *
      (1 + (Real.exp (score θ l h x i j) * (1 + 1 / ((A.vis l h i).card : ℝ) *
        ∑ k ∈ A.vis l h i, (Real.exp (score θ l h x i k) - 1))⁻¹ - 1)) := by
  have hm := card_vis_pos (A := A) l h i
  have e1 : 1 + 1 / ((A.vis l h i).card : ℝ) * ∑ k ∈ A.vis l h i, (Real.exp (score θ l h x i k) - 1)
      = 1 / ((A.vis l h i).card : ℝ) * ∑ k ∈ A.vis l h i, Real.exp (score θ l h x i k) := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one]
    field_simp
    ring
  rw [e1, attnWeight]
  simp only [one_mul]
  have hs : 0 < ∑ k ∈ A.vis l h i, Real.exp (score θ l h x i k) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) (A.vis_nonempty l h i)
  field_simp
  ring

end Forward

/-! ## Attention costs -/

section Attention

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n} {B : Set (Params Sym K A)} {κ ε : ℝ}

lemma inv_sqrt_le_one (d : ℕ) : (Real.sqrt d)⁻¹ ≤ 1 := by
  rcases Nat.eq_zero_or_pos d with h0 | hpos
  · simp [h0]
  · exact inv_le_one_of_one_le₀ (Real.one_le_sqrt.2 (by exact_mod_cast hpos))

lemma exp_sub_one_le_two_mul {β : ℝ} (h0 : 0 ≤ β) (h : β ≤ 1) : Real.exp β - 1 ≤ 2 * β := by
  have := Real.abs_exp_sub_one_le (x := β) (by rw [abs_of_nonneg h0]; exact h)
  rw [abs_of_nonneg h0] at this
  exact (le_abs_self _).trans this

/-- INTERNAL: an attention score costs (query cost) × (key cost). -/
lemma bd_score (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) (l : Fin 2) (h : Fin nH)
    (i j : Fin (n + 4)) {X : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    {wq wk cQ cK : ℝ} (hQ : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wQ l h)) wq)
    (hK : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wK l h)) wk)
    (hXi : Bd B κ ε (fun θ x => WithLp.toLp 2 (X θ x i)) cQ)
    (hXj : Bd B κ ε (fun θ x => WithLp.toLp 2 (X θ x j)) cK) :
    Bd B κ ε (fun θ x => score θ l h (X θ x) i j) (wq * cQ * (wk * cK)) := by
  have hq := Bd.app hκ hκ1 hε.le hQ hXi
  have hk := Bd.app hκ hκ1 hε.le hK hXj
  have hin := Bd.bilin hκ hκ1 hε.le
    (innerSL ℝ : EuclideanSpace ℝ (Fin A.dh) →L[ℝ] EuclideanSpace ℝ (Fin A.dh) →L[ℝ] ℝ) hq hk
  have hs := hin.smul (Real.sqrt A.dh)⁻¹
  refine (hs.congr fun θ _ x => ?_).mono ?_
  · simp only [smul_eq_mul, score_eq]; rfl
  · have h1 := hq.nonneg hκ hε.le
    have h2 := hk.nonneg hκ hε.le
    have hn := norm_innerSL_le (𝕜 := ℝ) (E := EuclideanSpace ℝ (Fin A.dh))
    have hc : |(Real.sqrt A.dh)⁻¹| ≤ 1 := by
      rw [abs_of_nonneg (by positivity)]; exact inv_sqrt_le_one _
    calc |(Real.sqrt A.dh)⁻¹| * (‖(innerSL ℝ : EuclideanSpace ℝ (Fin A.dh) →L[ℝ]
          EuclideanSpace ℝ (Fin A.dh) →L[ℝ] ℝ)‖ * (wq * cQ) * (wk * cK))
        ≤ 1 * (1 * (wq * cQ) * (wk * cK)) := by gcongr
      _ = _ := by ring

/-- INTERNAL: the softmax correction `δ = e^{ψ_j} (1 + m⁻¹ ∑_k (e^{ψ_k} − 1))⁻¹ − 1` costs at
most `8β` when every score costs `β ≤ 1/4`. -/
lemma bd_softmax_delta (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) {ι' : Type*}
    (V : Finset ι') (hV : V.Nonempty) (ψ : ι' → Params Sym K A → (Fin n → Sym) → ℝ) {β : ℝ}
    (hψ : ∀ k, Bd B κ ε (ψ k) β) (hβ : β ≤ 1 / 4) (j : ι') :
    Bd B κ ε (fun θ x => Real.exp (ψ j θ x) * (1 + 1 / (V.card : ℝ) *
      ∑ k ∈ V, (Real.exp (ψ k θ x) - 1))⁻¹ - 1) (8 * β) := by
  have hβ0 : 0 ≤ β := (hψ j).nonneg hκ hε.le
  have hm : 0 < (V.card : ℝ) := by exact_mod_cast hV.card_pos
  have hE : ∀ k, Bd B κ ε (fun θ x => Real.exp (ψ k θ x) - 1) (2 * β) := fun k =>
    ((hψ k).exp_sub_one hκ hκ1 hε).mono (exp_sub_one_le_two_mul hβ0 (by linarith))
  have hu : Bd B κ ε (fun θ x => 1 / (V.card : ℝ) * ∑ k ∈ V, (Real.exp (ψ k θ x) - 1))
      (2 * β) := by
    have := (Bd.sum V (fun k _ => hE k)).smul (1 / (V.card : ℝ))
    refine (this.congr fun θ _ x => by simp only [smul_eq_mul]).mono (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul, abs_of_pos (by positivity)]
    field_simp
  have hw := hu.inv_one_add_sub_one hκ hκ1 hε (by linarith)
  have hw' : Bd B κ ε (fun θ x => (1 + 1 / (V.card : ℝ) *
      ∑ k ∈ V, (Real.exp (ψ k θ x) - 1))⁻¹ - 1) (4 * β) := by
    refine hw.mono ?_
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hsum := ((hE j).add hw').add (Bd.mul hκ hκ1 hε.le (hE j) hw')
  refine (hsum.congr fun θ _ x => by ring).mono ?_
  nlinarith

/-- INTERNAL: the softmax correction of `attnWeight_one`. -/
noncomputable def sdelta (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) : ℝ :=
  Real.exp (score θ l h x i j) * (1 + 1 / ((A.vis l h i).card : ℝ) *
    ∑ k ∈ A.vis l h i, (Real.exp (score θ l h x i k) - 1))⁻¹ - 1

lemma attnWeight_one' (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) :
    attnWeight 1 θ l h x i j = 1 / ((A.vis l h i).card : ℝ) * (1 + sdelta θ l h x i j) :=
  attnWeight_one θ l h x i j

/-- INTERNAL: one head's output at `ζ = 1`, `ζ = 0`, and their difference. -/
lemma bd_head (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) (l : Fin 2) (h : Fin nH)
    (i : Fin (n + 4)) {X1 X0 : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    {wv β cK cK0 d : ℝ} (hV : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wV l h)) wv)
    (hS : ∀ k, Bd B κ ε (fun θ x => score θ l h (X1 θ x) i k) β) (hβ : β ≤ 1 / 8)
    (h1 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X1 θ x j)) cK)
    (h0 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X0 θ x j)) cK0)
    (hd : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X1 θ x j) - WithLp.toLp 2 (X0 θ x j)) d) :
    Bd B κ ε (fun θ x => ∑ j ∈ A.vis l h i,
        attnWeight 1 θ l h (X1 θ x) i j • toE (θ.wV l h) (WithLp.toLp 2 (X1 θ x j)))
      (2 * (wv * cK)) ∧
    Bd B κ ε (fun θ x => ∑ j ∈ A.vis l h i,
        attnWeight 0 θ l h (X0 θ x) i j • toE (θ.wV l h) (WithLp.toLp 2 (X0 θ x j)))
      (wv * cK0) ∧
    Bd B κ ε (fun θ x => ∑ j ∈ A.vis l h i,
        attnWeight 1 θ l h (X1 θ x) i j • toE (θ.wV l h) (WithLp.toLp 2 (X1 θ x j)) -
        ∑ j ∈ A.vis l h i,
        attnWeight 0 θ l h (X0 θ x) i j • toE (θ.wV l h) (WithLp.toLp 2 (X0 θ x j)))
      (8 * β * (wv * cK) + wv * d) := by
  have hm := card_vis_pos (A := A) l h i
  set m : ℝ := ((A.vis l h i).card : ℝ) with hmdef
  have hβ0 : 0 ≤ β := (hS i).nonneg hκ hε.le
  have hδ : ∀ j, Bd B κ ε (fun θ x => sdelta θ l h (X1 θ x) i j) (8 * β) := fun j =>
    bd_softmax_delta hκ hκ1 hε (A.vis l h i) (A.vis_nonempty l h i)
      (fun k θ x => score θ l h (X1 θ x) i k) hS (by linarith) j
  have hv1 : ∀ j, Bd B κ ε (fun θ x => toE (θ.wV l h) (WithLp.toLp 2 (X1 θ x j))) (wv * cK) :=
    fun j => Bd.app hκ hκ1 hε.le hV (h1 j)
  have hv0 : ∀ j, Bd B κ ε (fun θ x => toE (θ.wV l h) (WithLp.toLp 2 (X0 θ x j))) (wv * cK0) :=
    fun j => Bd.app hκ hκ1 hε.le hV (h0 j)
  have hvd : ∀ j, Bd B κ ε (fun θ x => toE (θ.wV l h)
      (WithLp.toLp 2 (X1 θ x j) - WithLp.toLp 2 (X0 θ x j))) (wv * d) :=
    fun j => Bd.app hκ hκ1 hε.le hV (hd j)
  have hvc : 0 ≤ wv * cK := (hv1 i).nonneg hκ hε.le
  have hcard : ∑ _j ∈ A.vis l h i, (1 : ℝ) = m := by simp [hmdef]
  refine ⟨?_, ?_, ?_⟩
  · have hterm : ∀ j, Bd B κ ε (fun θ x => (1 + sdelta θ l h (X1 θ x) i j) •
        toE (θ.wV l h) (WithLp.toLp 2 (X1 θ x j))) ((1 + 8 * β) * (wv * cK)) := fun j => by
      have := Bd.smul' hκ hκ1 hε.le ((Bd.const (1 : ℝ)).add (hδ j)) (hv1 j)
      simpa using this
    have hs := (Bd.sum (A.vis l h i) (fun j _ => hterm j)).smul (1 / m)
    refine (hs.congr fun θ _ x => ?_).mono ?_
    · simp only [Finset.smul_sum, smul_smul, attnWeight_one', hmdef]
    · rw [Finset.sum_const, nsmul_eq_mul, abs_of_pos (by positivity)]
      calc 1 / m * (↑(A.vis l h i).card * ((1 + 8 * β) * (wv * cK))) = (1 + 8 * β) * (wv * cK) := by
            rw [← hmdef]; field_simp
        _ ≤ 2 * (wv * cK) := by nlinarith
  · have hs := (Bd.sum (A.vis l h i) (fun j _ => hv0 j)).smul (1 / m)
    refine (hs.congr fun θ _ x => ?_).mono (le_of_eq ?_)
    · simp only [Finset.smul_sum, attnWeight_zero, hmdef]
    · rw [Finset.sum_const, nsmul_eq_mul, abs_of_pos (by positivity), ← hmdef]
      field_simp
  · have hterm : ∀ j, Bd B κ ε (fun θ x => sdelta θ l h (X1 θ x) i j •
        toE (θ.wV l h) (WithLp.toLp 2 (X1 θ x j)) + toE (θ.wV l h)
          (WithLp.toLp 2 (X1 θ x j) - WithLp.toLp 2 (X0 θ x j)))
        (8 * β * (wv * cK) + wv * d) := fun j =>
      (Bd.smul' hκ hκ1 hε.le (hδ j) (hv1 j)).add (hvd j)
    have hs := (Bd.sum (A.vis l h i) (fun j _ => hterm j)).smul (1 / m)
    refine (hs.congr fun θ _ x => ?_).mono (le_of_eq ?_)
    · simp only [Finset.smul_sum, attnWeight_one', attnWeight_zero, ← hmdef,
        ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [map_sub, smul_add, smul_sub, add_smul, smul_smul, mul_add, mul_one]
      abel
    · rw [Finset.sum_const, nsmul_eq_mul, abs_of_pos (by positivity), ← hmdef]
      field_simp

/-- INTERNAL: the attention sublayer at position `i`: costs at `ζ = 1`, at `ζ = 0`, and of the
score-dependent difference. -/
lemma bd_attn (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) (l : Fin 2) (i : Fin (n + 4))
    {X1 X0 : Params Sym K A → (Fin n → Sym) → Fin (n + 4) → Fin A.dres → ℝ}
    {wq wk wv cQ cK cQ0 cK0 d : ℝ}
    (hQ : ∀ h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wQ l h)) wq)
    (hK : ∀ h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wK l h)) wk)
    (hV : ∀ h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wV l h)) wv)
    (h1i : Bd B κ ε (fun θ x => WithLp.toLp 2 (X1 θ x i)) cQ)
    (h1 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X1 θ x j)) cK)
    (h0i : Bd B κ ε (fun θ x => WithLp.toLp 2 (X0 θ x i)) cQ0)
    (h0 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X0 θ x j)) cK0)
    (hd : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (X1 θ x j) - WithLp.toLp 2 (X0 θ x j)) d)
    (hβ : wq * cQ * (wk * cK) ≤ 1 / 8) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 1 θ l (X1 θ x) i))
      (cQ + nH * (2 * (wv * cK))) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 0 θ l (X0 θ x) i))
      (cQ0 + nH * (wv * cK0)) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 1 θ l (X1 θ x) i) -
        WithLp.toLp 2 (attnOut 0 θ l (X0 θ x) i))
      (d + nH * (8 * (wq * cQ * (wk * cK)) * (wv * cK) + wv * d)) := by
  have hH := fun h => bd_head hκ hκ1 hε l h i (hV h)
    (fun k => bd_score hκ hκ1 hε l h i k (hQ h) (hK h) h1i (h1 k)) hβ h1 h0 hd
  refine ⟨?_, ?_, ?_⟩
  · have := h1i.add (Bd.sum Finset.univ fun h _ => (hH h).1)
    refine (this.congr fun θ _ x => by rw [vec_attnOut]).mono (le_of_eq ?_)
    simp
  · have := h0i.add (Bd.sum Finset.univ fun h _ => (hH h).2.1)
    refine (this.congr fun θ _ x => by rw [vec_attnOut]).mono (le_of_eq ?_)
    simp
  · have := (hd i).add (Bd.sum Finset.univ fun h _ => (hH h).2.2)
    refine (this.congr fun θ _ x => ?_).mono (le_of_eq ?_)
    · rw [vec_attnOut, vec_attnOut, Finset.sum_sub_distrib]; abel
    · simp; ring

end Attention

/-! ## The ReLU sublayer with a fixed activation pattern -/

section MLP

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n} {B : Set (Params Sym K A)} {κ ε : ℝ}

/-- INTERNAL: the activation mask `v ↦ (pat m ? v_m : 0)`. -/
noncomputable def maskE {d : ℕ} (pat : Fin d → Prop) [DecidablePred pat] :
    EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d) :=
  toE (Matrix.diagonal fun m => if pat m then 1 else 0)

lemma maskE_toLp {d : ℕ} (pat : Fin d → Prop) [DecidablePred pat] (v : Fin d → ℝ) :
    maskE pat (WithLp.toLp 2 v) = WithLp.toLp 2 (fun m => if pat m then v m else 0) := by
  rw [maskE, toE_toLp]
  congr 1; ext m; rw [Matrix.mulVec_diagonal]; split_ifs <;> simp

lemma norm_maskE_le {d : ℕ} (pat : Fin d → Prop) [DecidablePred pat] : ‖maskE pat‖ ≤ 1 := by
  refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun v => ?_
  have hv : v = WithLp.toLp 2 v.ofLp := rfl
  rw [hv, maskE_toLp, one_mul, EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun m _ => ?_)
  simp only [Real.norm_eq_abs, sq_abs]
  show (if pat m then v.ofLp m else 0) ^ 2 ≤ v.ofLp m ^ 2
  split_ifs <;> simp [sq_nonneg]

/-- INTERNAL: the residual ReLU sublayer `Y + W_out ReLU(W_in Y + b)` with the activation
pattern `pat` frozen. -/
lemma vec_mlp (θ : Params Sym K A) (l : Fin 2) (Y : Fin A.dres → ℝ) (pat : Fin A.dmlp → Prop)
    [DecidablePred pat] (hpat : ∀ m, max ((θ.wIn l *ᵥ Y) m + θ.bIn l m) 0 =
      if pat m then (θ.wIn l *ᵥ Y) m + θ.bIn l m else 0) :
    WithLp.toLp 2 (Y + θ.wOut l *ᵥ fun m => max ((θ.wIn l *ᵥ Y) m + θ.bIn l m) 0) =
      WithLp.toLp 2 Y + toE (θ.wOut l)
        (maskE pat (toE (θ.wIn l) (WithLp.toLp 2 Y) + biasE θ l)) := by
  rw [toE_toLp, biasE, ← WithLp.toLp_add, maskE_toLp, toE_toLp, ← WithLp.toLp_add]
  congr 3
  funext m
  rw [hpat m]; rfl

/-- INTERNAL: costs through the frozen-pattern ReLU sublayer, and of a difference of two
inputs sharing the pattern (the bias cancels). -/
lemma bd_mlp (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) (l : Fin 2)
    (pat : Fin A.dmlp → Prop) [DecidablePred pat] {Y1 Y0 : Params Sym K A → (Fin n → Sym) → Fin A.dres → ℝ}
    {w cb c1 c0 d : ℝ}
    (hIn : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wIn l)) w)
    (hOut : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wOut l)) w)
    (hb : Bd B κ ε (fun θ (_ : Fin n → Sym) => biasE θ l) cb)
    (hp1 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn l *ᵥ Y1 θ x) m + θ.bIn l m) 0 =
      if pat m then (θ.wIn l *ᵥ Y1 θ x) m + θ.bIn l m else 0)
    (hp0 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn l *ᵥ Y0 θ x) m + θ.bIn l m) 0 =
      if pat m then (θ.wIn l *ᵥ Y0 θ x) m + θ.bIn l m else 0)
    (h1 : Bd B κ ε (fun θ x => WithLp.toLp 2 (Y1 θ x)) c1)
    (h0 : Bd B κ ε (fun θ x => WithLp.toLp 2 (Y0 θ x)) c0)
    (hd : Bd B κ ε (fun θ x => WithLp.toLp 2 (Y1 θ x) - WithLp.toLp 2 (Y0 θ x)) d) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (Y1 θ x + θ.wOut l *ᵥ
      fun m => max ((θ.wIn l *ᵥ Y1 θ x) m + θ.bIn l m) 0)) (c1 + w * (w * c1 + cb)) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (Y0 θ x + θ.wOut l *ᵥ
      fun m => max ((θ.wIn l *ᵥ Y0 θ x) m + θ.bIn l m) 0)) (c0 + w * (w * c0 + cb)) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (Y1 θ x + θ.wOut l *ᵥ
      fun m => max ((θ.wIn l *ᵥ Y1 θ x) m + θ.bIn l m) 0) -
        WithLp.toLp 2 (Y0 θ x + θ.wOut l *ᵥ
      fun m => max ((θ.wIn l *ᵥ Y0 θ x) m + θ.bIn l m) 0)) (d + w * (w * d)) := by
  have hM : Bd B κ ε (fun (_ : Params Sym K A) (_ : Fin n → Sym) => maskE pat) 1 :=
    (Bd.const _).mono (norm_maskE_le pat)
  have hw := hIn.nonneg hκ hε.le
  have one_mul' : ∀ t : ℝ, 0 ≤ t → 1 * t ≤ t := fun t _ => by rw [one_mul]
  refine ⟨?_, ?_, ?_⟩
  · have := h1.add (Bd.app hκ hκ1 hε.le hOut (Bd.app hκ hκ1 hε.le hM
      ((Bd.app hκ hκ1 hε.le hIn h1).add hb)))
    refine (this.congr fun θ hθ x => by rw [vec_mlp θ l _ pat (hp1 θ hθ x)]).mono (le_of_eq ?_)
    ring
  · have := h0.add (Bd.app hκ hκ1 hε.le hOut (Bd.app hκ hκ1 hε.le hM
      ((Bd.app hκ hκ1 hε.le hIn h0).add hb)))
    refine (this.congr fun θ hθ x => by rw [vec_mlp θ l _ pat (hp0 θ hθ x)]).mono (le_of_eq ?_)
    ring
  · have := hd.add (Bd.app hκ hκ1 hε.le hOut (Bd.app hκ hκ1 hε.le hM
      (Bd.app hκ hκ1 hε.le hIn hd)))
    refine (this.congr fun θ hθ x => ?_).mono (le_of_eq ?_)
    · rw [vec_mlp θ l _ pat (hp1 θ hθ x), vec_mlp θ l _ pat (hp0 θ hθ x)]
      simp only [map_sub, map_add]
      abel
    · ring

/-- INTERNAL: a frozen ReLU pattern. If `‖W_in Y‖ < 3ε/4` on the ball, every unit has the
sign of its bias `b̃_m` (Paper/appendix.tex:338-340, `|b_m| ≥ 3ε/4 > |(W_in h)_m|`). -/
lemma relu_pattern {θt θ : Params Sym K A} {a0 cemb Nb : ℝ} (hI : InitHyp θt a0 cemb Nb)
    (hε : 0 ≤ ε) (hθ : ‖θ - ε • θt‖ ≤ ε / 4) (l : Fin 2) (Y : Fin A.dres → ℝ)
    (hY : ‖toE (θ.wIn l) (WithLp.toLp 2 Y)‖ < 3 * ε / 4) (m : Fin A.dmlp) :
    max ((θ.wIn l *ᵥ Y) m + θ.bIn l m) 0 =
      if 0 < θt.bIn l m then (θ.wIn l *ᵥ Y) m + θ.bIn l m else 0 := by
  have hc : |(θ.wIn l *ᵥ Y) m| < 3 * ε / 4 := by
    have := PiLp.norm_apply_le (toE (θ.wIn l) (WithLp.toLp 2 Y)) m
    rw [toE_toLp, Real.norm_eq_abs] at this
    rw [toE_toLp] at hY
    exact lt_of_le_of_lt this hY
  rw [abs_lt] at hc
  obtain ⟨hp, hn⟩ := bias_sign hI hε hθ l m
  split_ifs with h
  · exact max_eq_left (by linarith [hp h])
  · exact max_eq_right (by linarith [hn h])

end MLP

/-! ## The input embedding -/

section Embed

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n} {B : Set (Params Sym K A)} {κ ε : ℝ}

/-- INTERNAL: the letters read by prompt position `i`: `w_t` sits at `t + 2`. -/
def letSet (i : Fin (n + 4)) : Finset (Fin n) := Finset.univ.filter fun t => t.val + 2 = i.val

lemma card_letSet_le (i : Fin (n + 4)) : (letSet i).card ≤ 1 :=
  Finset.card_le_one.2 fun a ha b hb => by
    simp only [letSet, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    exact Fin.ext (by omega)

lemma letSet_last : letSet (n := n) (Fin.last (n + 3)) = ∅ := by
  ext t; simp only [letSet, Finset.mem_filter, Finset.mem_univ, true_and, Fin.val_last,
    Finset.notMem_empty, iff_false]
  omega

lemma prompt_dependsOn (T : Task K Sym) (i : Fin (n + 4)) :
    DependsOn (fun w : Fin n → Sym => T.outcomePrompt w i) (letSet i : Set (Fin n)) := by
  intro w w' h
  simp only [Task.outcomePrompt]
  split_ifs with h0 h1 h2
  · rfl
  · rfl
  · congr 2
    apply h
    simp only [letSet, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq]
    omega
  · rfl
  · rfl

lemma vec_embed (T : Task K Sym) (θ : Params Sym K A) (w : Fin n → Sym) (i : Fin (n + 4)) :
    WithLp.toLp 2 (embed T θ w i) =
      toE θ.tokEmb (unit (T.outcomePrompt w i)) + toE θ.posEmb (unit i) := by
  rw [toE_unit, toE_unit, ← WithLp.toLp_add]
  rfl

/-- INTERNAL: an input vector costs `aε κ^{-|S_i|} + aε`; it reads at most the one letter at
its position. -/
lemma bd_embed (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε)
    (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) (i : Fin (n + 4)) :
    Bd B κ ε (fun θ w => WithLp.toLp 2 (embed T θ w i))
      ((|a0| + 2) * ε * κ⁻¹ ^ (letSet i).card + (|a0| + 2) * ε) := by
  have hu : Bd B κ ε (fun (_ : Params Sym K A) (w : Fin n → Sym) =>
      (unit (T.outcomePrompt w i) : EuclideanSpace ℝ (Token Sym K)))
      (κ⁻¹ ^ (letSet i).card * (1 + ε * 0)) :=
    Bd.single (letSet i) zero_le_one le_rfl
      (fun θ x y hxy => congrArg unit (prompt_dependsOn T i hxy))
      (fun θ _ x => by rw [norm_unit]) (fun θ _ θ' _ x => by simp)
  have hp : Bd B κ ε (fun (_ : Params Sym K A) (_ : Fin n → Sym) =>
      (unit i : EuclideanSpace ℝ (Fin (n + 4)))) 1 := by
    simpa [norm_unit] using (Bd.const (B := B) (κ := κ) (r := ε) (ι := Fin n) (α := Sym)
      (unit i : EuclideanSpace ℝ (Fin (n + 4))))
  have := (Bd.app hκ hκ1 hε.le (bd_tokEmb hI hε.le hB) hu).add
    (Bd.app hκ hκ1 hε.le (bd_posEmb hI hε.le hB) hp)
  refine (this.congr fun θ _ w => by rw [vec_embed]).mono (le_of_eq ?_)
  ring

end Embed

/-! ## The chain through the two blocks -/

section Chain

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n} {B : Set (Params Sym K A)} {κ ε : ℝ}

/-- INTERNAL: the costs of all parameter blocks and inputs, at weight `κ` and scale `ε`. -/
structure Costs (T : Task K Sym) (B : Set (Params Sym K A)) (κ ε a cb : ℝ) : Prop where
  wQ : ∀ l h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wQ l h)) (a * ε)
  wK : ∀ l h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wK l h)) (a * ε)
  wV : ∀ l h, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wV l h)) (a * ε)
  wIn : ∀ l, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wIn l)) (a * ε)
  wOut : ∀ l, Bd B κ ε (fun θ (_ : Fin n → Sym) => toE (θ.wOut l)) (a * ε)
  readout : Bd B κ ε (fun θ (_ : Fin n → Sym) => toE θ.readout) (a * ε)
  bias : ∀ l, Bd B κ ε (fun θ (_ : Fin n → Sym) => biasE θ l) cb
  emb : ∀ i, Bd B κ ε (fun θ w => WithLp.toLp 2 (embed T θ w i))
    (a * ε * κ⁻¹ ^ (letSet i).card + a * ε)

lemma costs_of_init (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hε : 0 < ε) (hε1 : ε ≤ 1)
    (hNb : ε * Nb ≤ 1) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    Costs T B κ ε (|a0| + 2) 3 where
  wQ l h := bd_wQ hI hε.le hB l h
  wK l h := bd_wK hI hε.le hB l h
  wV l h := bd_wV hI hε.le hB l h
  wIn l := bd_wIn hI hε.le hB l
  wOut l := bd_wOut hI hε.le hB l
  readout := bd_readout hI hε.le hB
  bias l := (bd_bias hI hε.le hB l).mono (by linarith)
  emb i := bd_embed T hI hκ hκ1 hε hB i

lemma inv_pow_card_le (hκ : 0 < κ) (hκ1 : κ ≤ 1) (i : Fin (n + 4)) :
    κ⁻¹ ^ (letSet i).card ≤ κ⁻¹ := by
  have hk : 1 ≤ κ⁻¹ := (one_le_inv₀ hκ).2 hκ1
  calc κ⁻¹ ^ (letSet i).card ≤ κ⁻¹ ^ 1 := pow_le_pow_right₀ hk (card_letSet_le i)
    _ = κ⁻¹ := pow_one _

lemma Costs.emb_le (T : Task K Sym) {a cb : ℝ} (hC : Costs T B κ ε a cb) (hκ : 0 < κ)
    (hκ1 : κ ≤ 1) (hε : 0 < ε) (ha : 0 ≤ a) (i : Fin (n + 4)) :
    Bd B κ ε (fun θ w => WithLp.toLp 2 (embed T θ w i)) (2 * a * ε * κ⁻¹) := by
  refine (hC.emb i).mono ?_
  have h1 := inv_pow_card_le hκ hκ1 i
  have hk : 1 ≤ κ⁻¹ := (one_le_inv₀ hκ).2 hκ1
  have hae : 0 ≤ a * ε := mul_nonneg ha hε.le
  nlinarith

/-- INTERNAL: block 1, attention sublayer (`X1 = X0 =` the embedding, no difference yet). -/
lemma chainA (T : Task K Sym) {a cb : ℝ} (hC : Costs T B κ ε a cb) (hκ : 0 < κ) (hκ1 : κ ≤ 1)
    (hε : 0 < ε) (ha : 0 ≤ a) (hβ : (a * ε * (2 * a * ε * κ⁻¹)) ^ 2 ≤ 1 / 8)
    (i : Fin (n + 4)) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 1 θ 0 (embed T θ x) i))
      (a * ε * κ⁻¹ ^ (letSet i).card + a * ε + 4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 0 θ 0 (embed T θ x) i))
      (a * ε * κ⁻¹ ^ (letSet i).card + a * ε + 4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (attnOut 1 θ 0 (embed T θ x) i) -
        WithLp.toLp 2 (attnOut 0 θ 0 (embed T θ x) i))
      (64 * nH * a ^ 6 * ε ^ 6 * κ⁻¹ ^ 3) := by
  have hX := fun j => hC.emb_le T hκ hκ1 hε ha j
  have hd : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (embed T θ x j) -
      WithLp.toLp 2 (embed T θ x j)) 0 := fun j =>
    (Bd.zero (B := B) (κ := κ) (r := ε)).congr fun θ _ x => by simp
  have hk : 1 ≤ κ⁻¹ := (one_le_inv₀ hκ).2 hκ1
  have hci : a * ε * κ⁻¹ ^ (letSet i).card + a * ε ≤ 2 * a * ε * κ⁻¹ := by
    have h1 := inv_pow_card_le hκ hκ1 i
    have hae : 0 ≤ a * ε := mul_nonneg ha hε.le
    nlinarith
  have hci0 : 0 ≤ a * ε * κ⁻¹ ^ (letSet i).card + a * ε := (hC.emb i).nonneg hκ hε.le
  have hw : 0 ≤ a * ε := mul_nonneg ha hε.le
  have hX0 : 0 ≤ 2 * a * ε * κ⁻¹ := by positivity
  have hβ' : a * ε * (a * ε * κ⁻¹ ^ (letSet i).card + a * ε) * (a * ε * (2 * a * ε * κ⁻¹))
      ≤ 1 / 8 := by
    refine le_trans ?_ hβ
    rw [sq]
    gcongr
  obtain ⟨h1, h0, hΔ⟩ := bd_attn (X1 := fun θ x => embed T θ x) (X0 := fun θ x => embed T θ x)
    hκ hκ1 hε 0 i (hC.wQ 0) (hC.wK 0) (hC.wV 0) (hC.emb i) hX (hC.emb i) hX hd hβ'
  refine ⟨h1.mono (le_of_eq (by ring)), h0.mono ?_, hΔ.mono ?_⟩
  · have : (0 : ℝ) ≤ nH := Nat.cast_nonneg _
    nlinarith [mul_nonneg this (mul_nonneg hw hX0)]
  · have hn : (0 : ℝ) ≤ nH := Nat.cast_nonneg _
    have e1 : a * ε * (a * ε * κ⁻¹ ^ (letSet i).card + a * ε) ≤ a * ε * (2 * a * ε * κ⁻¹) :=
      mul_le_mul_of_nonneg_left hci hw
    have e2 : 0 ≤ a * ε * (2 * a * ε * κ⁻¹) := by positivity
    calc (0 : ℝ) + nH * (8 * (a * ε * (a * ε * κ⁻¹ ^ (letSet i).card + a * ε) *
          (a * ε * (2 * a * ε * κ⁻¹))) * (a * ε * (2 * a * ε * κ⁻¹)) + a * ε * 0)
        ≤ nH * (8 * (a * ε * (2 * a * ε * κ⁻¹) * (a * ε * (2 * a * ε * κ⁻¹))) *
          (a * ε * (2 * a * ε * κ⁻¹))) := by
          rw [zero_add, mul_zero, add_zero]
          gcongr
      _ = _ := by ring

/-- INTERNAL: a sup bound on `‖W_in Y‖` from costs at weight `1`. -/
lemma norm_wIn_apply_lt {a cb : ℝ} (T : Task K Sym) (hC : Costs T B 1 ε a cb) (hε : 0 < ε)
    (l : Fin 2) {Y : Params Sym K A → (Fin n → Sym) → Fin A.dres → ℝ} {c : ℝ}
    (hY : Bd B 1 ε (fun θ x => WithLp.toLp 2 (Y θ x)) c) (hsmall : a * ε * c < 3 * ε / 4)
    {θ : Params Sym K A} (hθ : θ ∈ B) (x : Fin n → Sym) :
    ‖toE (θ.wIn l) (WithLp.toLp 2 (Y θ x))‖ < 3 * ε / 4 := by
  have h1 := (hC.wIn l).norm_le one_pos le_rfl hε.le hθ x
  have h2 := hY.norm_le one_pos le_rfl hε.le hθ x
  calc ‖toE (θ.wIn l) (WithLp.toLp 2 (Y θ x))‖
      ≤ ‖toE (θ.wIn l)‖ * ‖WithLp.toLp 2 (Y θ x)‖ := (toE (θ.wIn l)).le_opNorm _
    _ ≤ a * ε * c := mul_le_mul h1 h2 (norm_nonneg _) ((norm_nonneg _).trans h1)
    _ < 3 * ε / 4 := hsmall

/-- INTERNAL: the block-1 ReLU pattern is frozen on the ball, at `ζ = 1` and `ζ = 0`. -/
lemma pattern1 (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb a cb : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hε : 0 < ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (hC : Costs T B 1 ε a cb) (ha : 0 ≤ a) (hβ : (a * ε * (2 * a * ε)) ^ 2 ≤ 1 / 8)
    (hsmall : a * ε * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2) < 3 * ε / 4) :
    ∀ θ ∈ B, ∀ x i m,
      max ((θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if 0 < θt.bIn 0 m then (θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m
          else 0) ∧
      max ((θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if 0 < θt.bIn 0 m then (θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m
          else 0) := by
  intro θ hθ x i m
  have hβ1 : (a * ε * (2 * a * ε * (1 : ℝ)⁻¹)) ^ 2 ≤ 1 / 8 := by simpa using hβ
  obtain ⟨h1, h0, -⟩ := chainA T hC one_pos le_rfl hε ha hβ1 i
  have e : a * ε * (1 : ℝ)⁻¹ ^ (letSet i).card + a * ε + 4 * nH * a ^ 2 * ε ^ 2 * (1 : ℝ)⁻¹ =
      2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2 := by simp; ring
  rw [e] at h1 h0
  have hs : a * ε * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2) < 3 * ε / 4 := hsmall
  exact ⟨relu_pattern hI hε.le (hB θ hθ) 0 _
      (norm_wIn_apply_lt (Y := fun θ x => attnOut 1 θ 0 (embed T θ x) i) T hC hε 0 h1 hs hθ x) m,
    relu_pattern hI hε.le (hB θ hθ) 0 _
      (norm_wIn_apply_lt (Y := fun θ x => attnOut 0 θ 0 (embed T θ x) i) T hC hε 0 h0 hs hθ x) m⟩

/-- INTERNAL: block 1, output. -/
lemma chainB (T : Task K Sym) {a cb : ℝ} (hC : Costs T B κ ε a cb) (hκ : 0 < κ) (hκ1 : κ ≤ 1)
    (hε : 0 < ε) (ha : 0 ≤ a) (hae : a * ε ≤ 1)
    (hβ : (a * ε * (2 * a * ε * κ⁻¹)) ^ 2 ≤ 1 / 8) (pat : Fin A.dmlp → Prop) [DecidablePred pat]
    (hp : ∀ θ ∈ B, ∀ x i m,
      max ((θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if pat m then (θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m else 0) ∧
      max ((θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if pat m then (θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m else 0))
    (i : Fin (n + 4)) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (block 1 θ 0 (embed T θ x) i))
      (2 * (a * ε * κ⁻¹ ^ (letSet i).card + a * ε + 4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) +
        a * ε * cb) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (block 0 θ 0 (embed T θ x) i))
      (2 * (a * ε * κ⁻¹ ^ (letSet i).card + a * ε + 4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) +
        a * ε * cb) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2 (block 1 θ 0 (embed T θ x) i) -
        WithLp.toLp 2 (block 0 θ 0 (embed T θ x) i))
      (128 * nH * a ^ 6 * ε ^ 6 * κ⁻¹ ^ 3) := by
  obtain ⟨h1, h0, hΔ⟩ := chainA T hC hκ hκ1 hε ha hβ i
  obtain ⟨g1, g0, gΔ⟩ := bd_mlp (Y1 := fun θ x => attnOut 1 θ 0 (embed T θ x) i)
    (Y0 := fun θ x => attnOut 0 θ 0 (embed T θ x) i) hκ hκ1 hε 0 pat (hC.wIn 0) (hC.wOut 0)
    (hC.bias 0) (fun θ hθ x m => (hp θ hθ x i m).1) (fun θ hθ x m => (hp θ hθ x i m).2) h1 h0 hΔ
  have hw : 0 ≤ a * ε := mul_nonneg ha hε.le
  have hw1 : (a * ε) * (a * ε) ≤ 1 := by nlinarith
  have hc0 := h1.nonneg hκ hε.le
  have hd0 := hΔ.nonneg hκ hε.le
  refine ⟨g1.mono ?_, g0.mono ?_, gΔ.mono ?_⟩
  · nlinarith [mul_le_mul_of_nonneg_right hw1 hc0]
  · nlinarith [mul_le_mul_of_nonneg_right hw1 hc0]
  · nlinarith [mul_le_mul_of_nonneg_right hw1 hd0]

lemma card_letSet_last : (letSet (n := n) (Fin.last (n + 3))).card = 0 := by
  rw [letSet_last]; rfl

/-- INTERNAL: block 2, attention sublayer at the answer position, given the block-1 bounds:
`cK` for every position, `cQ` at the answer position, `d` for the score-dependent part. -/
lemma chainC (T : Task K Sym) {a cb cK cQ d : ℝ} (hC : Costs T B κ ε a cb) (hκ : 0 < κ)
    (hκ1 : κ ≤ 1) (hε : 0 < ε)
    (hK1 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (block 1 θ 0 (embed T θ x) j)) cK)
    (hK0 : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (block 0 θ 0 (embed T θ x) j)) cK)
    (hQ1 : Bd B κ ε (fun θ x => WithLp.toLp 2 (block 1 θ 0 (embed T θ x) (Fin.last (n + 3)))) cQ)
    (hQ0 : Bd B κ ε (fun θ x => WithLp.toLp 2 (block 0 θ 0 (embed T θ x) (Fin.last (n + 3)))) cQ)
    (hd : ∀ j, Bd B κ ε (fun θ x => WithLp.toLp 2 (block 1 θ 0 (embed T θ x) j) -
      WithLp.toLp 2 (block 0 θ 0 (embed T θ x) j)) d)
    (hβ : a * ε * cQ * (a * ε * cK) ≤ 1 / 8) :
    Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))))
      (cQ + nH * (2 * (a * ε * cK))) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))))
      (cQ + nH * (2 * (a * ε * cK))) ∧
    Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) -
        WithLp.toLp 2 (attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))))
      (d + nH * (8 * (a * ε * cQ * (a * ε * cK)) * (a * ε * cK) + a * ε * d)) := by
  obtain ⟨h1, h0, hΔ⟩ := bd_attn (X1 := fun θ x => block 1 θ 0 (embed T θ x))
    (X0 := fun θ x => block 0 θ 0 (embed T θ x)) hκ hκ1 hε 1 (Fin.last (n + 3))
    (hC.wQ 1) (hC.wK 1) (hC.wV 1) hQ1 hK1 hQ0 hK0 hd hβ
  refine ⟨h1, h0.mono ?_, hΔ⟩
  have hn : (0 : ℝ) ≤ nH := Nat.cast_nonneg _
  have : 0 ≤ a * ε * cK := (Bd.app hκ hκ1 hε.le (hC.wIn 1) (hK1 0)).nonneg hκ hε.le
  nlinarith

/-- INTERNAL: a frozen ReLU pattern from a cost bound at weight `1`. -/
lemma pattern_of_bd (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb a cb c : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hε : 0 < ε) (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (hC : Costs T B 1 ε a cb) (l : Fin 2) {Y : Params Sym K A → (Fin n → Sym) → Fin A.dres → ℝ}
    (hY : Bd B 1 ε (fun θ x => WithLp.toLp 2 (Y θ x)) c) (hsmall : a * ε * c < 3 * ε / 4) :
    ∀ θ ∈ B, ∀ x m, max ((θ.wIn l *ᵥ Y θ x) m + θ.bIn l m) 0 =
      (if 0 < θt.bIn l m then (θ.wIn l *ᵥ Y θ x) m + θ.bIn l m else 0) := fun θ hθ x m =>
  relu_pattern hI hε.le (hB θ hθ) l _ (norm_wIn_apply_lt T hC hε l hY hsmall hθ x) m

lemma vec_logits_sub (T : Task K Sym) (θ : Params Sym K A) (x : Fin n → Sym) :
    WithLp.toLp 2 (logits T A 1 θ x - logits T A 0 θ x) = toE θ.readout
      (WithLp.toLp 2 (block 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) -
        WithLp.toLp 2 (block 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3)))) := by
  rw [map_sub, toE_toLp, toE_toLp, ← WithLp.toLp_sub]
  rfl

/-- INTERNAL: block 2 at the answer position, then the readout: the score-dependent part of
the answer logits. -/
lemma chainD (T : Task K Sym) {a cb c1 c0 D : ℝ} (hC : Costs T B κ ε a cb) (hκ : 0 < κ)
    (hκ1 : κ ≤ 1) (hε : 0 < ε) (pat : Fin A.dmlp → Prop) [DecidablePred pat]
    (hp1 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn 1 *ᵥ
      attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m) 0 =
      (if pat m then (θ.wIn 1 *ᵥ
        attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m else 0))
    (hp0 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn 1 *ᵥ
      attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m) 0 =
      (if pat m then (θ.wIn 1 *ᵥ
        attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m else 0))
    (h1 : Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3)))) c1)
    (h0 : Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3)))) c0)
    (hΔ : Bd B κ ε (fun θ x => WithLp.toLp 2
        (attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) -
        WithLp.toLp 2 (attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3)))) D) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (logits T A 1 θ x - logits T A 0 θ x))
      (a * ε * (D + a * ε * (a * ε * D))) := by
  obtain ⟨-, -, gΔ⟩ := bd_mlp hκ hκ1 hε 1 pat (hC.wIn 1) (hC.wOut 1) (hC.bias 1) hp1 hp0 h1 h0 hΔ
  have := Bd.app hκ hκ1 hε.le hC.readout gΔ
  exact this.congr fun θ _ x => by rw [vec_logits_sub]; rfl

/-- INTERNAL: on the ball, every MLP pre-activation stays strictly inside `|·| < 3ε/4` around its
bias, in both blocks (block 2 at the answer position), at `ζ = 1` and `ζ = 0`
(Paper/appendix.tex:338-340). -/
lemma small_preacts (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hε : 0 < ε)
    (hεc : ε ≤ 1 / (50 * ((nH : ℝ) + 1) ^ 2 * (|a0| + 2) ^ 3)) (hNb : ε * Nb ≤ 1)
    (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4) :
    ∀ θ ∈ B, ∀ x,
      (∀ i, ‖toE (θ.wIn 0) (WithLp.toLp 2 (attnOut 1 θ 0 (embed T θ x) i))‖ < 3 * ε / 4 ∧
        ‖toE (θ.wIn 0) (WithLp.toLp 2 (attnOut 0 θ 0 (embed T θ x) i))‖ < 3 * ε / 4) ∧
      ‖toE (θ.wIn 1) (WithLp.toLp 2
        (attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))))‖ < 3 * ε / 4 ∧
      ‖toE (θ.wIn 1) (WithLp.toLp 2
        (attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))))‖ < 3 * ε / 4 := by
  obtain ⟨a, ha_def⟩ : ∃ a : ℝ, a = |a0| + 2 := ⟨_, rfl⟩
  obtain ⟨H, hH_def⟩ : ∃ H : ℝ, H = (nH : ℝ) + 1 := ⟨_, rfl⟩
  rw [← ha_def, ← hH_def] at hεc
  have ha2 : 2 ≤ a := by rw [ha_def]; linarith [abs_nonneg a0]
  have hn : (0 : ℝ) ≤ nH := Nat.cast_nonneg _
  have hH1 : 1 ≤ H := by rw [hH_def]; linarith
  have hnH : (nH : ℝ) ≤ H := by rw [hH_def]; linarith
  -- elementary smallness
  have hden : 0 < 50 * H ^ 2 * a ^ 3 := by positivity
  have hεa : 50 * (H ^ 2 * a ^ 3) * ε ≤ 1 := by
    rw [le_div_iff₀ hden] at hεc; linarith
  have ha2' : 4 ≤ a ^ 2 := by nlinarith
  have hHH : H ≤ H ^ 2 := by nlinarith
  have hP3 : a ^ 2 ≤ H ^ 2 * a ^ 3 := by
    have h1 : a ^ 2 ≤ a ^ 3 := by nlinarith
    have h2 : a ^ 3 ≤ H ^ 2 * a ^ 3 := le_mul_of_one_le_left (by positivity) (by nlinarith)
    linarith
  have hP2 : 4 * H * a ≤ H ^ 2 * a ^ 3 := by
    have h1 : 4 * a ≤ a ^ 3 := by nlinarith
    calc 4 * H * a = H * (4 * a) := by ring
      _ ≤ H ^ 2 * a ^ 3 := mul_le_mul hHH h1 (by positivity) (by positivity)
  have hP1 : 4 * a ≤ H ^ 2 * a ^ 3 := by nlinarith
  have hw : a * ε ≤ 1 / 200 := by nlinarith
  have hε1 : ε ≤ 1 := by nlinarith
  have hw0 : 0 ≤ a * ε := by positivity
  have hnw : (nH : ℝ) * (a * ε) ≤ 1 / 200 := by
    have : (nH : ℝ) * (a * ε) ≤ H * a * ε := by
      rw [← mul_assoc]; exact mul_le_mul_of_nonneg_right (by nlinarith) hε.le
    nlinarith
  have haw : a * (a * ε) ≤ 1 / 50 := by nlinarith
  have hβ : (a * ε * (2 * a * ε * (1 : ℝ)⁻¹)) ^ 2 ≤ 1 / 8 := by
    rw [inv_one, mul_one]
    have e : a * ε * (2 * a * ε) = 2 * (a * ε) ^ 2 := by ring
    have h1 : (a * ε) ^ 2 ≤ (1 / 200) ^ 2 := pow_le_pow_left₀ hw0 hw 2
    rw [e]
    have h2 : 0 ≤ 2 * (a * ε) ^ 2 := by positivity
    have h3 : 2 * (a * ε) ^ 2 ≤ 1 / 4 := by nlinarith
    nlinarith
  have hsm1 : a * ε * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2) < 3 * ε / 4 := by
    have e : a * ε * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2) =
        ε * (2 * (a * (a * ε)) + 4 * ((nH : ℝ) * (a * ε)) * (a * (a * ε))) := by ring
    rw [e]
    have h1 : 4 * ((nH : ℝ) * (a * ε)) * (a * (a * ε)) ≤ 4 * (1 / 200) * (1 / 50) := by
      have := mul_le_mul hnw haw (by positivity) (by norm_num)
      linarith
    have h2 : 2 * (a * (a * ε)) + 4 * ((nH : ℝ) * (a * ε)) * (a * (a * ε)) < 3 / 4 := by
      linarith
    nlinarith
  obtain ⟨c1, hc1⟩ : ∃ c1 : ℝ, c1 = 7 * (a * ε) + 8 * nH * (a * ε) ^ 2 := ⟨_, rfl⟩
  have hc10 : 0 ≤ c1 := by rw [hc1]; positivity
  have hwc1 : a * ε * c1 ≤ 1 / 20 := by
    have e : a * ε * c1 = 7 * (a * ε) ^ 2 + 8 * ((nH : ℝ) * (a * ε)) * (a * ε) ^ 2 := by
      rw [hc1]; ring
    have h1 : (a * ε) ^ 2 ≤ (1 / 200) ^ 2 := pow_le_pow_left₀ hw0 hw 2
    have h2 : 8 * ((nH : ℝ) * (a * ε)) * (a * ε) ^ 2 ≤ 8 * (1 / 200) * (1 / 200) ^ 2 := by
      have := mul_le_mul hnw h1 (by positivity) (by norm_num)
      linarith
    rw [e]; linarith
  have hwc10 : 0 ≤ a * ε * c1 := by positivity
  have hβ2 : a * ε * c1 * (a * ε * c1) ≤ 1 / 8 := by
    calc a * ε * c1 * (a * ε * c1) ≤ (1 / 20) * (1 / 20) :=
          mul_le_mul hwc1 hwc1 hwc10 (by norm_num)
      _ ≤ 1 / 8 := by norm_num
  have hac1 : a * c1 ≤ 1 / 5 := by
    have e : a * c1 = 7 * (a * (a * ε)) + 8 * ((nH : ℝ) * (a * ε)) * (a * (a * ε)) := by
      rw [hc1]; ring
    have h1 : 8 * ((nH : ℝ) * (a * ε)) * (a * (a * ε)) ≤ 8 * (1 / 200) * (1 / 50) := by
      have := mul_le_mul hnw haw (by positivity) (by norm_num)
      linarith
    rw [e]; linarith
  have hs2 : a * ε * (c1 + nH * (2 * (a * ε * c1))) < 3 * ε / 4 := by
    have h1 : (nH : ℝ) * (2 * (a * ε * c1)) ≤ c1 := by
      have : (nH : ℝ) * (a * ε) * c1 ≤ 1 / 200 * c1 := mul_le_mul_of_nonneg_right hnw hc10
      have e : (nH : ℝ) * (2 * (a * ε * c1)) = 2 * ((nH : ℝ) * (a * ε) * c1) := by ring
      rw [e]; linarith
    have h2 : a * ε * (c1 + nH * (2 * (a * ε * c1))) ≤ a * ε * (2 * c1) := by
      apply mul_le_mul_of_nonneg_left _ hw0; linarith
    calc a * ε * (c1 + nH * (2 * (a * ε * c1))) ≤ a * ε * (2 * c1) := h2
      _ = ε * (2 * (a * c1)) := by ring
      _ ≤ ε * (2 * (1 / 5)) := by gcongr
      _ < 3 * ε / 4 := by linarith
  have hcard : ∀ i : Fin (n + 4), a * ε * (1 : ℝ)⁻¹ ^ (letSet i).card + a * ε +
      4 * nH * a ^ 2 * ε ^ 2 * (1 : ℝ)⁻¹ = 2 * (a * ε) + 4 * nH * (a * ε) ^ 2 := by
    intro i; simp only [inv_one, one_pow, mul_one]; ring
  have hs1 : ∀ i : Fin (n + 4), a * ε * (a * ε * (1 : ℝ)⁻¹ ^ (letSet i).card + a * ε +
      4 * nH * a ^ 2 * ε ^ 2 * (1 : ℝ)⁻¹) < 3 * ε / 4 := by
    intro i; rw [hcard]
    have e : a * ε * (2 * (a * ε) + 4 * nH * (a * ε) ^ 2) =
        a * ε * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2) := by ring
    rw [e]; exact hsm1
  have hK : ∀ j : Fin (n + 4), (2 * (a * ε * (1 : ℝ)⁻¹ ^ (letSet j).card + a * ε +
      4 * nH * a ^ 2 * ε ^ 2 * (1 : ℝ)⁻¹) + a * ε * 3) = c1 := by
    intro j; rw [hcard, hc1]; ring
  -- the network at weight 1
  have hC := costs_of_init T hI one_pos le_rfl hε hε1 hNb hB
  rw [← ha_def] at hC
  have hpat := pattern1 T hI hε hB hC (by linarith) (by simpa using hβ) hsm1
  have hA := fun i => chainA T hC one_pos le_rfl hε (by linarith) hβ i
  have hB1 := fun i => chainB T hC one_pos le_rfl hε (by linarith) (by linarith) hβ
    (fun m => 0 < θt.bIn 0 m) hpat i
  have hK1 := fun j : Fin (n + 4) => (hB1 j).1.mono (le_of_eq (hK j))
  have hK0 := fun j : Fin (n + 4) => (hB1 j).2.1.mono (le_of_eq (hK j))
  have hd := fun j : Fin (n + 4) => (hB1 j).2.2
  obtain ⟨g1, g0, -⟩ := chainC T hC one_pos le_rfl hε hK1 hK0 (hK1 _) (hK0 _) hd hβ2
  intro θ hθ x
  exact ⟨fun i => ⟨norm_wIn_apply_lt T hC hε 0 (hA i).1 (hs1 i) hθ x,
    norm_wIn_apply_lt T hC hε 0 (hA i).2.1 (hs1 i) hθ x⟩,
    norm_wIn_apply_lt T hC hε 1 g1 hs2 hθ x, norm_wIn_apply_lt T hC hε 1 g0 hs2 hθ x⟩

/-- INTERNAL: the constant bookkeeping at the end of `bd_logits_sub`. -/
lemma final_arith {a ε H nn S σ X Y d : ℝ} (hw0 : 0 ≤ a * ε) (hw1 : a * ε ≤ 1) (hn : 0 ≤ nn)
    (hnH : nn ≤ H) (hH1 : 1 ≤ H) (hS0 : 0 ≤ S) (hS_def : S = H * a ^ 3 * σ) (ha : 0 ≤ a)
    (hε : 0 ≤ ε) (hσ0 : 0 ≤ σ) (hX0 : 0 ≤ X) (hX : X ≤ 15 * S) (hY0 : 0 ≤ Y) (hY : Y ≤ 15 * S)
    (hd0 : 0 ≤ d) (hdS : d ≤ 128 * S ^ 3) :
    a * ε * ((d + nn * (8 * (X * Y) * Y + a * ε * d)) +
      a * ε * (a * ε * (d + nn * (8 * (X * Y) * Y + a * ε * d)))) ≤
      60000 * H ^ 4 * a ^ 10 * ε * σ ^ 3 := by
  have hS3 : 0 ≤ S ^ 3 := by positivity
  have hXYY : 8 * (X * Y) * Y ≤ 27000 * S ^ 3 := by
    have h1 : X * Y ≤ 15 * S * (15 * S) := mul_le_mul hX hY hY0 (by positivity)
    have h2 : X * Y * Y ≤ 15 * S * (15 * S) * (15 * S) :=
      mul_le_mul h1 hY hY0 (by positivity)
    have e : 15 * S * (15 * S) * (15 * S) = 3375 * S ^ 3 := by ring
    linarith
  obtain ⟨D, hD⟩ : ∃ D : ℝ, D = d + nn * (8 * (X * Y) * Y + a * ε * d) := ⟨_, rfl⟩
  rw [← hD]
  have hDle : D ≤ 27256 * H * S ^ 3 := by
    have h1 : a * ε * d ≤ d := by
      have := mul_le_mul_of_nonneg_right hw1 hd0; linarith
    have h2 : nn * (8 * (X * Y) * Y + a * ε * d) ≤ H * (27000 * S ^ 3 + 128 * S ^ 3) := by
      apply mul_le_mul hnH _ (by positivity) (by linarith)
      linarith
    have h3 : 128 * S ^ 3 ≤ 128 * H * S ^ 3 := by
      have := mul_le_mul_of_nonneg_right hH1 hS3; linarith
    rw [hD]; linarith
  have hD0 : 0 ≤ D := by rw [hD]; positivity
  have hfinal : a * ε * (D + a * ε * (a * ε * D)) ≤ 2 * (a * ε) * D := by
    have h1 : a * ε * (a * ε) ≤ 1 := by
      have := mul_le_mul hw1 hw1 hw0 zero_le_one; linarith
    have : a * ε * (a * ε * D) ≤ D := by
      have := mul_le_mul_of_nonneg_right h1 hD0
      calc a * ε * (a * ε * D) = a * ε * (a * ε) * D := by ring
        _ ≤ 1 * D := this
        _ = D := one_mul D
    have : a * ε * (D + a * ε * (a * ε * D)) ≤ a * ε * (D + D) :=
      mul_le_mul_of_nonneg_left (by linarith) hw0
    linarith
  have e : 2 * (a * ε) * (27256 * H * S ^ 3) = 54512 * (H ^ 4 * a ^ 10 * ε * σ ^ 3) := by
    rw [hS_def]; ring
  have hpos : 0 ≤ H ^ 4 * a ^ 10 * ε * σ ^ 3 := by positivity
  calc a * ε * (D + a * ε * (a * ε * D)) ≤ 2 * (a * ε) * D := hfinal
    _ ≤ 2 * (a * ε) * (27256 * H * S ^ 3) := mul_le_mul_of_nonneg_left hDle (by positivity)
    _ = 54512 * (H ^ 4 * a ^ 10 * ε * σ ^ 3) := e
    _ ≤ 60000 * H ^ 4 * a ^ 10 * ε * σ ^ 3 := by linarith

set_option maxHeartbeats 1000000 in
/-- INTERNAL: **the score-dependent part of the answer logits** costs
`60000 (nH + 1)⁴ a¹⁰ ε (ε²/κ)³` at weight `κ` and scale `ε` on the doubled ball, with
`a = |a0| + 2`. This is the content of `prop:score-degree` (ii)–(iii) summed as in
`lem:score-sum` (Paper/appendix.tex:308-319, 505-521): one score costs `ε⁴/κ²` and the
leading term reads three letters at size `ε⁷`. -/
theorem bd_logits_sub (T : Task K Sym) {θt : Params Sym K A} {a0 cemb Nb : ℝ}
    (hI : InitHyp θt a0 cemb Nb) (hε : 0 < ε) (hκ : 0 < κ) (hκ1 : κ ≤ 1)
    (hεc : ε ≤ 1 / (50 * ((nH : ℝ) + 1) ^ 2 * (|a0| + 2) ^ 3)) (hNb : ε * Nb ≤ 1)
    (hB : ∀ θ ∈ B, ‖θ - ε • θt‖ ≤ ε / 4)
    (hσ : ε ^ 2 * κ⁻¹ ≤ 1 / (64 * ((nH : ℝ) + 1) * (|a0| + 2) ^ 3)) :
    Bd B κ ε (fun θ x => WithLp.toLp 2 (logits T A 1 θ x - logits T A 0 θ x))
      (60000 * ((nH : ℝ) + 1) ^ 4 * (|a0| + 2) ^ 10 * ε * (ε ^ 2 * κ⁻¹) ^ 3) := by
  have hpre := small_preacts T hI hε hεc hNb hB
  obtain ⟨a, ha_def⟩ : ∃ a : ℝ, a = |a0| + 2 := ⟨_, rfl⟩
  obtain ⟨H, hH_def⟩ : ∃ H : ℝ, H = (nH : ℝ) + 1 := ⟨_, rfl⟩
  obtain ⟨q, hq_def⟩ : ∃ q : ℝ, q = κ⁻¹ := ⟨_, rfl⟩
  rw [← ha_def, ← hH_def] at hεc hσ ⊢
  rw [← hq_def] at hσ ⊢
  have ha2 : 2 ≤ a := by rw [ha_def]; linarith [abs_nonneg a0]
  have hn : (0 : ℝ) ≤ nH := Nat.cast_nonneg _
  have hH1 : 1 ≤ H := by rw [hH_def]; linarith
  have hnH : (nH : ℝ) ≤ H := by rw [hH_def]; linarith
  have hq : 1 ≤ q := by rw [hq_def]; exact (one_le_inv₀ hκ).2 hκ1
  -- elementary smallness
  have hden : 0 < 50 * H ^ 2 * a ^ 3 := by positivity
  have hεa : 50 * (H ^ 2 * a ^ 3) * ε ≤ 1 := by
    rw [le_div_iff₀ hden] at hεc; linarith
  have ha2' : 4 ≤ a ^ 2 := by nlinarith
  have hHH : H ≤ H ^ 2 := by nlinarith
  have hP1 : 4 * a ≤ H ^ 2 * a ^ 3 := by
    have h1 : 4 * a ≤ a ^ 3 := by nlinarith
    have h2 : a ^ 3 ≤ H ^ 2 * a ^ 3 := le_mul_of_one_le_left (by positivity) (by nlinarith)
    linarith
  have hw : a * ε ≤ 1 / 200 := by nlinarith
  have hε1 : ε ≤ 1 := by nlinarith
  have hw0 : 0 ≤ a * ε := by positivity
  have hw1 : a * ε ≤ 1 := by linarith
  obtain ⟨σ, hσ_def⟩ : ∃ σ : ℝ, σ = ε ^ 2 * q := ⟨_, rfl⟩
  rw [← hσ_def] at hσ ⊢
  have hσ0 : 0 ≤ σ := by rw [hσ_def]; positivity
  have hε2σ : ε ^ 2 ≤ σ := by rw [hσ_def]; nlinarith [sq_nonneg ε]
  have hε3σ : ε ^ 3 * q ≤ σ := by
    rw [hσ_def]
    have : ε ^ 3 ≤ ε ^ 2 := pow_le_pow_of_le_one hε.le hε1 (by norm_num)
    exact mul_le_mul_of_nonneg_right this (by linarith)
  obtain ⟨S, hS_def⟩ : ∃ S : ℝ, S = H * a ^ 3 * σ := ⟨_, rfl⟩
  have hS0 : 0 ≤ S := by rw [hS_def]; positivity
  have hS : S ≤ 1 / 64 := by
    rw [hS_def]
    have hd : 0 < 64 * H * a ^ 3 := by positivity
    rw [le_div_iff₀ hd] at hσ
    linarith
  -- block 1
  have hβ : (a * ε * (2 * a * ε * κ⁻¹)) ^ 2 ≤ 1 / 8 := by
    rw [← hq_def]
    have e : a * ε * (2 * a * ε * q) = 2 * a ^ 2 * σ := by rw [hσ_def]; ring
    rw [e]
    have h1 : 2 * a ^ 2 * σ ≤ 1 / 16 := by
      have : 2 * a ^ 2 * σ ≤ S := by
        rw [hS_def]
        have h1 : 2 * a ^ 2 ≤ a ^ 3 := by
          have : a ^ 3 = a * a ^ 2 := by ring
          rw [this]; exact mul_le_mul_of_nonneg_right ha2 (by positivity)
        have h2 : a ^ 3 ≤ H * a ^ 3 := le_mul_of_one_le_left (by positivity) hH1
        exact mul_le_mul_of_nonneg_right (h1.trans h2) hσ0
      linarith
    have h0 : 0 ≤ 2 * a ^ 2 * σ := by positivity
    nlinarith
  -- block 2 inputs
  obtain ⟨cK, hcK⟩ : ∃ cK : ℝ, cK = 2 * (2 * a * ε * q + 4 * nH * a ^ 2 * ε ^ 2 * q) + a * ε * 3 :=
    ⟨_, rfl⟩
  obtain ⟨cQ, hcQ⟩ : ∃ cQ : ℝ, cQ = 2 * (2 * a * ε + 4 * nH * a ^ 2 * ε ^ 2 * q) + a * ε * 3 :=
    ⟨_, rfl⟩
  -- the two products `X = aε cQ`, `Y = aε cK` are at most `15 S`
  have ha32 : 2 * a ^ 2 ≤ a ^ 3 := by
    have : a ^ 3 = a * a ^ 2 := by ring
    rw [this]; exact mul_le_mul_of_nonneg_right ha2 (by positivity)
  have hnHa3 : (nH : ℝ) * a ^ 3 ≤ H * a ^ 3 := mul_le_mul_of_nonneg_right hnH (by positivity)
  have hHa3 : a ^ 3 ≤ H * a ^ 3 := le_mul_of_one_le_left (by positivity) hH1
  have hX : a * ε * cQ ≤ 15 * S := by
    have e : a * ε * cQ = 7 * a ^ 2 * ε ^ 2 + 8 * ((nH : ℝ) * a ^ 3) * (ε ^ 3 * q) := by
      rw [hcQ]; ring
    rw [e, hS_def]
    have h1 : 7 * a ^ 2 * ε ^ 2 ≤ 7 * a ^ 2 * σ := by
      have := mul_le_mul_of_nonneg_left hε2σ (by positivity : (0 : ℝ) ≤ 7 * a ^ 2); linarith
    have h2 : 8 * ((nH : ℝ) * a ^ 3) * (ε ^ 3 * q) ≤ 8 * (H * a ^ 3) * σ := by
      have := mul_le_mul hnHa3 hε3σ (by positivity) (by positivity); linarith
    have h3 : (2 * a ^ 2) * σ ≤ (H * a ^ 3) * σ :=
      mul_le_mul_of_nonneg_right (ha32.trans hHa3) hσ0
    have h4 : 0 ≤ H * a ^ 3 * σ := by positivity
    linarith
  have hY : a * ε * cK ≤ 15 * S := by
    have e : a * ε * cK = 4 * a ^ 2 * (ε ^ 2 * q) + 8 * ((nH : ℝ) * a ^ 3) * (ε ^ 3 * q) +
        3 * a ^ 2 * ε ^ 2 := by
      rw [hcK]; ring
    rw [e, hS_def, ← hσ_def]
    have h1 : 3 * a ^ 2 * ε ^ 2 ≤ 3 * a ^ 2 * σ := by
      have := mul_le_mul_of_nonneg_left hε2σ (by positivity : (0 : ℝ) ≤ 3 * a ^ 2); linarith
    have h2 : 8 * ((nH : ℝ) * a ^ 3) * (ε ^ 3 * q) ≤ 8 * (H * a ^ 3) * σ := by
      have := mul_le_mul hnHa3 hε3σ (by positivity) (by positivity); linarith
    have h3 : (2 * a ^ 2) * σ ≤ (H * a ^ 3) * σ :=
      mul_le_mul_of_nonneg_right (ha32.trans hHa3) hσ0
    have h4 : 0 ≤ H * a ^ 3 * σ := by positivity
    linarith
  have hX0 : 0 ≤ a * ε * cQ := by rw [hcQ]; positivity
  have hY0 : 0 ≤ a * ε * cK := by rw [hcK]; positivity
  have hβ2 : a * ε * cQ * (a * ε * cK) ≤ 1 / 8 := by
    calc a * ε * cQ * (a * ε * cK) ≤ (15 * S) * (15 * S) := mul_le_mul hX hY hY0 (by positivity)
      _ ≤ (15 * (1 / 64)) * (15 * (1 / 64)) := by gcongr
      _ ≤ 1 / 8 := by norm_num
  have hdS : 128 * nH * a ^ 6 * ε ^ 6 * q ^ 3 ≤ 128 * S ^ 3 := by
    have e : 128 * nH * a ^ 6 * ε ^ 6 * q ^ 3 = 128 * ((nH : ℝ) * a ^ 6) * σ ^ 3 := by
      rw [hσ_def]; ring
    have e2 : S ^ 3 = H ^ 3 * a ^ 9 * σ ^ 3 := by rw [hS_def]; ring
    rw [e, e2]
    have h1 : (nH : ℝ) * a ^ 6 ≤ H ^ 3 * a ^ 9 := by
      have hH3 : H ≤ H ^ 3 := by
        have : H ^ 3 = H * H ^ 2 := by ring
        rw [this]; exact le_mul_of_one_le_right (by linarith) (by nlinarith)
      have : (nH : ℝ) ≤ H ^ 3 := hnH.trans hH3
      have : a ^ 6 ≤ a ^ 9 := pow_le_pow_right₀ (by linarith) (by norm_num)
      exact mul_le_mul ‹_› this (by positivity) (by positivity)
    have := mul_le_mul_of_nonneg_right h1 (by positivity : (0 : ℝ) ≤ σ ^ 3)
    linarith
  have hC := costs_of_init T hI hκ hκ1 hε hε1 hNb hB
  rw [← ha_def] at hC
  have hpat1 : ∀ θ ∈ B, ∀ x i m,
      max ((θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if 0 < θt.bIn 0 m then (θ.wIn 0 *ᵥ attnOut 1 θ 0 (embed T θ x) i) m + θ.bIn 0 m
          else 0) ∧
      max ((θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m) 0 =
        (if 0 < θt.bIn 0 m then (θ.wIn 0 *ᵥ attnOut 0 θ 0 (embed T θ x) i) m + θ.bIn 0 m
          else 0) := fun θ hθ x i m =>
    ⟨relu_pattern hI hε.le (hB θ hθ) 0 _ ((hpre θ hθ x).1 i).1 m,
      relu_pattern hI hε.le (hB θ hθ) 0 _ ((hpre θ hθ x).1 i).2 m⟩
  have hB1 := fun i => chainB T hC hκ hκ1 hε (by linarith) hw1 hβ
    (fun m => 0 < θt.bIn 0 m) hpat1 i
  have hKle : ∀ j : Fin (n + 4), 2 * (a * ε * κ⁻¹ ^ (letSet j).card + a * ε +
      4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) + a * ε * 3 ≤ cK := by
    intro j
    have h1 := inv_pow_card_le hκ hκ1 j
    rw [← hq_def] at h1 ⊢
    rw [hcK]
    have : a * ε * q ^ (letSet j).card ≤ a * ε * q := mul_le_mul_of_nonneg_left h1 hw0
    have : a * ε ≤ a * ε * q := le_mul_of_one_le_right hw0 hq
    linarith
  have hQeq : 2 * (a * ε * κ⁻¹ ^ (letSet (n := n) (Fin.last (n + 3))).card + a * ε +
      4 * nH * a ^ 2 * ε ^ 2 * κ⁻¹) + a * ε * 3 = cQ := by
    rw [card_letSet_last, pow_zero, ← hq_def, hcQ]; ring
  have hK1 := fun j : Fin (n + 4) => (hB1 j).1.mono (hKle j)
  have hK0 := fun j : Fin (n + 4) => (hB1 j).2.1.mono (hKle j)
  have hQ1 := (hB1 (Fin.last (n + 3))).1.mono (le_of_eq hQeq)
  have hQ0 := (hB1 (Fin.last (n + 3))).2.1.mono (le_of_eq hQeq)
  have hd := fun j : Fin (n + 4) => (hB1 j).2.2
  obtain ⟨g1, g0, gΔ⟩ := chainC T hC hκ hκ1 hε hK1 hK0 hQ1 hQ0 hd hβ2
  have hp2 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn 1 *ᵥ
      attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m) 0 =
      (if 0 < θt.bIn 1 m then (θ.wIn 1 *ᵥ
        attnOut 1 θ 1 (block 1 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m else 0) :=
    fun θ hθ x m => relu_pattern hI hε.le (hB θ hθ) 1 _ (hpre θ hθ x).2.1 m
  have hp0 : ∀ θ ∈ B, ∀ x m, max ((θ.wIn 1 *ᵥ
      attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m) 0 =
      (if 0 < θt.bIn 1 m then (θ.wIn 1 *ᵥ
        attnOut 0 θ 1 (block 0 θ 0 (embed T θ x)) (Fin.last (n + 3))) m + θ.bIn 1 m else 0) :=
    fun θ hθ x m => relu_pattern hI hε.le (hB θ hθ) 1 _ (hpre θ hθ x).2.2 m
  have hfin := chainD T hC hκ hκ1 hε (fun m => 0 < θt.bIn 1 m) hp2 hp0 g1 g0 gΔ
  refine hfin.mono ?_
  have hd0 : 0 ≤ 128 * nH * a ^ 6 * ε ^ 6 * κ⁻¹ ^ 3 := by positivity
  rw [← hq_def] at hd0 ⊢
  exact final_arith hw0 hw1 hn hnH hH1 hS0 hS_def (by linarith) hε.le hσ0 hX0 hX hY0 hY hd0 hdS

end Chain

end Trace.Proof.SN
