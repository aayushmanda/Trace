import Trace.Model.Prelude

/-!
# Parameter blocks are bounded by the parameter norm

Every weight matrix of `θ` is a block of coordinates of `θ`, so its `ℓ²` operator norm is at
most its Frobenius norm, which is at most the Euclidean norm `‖θ‖` of the whole parameter
vector (rem:norms, Paper/appendix.tex:76-81). The same holds for every bias vector and every
embedding column. Each block is linear in `θ`, which turns these into Lipschitz bounds.

`mat M` is the matrix `M` as a continuous linear map of Euclidean spaces; `‖mat M‖` is
`opNorm M` by definition.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.ParamNorm

open Matrix

variable {m k ι : Type*} [Fintype m] [Fintype k] [DecidableEq k] [Fintype ι]

/-- INTERNAL: a matrix as a continuous linear map between Euclidean spaces.
TEXLINE: Paper/appendix.tex:76-81 -/
noncomputable def mat (M : Matrix m k ℝ) : EuclideanSpace ℝ k →L[ℝ] EuclideanSpace ℝ m :=
  LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin M)

/-- INTERNAL: `‖mat M‖ = opNorm M`. -/
theorem norm_mat (M : Matrix m k ℝ) : ‖mat M‖ = opNorm M := rfl

/-- INTERNAL: `mat M` acts by `mulVec`. -/
theorem mat_toLp (M : Matrix m k ℝ) (v : k → ℝ) :
    mat M (WithLp.toLp 2 v) = WithLp.toLp 2 (M *ᵥ v) := rfl

/-- INTERNAL: `mat` is additive. -/
theorem mat_add (M N : Matrix m k ℝ) : mat (M + N) = mat M + mat N := by
  ext1 v; simp [mat]

/-- INTERNAL: `mat` is subtractive. -/
theorem mat_sub (M N : Matrix m k ℝ) : mat (M - N) = mat M - mat N := by
  ext1 v; simp [mat]

/-- INTERNAL: `mat` is homogeneous. -/
theorem mat_smul (c : ℝ) (M : Matrix m k ℝ) : mat (c • M) = c • mat M := by
  ext1 v; simp [mat]

/-- INTERNAL: `mat` turns products into compositions. -/
theorem mat_mul {l : Type*} [Fintype l] [DecidableEq l] [DecidableEq m] (M : Matrix m k ℝ)
    (N : Matrix k l ℝ) : mat (M * N) = (mat M).comp (mat N) := by
  ext1 v
  show WithLp.toLp 2 ((M * N) *ᵥ v.ofLp) = WithLp.toLp 2 (M *ᵥ (N *ᵥ v.ofLp))
  rw [Matrix.mulVec_mulVec]

/-- INTERNAL: triangle inequality for `opNorm`. -/
theorem opNorm_add_le (M N : Matrix m k ℝ) : opNorm (M + N) ≤ opNorm M + opNorm N := by
  rw [← norm_mat, ← norm_mat, ← norm_mat, mat_add]; exact norm_add_le _ _

/-- INTERNAL: homogeneity of `opNorm`. -/
theorem opNorm_smul (c : ℝ) (M : Matrix m k ℝ) : opNorm (c • M) = |c| * opNorm M := by
  rw [← norm_mat, ← norm_mat, mat_smul, norm_smul, Real.norm_eq_abs]

/-- INTERNAL: `opNorm` is nonnegative. -/
theorem opNorm_nonneg (M : Matrix m k ℝ) : 0 ≤ opNorm M := norm_nonneg (mat M)

/-- INTERNAL: a sum of squares over an injectively embedded index set is at most the full one. -/
theorem sum_sq_comp_le {κ : Type*} [Fintype κ] (e : κ → ι) (he : Function.Injective e)
    (f : ι → ℝ) : ∑ x, f (e x) ^ 2 ≤ ∑ i, f i ^ 2 := by
  classical
  rw [← Finset.sum_image (f := fun i => f i ^ 2) (fun x _ y _ h => he h)]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    (fun i _ _ => sq_nonneg _)

/-- INTERNAL: a matrix whose entries are distinct coordinates of `θ` has operator norm at most
`‖θ‖` (operator norm ≤ Frobenius norm ≤ parameter norm). TEXLINE: Paper/appendix.tex:76-81 -/
theorem opNorm_entries_le (θ : EuclideanSpace ℝ ι) (e : m × k → ι)
    (he : Function.Injective e) : opNorm (Matrix.of fun a b => θ (e (a, b))) ≤ ‖θ‖ := by
  rw [← norm_mat]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun v => ?_
  refine le_of_sq_le_sq ?_ (by positivity)
  have hv : mat (Matrix.of fun a b => θ (e (a, b))) v =
      WithLp.toLp 2 ((Matrix.of fun a b => θ (e (a, b))) *ᵥ v.ofLp) := rfl
  rw [hv, EuclideanSpace.real_norm_sq_eq, mul_pow, EuclideanSpace.real_norm_sq_eq,
    EuclideanSpace.real_norm_sq_eq]
  calc ∑ a, ((Matrix.of fun a b => θ (e (a, b))) *ᵥ v.ofLp) a ^ 2
      ≤ ∑ a, (∑ b, θ (e (a, b)) ^ 2) * ∑ b, v b ^ 2 := by
        gcongr with a
        exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ = (∑ p : m × k, θ (e p) ^ 2) * ∑ b, v b ^ 2 := by
        rw [← Finset.sum_mul, ← Finset.sum_product']; rfl
    _ ≤ (∑ i, θ i ^ 2) * ∑ b, v b ^ 2 := by
        gcongr
        · exact Finset.sum_nonneg fun _ _ => sq_nonneg _
        · exact sum_sq_comp_le e he _

/-- INTERNAL: a vector whose entries are distinct coordinates of `θ` has norm at most `‖θ‖`. -/
theorem norm_entries_le (θ : EuclideanSpace ℝ ι) (e : k → ι) (he : Function.Injective e) :
    ‖(WithLp.toLp 2 (fun b => θ (e b)) : EuclideanSpace ℝ k)‖ ≤ ‖θ‖ := by
  refine le_of_sq_le_sq ?_ (norm_nonneg _)
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  exact sum_sq_comp_le e he _

/-- INTERNAL: a matrix-valued function of the parameters reading distinct coordinates.
TEXLINE: Paper/appendix.tex:76-81 -/
def IsBlock (F : EuclideanSpace ℝ ι → Matrix m k ℝ) : Prop :=
  ∃ e : m × k → ι, Function.Injective e ∧ ∀ θ a b, F θ a b = θ (e (a, b))

/-- INTERNAL: a vector-valued function of the parameters reading distinct coordinates. -/
def IsVBlock (F : EuclideanSpace ℝ ι → k → ℝ) : Prop :=
  ∃ e : k → ι, Function.Injective e ∧ ∀ θ b, F θ b = θ (e b)

namespace IsBlock

variable {F : EuclideanSpace ℝ ι → Matrix m k ℝ}

/-- INTERNAL. -/
theorem opNorm_le (hF : IsBlock F) (θ : EuclideanSpace ℝ ι) : opNorm (F θ) ≤ ‖θ‖ := by
  obtain ⟨e, he, hFe⟩ := hF
  have : F θ = Matrix.of fun a b => θ (e (a, b)) := by ext a b; exact hFe θ a b
  rw [this]; exact opNorm_entries_le θ e he

/-- INTERNAL. -/
theorem map_sub (hF : IsBlock F) (x y : EuclideanSpace ℝ ι) : F (x - y) = F x - F y := by
  obtain ⟨e, -, hFe⟩ := hF
  ext a b; simp [hFe]

/-- INTERNAL. -/
theorem map_smul (hF : IsBlock F) (c : ℝ) (x : EuclideanSpace ℝ ι) : F (c • x) = c • F x := by
  obtain ⟨e, -, hFe⟩ := hF
  ext a b; simp [hFe]

/-- INTERNAL: `‖mat (F x) − mat (F y)‖ ≤ ‖x − y‖`. -/
theorem lip (hF : IsBlock F) (x y : EuclideanSpace ℝ ι) : ‖mat (F x) - mat (F y)‖ ≤ ‖x - y‖ := by
  rw [← mat_sub, ← hF.map_sub, norm_mat]; exact hF.opNorm_le _

/-- INTERNAL: on a ball around `ε • θt`, `opNorm (F θ) ≤ ε opNorm (F θt) + ‖θ − ε • θt‖`. -/
theorem opNorm_ball_le (hF : IsBlock F) {ε : ℝ} (hε : 0 ≤ ε) (θt θ : EuclideanSpace ℝ ι) :
    opNorm (F θ) ≤ ε * opNorm (F θt) + ‖θ - ε • θt‖ := by
  have h : F θ = ε • F θt + F (θ - ε • θt) := by
    rw [hF.map_sub, hF.map_smul]; abel
  rw [h]
  refine (opNorm_add_le _ _).trans (add_le_add ?_ (hF.opNorm_le _))
  rw [opNorm_smul, abs_of_nonneg hε]

end IsBlock

namespace IsVBlock

variable {F : EuclideanSpace ℝ ι → k → ℝ}

/-- INTERNAL. -/
theorem norm_le (hF : IsVBlock F) (θ : EuclideanSpace ℝ ι) :
    ‖(WithLp.toLp 2 (F θ) : EuclideanSpace ℝ k)‖ ≤ ‖θ‖ := by
  obtain ⟨e, he, hFe⟩ := hF
  have : F θ = fun b => θ (e b) := by ext b; exact hFe θ b
  rw [this]; exact norm_entries_le θ e he

/-- INTERNAL. -/
theorem map_sub (hF : IsVBlock F) (x y : EuclideanSpace ℝ ι) : F (x - y) = F x - F y := by
  obtain ⟨e, -, hFe⟩ := hF
  ext b; simp [hFe]

/-- INTERNAL. -/
theorem map_smul (hF : IsVBlock F) (c : ℝ) (x : EuclideanSpace ℝ ι) : F (c • x) = c • F x := by
  obtain ⟨e, -, hFe⟩ := hF
  ext b; simp [hFe]

/-- INTERNAL: Lipschitz bound. -/
theorem lip (hF : IsVBlock F) (x y : EuclideanSpace ℝ ι) :
    ‖(WithLp.toLp 2 (F x) : EuclideanSpace ℝ k) - WithLp.toLp 2 (F y)‖ ≤ ‖x - y‖ := by
  rw [← WithLp.toLp_sub, ← hF.map_sub]; exact hF.norm_le _

/-- INTERNAL: on a ball around `ε • θt`. -/
theorem norm_ball_le (hF : IsVBlock F) {ε : ℝ} (hε : 0 ≤ ε) (θt θ : EuclideanSpace ℝ ι) :
    ‖(WithLp.toLp 2 (F θ) : EuclideanSpace ℝ k)‖ ≤
      ε * ‖(WithLp.toLp 2 (F θt) : EuclideanSpace ℝ k)‖ + ‖θ - ε • θt‖ := by
  have h : F θ = ε • F θt + F (θ - ε • θt) := by
    rw [hF.map_sub, hF.map_smul]; abel
  rw [h, WithLp.toLp_add, WithLp.toLp_smul]
  refine (norm_add_le _ _).trans (add_le_add ?_ (hF.norm_le _))
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hε]

end IsVBlock

/-- INTERNAL: a column of a matrix has norm at most its operator norm. -/
theorem norm_col_le (M : Matrix m k ℝ) (b : k) :
    ‖(WithLp.toLp 2 (fun a => M a b) : EuclideanSpace ℝ m)‖ ≤ opNorm M := by
  have h : (WithLp.toLp 2 (fun a => M a b) : EuclideanSpace ℝ m) =
      mat M (EuclideanSpace.single b 1) := by
    rw [mat]
    ext a
    simp [Matrix.mulVec, dotProduct, Pi.single_apply]
  rw [h, ← norm_mat]
  refine ((mat M).le_opNorm _).trans ?_
  simp

section Params

variable {Sym : Type u} {K nH n : ℕ} {A : Arch nH n}

/-- INTERNAL. -/
theorem isBlock_tokEmb : IsBlock (fun θ : Params Sym K A => θ.tokEmb) :=
  ⟨fun p => Sum.inl p, fun p q h => by simpa using h, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_posEmb : IsBlock (fun θ : Params Sym K A => θ.posEmb) :=
  ⟨fun p => Sum.inr (Sum.inl p), fun p q h => by simpa using h, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_wQ (l : Fin 2) (h : Fin nH) : IsBlock (fun θ : Params Sym K A => θ.wQ l h) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2))),
    fun p q hpq => by simpa [Prod.ext_iff] using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_wK (l : Fin 2) (h : Fin nH) : IsBlock (fun θ : Params Sym K A => θ.wK l h) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2)))),
    fun p q hpq => by simpa [Prod.ext_iff] using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_wV (l : Fin 2) (h : Fin nH) : IsBlock (fun θ : Params Sym K A => θ.wV l h) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, p.1, p.2))))),
    fun p q hpq => by simpa [Prod.ext_iff] using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_wIn (l : Fin 2) : IsBlock (fun θ : Params Sym K A => θ.wIn l) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, p.1, p.2)))))),
    fun p q hpq => by simpa [Prod.ext_iff] using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_wOut (l : Fin 2) : IsBlock (fun θ : Params Sym K A => θ.wOut l) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl
      (l, p.1, p.2)))))))),
    fun p q hpq => by simpa [Prod.ext_iff] using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isBlock_readout : IsBlock (fun θ : Params Sym K A => θ.readout) :=
  ⟨fun p => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr p))))))),
    fun p q hpq => by simpa using hpq, fun _ _ _ => rfl⟩

/-- INTERNAL. -/
theorem isVBlock_bIn (l : Fin 2) : IsVBlock (fun θ : Params Sym K A => θ.bIn l) :=
  ⟨fun m => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m))))))),
    fun p q hpq => by simpa using hpq, fun _ _ => rfl⟩

/-- INTERNAL. -/
theorem isVBlock_tokCol (v : Token Sym K) :
    IsVBlock (fun θ : Params Sym K A => fun d => θ.tokEmb d v) :=
  ⟨fun d => Sum.inl (d, v), fun p q hpq => by simpa using hpq, fun _ _ => rfl⟩

/-- INTERNAL. -/
theorem isVBlock_posCol (i : Fin (n + 4)) :
    IsVBlock (fun θ : Params Sym K A => fun d => θ.posEmb d i) :=
  ⟨fun d => Sum.inr (Sum.inl (d, i)), fun p q hpq => by simpa using hpq, fun _ _ => rfl⟩

end Params

end Trace.Proof.ParamNorm
