import Trace.Model.Prelude
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# The zero-order network with a frozen ReLU pattern

At `ζ = 0` every attention weight is `1/|vis|`, so a block is
`a_i = x_i + ∑_h |vis|⁻¹ ∑_{j ∈ vis} W_V x_j` followed by the MLP
`a_i + W_out ReLU(W_in a_i + b_in)`. If the sign of every MLP pre-activation is known, the ReLU
is multiplication by a fixed `0/1` vector `D` (the "fixed ReLU pattern on `B_ε`" of
prop:score-degree, Paper/appendix.tex:325-489), and the network becomes a polynomial in `θ`.

* `attnRes`, `mlpPat`, `patLogits`, `zeroPat`: the uniform-attention residual update, the MLP
  with pattern `D`, the resulting answer logits and credit functional.
* `block_zero_eq_mlpPat`: `block 0 = mlpPat D ∘ attnRes` whenever `D` is the sign pattern of the
  pre-activations.
* `differentiable_zeroPat`: the pattern functional is differentiable everywhere.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.ZeroOrder

open Matrix

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K] {A : Arch nH n}

/-- INTERNAL: the uniform-attention residual update `x_i + ∑_h |vis|⁻¹ ∑_{j ∈ vis} W_V x_j`.
TEXLINE: Paper/appendix.tex:95-96 -/
noncomputable def attnRes (θ : Params Sym K A) (l : Fin 2) (x : Fin (n + 4) → Fin A.dres → ℝ) :
    Fin (n + 4) → Fin A.dres → ℝ :=
  fun i => x i + ∑ h, ∑ j ∈ A.vis l h i, ((A.vis l h i).card : ℝ)⁻¹ • (θ.wV l h *ᵥ x j)

/-- INTERNAL: the MLP with the ReLU replaced by the `0/1` pattern `D`.
TEXLINE: Paper/appendix.tex:325-489 -/
noncomputable def mlpPat (D : Fin A.dmlp → ℝ) (θ : Params Sym K A) (l : Fin 2)
    (y : Fin (n + 4) → Fin A.dres → ℝ) : Fin (n + 4) → Fin A.dres → ℝ :=
  fun i => y i + θ.wOut l *ᵥ fun m => D m * ((θ.wIn l *ᵥ y i) m + θ.bIn l m)

/-- INTERNAL: the answer logits of the zero-order network with ReLU patterns `D 0`, `D 1`. -/
noncomputable def patLogits (T : Task K Sym) (A : Arch nH n) (D : Fin 2 → Fin A.dmlp → ℝ)
    (θ : Params Sym K A) (w : Fin n → Sym) : ZMod K → ℝ :=
  θ.readout *ᵥ mlpPat (D 1) θ 1 (attnRes θ 1 (mlpPat (D 0) θ 0 (attnRes θ 0 (embed T θ w))))
    (Fin.last (n + 3))

/-- INTERNAL: the zero-order credit functional of the pattern network. -/
noncomputable def zeroPat (T : Task K Sym) (A : Arch nH n) (D : Fin 2 → Fin A.dmlp → ℝ)
    (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => patLogits T A D θ w ⬝ᵥ T.targetSignal w

/-- INTERNAL: at `ζ = 0` the attention weights are uniform on the visible set. -/
theorem attnWeight_zero (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) :
    attnWeight 0 θ l h x i j = ((A.vis l h i).card : ℝ)⁻¹ := by
  simp [attnWeight]

/-- INTERNAL: `block 0` is the uniform-attention update followed by the ReLU MLP. -/
theorem block_zero_eq (θ : Params Sym K A) (l : Fin 2) (x : Fin (n + 4) → Fin A.dres → ℝ) :
    block 0 θ l x = fun i => attnRes θ l x i +
      θ.wOut l *ᵥ fun m => max ((θ.wIn l *ᵥ attnRes θ l x i) m + θ.bIn l m) 0 := by
  funext i
  simp only [block, attnWeight_zero, attnRes]

/-- INTERNAL: with the right pattern, `block 0 = mlpPat D ∘ attnRes`.
TEXLINE: Paper/appendix.tex:325-489 -/
theorem block_zero_eq_mlpPat (D : Fin A.dmlp → ℝ) (θ : Params Sym K A) (l : Fin 2)
    (x : Fin (n + 4) → Fin A.dres → ℝ)
    (hD : ∀ i m, max ((θ.wIn l *ᵥ attnRes θ l x i) m + θ.bIn l m) 0 =
      D m * ((θ.wIn l *ᵥ attnRes θ l x i) m + θ.bIn l m)) :
    block 0 θ l x = mlpPat D θ l (attnRes θ l x) := by
  rw [block_zero_eq]
  funext i
  simp only [mlpPat, hD]

/-! ### Differentiability of the pattern network -/

/-- INTERNAL: a coordinate of a Euclidean vector is differentiable in the vector. -/
@[fun_prop]
theorem differentiable_coord {ι : Type*} [Fintype ι] (i : ι) :
    Differentiable ℝ (fun θ : EuclideanSpace ℝ ι => θ.ofLp i) :=
  (PiLp.proj 2 (fun _ : ι => ℝ) i).differentiable

/-- INTERNAL. -/
theorem differentiable_attnRes (l : Fin 2) (x : Params Sym K A → Fin (n + 4) → Fin A.dres → ℝ)
    (hx : ∀ i d, Differentiable ℝ fun θ => x θ i d) (i : Fin (n + 4)) (d : Fin A.dres) :
    Differentiable ℝ fun θ => attnRes θ l (x θ) i d := by
  simp only [attnRes, Pi.add_apply, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, mulVec,
    dotProduct, Params.wV]
  fun_prop

/-- INTERNAL. -/
theorem differentiable_mlpPat (D : Fin A.dmlp → ℝ) (l : Fin 2)
    (y : Params Sym K A → Fin (n + 4) → Fin A.dres → ℝ)
    (hy : ∀ i d, Differentiable ℝ fun θ => y θ i d) (i : Fin (n + 4)) (d : Fin A.dres) :
    Differentiable ℝ fun θ => mlpPat D θ l (y θ) i d := by
  simp only [mlpPat, Pi.add_apply, mulVec, dotProduct, Params.wOut, Params.wIn, Params.bIn]
  fun_prop

/-- INTERNAL. -/
theorem differentiable_embed (T : Task K Sym) (w : Fin n → Sym) (i : Fin (n + 4))
    (d : Fin A.dres) : Differentiable ℝ fun θ : Params Sym K A => embed T θ w i d := by
  simp only [embed, Params.tokEmb, Params.posEmb]
  fun_prop

/-- INTERNAL: each pattern logit is differentiable in `θ`. -/
theorem differentiable_patLogits (T : Task K Sym) (D : Fin 2 → Fin A.dmlp → ℝ)
    (w : Fin n → Sym) (a : ZMod K) :
    Differentiable ℝ fun θ : Params Sym K A => patLogits T A D θ w a := by
  have h0 := differentiable_attnRes (Sym := Sym) (K := K) (A := A) 0 _
    (differentiable_embed T w)
  have h1 := differentiable_mlpPat (Sym := Sym) (K := K) (A := A) (D 0) 0 _ h0
  have h2 := differentiable_attnRes (Sym := Sym) (K := K) (A := A) 1 _ h1
  have h3 := differentiable_mlpPat (Sym := Sym) (K := K) (A := A) (D 1) 1 _ h2
  simp only [patLogits, mulVec, dotProduct, Params.readout]
  fun_prop

/-- INTERNAL: the pattern credit functional is differentiable everywhere. -/
theorem differentiable_zeroPat (T : Task K Sym) (D : Fin 2 → Fin A.dmlp → ℝ) :
    Differentiable ℝ (zeroPat T A D) := by
  have h := differentiable_patLogits (A := A) T D
  unfold zeroPat Task.Ex
  simp only [dotProduct, smul_eq_mul]
  fun_prop

end Trace.Proof.ZeroOrder
