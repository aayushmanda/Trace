import Trace.Model.Prelude
import Trace.Analysis.ScoreNetwork
import Mathlib.Analysis.Calculus.FDeriv.WithLp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# Differentiability of the forward pass where no ReLU unit sits at its kink

INTERNAL: the answer logits `θ ↦ logits T A ζ θ w` are differentiable at every parameter point
where all MLP pre-activations that reach the answer position are nonzero. On the ball `B_ε`
this holds by the frozen activation pattern (Paper/appendix.tex:227-230, 338-340), so the
`fderiv` in `Trace.scoreCredit` is a genuine derivative and not Mathlib's junk value `0`.
-/

universe u

set_option linter.unusedSectionVars false

namespace Trace.Proof.SD

open Trace Trace.Proof.SN Matrix

/-- INTERNAL: a coordinate of a Euclidean vector is differentiable. -/
@[fun_prop] lemma differentiableAt_coord {ι : Type*} [Fintype ι] (i : ι)
    (x : EuclideanSpace ℝ ι) : DifferentiableAt ℝ (fun θ : EuclideanSpace ℝ ι => θ i) x :=
  (PiLp.hasFDerivAt_apply 2 x i).differentiableAt

/-- INTERNAL: `max (f ·) 0` is differentiable where `f` is and does not vanish. -/
lemma differentiableAt_max_zero {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : E → ℝ} {x : E} (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    DifferentiableAt ℝ (fun y => max (f y) 0) x := by
  rcases lt_or_gt_of_ne hx with h | h
  · have hev : ∀ᶠ y in nhds x, f y < 0 := hf.continuousAt.eventually_lt continuousAt_const h
    refine (differentiableAt_const (0 : ℝ)).congr_of_eventuallyEq ?_
    filter_upwards [hev] with y hy using max_eq_right hy.le
  · have hev : ∀ᶠ y in nhds x, 0 < f y := continuousAt_const.eventually_lt hf.continuousAt h
    refine hf.congr_of_eventuallyEq ?_
    filter_upwards [hev] with y hy using max_eq_left hy.le

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
  {A : Arch nH n}

lemma differentiableAt_embed (T : Task K Sym) (w : Fin n → Sym) (θ0 : Params Sym K A)
    (j : Fin (n + 4)) (d : Fin A.dres) :
    DifferentiableAt ℝ (fun θ : Params Sym K A => embed T θ w j d) θ0 := by
  unfold embed Params.tokEmb Params.posEmb
  fun_prop

section Stream

variable {X : Params Sym K A → Fin (n + 4) → Fin A.dres → ℝ} {θ0 : Params Sym K A}

lemma differentiableAt_score (hX : ∀ j d, DifferentiableAt ℝ (fun θ => X θ j d) θ0)
    (l : Fin 2) (h : Fin nH) (i j : Fin (n + 4)) :
    DifferentiableAt ℝ (fun θ => score θ l h (X θ) i j) θ0 := by
  unfold score Params.wQ Params.wK
  simp only [mulVec, dotProduct]
  fun_prop

lemma differentiableAt_attnWeight (hX : ∀ j d, DifferentiableAt ℝ (fun θ => X θ j d) θ0)
    (ζ : ℝ) (l : Fin 2) (h : Fin nH) (i j : Fin (n + 4)) :
    DifferentiableAt ℝ (fun θ => attnWeight ζ θ l h (X θ) i j) θ0 := by
  unfold attnWeight
  have hs := fun k => differentiableAt_score hX l h i k
  refine DifferentiableAt.div (by fun_prop) (by fun_prop) ?_
  exact (Finset.sum_pos (fun _ _ => Real.exp_pos _) (A.vis_nonempty l h i)).ne'

lemma differentiableAt_attnOut (hX : ∀ j d, DifferentiableAt ℝ (fun θ => X θ j d) θ0)
    (ζ : ℝ) (l : Fin 2) (i : Fin (n + 4)) (d : Fin A.dres) :
    DifferentiableAt ℝ (fun θ => attnOut ζ θ l (X θ) i d) θ0 := by
  have hw := fun h j => differentiableAt_attnWeight hX ζ l h i j
  unfold attnOut Params.wV
  simp only [Pi.add_apply, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, mulVec, dotProduct]
  fun_prop

lemma differentiableAt_block (hX : ∀ j d, DifferentiableAt ℝ (fun θ => X θ j d) θ0)
    (ζ : ℝ) (l : Fin 2) (i : Fin (n + 4))
    (hne : ∀ m, (θ0.wIn l *ᵥ attnOut ζ θ0 l (X θ0) i) m + θ0.bIn l m ≠ 0) (d : Fin A.dres) :
    DifferentiableAt ℝ (fun θ => block ζ θ l (X θ) i d) θ0 := by
  have ha := fun d' => differentiableAt_attnOut hX ζ l i d'
  have hpre : ∀ m, DifferentiableAt ℝ
      (fun θ => (θ.wIn l *ᵥ attnOut ζ θ l (X θ) i) m + θ.bIn l m) θ0 := by
    intro m
    unfold Params.wIn Params.bIn
    simp only [mulVec, dotProduct]
    fun_prop
  have hmax := fun m => differentiableAt_max_zero (hpre m) (hne m)
  simp only [block_eq, Pi.add_apply]
  unfold Params.wOut
  simp only [mulVec, dotProduct]
  fun_prop

end Stream

end Trace.Proof.SD
