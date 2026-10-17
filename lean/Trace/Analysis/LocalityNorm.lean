import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.Normed.Operator.Bilinear
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Logic.Function.DependsOn

/-!
# Locality-weighted Lipschitz bounds for families of letter functions

INTERNAL: the bookkeeping device behind the power counting of `prop:score-degree` and the
summation of `lem:score-sum` (Paper/appendix.tex:308-319, 491-536).

A *family* is a map `f : P → (ι → α) → E`: a parameter `θ : P` and a letter sequence
`x : ι → α` give a value in a normed space. On a parameter set `B`, a *decomposition* of
`f` writes `f θ = ∑_S f_S θ` (sum over all `S : Finset ι`) where each piece `f_S θ`
depends only on the letters in `S`, is bounded by `s_S` and is `l_S`-Lipschitz in `θ`
on `B`. Its *cost* at weight `κ` and scale `r` is `∑_S κ^{-|S|} (s_S + r l_S)`.
`Bd B κ r f M` says that `f` has decompositions of cost arbitrarily close to `M`.

For `0 < κ ≤ 1` and `0 ≤ r` the cost is submultiplicative under bounded bilinear maps
(a piece on `S` times a piece on `T` is a piece on `S ∪ T`, and
`κ^{-|S ∪ T|} ≤ κ^{-|S|} κ^{-|T|}`), and it is closed under `exp(·) - 1` and
`(1 + ·)⁻¹`. This is the whole of the calculus the forward pass needs: no derivative
is ever computed, the `θ`-Lipschitz constants are carried along instead.
-/

set_option linter.unusedSectionVars false

namespace Trace.Loc

open Finset

variable {ι α P : Type*} [Fintype ι] [DecidableEq ι] [PseudoMetricSpace P]
variable {E F G : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
  [SeminormedAddCommGroup F] [NormedSpace ℝ F] [SeminormedAddCommGroup G] [NormedSpace ℝ G]

/-- INTERNAL: a locality decomposition of the family `f` on the parameter set `B`. -/
structure Dec (B : Set P) (f : P → (ι → α) → E) where
  /-- The piece reading the letters in `S`. -/
  piece : Finset ι → P → (ι → α) → E
  /-- A sup bound for the piece on `S`. -/
  s : Finset ι → ℝ
  /-- A Lipschitz constant in `θ` for the piece on `S`. -/
  l : Finset ι → ℝ
  s_nonneg : ∀ S, 0 ≤ s S
  l_nonneg : ∀ S, 0 ≤ l S
  eq : ∀ θ ∈ B, ∀ x, f θ x = ∑ S, piece S θ x
  dep : ∀ S θ, DependsOn (piece S θ) (S : Set ι)
  sup : ∀ S, ∀ θ ∈ B, ∀ x, ‖piece S θ x‖ ≤ s S
  lip : ∀ S, ∀ θ ∈ B, ∀ θ' ∈ B, ∀ x, ‖piece S θ x - piece S θ' x‖ ≤ l S * dist θ θ'

variable {B : Set P} {κ r : ℝ}

/-- INTERNAL: the cost `∑_S κ^{-|S|} (s_S + r l_S)` of a decomposition. -/
noncomputable def Dec.cost (κ r : ℝ) {f : P → (ι → α) → E} (d : Dec B f) : ℝ :=
  ∑ S, κ⁻¹ ^ S.card * (d.s S + r * d.l S)

/-- INTERNAL: `f` has decompositions on `B` of cost arbitrarily close to `M`. -/
def Bd (B : Set P) (κ r : ℝ) (f : P → (ι → α) → E) (M : ℝ) : Prop :=
  ∀ δ > 0, ∃ d : Dec B f, d.cost κ r ≤ M + δ

section Basic

variable {f g : P → (ι → α) → E} {M M' : ℝ}

lemma Dec.cost_nonneg (hκ : 0 < κ) (hr : 0 ≤ r) (d : Dec B f) : 0 ≤ d.cost κ r :=
  Finset.sum_nonneg fun S _ => mul_nonneg (pow_nonneg (inv_nonneg.2 hκ.le) _)
    (add_nonneg (d.s_nonneg S) (mul_nonneg hr (d.l_nonneg S)))

lemma Bd.nonneg (hκ : 0 < κ) (hr : 0 ≤ r) (h : Bd B κ r f M) : 0 ≤ M := by
  by_contra hM
  rw [not_le] at hM
  obtain ⟨d, hd⟩ := h (-M / 2) (by linarith)
  have := d.cost_nonneg hκ hr
  linarith

lemma Bd.mono (h : Bd B κ r f M) (hMM : M ≤ M') : Bd B κ r f M' := fun δ hδ => by
  obtain ⟨d, hd⟩ := h δ hδ
  exact ⟨d, by linarith⟩

lemma Bd.of_forall_gt (h : ∀ M' > M, Bd B κ r f M') : Bd B κ r f M := fun δ hδ => by
  obtain ⟨d, hd⟩ := h (M + δ / 2) (by linarith) (δ / 2) (by linarith)
  exact ⟨d, by linarith⟩

/-- INTERNAL: transport a decomposition along equality on `B`. -/
def Dec.congr (d : Dec B f) (hfg : ∀ θ ∈ B, ∀ x, f θ x = g θ x) : Dec B g where
  piece := d.piece
  s := d.s
  l := d.l
  s_nonneg := d.s_nonneg
  l_nonneg := d.l_nonneg
  eq θ hθ x := (hfg θ hθ x).symm.trans (d.eq θ hθ x)
  dep := d.dep
  sup := d.sup
  lip := d.lip

lemma Bd.congr (h : Bd B κ r f M) (hfg : ∀ θ ∈ B, ∀ x, f θ x = g θ x) : Bd B κ r g M :=
  fun δ hδ => by
    obtain ⟨d, hd⟩ := h δ hδ
    exact ⟨d.congr hfg, hd⟩

/-- INTERNAL: a single piece on `S₀`. -/
def Dec.single (S₀ : Finset ι) (s l : ℝ) (hs : 0 ≤ s) (hl : 0 ≤ l)
    (hdep : ∀ θ, DependsOn (f θ) (S₀ : Set ι))
    (hsup : ∀ θ ∈ B, ∀ x, ‖f θ x‖ ≤ s)
    (hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ∀ x, ‖f θ x - f θ' x‖ ≤ l * dist θ θ') : Dec B f where
  piece S := if S = S₀ then f else 0
  s S := if S = S₀ then s else 0
  l S := if S = S₀ then l else 0
  s_nonneg S := by split_ifs <;> simp [hs]
  l_nonneg S := by split_ifs <;> simp [hl]
  eq θ _ x := by
    rw [Finset.sum_eq_single S₀]
    · simp
    · intro b _ hb; simp [hb]
    · simp
  dep S θ := by
    split_ifs with h
    · subst h; exact hdep θ
    · exact fun _ _ _ => rfl
  sup S θ hθ x := by
    split_ifs
    · exact hsup θ hθ x
    · simp
  lip S θ hθ θ' hθ' x := by
    split_ifs
    · exact hlip θ hθ θ' hθ' x
    · simp

lemma Dec.single_cost (S₀ : Finset ι) (s l : ℝ) (hs : 0 ≤ s) (hl : 0 ≤ l)
    (hdep : ∀ θ, DependsOn (f θ) (S₀ : Set ι))
    (hsup : ∀ θ ∈ B, ∀ x, ‖f θ x‖ ≤ s)
    (hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ∀ x, ‖f θ x - f θ' x‖ ≤ l * dist θ θ') :
    (Dec.single S₀ s l hs hl hdep hsup hlip).cost κ r = κ⁻¹ ^ S₀.card * (s + r * l) := by
  unfold Dec.cost Dec.single
  rw [Finset.sum_eq_single S₀]
  · simp
  · intro b _ hb; simp [hb]
  · simp

/-- INTERNAL: a family depending only on the letters in `S₀`, bounded by `s` and
`l`-Lipschitz on `B`, costs `κ^{-|S₀|} (s + r l)`. -/
lemma Bd.single (S₀ : Finset ι) {s l : ℝ} (hs : 0 ≤ s) (hl : 0 ≤ l)
    (hdep : ∀ θ, DependsOn (f θ) (S₀ : Set ι))
    (hsup : ∀ θ ∈ B, ∀ x, ‖f θ x‖ ≤ s)
    (hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ∀ x, ‖f θ x - f θ' x‖ ≤ l * dist θ θ') :
    Bd B κ r f (κ⁻¹ ^ S₀.card * (s + r * l)) := fun δ hδ =>
  ⟨Dec.single S₀ s l hs hl hdep hsup hlip, by rw [Dec.single_cost]; linarith⟩

/-- INTERNAL: a family not reading the letters, bounded by `s`, `l`-Lipschitz on `B`. -/
lemma Bd.param {g : P → E} {s l : ℝ} (hs : 0 ≤ s) (hl : 0 ≤ l)
    (hsup : ∀ θ ∈ B, ‖g θ‖ ≤ s) (hlip : ∀ θ ∈ B, ∀ θ' ∈ B, ‖g θ - g θ'‖ ≤ l * dist θ θ') :
    Bd B κ r (fun θ (_ : ι → α) => g θ) (s + r * l) := by
  have := Bd.single (B := B) (κ := κ) (r := r) (f := fun θ (_ : ι → α) => g θ) ∅ hs hl
    (fun θ => fun _ _ _ => rfl) (fun θ hθ _ => hsup θ hθ) (fun θ hθ θ' hθ' _ => hlip θ hθ θ' hθ')
  simpa using this

/-- INTERNAL: a constant family. -/
lemma Bd.const (c : E) : Bd B κ r (fun (_ : P) (_ : ι → α) => c) ‖c‖ := by
  have := Bd.param (B := B) (κ := κ) (r := r) (ι := ι) (α := α) (g := fun _ => c)
    (norm_nonneg c) le_rfl (fun _ _ => le_rfl) (fun _ _ _ _ => by simp)
  simpa using this

/-- INTERNAL: sum of two decompositions. -/
def Dec.add (df : Dec B f) (dg : Dec B g) : Dec B (fun θ x => f θ x + g θ x) where
  piece S θ x := df.piece S θ x + dg.piece S θ x
  s S := df.s S + dg.s S
  l S := df.l S + dg.l S
  s_nonneg S := add_nonneg (df.s_nonneg S) (dg.s_nonneg S)
  l_nonneg S := add_nonneg (df.l_nonneg S) (dg.l_nonneg S)
  eq θ hθ x := by rw [df.eq θ hθ x, dg.eq θ hθ x, Finset.sum_add_distrib]
  dep S θ := fun x y h => by
    simp only
    rw [df.dep S θ h, dg.dep S θ h]
  sup S θ hθ x := (norm_add_le _ _).trans (add_le_add (df.sup S θ hθ x) (dg.sup S θ hθ x))
  lip S θ hθ θ' hθ' x := by
    have h1 := df.lip S θ hθ θ' hθ' x
    have h2 := dg.lip S θ hθ θ' hθ' x
    calc ‖df.piece S θ x + dg.piece S θ x - (df.piece S θ' x + dg.piece S θ' x)‖
        = ‖(df.piece S θ x - df.piece S θ' x) + (dg.piece S θ x - dg.piece S θ' x)‖ := by
          congr 1; abel
      _ ≤ _ := norm_add_le _ _
      _ ≤ _ := add_le_add h1 h2
      _ = _ := by ring

lemma Bd.add (hf : Bd B κ r f M) (hg : Bd B κ r g M') :
    Bd B κ r (fun θ x => f θ x + g θ x) (M + M') := fun δ hδ => by
  obtain ⟨df, hdf⟩ := hf (δ / 2) (by linarith)
  obtain ⟨dg, hdg⟩ := hg (δ / 2) (by linarith)
  refine ⟨df.add dg, ?_⟩
  have : (df.add dg).cost κ r = df.cost κ r + dg.cost κ r := by
    unfold Dec.cost Dec.add
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun S _ => ?_
    ring
  rw [this]; linarith

/-- INTERNAL: scalar multiple of a decomposition. -/
def Dec.smul (c : ℝ) (df : Dec B f) : Dec B (fun θ x => c • f θ x) where
  piece S θ x := c • df.piece S θ x
  s S := |c| * df.s S
  l S := |c| * df.l S
  s_nonneg S := mul_nonneg (abs_nonneg c) (df.s_nonneg S)
  l_nonneg S := mul_nonneg (abs_nonneg c) (df.l_nonneg S)
  eq θ hθ x := by rw [df.eq θ hθ x, Finset.smul_sum]
  dep S θ := fun x y h => by
    simp only
    rw [df.dep S θ h]
  sup S θ hθ x := by
    rw [norm_smul, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left (df.sup S θ hθ x) (abs_nonneg c)
  lip S θ hθ θ' hθ' x := by
    rw [← smul_sub, norm_smul, Real.norm_eq_abs, mul_assoc]
    exact mul_le_mul_of_nonneg_left (df.lip S θ hθ θ' hθ' x) (abs_nonneg c)

lemma Bd.smul (c : ℝ) (hf : Bd B κ r f M) : Bd B κ r (fun θ x => c • f θ x) (|c| * M) :=
  fun δ hδ => by
    obtain ⟨df, hdf⟩ := hf (δ / (|c| + 1)) (by positivity)
    refine ⟨df.smul c, ?_⟩
    have : (df.smul c).cost κ r = |c| * df.cost κ r := by
      unfold Dec.cost Dec.smul
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun S _ => ?_
      ring
    rw [this]
    have hc := abs_nonneg c
    calc |c| * df.cost κ r ≤ |c| * (M + δ / (|c| + 1)) := mul_le_mul_of_nonneg_left hdf hc
      _ = |c| * M + δ * (|c| / (|c| + 1)) := by ring
      _ ≤ |c| * M + δ * 1 := by
          gcongr
          rw [div_le_one (by positivity)]; linarith
      _ = _ := by ring

lemma Bd.neg (hf : Bd B κ r f M) : Bd B κ r (fun θ x => -f θ x) M := by
  have := hf.smul (-1)
  simpa using this

lemma Bd.sub (hf : Bd B κ r f M) (hg : Bd B κ r g M') :
    Bd B κ r (fun θ x => f θ x - g θ x) (M + M') := by
  have := hf.add hg.neg
  simpa [sub_eq_add_neg] using this

lemma Bd.zero : Bd B κ r (fun (_ : P) (_ : ι → α) => (0 : E)) 0 := by
  simpa using (Bd.const (B := B) (κ := κ) (r := r) (ι := ι) (α := α) (0 : E))

lemma Bd.sum {β : Type*} (t : Finset β) {f : β → P → (ι → α) → E} {M : β → ℝ}
    (h : ∀ b ∈ t, Bd B κ r (f b) (M b)) :
    Bd B κ r (fun θ x => ∑ b ∈ t, f b θ x) (∑ b ∈ t, M b) := by
  classical
  induction t using Finset.induction_on with
  | empty => simpa using (Bd.zero (B := B) (κ := κ) (r := r) (ι := ι) (α := α) (E := E))
  | insert b t hb ih =>
    simp only [Finset.sum_insert hb]
    exact (h b (Finset.mem_insert_self b t)).add
      (ih fun b' hb' => h b' (Finset.mem_insert_of_mem hb'))

end Basic

section Pointwise

variable {f : P → (ι → α) → E} {M : ℝ}

lemma Dec.norm_le (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) (d : Dec B f) {θ : P} (hθ : θ ∈ B)
    (x : ι → α) : ‖f θ x‖ ≤ d.cost κ r := by
  rw [d.eq θ hθ x]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun S _ => ?_)
  have h1 : (1 : ℝ) ≤ κ⁻¹ ^ S.card := one_le_pow₀ (one_le_inv₀ hκ |>.2 hκ1)
  have := d.sup S θ hθ x
  have := d.l_nonneg S
  have := d.s_nonneg S
  nlinarith [mul_nonneg hr (d.l_nonneg S)]

lemma Dec.lip_le (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 < r) (d : Dec B f) {θ θ' : P}
    (hθ : θ ∈ B) (hθ' : θ' ∈ B) (x : ι → α) :
    ‖f θ x - f θ' x‖ ≤ d.cost κ r / r * dist θ θ' := by
  rw [d.eq θ hθ x, d.eq θ' hθ' x, ← Finset.sum_sub_distrib]
  refine (norm_sum_le _ _).trans ?_
  rw [Dec.cost, Finset.sum_div, Finset.sum_mul]
  refine Finset.sum_le_sum fun S _ => ?_
  have h1 : (1 : ℝ) ≤ κ⁻¹ ^ S.card := one_le_pow₀ (one_le_inv₀ hκ |>.2 hκ1)
  refine (d.lip S θ hθ θ' hθ' x).trans ?_
  have hd := dist_nonneg (x := θ) (y := θ')
  have hl := d.l_nonneg S
  have hs := d.s_nonneg S
  rw [div_mul_eq_mul_div, le_div_iff₀ hr]
  nlinarith [mul_nonneg (mul_nonneg hs hd) (sub_nonneg.2 h1), mul_nonneg hs hd,
    mul_nonneg (mul_nonneg (mul_nonneg hl hd) hr.le) (sub_nonneg.2 h1)]

lemma Bd.norm_le (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) (h : Bd B κ r f M) {θ : P}
    (hθ : θ ∈ B) (x : ι → α) : ‖f θ x‖ ≤ M := by
  refine le_of_forall_pos_le_add fun δ hδ => ?_
  obtain ⟨d, hd⟩ := h δ hδ
  exact (d.norm_le hκ hκ1 hr hθ x).trans hd

lemma Bd.lip_le (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 < r) (h : Bd B κ r f M) {θ θ' : P}
    (hθ : θ ∈ B) (hθ' : θ' ∈ B) (x : ι → α) : ‖f θ x - f θ' x‖ ≤ M / r * dist θ θ' := by
  refine le_of_forall_pos_le_add fun δ hδ => ?_
  obtain ⟨d, hd⟩ := h (δ * r / (dist θ θ' + 1)) (by positivity)
  refine (d.lip_le hκ hκ1 hr hθ hθ' x).trans ?_
  have hD := dist_nonneg (x := θ) (y := θ')
  have : d.cost κ r / r ≤ M / r + δ / (dist θ θ' + 1) := by
    have hD1 : 0 < dist θ θ' + 1 := by positivity
    rw [div_add_div _ _ hr.ne' hD1.ne', le_div_iff₀ (by positivity)]
    calc d.cost κ r / r * (r * (dist θ θ' + 1)) = d.cost κ r * (dist θ θ' + 1) := by
          field_simp
      _ ≤ (M + δ * r / (dist θ θ' + 1)) * (dist θ θ' + 1) :=
          mul_le_mul_of_nonneg_right hd hD1.le
      _ = M * (dist θ θ' + 1) + r * δ := by field_simp
  calc d.cost κ r / r * dist θ θ' ≤ (M / r + δ / (dist θ θ' + 1)) * dist θ θ' :=
        mul_le_mul_of_nonneg_right this hD
    _ = M / r * dist θ θ' + δ * (dist θ θ' / (dist θ θ' + 1)) := by ring
    _ ≤ M / r * dist θ θ' + δ * 1 := by
        gcongr
        rw [div_le_one (by positivity)]; linarith
    _ = _ := by ring

end Pointwise

end Trace.Loc
