import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.Normed.Operator.Bilinear
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Logic.Function.DependsOn

/-!
# Locality costs: sup bounds and parameter-Lipschitz constants per set of letters

INTERNAL: the power-counting device behind `prop:score-degree` and the summation of
`lem:score-sum` (Paper/appendix.tex:308-319, 491-536), in the form the proof of
`Trace.Proof.unseenLettersScoreCredit` uses.

A *family* is a map `f : P → (ι → α) → E`: a parameter `θ : P` and a letter sequence
`x : ι → α` give a value in a normed space. On a parameter set `B`, a *decomposition* of
`f` writes `f θ = ∑_S f_S θ` (sum over all `S : Finset ι`) where each piece `f_S θ`
depends only on the letters in `S`, is bounded by `s_S` and is `l_S`-Lipschitz in `θ`
on `B`. Its *cost* at weight `κ` and scale `r` is `∑_S κ^{-|S|} (s_S + r l_S)`.
`Bd B κ r f M` says that `f` has decompositions of cost arbitrarily close to `M`.

For `0 < κ ≤ 1` and `0 ≤ r` the cost is submultiplicative under bounded bilinear maps
(`Bd.bilin`: a piece on `S` times a piece on `T` is a piece on `S ∪ T`, and
`κ^{-|S ∪ T|} ≤ κ^{-|S|} κ^{-|T|}`), closed under convergent series (`Bd.hasSum`, the tail
is one piece on all letters and its cost tends to `0`), and hence under `exp(·) - 1`
(`Bd.exp_sub_one`) and `(1 + ·)⁻¹ - 1` (`Bd.inv_one_add_sub_one`). This is the whole calculus
the forward pass needs: no derivative is computed, the `θ`-Lipschitz constants are carried
along instead. The basic closure lemmas repeat those of `Trace.Loc` in
`Trace/Analysis/LocalityNorm.lean`, a file owned by another proof.
-/

set_option linter.unusedSectionVars false

namespace Trace.Proof.SL

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


section Bilin

variable {f : P → (ι → α) → E} {g : P → (ι → α) → F} {M M' : ℝ}

/-- INTERNAL: the product of two decompositions under a bounded bilinear map; the pieces on
`S` and `T` combine into a piece on `S ∪ T`. -/
noncomputable def Dec.bilin (L : E →L[ℝ] F →L[ℝ] G) (df : Dec B f) (dg : Dec B g) :
    Dec B (fun θ x => L (f θ x) (g θ x)) where
  piece U θ x := ∑ p ∈ (univ : Finset (Finset ι × Finset ι)) with p.1 ∪ p.2 = U,
    L (df.piece p.1 θ x) (dg.piece p.2 θ x)
  s U := ∑ p ∈ (univ : Finset (Finset ι × Finset ι)) with p.1 ∪ p.2 = U,
    ‖L‖ * (df.s p.1 * dg.s p.2)
  l U := ∑ p ∈ (univ : Finset (Finset ι × Finset ι)) with p.1 ∪ p.2 = U,
    ‖L‖ * (df.l p.1 * dg.s p.2 + df.s p.1 * dg.l p.2)
  s_nonneg U := Finset.sum_nonneg fun p _ =>
    mul_nonneg L.opNorm_nonneg (mul_nonneg (df.s_nonneg _) (dg.s_nonneg _))
  l_nonneg U := Finset.sum_nonneg fun p _ => mul_nonneg L.opNorm_nonneg
    (add_nonneg (mul_nonneg (df.l_nonneg _) (dg.s_nonneg _))
      (mul_nonneg (df.s_nonneg _) (dg.l_nonneg _)))
  eq θ hθ x := by
    rw [Finset.sum_fiberwise univ (fun p : Finset ι × Finset ι => p.1 ∪ p.2)
      (fun p => L (df.piece p.1 θ x) (dg.piece p.2 θ x)), df.eq θ hθ x, dg.eq θ hθ x,
      Fintype.sum_prod_type' (fun S T => L (df.piece S θ x) (dg.piece T θ x))]
    simp only [map_sum, ContinuousLinearMap.coe_sum', Finset.sum_apply]
    exact Finset.sum_comm
  dep U θ := fun x y hxy => by
    refine Finset.sum_congr rfl fun p hp => ?_
    have hU := (Finset.mem_filter.1 hp).2
    have h1 : ∀ i ∈ (p.1 : Set ι), x i = y i := fun i hi =>
      hxy i (by rw [← hU]; exact Finset.mem_coe.2 (Finset.mem_union_left _ hi))
    have h2 : ∀ i ∈ (p.2 : Set ι), x i = y i := fun i hi =>
      hxy i (by rw [← hU]; exact Finset.mem_coe.2 (Finset.mem_union_right _ hi))
    rw [df.dep p.1 θ h1, dg.dep p.2 θ h2]
  sup U θ hθ x := by
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun p _ => ?_)
    refine (L.le_opNorm₂ _ _).trans ?_
    rw [mul_assoc]
    exact mul_le_mul_of_nonneg_left (mul_le_mul (df.sup _ θ hθ x) (dg.sup _ θ hθ x)
      (norm_nonneg _) (df.s_nonneg _)) L.opNorm_nonneg
  lip U θ hθ θ' hθ' x := by
    rw [← Finset.sum_sub_distrib, Finset.sum_mul]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun p _ => ?_)
    have e : L (df.piece p.1 θ x) (dg.piece p.2 θ x) - L (df.piece p.1 θ' x) (dg.piece p.2 θ' x) =
        L (df.piece p.1 θ x - df.piece p.1 θ' x) (dg.piece p.2 θ x) +
          L (df.piece p.1 θ' x) (dg.piece p.2 θ x - dg.piece p.2 θ' x) := by
      simp only [map_sub, ContinuousLinearMap.sub_apply]; abel
    rw [e]
    refine (norm_add_le _ _).trans ?_
    have h1 := (L.le_opNorm₂ (df.piece p.1 θ x - df.piece p.1 θ' x) (dg.piece p.2 θ x))
    have h2 := (L.le_opNorm₂ (df.piece p.1 θ' x) (dg.piece p.2 θ x - dg.piece p.2 θ' x))
    have a1 := df.lip p.1 θ hθ θ' hθ' x
    have a2 := dg.lip p.2 θ hθ θ' hθ' x
    have b1 := df.sup p.1 θ' hθ' x
    have b2 := dg.sup p.2 θ hθ x
    have hL := L.opNorm_nonneg
    have hd := dist_nonneg (x := θ) (y := θ')
    have c1 : ‖L‖ * ‖df.piece p.1 θ x - df.piece p.1 θ' x‖ * ‖dg.piece p.2 θ x‖ ≤
        ‖L‖ * (df.l p.1 * dist θ θ') * dg.s p.2 :=
      mul_le_mul (mul_le_mul_of_nonneg_left a1 hL) b2 (norm_nonneg _)
        (mul_nonneg hL (mul_nonneg (df.l_nonneg _) hd))
    have c2 : ‖L‖ * ‖df.piece p.1 θ' x‖ * ‖dg.piece p.2 θ x - dg.piece p.2 θ' x‖ ≤
        ‖L‖ * df.s p.1 * (dg.l p.2 * dist θ θ') :=
      mul_le_mul (mul_le_mul_of_nonneg_left b1 hL) a2 (norm_nonneg _)
        (mul_nonneg hL (df.s_nonneg _))
    calc _ ≤ _ := add_le_add (h1.trans c1) (h2.trans c2)
      _ = _ := by ring

lemma Dec.bilin_cost (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) (L : E →L[ℝ] F →L[ℝ] G)
    (df : Dec B f) (dg : Dec B g) :
    (df.bilin L dg).cost κ r ≤ ‖L‖ * df.cost κ r * dg.cost κ r := by
  have h1 : (df.bilin L dg).cost κ r = ∑ p : Finset ι × Finset ι,
      κ⁻¹ ^ (p.1 ∪ p.2).card * (‖L‖ * (df.s p.1 * dg.s p.2) +
        r * (‖L‖ * (df.l p.1 * dg.s p.2 + df.s p.1 * dg.l p.2))) := by
    unfold Dec.cost Dec.bilin
    dsimp only
    rw [← Finset.sum_fiberwise univ (fun p : Finset ι × Finset ι => p.1 ∪ p.2)]
    refine Finset.sum_congr rfl fun U _ => ?_
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, Finset.mul_sum]
    refine Finset.sum_congr rfl fun p hp => ?_
    rw [(Finset.mem_filter.1 hp).2]
  have hk : 1 ≤ κ⁻¹ := (one_le_inv₀ hκ).2 hκ1
  rw [h1, Dec.cost, Dec.cost, mul_assoc, Finset.sum_mul_sum, Finset.mul_sum]
  simp_rw [Finset.mul_sum]
  rw [← Fintype.sum_prod_type' (fun S T => ‖L‖ * (κ⁻¹ ^ S.card * (df.s S + r * df.l S) *
      (κ⁻¹ ^ T.card * (dg.s T + r * dg.l T))))]
  refine Finset.sum_le_sum fun p _ => ?_
  have hp : κ⁻¹ ^ (p.1 ∪ p.2).card ≤ κ⁻¹ ^ p.1.card * κ⁻¹ ^ p.2.card := by
    rw [← pow_add]; exact pow_le_pow_right₀ hk (Finset.card_union_le _ _)
  have s1 := df.s_nonneg p.1
  have s2 := dg.s_nonneg p.2
  have l1 := df.l_nonneg p.1
  have l2 := dg.l_nonneg p.2
  have hL := L.opNorm_nonneg
  have q : ‖L‖ * (df.s p.1 * dg.s p.2) + r * (‖L‖ * (df.l p.1 * dg.s p.2 + df.s p.1 * dg.l p.2))
      ≤ ‖L‖ * ((df.s p.1 + r * df.l p.1) * (dg.s p.2 + r * dg.l p.2)) := by
    have : 0 ≤ ‖L‖ * (r * df.l p.1 * (r * dg.l p.2)) := by positivity
    nlinarith
  have q0 : 0 ≤ ‖L‖ * (df.s p.1 * dg.s p.2) +
      r * (‖L‖ * (df.l p.1 * dg.s p.2 + df.s p.1 * dg.l p.2)) := by positivity
  calc κ⁻¹ ^ (p.1 ∪ p.2).card * (‖L‖ * (df.s p.1 * dg.s p.2) +
        r * (‖L‖ * (df.l p.1 * dg.s p.2 + df.s p.1 * dg.l p.2)))
      ≤ (κ⁻¹ ^ p.1.card * κ⁻¹ ^ p.2.card) *
          (‖L‖ * ((df.s p.1 + r * df.l p.1) * (dg.s p.2 + r * dg.l p.2))) :=
        mul_le_mul hp q q0 (by positivity)
    _ = _ := by ring

/-- INTERNAL: submultiplicativity of the cost under a bounded bilinear map. -/
lemma Bd.bilin (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) (L : E →L[ℝ] F →L[ℝ] G)
    (hf : Bd B κ r f M) (hg : Bd B κ r g M') :
    Bd B κ r (fun θ x => L (f θ x) (g θ x)) (‖L‖ * M * M') := fun δ hδ => by
  have hM := hf.nonneg hκ hr
  have hM' := hg.nonneg hκ hr
  have hL := L.opNorm_nonneg
  set η := min 1 (δ / ((‖L‖ + 1) * (M + M' + 1))) with hη
  have hη0 : 0 < η := lt_min one_pos (by positivity)
  have hη1 : η ≤ 1 := min_le_left _ _
  have hη2 : η * ((‖L‖ + 1) * (M + M' + 1)) ≤ δ := by
    have := min_le_right 1 (δ / ((‖L‖ + 1) * (M + M' + 1)))
    rw [← hη] at this
    calc η * ((‖L‖ + 1) * (M + M' + 1)) ≤ δ / ((‖L‖ + 1) * (M + M' + 1)) *
          ((‖L‖ + 1) * (M + M' + 1)) := mul_le_mul_of_nonneg_right this (by positivity)
      _ = δ := by field_simp
  obtain ⟨df, hdf⟩ := hf η hη0
  obtain ⟨dg, hdg⟩ := hg η hη0
  refine ⟨df.bilin L dg, (df.bilin_cost hκ hκ1 hr L dg).trans ?_⟩
  have c1 := df.cost_nonneg hκ hr
  have c2 := dg.cost_nonneg hκ hr
  calc ‖L‖ * df.cost κ r * dg.cost κ r ≤ ‖L‖ * (M + η) * (M' + η) := by gcongr
    _ = ‖L‖ * M * M' + η * (‖L‖ * (M + M' + η)) := by ring
    _ ≤ ‖L‖ * M * M' + η * ((‖L‖ + 1) * (M + M' + 1)) := by
        gcongr
        nlinarith
    _ ≤ ‖L‖ * M * M' + δ := by linarith

/-- INTERNAL: a family of bounded linear maps applied to a family of vectors. -/
lemma Bd.app (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) {A : P → (ι → α) → E →L[ℝ] F}
    (hA : Bd B κ r A M) {v : P → (ι → α) → E} (hv : Bd B κ r v M') :
    Bd B κ r (fun θ x => A θ x (v θ x)) (M * M') := by
  have := Bd.bilin hκ hκ1 hr (ContinuousLinearMap.id ℝ (E →L[ℝ] F)) hA hv
  refine (this.mono ?_)
  have hM := hA.nonneg hκ hr
  have hM' := hv.nonneg hκ hr
  calc ‖ContinuousLinearMap.id ℝ (E →L[ℝ] F)‖ * M * M' ≤ 1 * M * M' := by
        gcongr; exact ContinuousLinearMap.norm_id_le
    _ = M * M' := by ring

/-- INTERNAL: products of real families. -/
lemma Bd.mul (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) {u v : P → (ι → α) → ℝ}
    (hu : Bd B κ r u M) (hv : Bd B κ r v M') :
    Bd B κ r (fun θ x => u θ x * v θ x) (M * M') := by
  have := Bd.bilin hκ hκ1 hr (ContinuousLinearMap.mul ℝ ℝ) hu hv
  refine (this.mono ?_)
  have hM := hu.nonneg hκ hr
  have hM' := hv.nonneg hκ hr
  calc ‖ContinuousLinearMap.mul ℝ ℝ‖ * M * M' ≤ 1 * M * M' := by
        gcongr; exact ContinuousLinearMap.opNorm_mul_le ℝ ℝ
    _ = M * M' := by ring

/-- INTERNAL: a real family times a vector family. -/
lemma Bd.smul' (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) {u : P → (ι → α) → ℝ}
    (hu : Bd B κ r u M) {v : P → (ι → α) → E} (hv : Bd B κ r v M') :
    Bd B κ r (fun θ x => u θ x • v θ x) (M * M') := by
  have := Bd.bilin hκ hκ1 hr (ContinuousLinearMap.lsmul ℝ ℝ : ℝ →L[ℝ] E →L[ℝ] E) hu hv
  refine (this.mono ?_)
  have hM := hu.nonneg hκ hr
  have hM' := hv.nonneg hκ hr
  calc ‖(ContinuousLinearMap.lsmul ℝ ℝ : ℝ →L[ℝ] E →L[ℝ] E)‖ * M * M' ≤ 1 * M * M' := by
        gcongr; exact ContinuousLinearMap.opNorm_lsmul_le
    _ = M * M' := by ring

/-- INTERNAL: powers of a real family. -/
lemma Bd.pow (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 ≤ r) {u : P → (ι → α) → ℝ}
    (hu : Bd B κ r u M) (k : ℕ) : Bd B κ r (fun θ x => u θ x ^ k) (M ^ k) := by
  induction k with
  | zero => simpa using (Bd.const (B := B) (κ := κ) (r := r) (ι := ι) (α := α) (1 : ℝ))
  | succ k ih =>
    have := Bd.mul hκ hκ1 hr ih hu
    simpa [pow_succ] using this

end Bilin

section Series

open Filter Topology

/-- INTERNAL: closure under convergent series. The first `N` terms are summed as
decompositions; the tail is one piece on all letters, bounded with its Lipschitz constant by
the tail of `∑ M k`, and its cost tends to `0`. -/
lemma Bd.hasSum (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 < r) {f : ℕ → P → (ι → α) → E}
    {F : P → (ι → α) → E} {M : ℕ → ℝ} {Mt : ℝ} (hM : HasSum M Mt)
    (hf : ∀ k, Bd B κ r (f k) (M k)) (hF : ∀ θ ∈ B, ∀ x, HasSum (fun k => f k θ x) (F θ x)) :
    Bd B κ r F Mt := by
  have hM0 : ∀ k, 0 ≤ M k := fun k => (hf k).nonneg hκ hr.le
  set T : ℕ → ℝ := fun N => Mt - ∑ k ∈ Finset.range N, M k with hT
  have hTsum : ∀ N, HasSum (fun k => M (k + N)) (T N) := fun N =>
    (hasSum_nat_add_iff' N).2 hM
  have hT0 : ∀ N, 0 ≤ T N := fun N => (hTsum N).nonneg fun k => hM0 _
  have hTlim : Tendsto T atTop (𝓝 0) := by
    have := (hM.tendsto_sum_nat).const_sub Mt
    simpa [hT] using this
  -- the bound for each truncation
  have key : ∀ N, Bd B κ r F (Mt + 2 * κ⁻¹ ^ (Finset.univ : Finset ι).card * T N) := by
    intro N
    set R : P → (ι → α) → E := fun θ x => F θ x - ∑ k ∈ Finset.range N, f k θ x with hR
    have hRsum : ∀ θ ∈ B, ∀ x, HasSum (fun k => f (k + N) θ x) (R θ x) := fun θ hθ x =>
      (hasSum_nat_add_iff' N).2 (hF θ hθ x)
    have hRsup : ∀ θ ∈ B, ∀ x, ‖R θ x‖ ≤ T N := fun θ hθ x =>
      (hRsum θ hθ x).norm_le_of_bounded (hTsum N) fun k =>
        (hf (k + N)).norm_le hκ hκ1 hr.le hθ x
    have hRlip : ∀ θ ∈ B, ∀ θ' ∈ B, ∀ x, ‖R θ x - R θ' x‖ ≤ T N / r * dist θ θ' := by
      intro θ hθ θ' hθ' x
      refine ((hRsum θ hθ x).sub (hRsum θ' hθ' x)).norm_le_of_bounded
        (((hTsum N).div_const r).mul_right (dist θ θ')) fun k => ?_
      exact (hf (k + N)).lip_le hκ hκ1 hr hθ hθ' x
    have hRB : Bd B κ r R (κ⁻¹ ^ (Finset.univ : Finset ι).card * (T N + r * (T N / r))) :=
      Bd.single Finset.univ (hT0 N) (div_nonneg (hT0 N) hr.le)
        (fun θ => (dependsOn_univ _).mono (by simp)) hRsup hRlip
    have hfin := Bd.sum (B := B) (κ := κ) (r := r) (Finset.range N) (f := f) (M := M)
      fun k _ => hf k
    have hsum := hfin.add hRB
    refine (hsum.congr fun θ _ x => by simp [hR]).mono ?_
    have : r * (T N / r) = T N := by field_simp
    rw [this]
    have hTN := hT0 N
    have : ∑ b ∈ Finset.range N, M b = Mt - T N := by simp only [hT]; ring
    rw [this]
    nlinarith [pow_nonneg (inv_nonneg.2 hκ.le) (Finset.univ : Finset ι).card]
  refine Bd.of_forall_gt fun M' hM' => ?_
  have hc : 0 ≤ 2 * κ⁻¹ ^ (Finset.univ : Finset ι).card := by positivity
  have hev : ∀ᶠ N in atTop, 2 * κ⁻¹ ^ (Finset.univ : Finset ι).card * T N < M' - Mt := by
    have := hTlim.const_mul (2 * κ⁻¹ ^ (Finset.univ : Finset ι).card)
    rw [mul_zero] at this
    exact this.eventually (gt_mem_nhds (by linarith))
  obtain ⟨N, hN⟩ := hev.exists
  exact (key N).mono (by linarith)

/-- INTERNAL: `exp(ψ) − 1` costs at most `exp(cost ψ) − 1`. -/
lemma Bd.exp_sub_one (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 < r) {ψ : P → (ι → α) → ℝ} {β : ℝ}
    (hψ : Bd B κ r ψ β) : Bd B κ r (fun θ x => Real.exp (ψ θ x) - 1) (Real.exp β - 1) := by
  have hβ := hψ.nonneg hκ hr.le
  have hser : ∀ y : ℝ, HasSum (fun k => ((k + 1).factorial : ℝ)⁻¹ * y ^ (k + 1))
      (Real.exp y - 1) := by
    intro y
    have h := NormedSpace.expSeries_div_hasSum_exp y
    rw [← Real.exp_eq_exp_ℝ] at h
    have h2 := (hasSum_nat_add_iff' 1).2 h
    simp only [Finset.range_one, Finset.sum_singleton, pow_zero, Nat.factorial_zero,
      Nat.cast_one, div_one] at h2
    refine h2.congr_fun fun k => ?_
    simp [div_eq_inv_mul]
  refine Bd.hasSum hκ hκ1 hr (hser β) (fun k => ?_) (fun θ _ x => hser (ψ θ x))
  have := (hψ.pow hκ hκ1 hr.le (k + 1)).smul (((k + 1).factorial : ℝ)⁻¹)
  simp only [smul_eq_mul] at this
  rwa [abs_of_nonneg (by positivity)] at this

/-- INTERNAL: `(1 + u)⁻¹ − 1` costs at most `γ / (1 − γ)` when `u` costs `γ < 1`. -/
lemma Bd.inv_one_add_sub_one (hκ : 0 < κ) (hκ1 : κ ≤ 1) (hr : 0 < r)
    {u : P → (ι → α) → ℝ} {γ : ℝ} (hu : Bd B κ r u γ) (hγ : γ < 1) :
    Bd B κ r (fun θ x => (1 + u θ x)⁻¹ - 1) (γ / (1 - γ)) := by
  have hγ0 := hu.nonneg hκ hr.le
  have hser : ∀ y : ℝ, |y| < 1 → HasSum (fun k => (-y) ^ (k + 1)) ((1 + y)⁻¹ - 1) := by
    intro y hy
    have h := hasSum_geometric_of_abs_lt_one (r := -y) (by rwa [abs_neg])
    have h2 := (hasSum_nat_add_iff' 1).2 h
    simpa [sub_neg_eq_add, add_comm] using h2
  have hserγ : HasSum (fun k => γ ^ (k + 1)) (γ / (1 - γ)) := by
    have h := hasSum_geometric_of_abs_lt_one (r := γ) (by rw [abs_of_nonneg hγ0]; exact hγ)
    have h2 := (hasSum_nat_add_iff' 1).2 h
    have e : (1 - γ)⁻¹ - ∑ i ∈ Finset.range 1, γ ^ i = γ / (1 - γ) := by
      have : (1 - γ) ≠ 0 := by linarith
      simp only [Finset.range_one, Finset.sum_singleton, pow_zero]
      field_simp; ring
    rwa [e] at h2
  refine Bd.hasSum hκ hκ1 hr hserγ (fun k => ?_) (fun θ hθ x => hser _ ?_)
  · have := hu.neg.pow hκ hκ1 hr.le (k + 1)
    exact this
  · have := hu.norm_le hκ hκ1 hr.le hθ x
    rw [Real.norm_eq_abs] at this
    linarith

end Series

end Trace.Proof.SL
