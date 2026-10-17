import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Analysis.Contraction
import Trace.Analysis.ZeroOrderTrack

/-!
# Credit of the zero-order (uniform-attention) network

The first line of the proof of `lem:score-sum` (Paper/appendix.tex:510-512): the part of the
outcome credit carried by `J^{[0]}`, the Jacobian of the `ζ = 0` network, is at most
`C ε λ^n + C ε² λ^{n-1}`.

The proof is the Lipschitz route rather than an explicit Jacobian. On the set
`U = {θ : ‖θ − ε θ̃‖ ≤ ε/4}` (a neighbourhood of every point of `B_ε`, as in the doubled-ball
remark at Paper/appendix.tex:227-230) every weight block has operator norm at most `(|a0|+1) ε`,
every MLP bias has margin `3ε/4` with the sign of `θ̃`'s bias, so the ReLU pattern is frozen and
the `ζ = 0` network equals the polynomial network `ZeroOrder.patLogits`
(`ZeroOrderPattern.lean`). Along that network the logits split, for `θ, θ' ∈ U`, as
`z(θ') − z(θ) = f₀ + ∑_t g_t(w_t)` with `‖f₀‖ ≤ C ε ‖θ' − θ‖` and `∑_t sup ‖g_t‖ ≤ C ε² ‖θ' − θ‖`
(`ZeroOrderTrack.lean`): a letter reaches the answer position only through a value map and the
readout. Part (i) of `thm:main` charges the two pieces `λ^n` and `λ^{n-1}`
(`LetterDec.credit_le`), which bounds the local Lipschitz constant of the credit functional,
hence its derivative (`norm_fderiv_le_of_lip'`).

Constants: `a = |a0| + 1`, `c₀ = 1 / (8 a r₃)`, `C = (a + 1) r₄`, with `r₃, r₄` explicit
polynomials in `a` and `nH` (`ZeroOrder.r3`, `ZeroOrder.r4`). The hypotheses `hK`, `0 < p < 1`
are used only through part (i); `cemb` enters only through the ball radius `min(1/8, cemb/4) ≤ 1/8`.
-/

universe u

namespace Trace.Proof

open Matrix

/-- The zero-order credit functional `E⟨z^{[0]}_θ, y − u⟩`, whose gradient is
`E[(J^{[0]})ᵀ(y − u)]`. INTERNAL: names the difference `outcomeFunctional − scoreFunctional`.
TEXLINE: Paper/appendix.tex:295-306 -/
noncomputable def zeroFunctional {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ}
    [NeZero K] (T : Task K Sym) (A : Arch nH n) (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => logits T A 0 θ w ⬝ᵥ T.targetSignal w

/-- **Zero-order credit.** On `B_ε` the `ζ = 0` functional is differentiable and its
gradient has norm at most `C (ε λ^n + ε² λ^{n-1})`.
PAPER: Paper/appendix.tex:510-512 (first display of the proof of lem:score-sum, "(i): 0 or 1
letter"). -/
theorem zeroOrderCredit (hprior : Prior) (a0 cemb : ℝ) (K : ℕ) [NeZero K] (hK : 2 ≤ K)
    (nH : ℕ) :
    ∃ c₀ C : ℝ, 0 < c₀ ∧ 0 < C ∧
      ∀ n : ℕ, ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (A : Arch nH n) (θt : Params Sym K A) (Nb : ℝ), InitHyp θt a0 cemb Nb →
      ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
      ∀ θ ∈ initBall θt ε cemb,
        DifferentiableAt ℝ (zeroFunctional T A) θ ∧
          ‖fderiv ℝ (zeroFunctional T A) θ‖ ≤
            C * (ε * mixrate K T.p ^ n + ε ^ 2 * mixrate K T.p ^ (n - 1)) := by
  open ZeroOrder ParamNorm in
  set a : ℝ := |a0| + 1 with hadef
  have ha : 1 ≤ a := by have := abs_nonneg a0; linarith
  have ha0 : a0 ≤ a - 1 := by have := le_abs_self a0; linarith
  have hN : (0 : ℝ) ≤ nH := Nat.cast_nonneg nH
  obtain ⟨h1, h2, h3, h4⟩ := r_nonneg (by linarith : (0 : ℝ) ≤ a) hN
  have hr2 : 6 ≤ r2 a nH := by
    have : (a + 1) * (2 + 1) ≥ 6 := by nlinarith
    unfold r2; nlinarith [mul_nonneg (sq_nonneg (a + 1)) h1]
  have hr13 : r1 a nH ≤ r3 a nH := by
    have e1 : r1 a nH ≤ r2 a nH := by
      unfold r2; nlinarith [mul_nonneg (sq_nonneg (a + 1)) h1, sq_nonneg a]
    have e2 : r2 a nH ≤ r3 a nH := by
      unfold r3; nlinarith [mul_nonneg h2 (mul_nonneg hN (by linarith : (0 : ℝ) ≤ a + 1))]
    linarith
  have hr3 : 6 ≤ r3 a nH := by
    unfold r3; nlinarith [mul_nonneg h2 (mul_nonneg hN (by linarith : (0 : ℝ) ≤ a + 1))]
  have hr4 : 0 < r4 a nH := by
    unfold r4; nlinarith [mul_nonneg (sq_nonneg (a + 1)) h3]
  refine ⟨1 / (8 * a * r3 a nH), (a + 1) * r4 a nH, by positivity, by positivity, ?_⟩
  intro n Sym _ _ T hp0 hp1 A θt Nb hinit ε hε hεc θ hθ
  -- smallness of `ε`
  set c₀ : ℝ := 1 / (8 * a * r3 a nH) with hc₀
  have hc₀pos : 0 < c₀ := by positivity
  have hc₀1 : c₀ ≤ 1 := by
    rw [hc₀, div_le_one (by positivity)]; nlinarith
  have hmin : 0 < min 1 (1 / Nb) := by
    by_contra hcon
    have : c₀ * min 1 (1 / Nb) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hc₀pos.le (not_lt.mp hcon)
    linarith
  have hNb : 0 < Nb := one_div_pos.mp (lt_of_lt_of_le hmin (min_le_right _ _))
  have hεc₀ : ε ≤ c₀ :=
    hεc.trans (mul_le_of_le_one_right hc₀pos.le (min_le_left _ _))
  have hε1 : ε ≤ 1 := hεc₀.trans hc₀1
  have hεNb : ε * Nb ≤ c₀ := by
    have : ε ≤ c₀ * (1 / Nb) := hεc.trans
      (mul_le_mul_of_nonneg_left (min_le_right _ _) hc₀pos.le)
    rw [mul_one_div, le_div_iff₀ hNb] at this
    exact this
  have hsmall3 : 8 * a * r3 a nH * ε ≤ 1 := by
    have := mul_le_mul_of_nonneg_left hεc₀ (by positivity : (0 : ℝ) ≤ 8 * a * r3 a nH)
    rwa [hc₀, mul_one_div_cancel (by positivity)] at this
  have hsmall1 : 8 * a * r1 a nH * ε ≤ 1 :=
    le_trans (by gcongr) hsmall3
  -- the neighbourhood `U`
  set U : Set (Params Sym K A) := Metric.closedBall (ε • θt) (ε / 4) with hUdef
  have hθ8 : ‖θ - ε • θt‖ ≤ ε / 8 := by
    have := Metric.mem_closedBall.mp hθ
    rw [dist_eq_norm] at this
    exact this.trans (mul_le_mul_of_nonneg_right (min_le_left _ _) hε.le |>.trans_eq (by ring))
  have hU : ∀ θ' ∈ U, ‖θ' - ε • θt‖ ≤ ε / 4 := fun θ' h' => by
    simpa [hUdef, dist_eq_norm] using h'
  have hθU : θ ∈ U := by
    rw [hUdef, Metric.mem_closedBall, dist_eq_norm]; linarith
  have hdiam : ∀ θ' ∈ U, ‖θ' - θ‖ ≤ ε := by
    intro θ' h'
    have := norm_sub_le_norm_sub_add_norm_sub θ' (ε • θt) θ
    rw [norm_sub_rev (ε • θt) θ] at this
    linarith [hU θ' h']
  have hUnhds : U ∈ nhds θ := by
    refine Filter.mem_of_superset (Metric.ball_mem_nhds θ (by positivity : (0 : ℝ) < ε / 8)) ?_
    intro y hy
    rw [Metric.mem_ball, dist_eq_norm] at hy
    rw [hUdef, Metric.mem_closedBall, dist_eq_norm]
    have := norm_sub_le_norm_sub_add_norm_sub y θ (ε • θt)
    linarith
  -- size hypotheses on `U`
  have blk : ∀ {m k : Type} [Fintype m] [Fintype k] [DecidableEq k]
      {F : Params Sym K A → Matrix m k ℝ}, IsBlock F → opNorm (F θt) ≤ a0 →
      ∀ θ' ∈ U, opNorm (F θ') ≤ a * ε := by
    intro m k _ _ _ F hF hb θ' h'
    refine (hF.opNorm_ball_le hε.le θt θ').trans ?_
    have := hU θ' h'
    nlinarith
  have vblk : ∀ {k : Type} [Fintype k] [DecidableEq k] {F : Params Sym K A → k → ℝ}, IsVBlock F →
      ‖(WithLp.toLp 2 (F θt) : EuclideanSpace ℝ k)‖ ≤ a0 →
      ∀ θ' ∈ U, ‖(WithLp.toLp 2 (F θ') : EuclideanSpace ℝ k)‖ ≤ a * ε := by
    intro k _ _ F hF hb θ' h'
    refine (hF.norm_ball_le hε.le θt θ').trans ?_
    have := hU θ' h'
    nlinarith
  have hH : Hyp U θ ε a 2 :=
    { mem := hθU
      wV := fun θ' h' l h => blk (isBlock_wV l h) (hinit.wV_le l h) θ' h'
      wIn := fun θ' h' l => blk (isBlock_wIn l) (hinit.wIn_le l) θ' h'
      wOut := fun θ' h' l => blk (isBlock_wOut l) (hinit.wOut_le l) θ' h'
      readout := fun θ' h' => blk isBlock_readout hinit.readout_le θ' h'
      tokCol := fun θ' h' v => vblk (isVBlock_tokCol v)
        ((norm_col_le _ v).trans hinit.tokEmb_le) θ' h'
      posCol := fun θ' h' i => vblk (isVBlock_posCol i)
        ((norm_col_le _ i).trans hinit.posEmb_le) θ' h'
      bIn := fun θ' h' l => by
        refine ((isVBlock_bIn l).norm_ball_le hε.le θt θ').trans ?_
        have := hU θ' h'
        have hb := hinit.bIn_norm_le l
        nlinarith }
  -- the frozen ReLU pattern
  set D : Fin 2 → Fin A.dmlp → ℝ := fun l m => if 0 < θt.bIn l m then 1 else 0 with hDdef
  have hD : ∀ l m, D l m = 0 ∨ D l m = 1 := by
    intro l m; simp only [hDdef]; split_ifs <;> simp
  have hmargin : ∀ l, ∀ θ' ∈ U, ∀ m, (D l m = 1 ∧ 3 * ε / 4 ≤ θ'.bIn l m) ∨
      (D l m = 0 ∧ θ'.bIn l m ≤ -(3 * ε / 4)) := by
    intro l θ' h' m
    have hcoord : |θ'.bIn l m - ε * θt.bIn l m| ≤ ε / 4 := by
      have e : θ'.bIn l m - ε * θt.bIn l m = (θ' - ε • θt).bIn l m := by
        rw [(isVBlock_bIn l).map_sub, (isVBlock_bIn l).map_smul]; rfl
      rw [e]
      have := PiLp.norm_apply_le (WithLp.toLp 2 ((θ' - ε • θt).bIn l) :
        EuclideanSpace ℝ (Fin A.dmlp)) m
      rw [Real.norm_eq_abs] at this
      exact this.trans (((isVBlock_bIn l).norm_le _).trans (hU θ' h'))
    rw [abs_le] at hcoord
    have hone := hinit.bIn_ge_one l m
    by_cases hpos : 0 < θt.bIn l m
    · left
      refine ⟨by simp [hDdef, hpos], ?_⟩
      have : 1 ≤ θt.bIn l m := by rwa [abs_of_pos hpos] at hone
      nlinarith
    · right
      refine ⟨by simp [hDdef, hpos], ?_⟩
      have : θt.bIn l m ≤ -1 := by
        rw [abs_of_nonpos (not_lt.mp hpos)] at hone; linarith
      nlinarith
  obtain ⟨tp0, tp1, tpL⟩ := track_patLogits hH ha hε.le hε1 T D hD
  have hlogits : ∀ θ' ∈ U, ∀ w, logits T A 0 θ' w = patLogits T A D θ' w := by
    intro θ' h' w
    have r0 := relu_frozen hH (by linarith) hε h1 hdiam hsmall1 0 (D 0) tp0 (hmargin 0)
      θ' h' w
    have r1' := relu_frozen hH (by linarith) hε h3 hdiam hsmall3 1 (D 1) tp1 (hmargin 1)
      θ' h' w
    unfold logits patLogits
    rw [block_zero_eq_mlpPat (D 0) θ' 0 (embed T θ' w) r0,
      block_zero_eq_mlpPat (D 1) θ' 1 _ r1']
  have hfun : ∀ θ' ∈ U, zeroFunctional T A θ' = zeroPat T A D θ' := by
    intro θ' h'
    unfold zeroFunctional zeroPat
    simp only [hlogits θ' h']
  have hEq : zeroFunctional T A =ᶠ[nhds θ] zeroPat T A D :=
    Filter.mem_of_superset hUnhds fun θ' h' => hfun θ' h'
  refine ⟨((differentiable_zeroPat T D) θ).congr_of_eventuallyEq hEq, ?_⟩
  -- the Lipschitz bound
  set lam := mixrate K T.p with hlam
  have hlam0 : 0 ≤ lam := norm_nonneg _
  have hlip : ∀ᶠ y in nhds θ, ‖zeroFunctional T A y - zeroFunctional T A θ‖ ≤
      (lam ^ n * ((a + 1) * r4 a nH * ε) + lam ^ (n - 1) * ((a + 1) * r4 a nH * ε ^ 2)) *
        ‖y - θ‖ := by
    filter_upwards [hUnhds] with y hy
    rw [hfun y hy, hfun θ hθU]
    have e : zeroPat T A D y - zeroPat T A D θ = T.Ex (fun w =>
        ((WithLp.toLp 2 (patLogits T A D y w) : EuclideanSpace ℝ (ZMod K)) -
          WithLp.toLp 2 (patLogits T A D θ w)).ofLp ⬝ᵥ T.targetSignal w) := by
      unfold zeroPat Task.Ex
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun w _ => ?_
      simp only [WithLp.ofLp_sub, sub_dotProduct, smul_sub]
    rw [e, Real.norm_eq_abs]
    have := LetterDec.credit_le hprior hK T hp0 hp1 (tpL.2 y hy)
    refine this.trans (le_of_eq ?_)
    ring
  have hC0 : 0 ≤ lam ^ n * ((a + 1) * r4 a nH * ε) +
      lam ^ (n - 1) * ((a + 1) * r4 a nH * ε ^ 2) := by
    positivity
  refine (norm_fderiv_le_of_lip' ℝ hC0 hlip).trans (le_of_eq ?_)
  ring

end Trace.Proof

/-! ### Run record
Newest first. History, not instruction — what this file claims is above.

* r2 · proved · `zeroOrderCredit` closed by the Lipschitz route (frozen ReLU pattern on the
  `ε/4`-ball, `Track` through `patLogits`, `LetterDec.credit_le`, `norm_fderiv_le_of_lip'`)
-/
