import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Meta.ModelClosure
import Mathlib.Logic.Function.DependsOn
import Trace.Analysis.ScoreCredit
import Trace.Analysis.ZeroOrderCredit
import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# Outcome credit (`thm:main` (ii-b), eq:thm1-full)

`Φ_out = Φ_score + Φ_0` pointwise, so the outcome credit is at most the score-mediated credit
(`unseenLettersScoreCredit`) plus the zero-order credit (`zeroOrderCredit`), as in the proof of
`lem:score-sum` (Paper/appendix.tex:522-528). With `ε ≤ cλ` and `c ≤ 1` one has `ε ≤ λ ≤ 1`, so
`ε⁶λ^{n-3} ≤ ελ^n` and `ε²λ^{n-1} ≤ ελ^n`.
-/

section
open scoped Matrix

namespace Trace.Proof

/-- `λ ≤ 1`. INTERNAL: triangle inequality for `|1 − p + p e^{2πi/K}|`.
TEXLINE: Paper/body.tex:89-96 -/
theorem mixrate_le_one (K : ℕ) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) : mixrate K p ≤ 1 := by
  unfold mixrate
  have he : ‖Complex.exp (2 * Real.pi * Complex.I / K)‖ = 1 := by
    have : 2 * Real.pi * Complex.I / K = ((2 * Real.pi / K : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [this, Complex.norm_exp_ofReal_mul_I]
  calc ‖(1 - p : ℂ) + p * Complex.exp (2 * Real.pi * Complex.I / K)‖
      ≤ ‖(1 - p : ℂ)‖ + ‖(p : ℂ) * Complex.exp (2 * Real.pi * Complex.I / K)‖ := norm_add_le _ _
    _ = 1 := by
      rw [norm_mul, he, mul_one]
      have h1 : ‖(1 - p : ℂ)‖ = 1 - p := by
        rw [show (1 - p : ℂ) = ((1 - p : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
          Real.norm_of_nonneg (by linarith)]
      rw [h1, Complex.norm_real, Real.norm_of_nonneg hp0]; ring

/-- `Φ_out = Φ_score + Φ_0`. INTERNAL: linearity of the expectation.
TEXLINE: Paper/appendix.tex:295-306 -/
theorem outcomeFunctional_eq {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ}
    [NeZero K] (T : Task K Sym) (A : Arch nH n) :
    outcomeFunctional T A = fun θ => scoreFunctional T A θ + zeroFunctional T A θ := by
  funext θ
  simp only [outcomeFunctional, scoreFunctional, zeroFunctional, Task.Ex, ← Finset.sum_add_distrib,
    ← smul_add, sub_dotProduct, sub_add_cancel]

end Trace.Proof

/-- Proof-side owner for `Trace.unseen_letters_outcome_credit`; its statement is fixed by the proof charter. -/
theorem Trace.Proof.unseenLettersOutcomeCredit (hprior : Trace.Prior)
    (a0 cemb : ℝ) (hcemb : 0 < cemb) (K : ℕ) [NeZero K] (hK : 2 ≤ K) (nH : ℕ) :
    ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
      ∀ n : ℕ, 2 ≤ n →
      ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Trace.Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (A : Trace.Arch nH n) (θt : Trace.Params Sym K A) (Nb : ℝ), Trace.InitHyp θt a0 cemb Nb →
      ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
      ∀ θ ∈ Trace.initBall θt ε cemb,
        ε ≤ c * Trace.mixrate K T.p →
          DifferentiableAt ℝ (Trace.outcomeFunctional T A) θ ∧
            Trace.outcomeCredit T A θ ≤ C * ε * Trace.mixrate K T.p ^ n := by
  obtain ⟨c₀s, cs, Cs, hc₀s, hcs, hCs, hS⟩ :=
    Trace.Proof.unseenLettersScoreCredit.{u} hprior a0 cemb hcemb K hK nH
  obtain ⟨c₀z, Cz, hc₀z, hCz, hZ⟩ := Trace.Proof.zeroOrderCredit.{u} hprior a0 cemb K hK nH
  refine ⟨min c₀s c₀z, min 1 cs, Cs + 2 * Cz, lt_min hc₀s hc₀z, lt_min one_pos hcs,
    by positivity, ?_⟩
  intro n hn Sym _ _ T hp0 hp1 A θt Nb hinit ε hε hεc₀ θ hθ hεc
  set lam := Trace.mixrate K T.p with hlam
  have hlam1 : lam ≤ 1 := Trace.Proof.mixrate_le_one K hp0.le hp1.le
  have hcle1 : min 1 cs ≤ 1 := min_le_left _ _
  have hεlam : ε ≤ lam := by
    have : 0 ≤ lam := norm_nonneg _
    nlinarith
  have hlam0 : 0 < lam := lt_of_lt_of_le hε hεlam
  have hε1 : ε ≤ 1 := hεlam.trans hlam1
  have hmin : min 1 (1 / Nb) ≥ 0 := by
    by_contra h
    have h' : min 1 (1 / Nb) < 0 := lt_of_not_ge h
    nlinarith [lt_min hc₀s hc₀z]
  have hεs : ε ≤ c₀s * min 1 (1 / Nb) :=
    hεc₀.trans (mul_le_mul_of_nonneg_right (min_le_left _ _) hmin)
  have hεz : ε ≤ c₀z * min 1 (1 / Nb) :=
    hεc₀.trans (mul_le_mul_of_nonneg_right (min_le_right _ _) hmin)
  have hε2 : ε ^ 2 ≤ cs * lam := by
    have h1 : ε ≤ cs * lam := hεc.trans (mul_le_mul_of_nonneg_right (min_le_right _ _) hlam0.le)
    nlinarith
  obtain ⟨hdS, hbS⟩ := hS n hn Sym T hp0 hp1 A θt Nb hinit ε hε hεs θ hθ hε2
  obtain ⟨hdZ, hbZ⟩ := hZ n Sym T hp0 hp1 A θt Nb hinit ε hε hεz θ hθ
  rw [Trace.Proof.outcomeFunctional_eq] at ⊢
  refine ⟨hdS.add hdZ, ?_⟩
  unfold Trace.outcomeCredit
  rw [Trace.Proof.outcomeFunctional_eq, fderiv_fun_add hdS hdZ]
  unfold Trace.scoreCredit at hbS
  -- the two exponent comparisons
  have hpow : lam ^ n = lam * lam ^ (n - 1) := by
    rw [← pow_succ']; congr 1; omega
  have hA : ε ^ 6 * lam ^ ((n : ℤ) - 3) ≤ ε * lam ^ n := by
    rw [zpow_sub₀ hlam0.ne', zpow_natCast, mul_div_assoc', div_le_iff₀ (by positivity)]
    have h5 : ε ^ 5 ≤ lam ^ 3 :=
      (pow_le_pow_left₀ hε.le hεlam 5).trans (pow_le_pow_of_le_one hlam0.le hlam1 (by norm_num))
    have hn0 : 0 < lam ^ n := by positivity
    have : ε ^ 6 * lam ^ n = (ε * lam ^ n) * ε ^ 5 := by ring
    rw [show (lam : ℝ) ^ (3 : ℤ) = lam ^ 3 by norm_cast, this]
    exact mul_le_mul_of_nonneg_left h5 (by positivity)
  have hB : ε ^ 2 * lam ^ (n - 1) ≤ ε * lam ^ n := by
    rw [hpow]
    have : ε ^ 2 * lam ^ (n - 1) = ε * (ε * lam ^ (n - 1)) := by ring
    rw [this]
    gcongr
  calc ‖fderiv ℝ (Trace.scoreFunctional T A) θ + fderiv ℝ (Trace.Proof.zeroFunctional T A) θ‖
      ≤ ‖fderiv ℝ (Trace.scoreFunctional T A) θ‖ + ‖fderiv ℝ (Trace.Proof.zeroFunctional T A) θ‖ :=
        norm_add_le _ _
    _ ≤ Cs * ε ^ 6 * lam ^ ((n : ℤ) - 3) + Cz * (ε * lam ^ n + ε ^ 2 * lam ^ (n - 1)) :=
        add_le_add hbS hbZ
    _ ≤ Cs * (ε * lam ^ n) + Cz * (ε * lam ^ n + ε * lam ^ n) := by
        rw [mul_assoc Cs]; gcongr
    _ = (Cs + 2 * Cz) * ε * lam ^ n := by ring

end
