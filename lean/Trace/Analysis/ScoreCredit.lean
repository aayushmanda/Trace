import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Meta.ModelClosure
import Mathlib.Logic.Function.DependsOn

section
open scoped Matrix

/-- Proof-side owner for `Trace.unseen_letters_score_credit`; its statement is fixed by the proof charter. -/
theorem Trace.Proof.unseenLettersScoreCredit (hprior : Trace.Prior)
    (a0 cemb : ℝ) (hcemb : 0 < cemb) (K : ℕ) [NeZero K] (hK : 2 ≤ K) (nH : ℕ) :
    ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
      ∀ n : ℕ, 2 ≤ n →
      ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Trace.Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (A : Trace.Arch nH n) (θt : Trace.Params Sym K A) (Nb : ℝ), Trace.InitHyp θt a0 cemb Nb →
      ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
      ∀ θ ∈ Trace.initBall θt ε cemb,
        ε ^ 2 ≤ c * Trace.mixrate K T.p →
          DifferentiableAt ℝ (Trace.scoreFunctional T A) θ ∧
            Trace.scoreCredit T A θ ≤ C * ε ^ 6 * Trace.mixrate K T.p ^ ((n : ℤ) - 3) := by
  sorry

end
