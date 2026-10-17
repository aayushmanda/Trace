import Trace.Model.Prelude
import Trace.Model.Prior
import Trace.Meta.ModelClosure
import Mathlib.Logic.Function.DependsOn
import Trace.Analysis.Contraction
import Trace.Analysis.ScoreCredit
import Trace.Analysis.OutcomeCredit

/-!
# Unseen letters erase outcome credit (`thm:main`, Paper/body.tex:191-199)

Under ass:arch, ass:init and ass:task, with the start count and the query fixed:

* **(i)** for every set `S` of letters and every `H` that depends only on the letters in
  `S`, `‖E[Hᵀ(y − u)]‖ ≤ λ^{n−|S|} ‖H‖_{L²}`
  (`unseen_letters_contraction`);
* **(ii)** for every `n ≥ 2` and `θ ∈ B_ε`: if `ε² ≤ cλ`, the score-mediated credit is at most
  `C ε⁶ λ^{n−3}` (`unseen_letters_score_credit`); if `ε ≤ cλ`, the whole outcome credit is at
  most `C ε λ^n` (`unseen_letters_outcome_credit`).

Departures from the literal text, each one a repair rather than a weakening:

* `0 < ε` is a hypothesis of (ii). `thm:main` omits it, but `lem:score-sum`, which its proof
  invokes, assumes it; at `ε = 0` the point `θ₀ = 0` has every MLP pre-activation exactly `0`,
  where ReLU is not differentiable, and `λ^{n−3}` at `n = 2` needs `λ > 0`, which `ε > 0` and
  `ε² ≤ cλ` supply.
* `λ^{n−3}` is an integer power `mixrate K T.p ^ ((n : ℤ) - 3)`. Natural subtraction would turn
  it into `λ⁰ = 1` at `n = 2, 3`, a different claim.
* `DifferentiableAt` is asserted alongside each credit bound, so that a bound on
  Mathlib's `fderiv` (which is `0` off the differentiable locus) cannot hold vacuously.
* The paper's "square-integrable" hypothesis in (i) is dropped: the alphabet is finite,
  so every `H` is square-integrable.

Constants. `c₀` (the threshold `ε₀ = c₀ min(1, 1/N_b)` of ass:init), `c` and `C` are chosen
after `a0`, `cemb`, `K` and the number of heads `nH` only, and before `n`, the widths, the masks,
the alphabet, `μ`, `p`, `s₀`, `w⋆`, `N_b`, `θ̃`, `ε` and `θ` (Paper/appendix.tex:60-61, 227).
(ii-a) and (ii-b) each carry their own constants; taking `c₀, c` to be the minima and `C` the
maximum of the two triples gives one triple serving both, since shrinking `c₀`, `c` and growing
`C` only weakens each hypothesis-conclusion pair, so the split loses nothing.
-/

universe u v

namespace Trace

open Matrix

/-- **Part (i): conditional contraction** (eq:credit-degree; lem:target-contraction,
Paper/appendix.tex:264-290). A feature that reads only the letters in `S` has credit at most
`λ^{n−|S|}` times its `L²` norm. Architecture-free. -/
theorem unseen_letters_contraction (hprior : Prior)
    {K : ℕ} [NeZero K] (hK : 2 ≤ K)
    {Sym : Type u} [Fintype Sym] [DecidableEq Sym] (T : Task K Sym)
    (hp0 : 0 < T.p) (hp1 : T.p < 1)
    (n : ℕ) (S : Finset (Fin n)) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) (hH : DependsOn H (S : Set (Fin n))) :
    featureCredit T H ≤ mixrate K T.p ^ (n - S.card) * T.l2Norm H := by
    exact Trace.Proof.unseenLettersContraction hprior (K := K) hK (Sym := Sym) T hp0 hp1 n S (ι := ι) H hH

/-- **Part (ii-a): score-mediated credit** (lem:score-sum, Paper/appendix.tex:494-499). -/
theorem unseen_letters_score_credit (hprior : Prior)
    (a0 cemb : ℝ) (hcemb : 0 < cemb) (K : ℕ) [NeZero K] (hK : 2 ≤ K) (nH : ℕ) :
    ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
      ∀ n : ℕ, 2 ≤ n →
      ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (A : Arch nH n) (θt : Params Sym K A) (Nb : ℝ), InitHyp θt a0 cemb Nb →
      ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
      ∀ θ ∈ initBall θt ε cemb,
        ε ^ 2 ≤ c * mixrate K T.p →
          DifferentiableAt ℝ (scoreFunctional T A) θ ∧
            scoreCredit T A θ ≤ C * ε ^ 6 * mixrate K T.p ^ ((n : ℤ) - 3) := by
    exact Trace.Proof.unseenLettersScoreCredit hprior a0 cemb hcemb K hK nH

/-- **Part (ii-b): outcome credit** (eq:thm1-full, Paper/appendix.tex:502-505). -/
theorem unseen_letters_outcome_credit (hprior : Prior)
    (a0 cemb : ℝ) (hcemb : 0 < cemb) (K : ℕ) [NeZero K] (hK : 2 ≤ K) (nH : ℕ) :
    ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
      ∀ n : ℕ, 2 ≤ n →
      ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (A : Arch nH n) (θt : Params Sym K A) (Nb : ℝ), InitHyp θt a0 cemb Nb →
      ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
      ∀ θ ∈ initBall θt ε cemb,
        ε ≤ c * mixrate K T.p →
          DifferentiableAt ℝ (outcomeFunctional T A) θ ∧
            outcomeCredit T A θ ≤ C * ε * mixrate K T.p ^ n := by
    exact Trace.Proof.unseenLettersOutcomeCredit hprior a0 cemb hcemb K hK nH

/-- **Unseen letters erase outcome credit** (`thm:main`): parts (i), (ii-a) and (ii-b). -/
theorem unseen_letters_erase_outcome_credit (hprior : Prior) :
    (∀ {K : ℕ} [NeZero K], 2 ≤ K →
      ∀ {Sym : Type u} [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
        0 < T.p → T.p < 1 →
      ∀ (n : ℕ) (S : Finset (Fin n)) {ι : Type v} [Fintype ι]
        (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ), DependsOn H (S : Set (Fin n)) →
        featureCredit T H ≤ mixrate K T.p ^ (n - S.card) * T.l2Norm H) ∧
    (∀ (a0 cemb : ℝ), 0 < cemb → ∀ (K : ℕ) [NeZero K], 2 ≤ K → ∀ nH : ℕ,
      ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
        ∀ n : ℕ, 2 ≤ n →
        ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
          0 < T.p → T.p < 1 →
        ∀ (A : Arch nH n) (θt : Params Sym K A) (Nb : ℝ), InitHyp θt a0 cemb Nb →
        ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
        ∀ θ ∈ initBall θt ε cemb,
          ε ^ 2 ≤ c * mixrate K T.p →
            DifferentiableAt ℝ (scoreFunctional T A) θ ∧
              scoreCredit T A θ ≤ C * ε ^ 6 * mixrate K T.p ^ ((n : ℤ) - 3)) ∧
    (∀ (a0 cemb : ℝ), 0 < cemb → ∀ (K : ℕ) [NeZero K], 2 ≤ K → ∀ nH : ℕ,
      ∃ c₀ c C : ℝ, 0 < c₀ ∧ 0 < c ∧ 0 < C ∧
        ∀ n : ℕ, 2 ≤ n →
        ∀ (Sym : Type u) [Fintype Sym] [DecidableEq Sym] (T : Task K Sym),
          0 < T.p → T.p < 1 →
        ∀ (A : Arch nH n) (θt : Params Sym K A) (Nb : ℝ), InitHyp θt a0 cemb Nb →
        ∀ ε : ℝ, 0 < ε → ε ≤ c₀ * min 1 (1 / Nb) →
        ∀ θ ∈ initBall θt ε cemb,
          ε ≤ c * mixrate K T.p →
            DifferentiableAt ℝ (outcomeFunctional T A) θ ∧
              outcomeCredit T A θ ≤ C * ε * mixrate K T.p ^ n) :=
  ⟨@unseen_letters_contraction hprior, @unseen_letters_score_credit hprior,
    @unseen_letters_outcome_credit hprior⟩

end Trace

#modelClosureOfType Trace.unseen_letters_erase_outcome_credit
#print axioms Trace.unseen_letters_erase_outcome_credit

#surplusIn Trace.Model from Trace.unseen_letters_erase_outcome_credit
