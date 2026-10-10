import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Calculus.FDeriv.Defs
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Data.ZMod.Defs

/-!
# Counting modulo `K`, the two-block Transformer, and its outcome credit

The vocabulary of `thm:main` (Paper/body.tex:191-199) and the quantities it bounds.

## Encodings chosen, and why

* **Finite alphabet, expectation as a finite weighted sum.** The paper never says the
  alphabet is finite; learned token embeddings presuppose it, so `Sym` carries
  `[Fintype Sym]`. The letter law is `n` i.i.d. draws from `μ : PMF Sym`, and
  `Task.Ex f = ∑ w, (∏ t, μ (w t)) • f w`. This fills a gap the paper leaves implicit,
  and it is exactly what makes the paper's "square-integrable" hypothesis in part (i)
  automatic, which is why part (i) is stated without it. Independence of the letters
  — the whole of `lem:target-contraction` — is the product form of `Task.letterWeight`,
  so the conditioning argument is a Fubini split of a finite sum over `S` and `Sᶜ`.
* **Start count and query are fixed.** `Task.s0` and `Task.wstar` are plain fields,
  and `Task.Ex` sums over the letters `w` only. This fixedness is load-bearing for the
  independence step of `lem:target-contraction` (Paper/body.tex:163, appendix.tex:227)
  and must not change in later refactors.
* **"Depends only on the letters in `S`"** is Mathlib's `DependsOn H ↑S`; on a finite
  product space this is `σ(w_t : t ∈ S)`-measurability.
* **Parameters** form a `EuclideanSpace ℝ ParamIdx`, so `‖θ‖` is the Euclidean norm of
  the vectorised parameters (Frobenius on each block), as rem:norms
  (Paper/appendix.tex:76-81) requires. The relative position bias `β` is fixed at `0`
  for `thm:main` and excluded from `θ` and from every gradient; it is therefore simply
  absent here. Parameters at positions past the answer position and readout rows of
  non-value tokens never reach the answer logits, so they are omitted too.
* **The forward pass** is transcribed from the display at Paper/appendix.tex:95-96, not
  from a free reading of ass:arch: per head `W_Q, W_K ∈ ℝ^{d_h×d_res}` and a single
  `W_V ∈ ℝ^{d_res×d_res}` with no separate output projection, scores scaled by
  `1/√d_h`, a residual ReLU MLP `W_out ReLU(W_in h + b_in)` with no output bias, no
  LayerNorm, and a linear readout. This specific formula is the one whose power
  counting `prop:score-degree` performs.
* **The score scale `ζ`** is an explicit real argument of `logits`. It multiplies every
  attention score in *both* blocks of one composed forward pass, so block 2 reads the
  residual stream of the `ζ`-scaled block 1 (Paper/appendix.tex:298-306,
  direct_proofs/conditional_outcome.tex:29). `z_θ = logits … 1 θ` and
  `z^{[0]}_θ = logits … 0 θ`. Having `ζ` as a real variable is what lets the
  `ζ`-Taylor expansion `eq:score-split` be taken literally.
* **Submasks.** Each `(block, head, position)` has a fixed visible set; `Arch` demands
  it be causal and nonempty. Nonemptiness is what makes every softmax denominator a
  nonempty sum of exponentials, hence positive: an `Arch` with an empty mask cannot be
  built, so no instance of the theorem silently uses Lean's `x / 0 = 0`. Ordinary
  attention is the full prefix.
* **Credits are gradient norms.** `outcomeCredit` and `scoreCredit` are operator norms of
  Mathlib `fderiv`s of the credit functionals; on a Euclidean parameter space this is
  the Euclidean norm of the gradient `E[Jᵀ(y − u)]`. Because `fderiv` is `0` where a
  map is not differentiable, the theorem also asserts `DifferentiableAt`.
-/

universe u v

namespace Trace

open Matrix

/-! ## Vocabulary -/

/-- **The counting task** (def:task, Paper/body.tex:78; ass:task, appendix.tex:55): states
`ZMod K`, a finite alphabet `Sym` with letter law `μ`, the query letter `w⋆`, and the start
count `s₀`. The query and the start count are fixed, not random. -/
structure Task (K : ℕ) (Sym : Type u) where
  /-- The law of one letter; the `n` letters are i.i.d. from it. -/
  μ : PMF Sym
  /-- The query letter `w⋆`. -/
  wstar : Sym
  /-- The start count `s₀`, fixed. -/
  s0 : ZMod K

namespace Task

variable {K : ℕ} {Sym : Type u} [Fintype Sym] [DecidableEq Sym] (T : Task K Sym) {n : ℕ}

/-- `p = μ(w⋆)`, the probability that a letter is the query. -/
noncomputable def p : ℝ := (T.μ T.wstar).toReal

/-- The answer `s_n = s₀ + ∑_t b_t (mod K)`, with `b_t = 1[w_t = w⋆]`. -/
def answer (w : Fin n → Sym) : ZMod K :=
  T.s0 + ∑ t, if w t = T.wstar then 1 else 0

/-- Probability of the letter sequence `w` under `n` i.i.d. draws from `μ`. -/
noncomputable def letterWeight (w : Fin n → Sym) : ℝ :=
  ∏ t, (T.μ (w t)).toReal

/-- Expectation over the letters only (start count and query fixed). -/
noncomputable def Ex {E : Type v} [AddCommGroup E] [Module ℝ E]
    (f : (Fin n → Sym) → E) : E :=
  ∑ w, T.letterWeight w • f w

/-- The outcome signal `G = y − u`: the one-hot answer `y = e_{s_n}` minus the uniform
vector `u = K⁻¹ 𝟙`. -/
noncomputable def targetSignal (w : Fin n → Sym) : ZMod K → ℝ :=
  fun a => (if a = T.answer w then 1 else 0) - 1 / (K : ℝ)

/-- `‖H‖_{L²} = (E‖H‖_F²)^{1/2}` for a feature `H` with `K` rows (lem:target-contraction). -/
noncomputable def l2Norm [NeZero K] {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) : ℝ :=
  Real.sqrt (T.Ex fun w => ∑ a, ∑ i, H w a i ^ 2)

end Task

/-- **The contraction coefficient** `λ = |1 − p + p e^{2πi/K}|` (def:lambda,
Paper/body.tex:89). The paper's other form `max_{1≤j<K} |1 − p + p e^{2πij/K}|` is a
proof obligation, not a definition. -/
noncomputable def mixrate (K : ℕ) (p : ℝ) : ℝ :=
  ‖(1 - p : ℂ) + p * Complex.exp (2 * Real.pi * Complex.I / K)‖

/-- The vocabulary: letters, value tokens `ZMod K`, and the three specials
`SEP = 0`, `":" = 1`, `EOS = 2`. -/
abbrev Token (Sym : Type u) (K : ℕ) := Sym ⊕ ZMod K ⊕ Fin 3

/-- The token of letter `a`. -/
def Token.letter {Sym : Type u} {K : ℕ} (a : Sym) : Token Sym K := Sum.inl a

/-- The token of count value `v`. -/
def Token.value {Sym : Type u} {K : ℕ} (v : ZMod K) : Token Sym K := Sum.inr (Sum.inl v)

/-- The separator `SEP`. -/
def Token.sep {Sym : Type u} {K : ℕ} : Token Sym K := Sum.inr (Sum.inr 0)

/-- The answer marker `":"`. -/
def Token.colon {Sym : Type u} {K : ℕ} : Token Sym K := Sum.inr (Sum.inr 1)

/-- **The outcome prompt** (Paper/appendix.tex:107-115), positions `0, …, n+3`:
`s₀@0, w⋆@1, w_t@(1+t), SEP@(n+2), ":"@(n+3)`. Position `n+3` predicts `s_n`. Only the
letter positions depend on `w`. -/
def Task.outcomePrompt {K : ℕ} {Sym : Type u} (T : Task K Sym) {n : ℕ}
    (w : Fin n → Sym) (i : Fin (n + 4)) : Token Sym K :=
  if i.val = 0 then Token.value T.s0
  else if i.val = 1 then Token.letter T.wstar
  else if h : i.val - 2 < n then Token.letter (w ⟨i.val - 2, h⟩)
  else if i.val = n + 2 then Token.sep
  else Token.colon

/-- **Architecture shape** (ass:arch, Paper/appendix.tex:34-41): `nH` heads in each of the two
blocks, residual width `dres`, head width `dh`, MLP width `dmlp`, and for each block, head and
position a fixed causal visible set (the full prefix for ordinary attention). Sequences have
the `n + 4` positions of the outcome prompt. -/
structure Arch (nH n : ℕ) where
  /-- Residual width `d_res`. -/
  dres : ℕ
  /-- Head width `d_h`. -/
  dh : ℕ
  /-- MLP width `w`. -/
  dmlp : ℕ
  /-- The positions head `h` of block `l` attends to from position `i`. -/
  vis : Fin 2 → Fin nH → Fin (n + 4) → Finset (Fin (n + 4))
  /-- Masks are causal. -/
  vis_causal : ∀ l h i, ∀ j ∈ vis l h i, j ≤ i
  /-- Masks are nonempty, so every softmax denominator is a nonempty sum of exponentials. -/
  vis_nonempty : ∀ l h i, (vis l h i).Nonempty

/-- Index of one scalar parameter, block by block:
token embedding `E_tok` (`d_res × vocabulary`), position embedding `E_pos` (`d_res × (n+4)`),
per block and head `W_Q`, `W_K` (`d_h × d_res`) and `W_V` (`d_res × d_res`), per block
`W_in` (`w × d_res`), `b_in` (`w`) and `W_out` (`d_res × w`), and the readout `V`
(`K × d_res`, the value-token rows). The relative bias `β` is not a parameter here. -/
abbrev ParamIdx (Sym : Type u) (K : ℕ) {nH n : ℕ} (A : Arch nH n) :=
  (Fin A.dres × Token Sym K)
  ⊕ (Fin A.dres × Fin (n + 4))
  ⊕ (Fin 2 × Fin nH × Fin A.dh × Fin A.dres)
  ⊕ (Fin 2 × Fin nH × Fin A.dh × Fin A.dres)
  ⊕ (Fin 2 × Fin nH × Fin A.dres × Fin A.dres)
  ⊕ (Fin 2 × Fin A.dmlp × Fin A.dres)
  ⊕ (Fin 2 × Fin A.dmlp)
  ⊕ (Fin 2 × Fin A.dres × Fin A.dmlp)
  ⊕ (ZMod K × Fin A.dres)

/-- **Parameters** `θ` (all but the relative bias), with the Euclidean norm of the
vectorisation (rem:norms). -/
abbrev Params (Sym : Type u) (K : ℕ) {nH n : ℕ} (A : Arch nH n) :=
  EuclideanSpace ℝ (ParamIdx Sym K A)

namespace Params

variable {Sym : Type u} {K nH n : ℕ} {A : Arch nH n} (θ : Params Sym K A)

/-- Token embedding `E_tok`, one column per token. -/
def tokEmb : Matrix (Fin A.dres) (Token Sym K) ℝ :=
  fun d v => θ (Sum.inl (d, v))

/-- Position embedding `E_pos`, one column per position. -/
def posEmb : Matrix (Fin A.dres) (Fin (n + 4)) ℝ :=
  fun d i => θ (Sum.inr (Sum.inl (d, i)))

/-- Query matrix `W_Q` of head `h` in block `l`. -/
def wQ (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dh) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inl (l, h, a, b))))

/-- Key matrix `W_K` of head `h` in block `l`. -/
def wK (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dh) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, a, b)))))

/-- Value matrix `W_V` of head `h` in block `l` (there is no separate output projection). -/
def wV (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dres) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, a, b))))))

/-- MLP input matrix `W_in` of block `l`. -/
def wIn (l : Fin 2) : Matrix (Fin A.dmlp) (Fin A.dres) ℝ :=
  fun m d => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m, d)))))))

/-- MLP bias `b_in` of block `l`. -/
def bIn (l : Fin 2) : Fin A.dmlp → ℝ :=
  fun m => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m))))))))

/-- MLP output matrix `W_out` of block `l` (there is no output bias). -/
def wOut (l : Fin 2) : Matrix (Fin A.dres) (Fin A.dmlp) ℝ :=
  fun d m =>
    θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, d, m)))))))))

/-- Readout `V`: the value-token (state) rows. -/
def readout : Matrix (ZMod K) (Fin A.dres) ℝ :=
  fun v d =>
    θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (v, d)))))))))

/-- The value-token embeddings `E_val`, the columns of `E_tok` at the `K` value tokens. -/
def valueEmb : Matrix (Fin A.dres) (ZMod K) ℝ :=
  fun d v => θ.tokEmb d (Token.value v)

end Params

section Forward

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]

/-- Input residual stream `x_i = E_tok(token_i) + E_pos(i)` of the outcome prompt. -/
def embed (T : Task K Sym) {A : Arch nH n} (θ : Params Sym K A) (w : Fin n → Sym) :
    Fin (n + 4) → Fin A.dres → ℝ :=
  fun i d => θ.tokEmb d (T.outcomePrompt w i) + θ.posEmb d i

/-- Attention score `ψ_{ij} = (1/√d_h) ⟨W_Q x_i, W_K x_j⟩` of head `h` in block `l`. -/
noncomputable def score {A : Arch nH n} (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) : ℝ :=
  (Real.sqrt A.dh)⁻¹ * ((θ.wQ l h *ᵥ x i) ⬝ᵥ (θ.wK l h *ᵥ x j))

/-- Attention weight `ω_{ij} = softmax_{j ∈ vis(i)}(ζ ψ_{ij})`, with every score scaled by `ζ`.
The denominator runs over the visible set `A.vis l h i`, nonempty by `Arch.vis_nonempty`. -/
noncomputable def attnWeight {A : Arch nH n} (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2)
    (h : Fin nH) (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) : ℝ :=
  Real.exp (ζ * score θ l h x i j) / ∑ k ∈ A.vis l h i, Real.exp (ζ * score θ l h x i k)

/-- One Transformer block `l` with score scale `ζ` (Paper/appendix.tex:95-96):
`a_i = x_i + ∑_h ∑_{j ∈ vis(i)} ω_{ij} W_V x_j`, then `a_i + W_out ReLU(W_in a_i + b_in)`. -/
noncomputable def block {A : Arch nH n} (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2)
    (x : Fin (n + 4) → Fin A.dres → ℝ) : Fin (n + 4) → Fin A.dres → ℝ :=
  let a : Fin (n + 4) → Fin A.dres → ℝ := fun i =>
    x i + ∑ h, ∑ j ∈ A.vis l h i, attnWeight ζ θ l h x i j • (θ.wV l h *ᵥ x j)
  fun i => a i + θ.wOut l *ᵥ fun m => max ((θ.wIn l *ᵥ a i) m + θ.bIn l m) 0

/-- **State logits at the answer position** `n + 3` of the outcome prompt, for the network
whose attention scores are all scaled by `ζ` in both blocks of one composed forward pass.
`z_θ = logits T A 1 θ`; `z^{[0]}_θ = logits T A 0 θ` (uniform attention over each visible set,
composed through both blocks). -/
noncomputable def logits (T : Task K Sym) (A : Arch nH n) (ζ : ℝ) (θ : Params Sym K A)
    (w : Fin n → Sym) : ZMod K → ℝ :=
  θ.readout *ᵥ block ζ θ 1 (block ζ θ 0 (embed T θ w)) (Fin.last (n + 3))

end Forward

/-- The `ℓ²` operator norm of a real matrix, `‖M‖ = sup_{‖x‖ ≤ 1} ‖M x‖`
(definitionally Mathlib's `Matrix.l2_opNorm_def` under `Matrix.Norms.L2Operator`). -/
noncomputable def opNorm {m k : Type*} [Fintype m] [Fintype k] [DecidableEq k]
    (M : Matrix m k ℝ) : ℝ :=
  ‖LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin M)‖

/-- **Assumption A2 on the unscaled initialisation `θ̃`** (ass:init, Paper/appendix.tex:43-52):
every weight matrix and embedding block has operator norm at most `a0`; the value-token
embeddings have smallest singular value at least `cemb` (`‖E_val x‖ ≥ cemb ‖x‖`); every MLP bias
coordinate has `|b̃_m| ≥ 1` and every MLP bias vector has norm at most `Nb`. -/
structure InitHyp {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
    {A : Arch nH n} (θt : Params Sym K A) (a0 cemb Nb : ℝ) : Prop where
  tokEmb_le : opNorm θt.tokEmb ≤ a0
  posEmb_le : opNorm θt.posEmb ≤ a0
  wQ_le : ∀ l h, opNorm (θt.wQ l h) ≤ a0
  wK_le : ∀ l h, opNorm (θt.wK l h) ≤ a0
  wV_le : ∀ l h, opNorm (θt.wV l h) ≤ a0
  wIn_le : ∀ l, opNorm (θt.wIn l) ≤ a0
  wOut_le : ∀ l, opNorm (θt.wOut l) ≤ a0
  readout_le : opNorm θt.readout ≤ a0
  valueEmb_sigmaMin : ∀ x : EuclideanSpace ℝ (ZMod K),
    cemb * ‖x‖ ≤ ‖Matrix.toEuclideanLin θt.valueEmb x‖
  bIn_ge_one : ∀ l m, 1 ≤ |θt.bIn l m|
  bIn_norm_le : ∀ l, ‖(WithLp.toLp 2 (θt.bIn l) : EuclideanSpace ℝ (Fin A.dmlp))‖ ≤ Nb

/-- **The ball `B_ε`** around `θ₀ = ε θ̃`, of radius `r₀ ε` with `r₀ = min(1/8, cemb/4)`
(ass:init); `β = 0` is built in because `β` is not a parameter. -/
def initBall {Sym : Type u} [Fintype Sym] {K nH n : ℕ} [NeZero K] {A : Arch nH n}
    (θt : Params Sym K A)
    (ε cemb : ℝ) : Set (Params Sym K A) :=
  Metric.closedBall (ε • θt) (min (1 / 8) (cemb / 4) * ε)

/-! ## The quantity -/

section Quantity

variable {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]

/-- **Credit of a feature** `H` with `K` rows: `‖E[Hᵀ(y − u)]‖`, the Euclidean norm
(the left side of eq:credit-degree, Paper/body.tex:178). -/
noncomputable def featureCredit (T : Task K Sym) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) : ℝ :=
  ‖(WithLp.toLp 2 (T.Ex fun w => (H w)ᵀ *ᵥ T.targetSignal w) : EuclideanSpace ℝ ι)‖

/-- The outcome credit functional `Φ_out(θ) = E⟨z_θ, y − u⟩` (def:credit, def:losses). -/
noncomputable def outcomeFunctional (T : Task K Sym) (A : Arch nH n)
    (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => logits T A 1 θ w ⬝ᵥ T.targetSignal w

/-- The score-mediated credit functional `E⟨z_θ − z^{[0]}_θ, y − u⟩`, whose gradient is
`E[J_scoreᵀ(y − u)]` with `J_score = J_z − J^{[0]}` (Paper/appendix.tex:26-30). -/
noncomputable def scoreFunctional (T : Task K Sym) (A : Arch nH n)
    (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => (logits T A 1 θ w - logits T A 0 θ w) ⬝ᵥ T.targetSignal w

/-- **Outcome credit** `‖∇_θ Φ_out(θ)‖ = ‖E[J_zᵀ(y − u)]‖`. -/
noncomputable def outcomeCredit (T : Task K Sym) (A : Arch nH n) (θ : Params Sym K A) : ℝ :=
  ‖fderiv ℝ (outcomeFunctional T A) θ‖

/-- **Score-mediated credit** `‖E[J_scoreᵀ(y − u)]‖`. -/
noncomputable def scoreCredit (T : Task K Sym) (A : Arch nH n) (θ : Params Sym K A) : ℝ :=
  ‖fderiv ℝ (scoreFunctional T A) θ‖

end Quantity

end Trace
