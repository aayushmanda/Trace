# Formalization blueprint — Unseen letters erase outcome credit

> Generated 2026-10-10T23:13:20.948Z by Claude subscription · claude-opus-5-5 from `Paper/body.tex:191` (`\begin{theorem}`).
> This is a **plan**, not a proof. Every Lean statement below is a proposal to be checked against the paper line by line before anyone starts proving it.

## What is being claimed

The 4 statements this development exists to prove, as `lean/Trace/Model/Theorem.lean` states them. Read these against the paper before anything else here: every section below inherits whatever is wrong in them, and a plan for the wrong statement looks exactly like a plan for the right one.

### `Trace.unseen_letters_contraction`

**Part (i): conditional contraction** (eq:credit-degree; lem:target-contraction, Paper/appendix.tex:264-290). A feature that reads only the letters in `S` has credit at most `λ^{n−|S|}` times its `L²` norm. Architecture-free.

```lean
theorem unseen_letters_contraction (hprior : Prior)
    {K : ℕ} [NeZero K] (hK : 2 ≤ K)
    {Sym : Type u} [Fintype Sym] [DecidableEq Sym] (T : Task K Sym)
    (hp0 : 0 < T.p) (hp1 : T.p < 1)
    (n : ℕ) (S : Finset (Fin n)) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) (hH : DependsOn H (S : Set (Fin n))) :
    featureCredit T H ≤ mixrate K T.p ^ (n - S.card) * T.l2Norm H
```

### `Trace.unseen_letters_score_credit`

**Part (ii-a): score-mediated credit** (lem:score-sum, Paper/appendix.tex:494-499).

```lean
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
            scoreCredit T A θ ≤ C * ε ^ 6 * mixrate K T.p ^ ((n : ℤ) - 3)
```

### `Trace.unseen_letters_outcome_credit`

**Part (ii-b): outcome credit** (eq:thm1-full, Paper/appendix.tex:502-505).

```lean
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
            outcomeCredit T A θ ≤ C * ε * mixrate K T.p ^ n
```

### `Trace.unseen_letters_erase_outcome_credit`

**Unseen letters erase outcome credit** (`thm:main`): parts (i), (ii-a) and (ii-b).

```lean
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
              outcomeCredit T A θ ≤ C * ε * mixrate K T.p ^ n)
```

#### The definitions these are stated in

Quoted because a statement cannot be checked against a paper without them: a
bound written as a named quantity is a bound nobody has read yet.

From `lean/Trace/Model/Prelude.lean`:

```lean
structure Task (K : ℕ) (Sym : Type u) where

noncomputable def p : ℝ := (T.μ T.wstar).toReal

def answer (w : Fin n → Sym) : ZMod K :=
  T.s0 + ∑ t, if w t = T.wstar then 1 else 0

noncomputable def letterWeight (w : Fin n → Sym) : ℝ :=
  ∏ t, (T.μ (w t)).toReal

noncomputable def Ex {E : Type v} [AddCommGroup E] [Module ℝ E]
    (f : (Fin n → Sym) → E) : E :=
  ∑ w, T.letterWeight w • f w

noncomputable def targetSignal (w : Fin n → Sym) : ZMod K → ℝ :=
  fun a => (if a = T.answer w then 1 else 0) - 1 / (K : ℝ)

noncomputable def l2Norm [NeZero K] {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) : ℝ :=
  Real.sqrt (T.Ex fun w => ∑ a, ∑ i, H w a i ^ 2)

noncomputable def mixrate (K : ℕ) (p : ℝ) : ℝ :=
  ‖(1 - p : ℂ) + p * Complex.exp (2 * Real.pi * Complex.I / K)‖

abbrev Token (Sym : Type u) (K : ℕ) := Sym ⊕ ZMod K ⊕ Fin 3

structure Arch (nH n : ℕ) where

abbrev ParamIdx (Sym : Type u) (K : ℕ) {nH n : ℕ} (A : Arch nH n) :=
  (Fin A.dres × Token Sym K)
  ⊕ (Fin A.dres × Fin (n + 4))
  ⊕ (Fin 2 × Fin nH × Fin A.dh × Fin A.dres)
  ⊕ (Fin 2 × Fin nH × Fin A.dh × Fin A.dres)
  ⊕ (Fin 2 × Fin nH × Fin A.dres × Fin A.dres)
  ⊕ (Fin 2 × Fin A.dmlp × Fin A.dres)
  ⊕ (Fin 2 × Fin A.dmlp)

abbrev Params (Sym : Type u) (K : ℕ) {nH n : ℕ} (A : Arch nH n) :=
  EuclideanSpace ℝ (ParamIdx Sym K A)

def tokEmb : Matrix (Fin A.dres) (Token Sym K) ℝ :=
  fun d v => θ (Sum.inl (d, v))

def posEmb : Matrix (Fin A.dres) (Fin (n + 4)) ℝ :=
  fun d i => θ (Sum.inr (Sum.inl (d, i)))

def wQ (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dh) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inl (l, h, a, b))))

def wK (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dh) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, a, b)))))

def wV (l : Fin 2) (h : Fin nH) : Matrix (Fin A.dres) (Fin A.dres) ℝ :=
  fun a b => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, h, a, b))))))

def wIn (l : Fin 2) : Matrix (Fin A.dmlp) (Fin A.dres) ℝ :=
  fun m d => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m, d)))))))

def bIn (l : Fin 2) : Fin A.dmlp → ℝ :=
  fun m => θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, m))))))))

def wOut (l : Fin 2) : Matrix (Fin A.dres) (Fin A.dmlp) ℝ :=
  fun d m =>
    θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl (l, d, m)))))))))

def readout : Matrix (ZMod K) (Fin A.dres) ℝ :=
  fun v d =>
    θ (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (v, d)))))))))

def embed (T : Task K Sym) {A : Arch nH n} (θ : Params Sym K A) (w : Fin n → Sym) :
    Fin (n + 4) → Fin A.dres → ℝ :=
  fun i d => θ.tokEmb d (T.outcomePrompt w i) + θ.posEmb d i

noncomputable def score {A : Arch nH n} (θ : Params Sym K A) (l : Fin 2) (h : Fin nH)
    (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) : ℝ :=
  (Real.sqrt A.dh)⁻¹ * ((θ.wQ l h *ᵥ x i) ⬝ᵥ (θ.wK l h *ᵥ x j))

noncomputable def attnWeight {A : Arch nH n} (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2)
    (h : Fin nH) (x : Fin (n + 4) → Fin A.dres → ℝ) (i j : Fin (n + 4)) : ℝ :=
  Real.exp (ζ * score θ l h x i j) / ∑ k ∈ A.vis l h i, Real.exp (ζ * score θ l h x i k)

noncomputable def block {A : Arch nH n} (ζ : ℝ) (θ : Params Sym K A) (l : Fin 2)
    (x : Fin (n + 4) → Fin A.dres → ℝ) : Fin (n + 4) → Fin A.dres → ℝ :=
  let a : Fin (n + 4) → Fin A.dres → ℝ := fun i =>
    x i + ∑ h, ∑ j ∈ A.vis l h i, attnWeight ζ θ l h x i j • (θ.wV l h *ᵥ x j)
  fun i => a i + θ.wOut l *ᵥ fun m => max ((θ.wIn l *ᵥ a i) m + θ.bIn l m) 0

noncomputable def logits (T : Task K Sym) (A : Arch nH n) (ζ : ℝ) (θ : Params Sym K A)
    (w : Fin n → Sym) : ZMod K → ℝ :=
  θ.readout *ᵥ block ζ θ 1 (block ζ θ 0 (embed T θ w)) (Fin.last (n + 3))

noncomputable def opNorm {m k : Type*} [Fintype m] [Fintype k] [DecidableEq k]
    (M : Matrix m k ℝ) : ℝ :=
  ‖LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin M)‖

structure InitHyp {Sym : Type u} [Fintype Sym] [DecidableEq Sym] {K nH n : ℕ} [NeZero K]
    {A : Arch nH n} (θt : Params Sym K A) (a0 cemb Nb : ℝ) : Prop where
  tokEmb_le : opNorm θt.tokEmb ≤ a0
  posEmb_le : opNorm θt.posEmb ≤ a0
  wQ_le : ∀ l h, opNorm (θt.wQ l h) ≤ a0
  wK_le : ∀ l h, opNorm (θt.wK l h) ≤ a0
  wV_le : ∀ l h, opNorm (θt.wV l h) ≤ a0
  wIn_le : ∀ l, opNorm (θt.wIn l) ≤ a0

def initBall {Sym : Type u} [Fintype Sym] {K nH n : ℕ} [NeZero K] {A : Arch nH n}
    (θt : Params Sym K A)
    (ε cemb : ℝ) : Set (Params Sym K A) :=
  Metric.closedBall (ε • θt) (min (1 / 8) (cemb / 4) * ε)

noncomputable def featureCredit (T : Task K Sym) {ι : Type v} [Fintype ι]
    (H : (Fin n → Sym) → Matrix (ZMod K) ι ℝ) : ℝ :=
  ‖(WithLp.toLp 2 (T.Ex fun w => (H w)ᵀ *ᵥ T.targetSignal w) : EuclideanSpace ℝ ι)‖

noncomputable def outcomeFunctional (T : Task K Sym) (A : Arch nH n)
    (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => logits T A 1 θ w ⬝ᵥ T.targetSignal w

noncomputable def scoreFunctional (T : Task K Sym) (A : Arch nH n)
    (θ : Params Sym K A) : ℝ :=
  T.Ex fun w => (logits T A 1 θ w - logits T A 0 θ w) ⬝ᵥ T.targetSignal w

noncomputable def outcomeCredit (T : Task K Sym) (A : Arch nH n) (θ : Params Sym K A) : ℝ :=
  ‖fderiv ℝ (outcomeFunctional T A) θ‖

noncomputable def scoreCredit (T : Task K Sym) (A : Arch nH n) (θ : Params Sym K A) : ℝ :=
  ‖fderiv ℝ (scoreFunctional T A) θ‖
```


## What this run produced

| Where | What it holds |
| --- | --- |
| `.blueprint/thm-main-5e5f26eaeb/BLUEPRINT.md` | This file: the whole plan, in reading order. |
| `.blueprint/thm-main-5e5f26eaeb/blueprint.json` | The same plan as data — every pass's output verbatim, including the source excerpt it was built from. |
| — | There are no briefs. This plan is executed **top-down**: agents go at the capstone and state the lemmas they turn out to need, so the work list is whatever still carries `sorry` in `lean/` at the start of each round. |
| `lean/Trace/Model/` | The **audit surface**: 2 files holding everything the headline theorem asserts, and nothing else. |
| `lean/README.md` | How to build the project, and the three checks that decide whether it is actually proved. |

### Read it in this order

1. **[What is being claimed](#what-is-being-claimed)** — the theorem as `lean/Trace/Model/Theorem.lean` states it, with the definitions it is stated in. This is the whole of what the run is trying to prove, quoted rather than described, so it is the first thing to check against the paper.

2. **[The result, restated](#the-result-restated)** — the same statement in words, with a verdict on each hypothesis and what the paper leaves unsaid. Every later section inherits whatever is wrong here, and a plan for the wrong statement looks exactly like a plan for the right one.

3. **The rest of the audit surface** — `lean/Trace/Model/Prelude.lean`. Everything else in the project is machinery serving these. This is the part a reader has to trust, so it is the part to read.

4. **[The crux](#the-crux)** — the 3 steps the proof actually turns on. If the architecture is wrong, it is wrong here, and it is cheaper to find that out before anything else is built.

5. **[Landing order](#landing-order)** — the sequence to prove things in, with what depends on what. Rebuild green after each one.

Sections between the restatement and the crux — the architecture and the reuse map — say what is being taken from Mathlib and arlib rather than proved here, and why the proof is split the way it is. Read them when a decision looks wrong, not before.

## The result, restated

Setting. Fix K ≥ 2, a finite alphabet Σ with a probability distribution μ on it, a query letter w* ∈ Σ with p := μ(w*) ∈ (0,1), a start count s0 ∈ Z_K and a horizon n ≥ 1 (n ≥ 2 in part (ii)). Letters w = (w_1,…,w_n) are i.i.d. μ; b_t = 1[w_t = w*]; the answer is s_n = s0 + Σ_t b_t (mod K); y = e_{s_n} ∈ R^K, u = (1/K)·1, G := y − u. All expectations E are over the letters only (s0 and w* fixed). λ := |1 − p + p·exp(2πi/K)| = max_{1≤j<K} |1 − p + p·exp(2πij/K)|, which lies in [0,1).

Part (i) (architecture-free). For every S ⊆ {1,…,n}, every finite index type ι, and every H : Σ^n → R^{K×ι} that depends only on (w_t)_{t∈S} (i.e. H(w) = H(w') whenever w_t = w'_t for all t ∈ S): ‖E[Hᵀ G]‖₂ ≤ λ^{n−|S|} · (E‖H‖_F²)^{1/2}. (Square-integrability is automatic for a finite alphabet.)

Part (ii). There exist constants c₀, c, C > 0 depending only on a0, c_emb, K and the number of heads per block (NOT on n, the widths d_res, d_h, w, the alphabet Σ, μ, p, s0, w*, the bias-norm bound N_b, the submasks, or ε) such that the following holds. Let θ̃ be any parameter vector of the two-block, no-LayerNorm, causal softmax-attention + ReLU-MLP Transformer of Assumption A1 (learned token and position embeddings, per-head W_Q, W_K ∈ R^{d_h×d_res}, W_V ∈ R^{d_res×d_res}, per-block MLP W_in ∈ R^{w×d_res}, b_in ∈ R^w, W_out ∈ R^{d_res×w}, readout V ∈ R^{K×d_res}; fixed nonempty causal per-head submasks allowed; relative position bias fixed at 0 and excluded from the parameters) such that every weight matrix and embedding block of θ̃ has operator norm ≤ a0, the value-token embeddings have smallest singular value ≥ c_emb > 0, every MLP bias coordinate satisfies |b̃_m| ≥ 1 and each MLP bias vector has Euclidean norm ≤ N_b. Let 0 < ε ≤ c₀·min(1, 1/N_b), θ0 := ε·θ̃, r0 := min(1/8, c_emb/4), and θ any parameter with ‖θ − θ0‖ ≤ r0·ε (Euclidean norm of the vectorized parameters, Frobenius per block). Let z_θ(w) ∈ R^K be the state (value-token) logits at the answer position (the ':' position n+3 of the prompt s0, w*, w_1,…,w_n, SEP, ':'), and z^{[0]}_θ(w) the same logits computed by the network in which every attention score in both blocks is multiplied by ζ = 0 (uniform attention over each head's visible set, composed through both blocks). Define the outcome credit OC(θ) := ‖∇_θ E[⟨z_θ, G⟩]‖ = ‖E[(∂z_θ/∂θ)ᵀ G]‖ and the score-mediated credit SC(θ) := ‖∇_θ E[⟨z_θ − z^{[0]}_θ, G⟩]‖ = ‖E[(∂z_θ/∂θ − ∂z^{[0]}_θ/∂θ)ᵀ G]‖. Then for every n ≥ 2:
  (ii-a) if ε² ≤ c·λ, then SC(θ) ≤ C·ε⁶·λ^{n−3} (an integer power; at n = 2 this is λ^{−1}, well defined because ε > 0 and ε² ≤ cλ force λ > 0);
  (ii-b) if ε ≤ c·λ, then OC(θ) ≤ C·ε·λ^n.

Reading provenance: `reader-agent`

### Canonical target

thm:main parts (i) and (ii) exactly as in informalStatement: (i) the conditional-contraction credit bound ‖E[Hᵀ(y−u)]‖ ≤ λ^{n−\|S\|}‖H‖_{L²} for S-local H; (ii-a) score-mediated credit ≤ Cε⁶λ^{n−3} when ε² ≤ cλ; (ii-b) outcome credit ≤ Cελ^n when ε ≤ cλ; with fixed start count and query, θ ∈ B_ε, β = 0, and constants uniform in n, widths, alphabet, p. (Paper/body.tex:191-199; proof Paper/appendix.tex:530-536 via lem:target-contraction (appendix.tex:264) and lem:score-sum (appendix.tex:494))

Non-goals:

- The bound \|Φ_out(θ)\| ≤ Cε²λ^n of eq:thm1-full (in lem:score-sum, not in thm:main).
- The identity/bound eq:horizon itself is a lemma for (i), not a separate headline claim.
- Averaging over the start state and query (lem:score-sum last sentence).
- Sharpness of the exponent n − \|S\| (rem:tight).
- thm:separation, thm:outcome-end, thm:joint, prop:represent (exact executors), lem:bias-uniform (trained relative bias).
- The claim that fan-in initialization satisfies A2 w.h.p. (appendix.tex:61-65).
- Any training dynamics (ass:gd is not a hypothesis of thm:main).

### Paper evidence

Target: Paper/body.tex:191-199 (thm:main), proof at Paper/appendix.tex:530-536

Conventions read:

- Paper/body.tex:78-96 def:task and def:lambda (λ = \|1 − p + p e^{2πi/K}\|)
- Paper/body.tex:117-128 formats, model summary, constants c, C independent of widths, n, ε
- Paper/body.tex:163-189 'Fix the start count', def:credit, eq:credit-degree, ζ-expansion paragraph defining score-mediated credit
- Paper/appendix.tex:11-30 def:losses, Φ_out = Φ_{y−u}, u = K^{-1}1, J_score = J_z − J^{[0]}
- Paper/appendix.tex:34-65 ass:arch, ass:init, ass:task, constants paragraph (c, C depend only on a0, c_emb, K, numbers of blocks and heads)
- Paper/appendix.tex:76-81 rem:norms: Euclidean norm of vectorized params; β fixed at 0 and excluded from θ in thm:main
- Paper/appendix.tex:89-115 forward-pass display (1/√d_h score scaling, single W_V per head, MLP without output bias) and token/position layout of the outcome prompt (':' at n+3 predicts s_n)
- Paper/appendix.tex:227-230 conventions: query and start fixed; bounds also on the doubled ball

Proof sections read:

- Paper/appendix.tex:237-248 lem:gradient-split
- Paper/appendix.tex:250-293 contraction paragraph, lem:target-contraction and its proof, sharpness remark
- Paper/appendix.tex:295-489 score expansion eq:score-split, prop:score-degree, prop:network and proofs
- Paper/appendix.tex:491-536 lem:score-sum, its proof, the c-choice remark at line 528, proof of thm:main
- Paper/direct_proofs/conditional_outcome.tex:1-107 (alternate, more explicit write-up of the same lemmas; clarifies the ζ-scaled network is the composed network and that remainder/zero-order bounds need ε ≤ 1)

Omitted or missing sources:

- The excerpt omitted def:lambda (λ's definition), def:credit, def:losses, rem:norms (β excluded from θ; Euclidean parameter norm), the forward-pass display and prompt layout, eq:score-split (definition of J^{[0]}), prop:network, and the constants paragraph; all were read from the sources.
- The paper never says whether the alphabet is finite; finiteness is implied by learned token embeddings.
- The paper never states ε > 0 or λ > 0 in thm:main; lem:score-sum assumes both.

### Claims in the headline

| Claim | Can it fail? | Note |
| --- | --- | --- |
| (i) conditional-contraction credit bound for S-local features | no — holds on every reachable outcome | Deterministic inequality about an expectation; no failure probability. |
| (ii-a) score-mediated credit ≤ Cε⁶λ^{n−3} | no — holds on every reachable outcome | Bound on a gradient norm, not on cost. |
| (ii-b) outcome credit ≤ Cελ^n | no — holds on every reachable outcome | Bound on a gradient norm, not on cost. |

Proposed shape, as the restatement pass wrote it:

```lean
Deterministic inequality theorem (no algorithm, no probability monad): (i) ∀ S-local H, ‖E[Hᵀ(y−u)]‖ ≤ λ^{n−|S|}‖H‖_{L²}; (ii) ∃ c₀ c C > 0 uniform in n/widths/alphabet such that ∀ n ≥ 2, ∀ admissible θ̃, ε ∈ (0, c₀ min(1,1/N_b)], θ ∈ B_ε: ε² ≤ cλ → scoreCredit ≤ Cε⁶λ^{n−3}, and ε ≤ cλ → outcomeCredit ≤ Cελ^n. Expectations are finite sums over the letter law.
```

Claim ledger:

- (i) For all S ⊆ [n] and every H depending only on letters in S: ‖E[Hᵀ(y − u)]‖ ≤ λ^{n−\|S\|}‖H‖_{L²}. (correctness, Model/Theorem; Paper/body.tex:195, eq:credit-degree body.tex:178; proof lem:target-contraction appendix.tex:264-290)
- (ii-a) n ≥ 2, ε² ≤ cλ, θ ∈ B_ε ⇒ ‖E[J_scoreᵀ(y − u)]‖ ≤ Cε⁶λ^{n−3}. (other, Model/Theorem; Paper/body.tex:196; lem:score-sum appendix.tex:497-499)
- (ii-b) n ≥ 2, ε ≤ cλ, θ ∈ B_ε ⇒ ‖∇_θ Φ_out(θ)‖ ≤ Cελ^n. (other, Model/Theorem; Paper/body.tex:196-197; eq:thm1-full appendix.tex:502-505)
- Constants c₀, c, C depend only on a0, c_emb, K, number of heads (blocks fixed at 2). (other, Model/Theorem; Paper/appendix.tex:60-61, 227-228; body.tex:127-128)
- eq:horizon: ‖E[G \| F_S]‖² = (1/K)Σ_{j=1}^{K−1}\|ρ_j\|^{2(n−\|S\|)} ≤ (1 − 1/K)λ^{2(n−\|S\|)} (pointwise in the conditioning letters). (correctness, Analysis; Paper/appendix.tex:267-270)
- prop:score-degree: ζ-Taylor decomposition of J_z into J^{[0]} (≤ 1 letter per term), J^{[k]} (≤ 2k+1 letters, total Cε²(C_att ε⁴)^k) and remainder; fixed ReLU pattern on B_ε. (other, Analysis; Paper/appendix.tex:308-319, 325-489)
- Credit equals gradient of the credit functional: ∇_θ E⟨z_θ, ξ⟩ = E[J_zᵀ ξ]. (other, Interface; Paper/body.tex:167-168; appendix.tex:242)

### Modelling choices

- **acceptable** Finite alphabet with a PMF; expectation as a finite weighted sum: Σ : Fintype, μ : PMF Σ; E f = Σ_w ∏_t μ(w_t) f(w). Risk: Part (i) is stated by the paper for general square-integrable H; on a finite space this is the full generality.
- **acceptable** 'depends only on letters in S' formalised as agreement on S implies equal values: DependsOnlyOn S H. Risk: If μ has zero-mass letters, the paper's F_S-measurable class is the same (σ-algebra measurability, not a.s.), so no gap.
- **modeling-note** Credits defined via fderiv of the scalar credit functional: OC = ‖fderiv ℝ (θ ↦ E⟨z_θ, G⟩) θ‖, SC = ‖fderiv ℝ (θ ↦ E⟨z_θ − z^{[0]}_θ, G⟩) θ‖. Risk: Mathlib's fderiv is 0 at non-differentiable points, which would make the bound vacuous there; a proof obligation must show differentiability on B_ε (fixed ReLU pattern) or the theorem should state DifferentiableAt as a conjunct so the bound is not trivially satisfied.
- **acceptable** Relative bias β omitted: No β in Params. Risk: None for this theorem.
- **acceptable** Readout restricted to K value-token rows; sequence truncated at n+4 positions: V : R^{K×d_res}; positions Fin (n+4). Risk: Low.
- **modeling-note** Single W_V per head (no output projection); score scale 1/√d_h; MLP without output bias: Follow appendix.tex:95-96 literally. Risk: A W_O would alter the power counting; handoff should note the chosen architecture.
- **modeling-note** Per-head fixed causal submasks: Arch carries vis sets, required nonempty and ⊆ prefix; constants uniform in them. Risk: Could specialise to full prefix to simplify, at the cost of generality the paper claims; recommend keeping masks.
- **modeling-note** Constants and the threshold c of ε0 are existential: ∀ a0 cemb K nH, ∃ c₀ c C > 0, ∀ n widths Σ μ masks θ̃ Nb ε θ, …. Risk: Quantifier order must put the constants before n, widths, alphabet, p, N_b, masks; putting them after would make the theorem near-trivial.
- **modeling-note** λ^{n−3} at n = 2: Use zpow (λ ^ ((n:ℤ) − 3)) or the equivalent λ³·SC ≤ Cε⁶λ^n. Risk: With ℕ subtraction n − 3 = 0 at n = 2, the bound would become Cε⁶, which is a different (stronger) claim — must not be used.
- **acceptable** Prompt layout: s0@0, w*@1, w_t@(1+t), SEP@(n+2), ':'@(n+3); answer logits at n+3. Risk: Low.
- **proof-obligation** ReLU differentiability: Prove the activation pattern is constant on a neighbourhood of every θ ∈ B_ε (the paper proves it on the doubled ball), so z is smooth there. Risk: If closedBall boundary points were not interior to a constant-pattern region, fderiv could be 0 there; the doubled-ball claim (appendix.tex:229-230) covers it.

### Downstream contract

- Program: Define the deterministic forward map logits (A : Arch) (ζ : ℝ) (θ : Params A) (w : Fin n → Σ) : EuclideanSpace ℝ (ZMod K) for the two-block, no-LayerNorm, causal softmax Transformer with per-head W_Q, W_K (d_h×d_res), W_V (d_res×d_res), scores (1/√d_h)⟨W_Q h_i, W_K h_j⟩ scaled by ζ, softmax over a fixed nonempty causal visible set, residual ReLU MLP W_out ReLU(W_in h + b_in), readout V (K×d_res), evaluated at position n+3 of the outcome prompt.
- Program: ζ must scale the scores in both blocks inside the composed computation (block-2 scores computed from the ζ-network's block-1 residuals).
- Program: No relative bias; Params is a EuclideanSpace over a Fintype index so ‖·‖ is the vectorized Euclidean norm.
- Program: Task: Fintype alphabet, PMF μ, query w*, s0 : ZMod K; answer = s0 + #{t : w_t = w*} mod K; G = onehot(answer) − (1/K)1; expectation = finite weighted sum.
- Pseudocode: May present the forward pass with ζ hidden (ζ = 1) and z^{[0]} as 'uniform attention'; must state that z^{[0]} is the composed ζ = 0 network.
- Pseudocode: May present masks as 'full prefix by default'.
- Theorem: One theorem with two parts (or two theorems sharing the setting): part (i) architecture-free: ∀ S H, DependsOnlyOn S H → ‖Ex (fun w => (H w)ᵀ *ᵥ G w)‖ ≤ mixrate K p ^ (n − S.card) * l2Norm H.
- Theorem: Part (ii): ∀ a0 cemb (K) nH, ∃ c₀ c C > 0, ∀ n ≥ 2, ∀ Task with that K and p ∈ (0,1), ∀ Arch with nH heads (any widths, masks), ∀ θ̃ Nb with InitHyp θ̃ a0 cemb Nb, ∀ ε, 0 < ε → ε ≤ c₀ * min 1 (1/Nb) → ∀ θ ∈ closedBall (ε • θ̃) (min (1/8) (cemb/4) * ε) → (ε^2 ≤ c * λ → scoreCredit θ ≤ C * ε^6 * λ ^ ((n:ℤ) − 3)) ∧ (ε ≤ c * λ → outcomeCredit θ ≤ C * ε * λ ^ n).
- Theorem: Constants must be quantified before n, widths, alphabet, μ, p, s0, w*, Nb, masks, θ̃, ε, θ.
- Theorem: Do not use ℕ subtraction for n − 3. Optionally add DifferentiableAt conjuncts so the fderiv bound is not vacuous.
- Handoff: Check quantifier order of constants (uniform in n, widths, N_b, p, alphabet).
- Handoff: Check λ is \|1 − p + p e^{2πi/K}\| and the exponent n − 3 is an integer power.
- Handoff: Check z^{[0]} is the ζ = 0 composed network, and that both credits exclude β.
- Handoff: Check part (i) uses the Euclidean norm on E[Hᵀ G] and the Frobenius L² norm on H.
- Handoff: Report that ε > 0 was added (implicit in the paper) and that eq:thm1-full's \|Φ_out\| bound is a non-goal.

### Hypotheses

| Hypothesis | Verdict | Why |
| --- | --- | --- |
| K ≥ 2 and p ∈ (0,1) | `required` | λ < 1 used in summation and in ε ≤ λ; stated in def:task / ass:task. |
| Start count and query fixed; letters i.i.d. | `required` | Basis of the conditioning argument. |
| β = 0 | `discharge-internally` | Omit β from the model. |
| θ̃ satisfies A2 (op-norm ≤ a0, σ_min(value embeddings) ≥ c_emb, \|b̃_m\| ≥ 1, ‖b̃‖ ≤ N_b) | `required` | Fixes ReLU pattern and the ε-power bounds; c_emb sets r0. |
| 0 < ε ≤ c₀ min(1, 1/N_b) | `required` | ε > 0 is implicit in the paper but needed (differentiability at θ0 = 0 fails, λ^{−1} at n = 2). |
| θ ∈ B_ε | `required` | Local result. |
| n ≥ 2 (part ii) | `required` | Stated; k0 ≥ 1. |
| ε² ≤ cλ (ii-a), ε ≤ cλ (ii-b) | `required` | Geometric decay C_att ε⁴ ≤ λ²/2 and collapse to one term. |
| H square-integrable | `unnecessary` | Finite alphabet. |
| Nonempty visible sets for masked heads | `required` | Softmax over an empty set is undefined; implicit in the paper. |

### Caller Conditions

- **derive_Catt_eps4_le_half_lambdaSq** — C_att · ε⁴ ≤ (1/2)·λ², to be derived (not assumed) from ε² ≤ c·λ by fixing the existential constant c := (2·C_att)⁻¹ᐟ² (or any 0 < c ≤ (2·C_att)⁻¹ᐟ²) before lem:score-sum's internal hypothesis is invoked. (Paper/appendix.tex:530-532 ('For (ii), take c ≤ (2C_att)^{-1/2}: ε²≤cλ ⟹ C_attε⁴ ≤ C_att c²λ² ≤ (1/2)λ²'); applies to Model/Theorem (assembly of thm:main part ii), Analysis/lem:score-sum); Lean binder `have hc : C_att * ε^4 ≤ (1/2) * λ^2 := by have := hε2 -- ε^2 ≤ c*λ with c chosen as (2*Catt)⁻¹ᐟ²; nlinarith/field_simp from that choice`.
- **derive_lambda_pos_from_eps_pos_and_eps2_le_c_lambda** — 0 < λ, derived from 0 < ε, c > 0 and ε² ≤ c·λ (so c·λ ≥ ε² > 0, hence λ > 0 since c > 0). (Paper/appendix.tex:494-496 (lem:score-sum assumes '0<λ<1'); ambiguity note on λ=0 at K=2,p=1/2 and the λ^{n-3} zpow at n=2; applies to Model/Theorem (assembly of thm:main part ii-a), Analysis/lem:score-sum); Lean binder `have hλpos : 0 < λ := by nlinarith [sq_nonneg ε, hεpos, hc_pos, h_eps2_le_cλ]`.
- **lambda_lt_one_standing_fact** — λ < 1, derivable once from K ≥ 2 and p ∈ (0,1) alone (no dependence on ε or θ). (Paper/ass:task appendix.tex:57 ('so λ<1'); def:lambda body.tex:89; applies to Model/ProblemSetting (mixrate cross-check), Analysis (lem:target-contraction, lem:score-sum)); Lean binder `theorem mixrate_lt_one (hK : 2 ≤ K) (hp : p ∈ Set.Ioo (0:ℝ) 1) : mixrate K p < 1`.
- **visible_set_nonempty_before_softmax** — ∀ block l, head h, position i: (vis l h i).Nonempty, must hold before the forward pass computes softmax at (l,h,i). (Paper/ass:arch appendix.tex:34-41 ('A head may have a fixed, input-independent causal submask'); applies to Model/Pseudocode (logits / forward pass), Model/Prelude (Arch structure)); Lean binder `already a structure field `vis_nonempty : ∀ l h i, (vis l h i).Nonempty`; must actually be consumed as a side condition at every softmax call site in the forward-pass definition, not merely declared`.
- **fixed_activation_pattern_before_differentiating** — For every θ in a neighbourhood of B_ε (the 'doubled ball') and every ζ ∈ [0,1], the sign of each MLP pre-activation (W_in h + b_in)_m equals sign(b̃_m), constant across all such θ, ζ, and inputs. (Paper/appendix.tex:229-230 ('we work on the doubled ball ... open neighbourhood'); ass:init bias condition \|b̃_m\|≥1 appendix.tex:47; applies to Analysis (prop:network, prop:score-degree), Model/Theorem (well-definedness of fderiv in outcomeCredit/scoreCredit)); Lean binder `lemma activation_pattern_fixed (A : Arch T n) (θt : Params A) (h : InitHyp A θt a0 cemb Nb) (hε : 0 < ε) (hε0 : ε ≤ c₀ * min 1 (1/Nb)) : ∃ r > 0, ∀ θ' ∈ Metric.ball (ε • θt) r, ∀ ζ ∈ Set.Icc (0:ℝ) 1, ∀ w, <constant sign of every pre-activation>`.
- **relu_pattern_invariant_across_blocks_and_zeta** — The sign of every MLP pre-activation in both block 1 and block 2 stays equal to sign(b̃_m) simultaneously for all ζ ∈ [0,1] and all θ in the (doubled) ball — including after block-1's ζ-scaled residual update feeds into block 2's pre-activations. (Paper/appendix.tex:229-230; the ζ-expansion paragraph before thm:main (body.tex, 'Score mediated credit' setup) which composes the ζ=0 network through both blocks; applies to Analysis/prop:network, Analysis/prop:score-degree); Lean binder `invariant carried as a hypothesis/output of an induction over block index l : Fin 2, e.g. `∀ l, PatternFixed A θt ε l` proved by induction l=1 then l=2 using the l=1 output bound`.
- **k_indexed_zeta_taylor_term_bound** — For every k = 0,...,k0 the ζ^k-order term J^[k] of the Taylor expansion reads at most 2k+1 letters and has sup-Frobenius-norm bound C·ε²·(C_att·ε⁴)^k, uniformly in n; for k=k0 this is an induction on the formal power-series expansion in ζ, not a closed-form identity. (Paper/appendix.tex:308-319 (prop:score-degree parts i-ii), proof of lem:score-sum appendix.tex:506-513; applies to Analysis/prop:score-degree, Analysis/lem:score-sum assembly); Lean binder `lemma score_term_bound (k : ℕ) (hk : k ≤ k0) : sup_term_bound A θt ε k ≤ C * ε^2 * (Catt * ε^4)^k, proved by induction on k alongside the letter-count bound reads_at_most (2*k+1) letters`.
- **geometric_ratio_bound_sustained_across_sum** — The ratio r := C_att·ε⁴/λ² ≤ 1/2 holds uniformly for every k in the sum (not just k=1), which is what licenses bounding Σ_{k≥1} r^{k-1} by the geometric-series constant 3 and absorbing the remainder term (k=k0+1) into the same bound. (Paper/appendix.tex:522-528 ('Since C_att ε⁴/λ² ≤ 1/2, the score-dependent terms sum to at most 3Cε⁶λ^{n-3}'); applies to Analysis/lem:score-sum assembly step); Lean binder `lemma geometric_closing (hr : Catt*ε^4/λ^2 ≤ 1/2) : ∑ k in Finset.Icc 1 k0, (Catt*ε^4/λ^2)^(k-1) ≤ 3, via `tsum_geometric_of_lt_one`/`geom_sum_le` style bound`.
- **k0_ceiling_arithmetic** — k0 := ⌈(n-1)/2⌉ satisfies 2·k0 ≥ n-1 (used to convert the remainder bound C_att·ε⁶·λ^{2k0}·r^{k0} into the Cε⁶λ^{n-3}·r^{k0} form via λ^{2k0} ≤ λ^{n-1} since 0<λ<1 and 2k0≥n-1 — note the inequality direction flips because exponentiating a number in (0,1) by a larger exponent gives a smaller value, so λ^{2k0} ≤ λ^{n-1} requires 2k0 ≥ n-1 together with λ<1). (Paper/appendix.tex:518-521 ('≤ Cε⁶λ^{n-3}(C_attε⁴/λ²)^{k0} since 2k0≥n-1'); applies to Analysis/lem:score-sum remainder step); Lean binder `lemma k0_ceil_ge (n : ℕ) (hn : 2 ≤ n) : 2 * (⌈((n:ℝ)-1)/2⌉₊) ≥ n - 1 -- or the Nat.ceil/Int version matching k0's actual Lean definition`.
- **lambda_in_open_unit_interval_sustained_for_negative_exponents** — 0 < λ < 1 must hold (not just λ ≤ 1) at every term of the sum where 2k+1 > n, because the claim 'λ^{n-2k-1} ≥ 1' (a zpow with negative exponent) is only true for λ strictly in (0,1); at λ=0 this term is 0 under Mathlib's zpow convention, which breaks the 'still bounds the credit' step. (Paper/appendix.tex:516 ('If 2k+1>n, the factor λ^{n-2k-1}≥1 still bounds the credit'); applies to Analysis/lem:score-sum, Analysis/prop:score-degree term-by-term step); Lean binder `lemma zpow_neg_ge_one (hλ0 : 0 < λ) (hλ1 : λ < 1) (m : ℤ) (hm : m ≤ 0) : 1 ≤ λ ^ m`.
- **the existential constant c in part (ii) vs. the proof's specific choice c := (2·C_att)^{-1/2}** (proof-obligation) — The restatement's downstreamContract leaves c as an unexplained existential; it does not record that c must be *defined from* C_att (itself a downstream output of prop:score-degree depending only on a0, c_emb, K, nH) rather than chosen independently. An executor who picks c first and only later tries to derive the C_attε⁴≤λ²/2 bound will get stuck or silently weaken the theorem (e.g. by adding an extra unstated hypothesis bounding C_att). Source: Paper/appendix.tex:530-532.
- **λ^((n:ℤ)-3) at n=2 under a possible λ=0 degenerate case** (proof-obligation) — The restatement already flags this as an ambiguity but does not spell out the concrete failure mode: with Mathlib's zpow convention (0^(-1)=0), if λ=0 slipped through, the RHS collapses to 0 and the stated bound would force scoreCredit=0, i.e. a different (and almost certainly false) claim rather than merely 'undefined.' The fix (deriving λ>0 from ε>0, c>0, ε²≤cλ) is captured above as a caller condition but must be applied *before* the zpow is formed in the Lean statement, not discovered later while discharging the proof. Source: Paper/appendix.tex:494-496, 530-532; ambiguityLedger entry on λ=0.
- **fderiv-based definitions of outcomeCredit / scoreCredit** (add-hypothesis) — Mathlib's `fderiv` returns the zero map at points where the function is not differentiable. If the 'fixed activation pattern on an open neighbourhood' invariant is established only pointwise at θ (not on a ball around θ), the headline bound could be discharged vacuously (fderiv silently 0, inequality trivially true) without ever proving the intended Jacobian bound — exactly the failure mode the restatement's own modelingChoices entry flags but does not resolve. Recommend the headline (or an immediately preceding lemma it cites) carry an explicit `DifferentiableAt ℝ (...) θ` conjunct so the numeric bound cannot be satisfied by vacuity. Source: Paper/appendix.tex:229-230; restatement modelingChoices entry on fderiv-based credit; ambiguityLedger proof-obligation on ReLU differentiability.
- **vis_nonempty field of Arch** (model-internal) — The field exists in the structure sketch but the restatement does not confirm it is actually *threaded into* every softmax normalization site inside the forward-pass (Pseudocode) definition. A structure field that is declared but never consumed by the definition it is meant to guard is a silent gap: an instantiation with an empty mask would typecheck (Lean reals give 0/0=0) and produce a degenerate, wrong network rather than failing visibly. Source: Paper/ass:arch appendix.tex:34-41.
- **k0 := ⌈(n-1)/2⌉ and its use in the remainder-term bound** (proof-obligation) — Off-by-one risk: Lean has several candidate definitions (Nat.ceil of a rational, `(n+1)/2` integer division family, `(n-1+1)/2`). Only the one matching the paper's ⌈(n-1)/2⌉ with the exact inequality 2k0≥n-1 will make the remainder-collapsing step of lem:score-sum go through at the boundary cases n=2,3 that part (ii) explicitly covers. Source: Paper/appendix.tex:494, 518-521.
- **Task.s0 and Task.wstar as fixed (non-summed) fields vs. the expectation Ex** (model-internal) — The whole conditional-contraction argument (lem:target-contraction, part i) and the F_S-measurability argument depend on s0 and w* being literal fixed constants outside the sum defining Ex, not fields that vary with w. If a later refactor of Task or of the credit definitions accidentally closes over s0/w* as part of the summed index (e.g. by folding them into the prompt-generation function without pinning them first), the independence argument underlying eq:horizon silently breaks. Source: Paper/body.tex:163, 193; appendix.tex:227 ('query and start fixed').

### What the paper leaves unstated

Lean will force each of these. Where one turns out to be false rather than
merely unstated, that is a finding about the paper, not a modelling problem.

- ε > 0 is not stated in thm:main (lem:score-sum assumes it). At ε = 0, θ0 = 0, all MLP pre-activations are exactly 0 and the ReLU is not differentiable; add 0 < ε.
- λ can be 0 (K = 2, p = 1/2) and at n = 2 the bound uses λ^{−1}. With ε > 0 the hypothesis ε² ≤ cλ forces λ > 0, so the statement is well defined once ε > 0 is added; Lean must use an integer power, not ℕ subtraction.
- Quantifier order / dependence of constants: the paper says c, C depend only on a0, c_emb, K and the number of blocks and heads; the c inside ε0 = c min(1,1/N_b) is also an existential constant. The Lean statement must fix these before n, widths, N_b, alphabet, p.
- Architecture details not pinned in ass:arch: whether heads have an output projection W_O, whether scores carry 1/√d_h, whether MLPs have an output bias. The appendix display (appendix.tex:95-96) fixes: no W_O, 1/√d_h, no output bias. Adopted.
- 'Every weight matrix and embedding block … operator norm ≤ a0': unclear whether the readout V counts as a weight matrix (assumed yes) and how embedding blocks are shaped (token embedding as d_res×|vocab|, position embedding as d_res×L, assumed).
- z^{[0]} / J^{[0]}: defined by multiplying every score by ζ in both blocks of the composed network; J^{[0]} = ∂_θ z(0). The body paragraph is informal; appendix eq:score-split and direct_proofs/conditional_outcome.tex:29 fix the composed reading.
- Part (i) norm conventions: E[Hᵀ G] uses the Euclidean norm, ‖H‖_{L²} uses Frobenius (lem:target-contraction). H has K rows and arbitrary column count.
- The alphabet is never stated to be finite; finiteness is assumed (learned token embeddings).
- Submasks are allowed by ass:arch but their visible sets are not required to be nonempty; we require nonempty.
- Differentiability: the credit is a gradient; the theorem implicitly needs z_θ differentiable at θ ∈ B_ε (fixed ReLU pattern on the doubled ball, appendix.tex:229-230).

## Architecture

This target is planned as a standalone theorem: the proof route, reusable
source theorems and statement checks are the architecture. No interface or
concrete instance is being invented for it.

### Route — `direct-proof`

No library proves any of the three parts. Part (i) is a short finite-sum argument: a resampling identity, Fourier analysis on ZMod K, then Cauchy–Schwarz. Parts (ii-a) and (ii-b) are bounds on a gradient. The paper gets them from a ζ-Taylor expansion of the Jacobian (prop:score-degree plus lem:score-sum). The plan proves the same power counting differently, and this route is easier to formalize: (1) a 'locality norm' N_κ on functions of the letters, where a term reading the letter set S has weight κ^{-|S|}; (2) a Lipschitz (two-point difference) bound instead of a Jacobian, turned into an fderiv bound by Mathlib's `norm_fderiv_le_of_lip'`. Each attention score costs ε⁴κ^{-2} and the value path costs ε²κ^{-1}, which matches the paper's ε^{2+4k} λ^{-(2k+1)} exactly. The surface threads `hprior : Trace.Prior` through every theorem. I have not seen `Trace/Model/Prior.lean`, so I cannot say what it supplies. The route below does not need it. If it turns out to state prop:score-degree directly for `Trace.logits`, a shorter lem:score-sum-style summation becomes possible (see knownBlockers). The paper proves every lemma it uses, so `borrowed-prior` is not the honest route label.

### Source theorems

| Declaration | Module | Why it applies |
| --- | --- | --- |
| `DependsOn` | `Mathlib.Logic.Function.DependsOn` | The surface's 'depends only on letters in S'. Use it in the resampling identity and in every locality decomposition. `DependsOn.mono` (s ⊆ t) is probably in the same file (medium confidence); if it is missing, prove it in one line. |
| `norm_fderiv_le_of_lip'` | `Mathlib.Analysis.Calculus.FDeriv.Basic` | Turns the two-point bound \|Φ(θ')−Φ(θ)\| ≤ L‖θ'−θ‖ on ball(θ, r₀ε) into scoreCredit / outcomeCredit ≤ L. No Jacobian is ever computed. High confidence on the name; check the exact implicit-argument order. |
| `DifferentiableAt.congr_of_eventuallyEq` | `Mathlib.Analysis.Calculus.FDeriv.Basic` | Proves the DifferentiableAt conjuncts. On the doubled ball the ReLU equals a fixed 0/1 mask (pattern lemma), so each functional agrees near θ with a max-free expression made of exp, division by positive sums, and polynomial operations. |
| `fderiv_add` | `Mathlib.Analysis.Calculus.FDeriv.Add` | (ii-b) splits outcomeFunctional = zeroOrderFunctional + scoreFunctional and reuses (ii-a). |
| `Metric.ball_mem_nhds` | `Mathlib.Topology.MetricSpace.Pseudo.Defs` | Supplies the `∀ᶠ θ' in 𝓝 θ` in norm_fderiv_le_of_lip'. Use ball(θ, r₀ε) ⊆ closedBall(εθ̃, 2r₀ε), the doubled ball. |
| `AddChar.sum_mulShift` | `Mathlib.NumberTheory.LegendreSymbol.AddCharacter (module path uncertain; it may have moved under Mathlib/Algebra/Group/AddChar or Mathlib/NumberTheory/...)` | Character orthogonality on ZMod K, the basis of the Parseval identity in part (i). Medium confidence on the exact name and hypotheses; grep `sum_mulShift`. Fallback: `IsPrimitiveRoot.geom_sum_eq_zero` with `Complex.isPrimitiveRoot_exp`. |
| `ZMod.stdAddChar / ZMod.isPrimitive_stdAddChar` | `Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar (medium confidence on module)` | The characters χ_j(a) = α^{ja} without hand-made mod-K bookkeeping, plus the link stdAddChar 1 = exp(2πi/K) to `Trace.mixrate`. Grep `stdAddChar_apply` and `toCircle_apply` for the exact form. |
| `Complex.isPrimitiveRoot_exp / IsPrimitiveRoot.geom_sum_eq_zero` | `Mathlib.RingTheory.RootsOfUnity.Complex / Mathlib.RingTheory.RootsOfUnity.PrimitiveRoots` | Fallback route to character orthogonality if the AddChar API fights back. |
| `Fintype.prod_sum (a.k.a. Finset.prod_univ_sum)` | `Mathlib.Algebra.BigOperators.Ring (exact file uncertain)` | Product form of the letter law. It gives Σ_w letterWeight w = 1 and E[χ_j(count over Sᶜ)] = ρ_j^{\|Sᶜ\|} (independence, used in the forward direction). |
| `Finset.prod_add` | `Mathlib.Algebra.BigOperators.Ring` | Finite (series-free) proof of the locality-norm exponential bound N(e^f − 1) ≤ e^{N f} − 1, via e^{Σ_S T_S} = Π_S(1 + (e^{T_S} − 1)). |
| `Equiv.sum_comp` | `Mathlib.Algebra.BigOperators.Group.Finset.Basic` | The resampling involution (w,v) ↦ (merge w v, merge v w) on (Fin n → Sym)², which proves E[HᵀG] = E[HᵀĜ] with no subtype equivalences. |
| `PMF.tsum_coe, tsum_fintype, ENNReal.toReal_sum` | `Mathlib.Probability.ProbabilityMassFunction.Basic; Mathlib.Topology.Algebra.InfiniteSum.Basic; Mathlib.Data.ENNReal.BigOperators` | Σ_x (μ x).toReal = 1, hence Σ_w letterWeight w = 1 and ρ_j = (1−p) + p χ_j(1). |
| `Real.cos_le_cos_of_nonneg_of_le_pi` | `Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic` | Gives \|ρ_j\| ≤ \|ρ_1\| = λ for 1 ≤ j.val ≤ K−1. Combine with cos(2π − x) = cos x for j.val > K/2. |
| `ContinuousLinearMap.le_opNorm / opNorm_le_bound` | `Mathlib.Analysis.Normed.Operator.Basic` | Unpacks `Trace.opNorm` (a CLM norm of Matrix.toEuclideanLin) into ‖M x‖₂ ≤ opNorm M ‖x‖₂. Also gives opNorm ≤ Frobenius ≤ the full parameter norm. |
| `Finset.sum_mul_sq_le_sq_mul_sq / inner_mul_le_norm_mul_norm` | `Mathlib.Algebra.Order.Chebyshev (or BigOperators) / Mathlib.Analysis.InnerProductSpace.Basic` | Cauchy–Schwarz in part (i) (‖Hᵀg‖ ≤ ‖H‖_F‖g‖, and Jensen Σπ‖H‖_F ≤ (Σπ‖H‖_F²)^{1/2}) and in the locality credit lemma. |

### Proof steps

- **0. Before anything else, open Trace/Model/Prior.lean and record what `Trace.Prior` asserts. Do not add to it. If it is a hypothesis the route below never uses, ignore it (`_hprior`). If it is strong or vacuous (e.g. implies False), report that immediately as a statement defect: every surface theorem is then conditional on it.** — `read the file; `#check @Trace.Prior`; `#print Trace.Prior``. The fixed surface threads `hprior` through all three theorems. Its content decides whether part (ii) can shortcut through a stated prop:score-degree.
- **1a. Part (i), resampling identity. Define merge S w v t := if t ∈ S then w t else v t and Ĝ(w) := Σ_v letterWeight v • targetSignal (merge S w v). Prove Ex (fun w => (H w)ᵀ *ᵥ G w) = Ex (fun w => (H w)ᵀ *ᵥ Ĝ w). Insert Σ_v letterWeight v = 1, then reindex by the involution e(w,v) = (merge S w v, merge S v w). It preserves letterWeight w * letterWeight v (the products split over S and Sᶜ), and H w = H (merge S w v) by hH.** — `rw [← mul_one (Ex …), ← sum_letterWeight_eq_one]; simp only [Ex, Finset.sum_mul, Finset.smul_sum]; rw [← Fintype.sum_prod_type']; exact (Equiv.sum_comp eInvol _).symm ▸ …   (eInvol built with `Function.Involutive.toPerm`)`. This is the finite-sum form of the tower property. It avoids Equiv.piEquivPiSubtypeProd and any conditional-expectation API.
- **1b. Part (i), Fourier form of Ĝ. For j : ZMod K, Σ_a Ĝ(w)_a · χ(j·a) = (if j = 0 then 0 else χ(j·(s0 + c_S(w))) · ρ_j^{\|Sᶜ\|}), where χ = ZMod.stdAddChar, ρ_j = (1−p) + p·χ(j) and c_S(w) = Σ_{t∈S}[w t = w⋆]. Proof: χ is additive, and Σ_v Π_t f_t(v t) = Π_t Σ_x f_t(x) (Fintype.prod_sum) with f_t = μ for t ∈ S. The j = 0 case is Σ_a q(a) − K·(1/K) = 0.** — `simp only [Ĝ, targetSignal, answer, AddChar.map_add_eq_mul, Finset.mul_sum]; rw [Fintype.prod_sum]; split_ifs; ring_nf; field_simp`. Independence of the letters. This is the only place the product law matters.
- **1c. Part (i), Parseval on ZMod K. For f : ZMod K → ℂ: Σ_a ‖f a‖² = (1/K) Σ_j ‖Σ_a f a · χ(j·a)‖². Prove it from character orthogonality (AddChar.sum_mulShift with ZMod.isPrimitive_stdAddChar), expanding \|·\|² = z·conj z. Then ‖Ĝ(w)‖₂² = (1/K) Σ_{j≠0} \|ρ_j\|^{2\|Sᶜ\|} ≤ λ^{2(n−\|S\|)}, using \|ρ_j\| ≤ λ (step 1d), the K−1 nonzero modes, and (K−1)/K ≤ 1.** — `own lemma `parseval_zmod`: simp only [Complex.sq_norm/normSq, map_sum, Finset.sum_comm, ← AddChar.map_neg_eq_conj]; rw [AddChar.sum_mulShift]; simp [Finset.sum_ite_eq]`. This is eq:horizon (appendix.tex:267). Check first whether Mathlib already has a ZMod.dft Parseval (grep `dft` in Mathlib/Analysis/Fourier/ZMod.lean). If so, use it.
- **1d. \|ρ_j\| ≤ mixrate K p for j ≠ 0. Use \|(1−p) + p z\|² = (1−p)² + p² + 2p(1−p)·Re z for \|z\| = 1, with Re χ(j) = cos(2π j.val/K) ≤ cos(2π/K). Also prove 0 ≤ mixrate ≤ 1 (triangle inequality), needed downstream.** — `rw [← sq_le_sq₀ …, Complex.sq_abs, Complex.normSq_apply]; simp [stdAddChar_apply, Complex.exp_re]; nlinarith [Real.cos_le_cos_of_nonneg_of_le_pi …, cos_two_pi_sub …]`. The paper's λ = max_j\|ρ_j\| is a proof obligation; the surface defines λ = \|ρ_1\|.
- **1e. Close part (i). ‖Ex(Hᵀ Ĝ)‖ ≤ Σ_w π(w)‖H w‖_F‖Ĝ w‖ ≤ λ^{n−\|S\|} Σ_w π(w)‖H w‖_F ≤ λ^{n−\|S\|}·sqrt(Σ_w π(w)‖H w‖_F²) = λ^{n−\|S\|}·l2Norm H. Note \|Sᶜ\| = n − S.card via Finset.card_compl and Fintype.card_fin, so ℕ subtraction is harmless here.** — `calc featureCredit T H = ‖toLp (Ex (Hᵀ Ĝ))‖ := by rw [resample] _ ≤ … := norm_sum_le … _ ≤ … := gcongr … _ ≤ … := Real.le_sqrt … (Cauchy–Schwarz with Σπ = 1)`. Delivers `Trace.Proof.unseenLettersContraction`. Uses K ≥ 1 only; hK, hp0 and hp1 are unused.
- **2. Locality calculus (proof-side file Trace/Analysis/Locality.lean). For a normed space E and κ ∈ (0,1], let a decomposition be T : Finset (Fin n) → ((Fin n → Sym) → E) with DependsOn (T S) S and f = Σ_S T S. Grouping terms by support set loses nothing. Define locNorm κ f := sInf {Σ_S κ⁻¹^{S.card}·‖T S‖_sup}. Lemmas: ‖f‖_sup ≤ locNorm ≤ κ^{-n}‖f‖_sup; triangle; scalar; constants (S = ∅); `DependsOn f S → locNorm ≤ κ^{-\|S\|}‖f‖`; bilinear bound locNorm(B f g) ≤ ‖B‖·locNorm f·locNorm g (regroup S ∪ S', using κ^{-\|S∪S'\|} ≤ κ^{-\|S\|}κ^{-\|S'\|}); exp bound locNorm(exp∘f − 1) ≤ exp(locNorm f) − 1 (finite product expansion, Finset.prod_add); inverse bound locNorm((1+β)⁻¹ − 1) ≤ N/(1−N) for N = locNorm β < 1 (Neumann series plus the sup-norm comparison to pass to the limit); softmax lemma: if max_{j∈V} locNorm ψ_j ≤ M ≤ 1/4 then Σ_{j∈V} locNorm(softmax_V(ψ)_j − 1/\|V\|) ≤ 8M, plus its two-point version Σ_j locNorm(ω_j(ψ') − ω_j(ψ)) ≤ C·max_j locNorm(ψ'_j − ψ_j).** — `a series of 15-40 line lemmas; `le_csInf`/`csInf_le` or a predicate `LocBound κ f B` with existential decompositions to avoid sInf bookkeeping`. This replaces the paper's 'term T_{k,ν} reads ≤ 2k+1 letters, Σ_ν sup‖T‖ ≤ …' enumeration with a submultiplicative seminorm. The softmax lemma is uniform in \|V\|, which gives uniformity in n.
- **3. Locality credit lemma: for f : (Fin n → Sym) → (ZMod K → ℝ) and λ = mixrate K T.p > 0, \|Ex (fun w => f w ⬝ᵥ targetSignal w)\| ≤ √K · λ^n · locNorm λ f. Apply part (i) with ι = Unit to each T S (H w = column T S w), use l2Norm ≤ sup and the Euclidean-vs-sup factor √K, then use λ^{n−\|S\|} = λ^n·λ^{-\|S\|}.** — `obtain ⟨T, hdep, rfl, hB⟩; simp only [Ex, dotProduct_sum…]; refine (abs_sum_le_sum_abs …).trans ?_; gcongr with S; have := Trace.Proof.unseenLettersContraction hprior hK T hp0 hp1 n S (H := fun w => Matrix.of fun a _ => T S w a) …`. This is how part (ii) consumes part (i). It is the formal version of 'apply lem:target-contraction term by term'.
- **4. Parameter bounds on the doubled ball D := closedBall(εθ̃, 2r₀ε) with 2r₀ ≤ 1/4. For θ' ∈ D: every accessor block has opNorm ≤ a₁ε with a₁ := max a0 0 + 1/4 (perturbation: opNorm ≤ Frobenius ≤ ‖θ'−εθ̃‖); for θ, θ' ∈ D the block differences have opNorm ≤ ‖θ'−θ‖; embedding columns have Euclidean norm ≤ a₁ε; \|bIn l m\| ≥ (3/4)ε and sign(bIn l m) = sign(θ̃.bIn l m); ‖bIn l‖ ≤ ε(Nb + 1/4).** — `generic lemma: for an injective index map g into ParamIdx, Σ_i θ (g i)^2 ≤ ‖θ‖^2 (Finset.sum_map + sum_le_univ_sum_of_nonneg); then one lemma per accessor via Sum.inl_injective/Sum.inr_injective`. Every ε power in the analysis comes from here. Only r₀ ≤ 1/8 is used; cemb enters only through r₀.
- **5. Pattern lemma, the coupling to a ReLU-free network. For θ' ∈ D, every w, every real ζ, l ∈ {0,1}, position i and unit m: max(pre,0) = (if 0 < θt.bIn l m then pre else 0), where pre = (wIn l *ᵥ a_i) m + bIn l m. Proof: attention weights are convex for any ζ, so the sup bounds ‖x_i‖ ≤ 2a₁ε, ‖a_i‖ ≤ Cε and ‖block-1 output‖ ≤ Cε use ε·Nb ≤ c₀. Then \|(wIn a)_m\| ≤ Cε² < (3/4)ε ≤ \|b_m\| once c₀ is small. Then prove DifferentiableAt of logits A ζ · w at θ (ζ ∈ {0,1}) through the max-free form plus congr_of_eventuallyEq, and lift to the two functionals by DifferentiableAt.sum/mul.** — `lemma relu_eq_mask …; DifferentiableAt: `refine (hLin).congr_of_eventuallyEq (Filter.eventually_of_mem (Metric.ball_mem_nhds θ hr) fun θ' h => …)`; hLin by `fun_prop` plus DifferentiableAt.div with `Finset.sum_pos (fun _ _ => Real.exp_pos _) (A.vis_nonempty …)``. Discharges both DifferentiableAt conjuncts and makes the MLPs exactly affine (W_out·D·(W_in a + b)) on D, as the paper assumes.
- **6. Network propagation (Trace/Analysis/NetworkBounds.lean), all with κ ∈ (0,1], hypothesis ε² ≤ c·κ, θ, θ' ∈ D and ρ := ‖θ'−θ‖/ε. Bound locNorm κ of: (value) embed ≤ Cεκ^{-1} (≤ Cε at the answer position); scores ψ ≤ Cε⁴κ^{-2}; block outputs ≤ Cεκ^{-1}, and ≤ Cε + Cε²κ^{-1} at position n+3; logits ≤ C(ε² + ε³κ^{-1}). (ζ-difference, ζ=1 minus ζ=0) attention weights Σ_j ≤ Cε⁴κ^{-2}; block-1 output ≤ Cε⁶κ^{-3}; logits ≤ Cε⁷κ^{-3}. (θ-difference) each of the above times Cρ. (mixed) logits ≤ Cρε⁷κ^{-3}. Use ω(ζ=0) = 1/\|vis\|, which is θ-independent, so the mixed difference of the weights is just the θ-difference of ω(1). Every C depends only on a0, K, nH.** — `one lemma per layer and per difference type (≈16 lemmas), each a `calc` chaining the bilinear bound, the softmax lemma and step-4 bounds; finish with `nlinarith`/`gcongr`, or `positivity` for side conditions`. This is the bulk. It reproduces the paper's power counting (score ε⁴λ^{-2}, value path ε²λ^{-1}) without the k-indexed expansion or the remainder J_rem.
- **7. Close (ii-a) (`Trace.Proof.unseenLettersScoreCredit`). Choose c₀, c, C from a0 and nH (and K, through √K) explicitly. Given the hypotheses, λ > 0 follows from ε > 0 and ε² ≤ cλ. For θ' ∈ ball(θ, r₀ε) ⊆ D: scoreFunctional θ' − scoreFunctional θ = Ex(⟨Δ_θΔ_ζ z, G⟩), which is ≤ √K λ^n·Cρε⁷λ^{-3} = C'ε⁶λ^{(n:ℤ)−3}‖θ'−θ‖. Apply norm_fderiv_le_of_lip'. DifferentiableAt comes from step 5.** — `refine ⟨c₀, c, C, …, fun n hn Sym _ _ T hp0 hp1 A θt Nb hI ε hε hε0 θ hθ hc => ⟨diff_score …, ?_⟩⟩; exact norm_fderiv_le_of_lip' (by positivity) (Filter.eventually_of_mem (Metric.ball_mem_nhds _ _) fun θ' h => by rw [zpow_sub₀ hλ.ne', zpow_natCast]; …)`. Uses credit lemma (step 3) with κ = λ and the mixed bound of step 6.
- **8. Close (ii-b) (`Trace.Proof.unseenLettersOutcomeCredit`). Get (c₀', c', C') from the score owner. Set c := min c' 1 (and c ≤ 1), so ε ≤ cλ gives ε ≤ λ ≤ 1 and ε² ≤ c'λ. Split outcomeFunctional = zeroOrderFunctional + scoreFunctional, where zeroOrderFunctional θ = Ex⟨logits A 0 θ w, G⟩ (Finset.sum_add_distrib, add_dotProduct). The fderiv of zeroOrderFunctional is bounded by the step-3/step-6 θ-difference bound at ζ = 0: √K λ^n·Cρ(ε² + ε³λ^{-1}) ≤ C''ελ^n‖θ'−θ‖. The score part satisfies C'ε⁶λ^{n−3} ≤ C'ελ^n because ε⁵ ≤ λ³. Combine with fderiv_add and norm_add_le.** — `obtain ⟨c₀', c', C', …, hS⟩ := Trace.Proof.unseenLettersScoreCredit hprior a0 cemb hcemb K hK nH; refine ⟨min c₀' c₀'', min c' (min 1 c''), …⟩; rw [outcome_eq_add, fderiv_add hd0 hdS]; exact (norm_add_le _ _).trans (add_le_add h0 hS') \|>.trans (by nlinarith …)`. This is lem:score-sum's 'adding the zero-order term'. Reusing (ii-a) avoids a second ζ-difference analysis.
- **9. Discharge the surface: replace each `sorry` in Trace/Model/Theorem.lean with `:= Trace.Proof.<owner> hprior …`. Theorem.lean imports the three Analysis files; that is allowed, since Theorem is the assembly file. The capstone is already `⟨…, …, …⟩`. Run `#print axioms Trace.unseen_letters_erase_outcome_credit` and `#modelClosureOfType`.** — `theorem unseen_letters_contraction … := Trace.Proof.unseenLettersContraction hprior hK T hp0 hp1 n S H hH`. Keeps the statement fixed and puts the proof content outside Model/.

### Statement checks

- **`hprior : Trace.Prior` may make the whole theorem conditional, or vacuous if Prior is unsatisfiable. That would be invisible in the headline's binder list.** — Print `Trace.Prior`. If the route never uses it, say so in the handoff and suggest dropping it from the surface in a later pass. If it states a paper lemma, record exactly which one and treat the result as conditional on it.
- **ℕ subtraction in λ^{n−3} would silently strengthen (ii-a) to Cε⁶ at n = 2, 3.** — The surface correctly uses `mixrate K T.p ^ ((n : ℤ) - 3)` (zpow). Keep it. Proofs need `zpow_sub₀` with λ ≠ 0, which follows from ε > 0 and ε² ≤ cλ.
- **Constants quantified in the wrong order (after n, the widths, Nb, the alphabet or p) would trivialize part (ii).** — The surface puts ∃ c₀ c C right after a0, cemb, K, nH. In the proof, define the constants explicitly from a0, nH and K before `intro n`. Never use n, A.dres, A.dh, A.dmlp, Fintype.card Sym, Nb or T.p in them.
- **fderiv is 0 at non-differentiable points, so a credit bound could hold vacuously.** — The DifferentiableAt conjuncts are present. Prove them from the pattern lemma (step 5), not from the Lipschitz bound.
- **Width-dependent constants can creep in through the Pi sup norm: residual vectors are typed `Fin A.dres → ℝ`, whose Mathlib norm is the sup norm, while `Trace.opNorm` is Euclidean.** — In every locality and sup bound, measure residual vectors in EuclideanSpace ℝ (Fin dres) via `WithLp.toLp 2`. Grep the proof files for norms of bare `Fin _ → ℝ` vectors. √K factors (logits) are allowed; √dres factors are not.
- **Part (i) needs λ^{n−\|S\|} with \|Sᶜ\| = n − S.card. A mismatch between `Finset.card_compl` and ℕ subtraction could look like an off-by-something.** — S.card ≤ n by Finset.card_le_univ and Fintype.card_fin, so the ℕ subtraction is exact. Prove `(Sᶜ).card = n - S.card` once and rewrite with it.
- **Degenerate Nb (dmlp = 0 allows any Nb, including Nb ≤ 0) makes `min 1 (1/Nb)` ≤ 0.** — The hypothesis ε ≤ c₀·min 1 (1/Nb) with ε > 0 then has no solutions, so the theorem is vacuous there, not false. Do not add Nb > 0. When Nb > 0, the proof uses ε·Nb ≤ c₀ (from ε ≤ c₀/Nb) and ε ≤ c₀.
- **Paper gaps the formalization will hit: (1) ε > 0 missing in thm:main (already repaired); (2) λ = 0 is possible (K = 2, p = 1/2), but ε² ≤ cλ with ε > 0 excludes it; (3) the restriction n ≥ 2 is not needed by this route.** — Keep the surface as-is. In the handoff, report that the proof never uses `2 ≤ n` (hn is unused), which is evidence that the paper's restriction is an artifact of k₀ = ⌈(n−1)/2⌉ in lem:score-sum.
- **Part (i) is stated with λ^{n−\|S\|}, which is weaker than eq:horizon's √(1−1/K)·λ^{n−\|S\|}. This matches eq:credit-degree, so it is fine, but do not 'strengthen' it in the surface.** — Statement matches eq:credit-degree (body.tex:178) and lem:target-contraction's second display.
- **z^{[0]} must be the composed ζ = 0 network, not frozen ζ = 1 block-1 residuals.** — `logits T A 0 θ` uses `block 0 θ 1 (block 0 θ 0 …)`, so ζ enters both blocks. The mixed-difference bound in step 6 relies on exactly this.


### Capstone — `Trace.unseen_letters_erase_outcome_credit`

```lean
Already assembled in Trace/Model/Theorem.lean as ⟨@unseen_letters_contraction hprior, @unseen_letters_score_credit hprior, @unseen_letters_outcome_credit hprior⟩. Discharging it means replacing the three surface `sorry`s with the proof-side owners Trace.Proof.unseenLettersContraction, Trace.Proof.unseenLettersScoreCredit and Trace.Proof.unseenLettersOutcomeCredit.
```

The interface models an idealized process, so the capstone is only honest once
`Trace.Proof.relu_eq_mask_on_doubledBall` connects it back to the real object:

```lean
∀ θ' ∈ closedBall (ε • θt) (2 * min (1/8) (cemb/4) * ε), ∀ w ζ l i m, max ((θ'.wIn l *ᵥ a_i) m + θ'.bIn l m) 0 = if 0 < θt.bIn l m then (θ'.wIn l *ᵥ a_i) m + θ'.bIn l m else 0, where a_i is the post-attention residual of block l in the ζ-network at θ' (given 0 < ε ≤ c₀ min 1 (1/Nb) and InitHyp).
```

This connects the real network, with its non-smooth ReLU, to the affine-MLP network that both the paper and the locality analysis reason about. It also supplies DifferentiableAt. It holds on the doubled ball, so it covers the neighbourhood ball(θ, r₀ε) used by norm_fderiv_le_of_lip'.

## The crux

Prove this first, in isolation. Once it is green the surrounding theorem
usually collapses into bookkeeping — and if it does not, the crux was
mis-identified and the architecture above needs another look.

### `Horizon bound via Parseval on ZMod K`

```lean
For every S and w: ‖Ĝ(w)‖₂ ≤ mixrate K p ^ (n − |S|), where Ĝ(w) = Σ_v letterWeight v • targetSignal(merge S w v). Equivalently Σ_a (P(s_n = a | w_S) − 1/K)² = (1/K)Σ_{j≠0}|ρ_j|^{2(n−|S|)} ≤ λ^{2(n−|S|)}.
```

**Why this is the heart:** All of part (i) and, through the credit lemma, every λ^n factor in part (ii) rest on it. It is the only place Fourier analysis, characters of ZMod K and the cos comparison \|ρ_j\| ≤ \|ρ_1\| are needed. Everything else is real inequalities.

**How to attack it:** (1) Grep Mathlib for a ZMod.dft Parseval or a `sum_mulShift`-type orthogonality. (2) Prove `parseval_zmod` (≈30 lines) from AddChar.sum_mulShift with ZMod.isPrimitive_stdAddChar. (3) Compute the transform of Ĝ by Fintype.prod_sum (step 1b). (4) Prove \|ρ_j\| ≤ λ by expanding normSq and Real.cos_le_cos_of_nonneg_of_le_pi. Sanity check: mixrate 2 p = \|1 − 2p\| (appendix.tex:253). Before step (2), spend 10 minutes checking Arlib.MarkovChains for a lazy cycle-walk spectral gap; no declaration name is known, so do not plan on it.

### `Locality-norm calculus with a width- and n-uniform softmax lemma`

```lean
locNorm κ is a seminorm on letter functions, ≥ the sup norm, submultiplicative under bounded bilinear maps (κ ≤ 1), with locNorm(e^f − 1) ≤ e^{locNorm f} − 1, locNorm((1+β)⁻¹ − 1) ≤ N/(1−N), and Σ_{j∈V} locNorm(softmax_V(ψ)_j − 1/|V|) ≤ 8 max_j locNorm ψ_j when that max is ≤ 1/4. A two-point version of each holds as well.
```

**Why this is the heart:** This is what makes the paper's term enumeration ('reads ≤ 2k+1 letters, costs ε^{2+4k}') into algebra. Without the \|V\|-uniform softmax lemma, constants pick up n through the averaging attention.

**How to attack it:** Group decompositions by support set (T : Finset (Fin n) → …). Prove the bilinear bound by regrouping over S ∪ S'. Prove exp via Finset.prod_add (no series). Prove the inverse via the Neumann series and locNorm ≤ κ^{-n}·sup to pass to the limit. Write softmax − 1/\|V\| = (α_j − β)/(\|V\|(1+β)) with α_j = e^{ψ_j} − 1 and β = mean α. Test the bilinear lemma on two single-letter functions first.

### `Network propagation: value, ζ-difference, θ-difference and mixed bounds`

```lean
For θ, θ' in the doubled ball, κ ∈ (0,1] and ε² ≤ cκ: locNorm κ of the logit differences satisfies (a) θ-difference at ζ = 0: ≤ Cρ(ε² + ε³κ^{-1}); (b) mixed θ/ζ difference: ≤ Cρε⁷κ^{-3}. Here ρ = ‖θ'−θ‖/ε and C depends only on a0, K, nH.
```

**Why this is the heart:** This is where all the ε powers and the uniformity in n, widths, alphabet, masks and Nb are earned. It is the largest body of work, about 16 layer lemmas.

**How to attack it:** Fix the order: parameter bounds → embed → block-1 scores → block-1 weights (softmax lemma) → block-1 attention output → block-1 MLP (affine via the pattern lemma; the bias part uses εNb ≤ c₀) → block-2 scores (on block-1 outputs) → block-2 weights → block-2 output → readout. Track separate bounds for the answer position n+3, which has no κ^{-1} on its own token. Use that ω(ζ=0) = 1/\|vis\| does not depend on θ. Land the value bounds first (they also give the sup bounds at κ = 1), then θ-differences, then ζ-differences, then mixed.

## Invariants to bake into the model

Cheaper as structure fields than as case splits later: Lean's `x / 0 = 0`
convention means a cancellation that ignores the zero case gets stuck on a
false-looking identity.

- `Euclidean residual norms`: Every residual or hidden vector is measured in EuclideanSpace ℝ (Fin d) (`WithLp.toLp 2`), never with the Pi sup norm of `Fin d → ℝ`. — Trace.opNorm and InitHyp are Euclidean. Mixing in sup norms introduces √dres or √dmlp factors and breaks width-uniformity of the constants.
- `Constants depend only on (a0, K, nH)`: a₁ := max a0 0 + 1/4, and every C, c, c₀ is an explicit expression in a₁, nH and √K. cemb enters only through r₀ ≤ 1/8. — Surface quantifier order. Any dependence on n, widths, \|Sym\|, p, Nb or masks makes the existential false to its spec.
- `Doubled-ball pattern`: On closedBall(εθ̃, 2r₀ε) (2r₀ ≤ 1/4): \|bIn l m\| ≥ 3ε/4, sign bIn l m = sign θ̃.bIn l m, and every MLP pre-activation for every w and every real ζ has the sign of θ̃.bIn l m. — Makes the MLPs affine with a fixed mask D_l = diag[0 < θ̃.bIn l m], which is independent of θ', w and ζ. Both the two-point algebra and DifferentiableAt depend on it.
- `Convex attention weights`: For every real ζ, attnWeight ζ θ l h x i · is nonnegative on vis(l,h,i) and sums to 1 there. At ζ = 0 it equals 1/\|vis(l,h,i)\|. — Gives ζ-independent sup bounds for the pattern lemma, and θ-independence of ω(0) for the mixed difference.
- `0 < λ ≤ 1 and κ = λ`: mixrate ≤ 1 always. Under the part-(ii) hypotheses, λ > 0. The locality weight used in the credit lemma is exactly κ = λ. — Submultiplicativity needs κ ≤ 1, λ^{n−\|S\|} = λ^n λ^{-\|S\|} needs λ > 0, and zpow manipulations need λ ≠ 0.
- `Start count and query fixed`: Task.Ex sums over the letters w only. s0 and wstar are constants of the prompt. — Positions 0, 1, n+2, n+3 contribute weight κ^0. Only letter positions carry κ^{-1}.

## Modelling decisions

### Prove part (ii) with a locality-weighted norm N_κ (weight κ^{-\|S\|} for a term reading letters S), not with the paper's ζ-Taylor expansion and term enumeration (prop:score-degree).

- **Rejected:** Formalize eq:score-split: Taylor coefficients J^{[k]}, a decomposition into T_{k,ν} each reading ≤ 2k+1 letters, the remainder J_rem with k₀ = ⌈(n−1)/2⌉, then lem:score-sum's geometric summation.
- **Because:** The paper's route needs explicit k-th ζ-derivatives of a two-block softmax network, combinatorial support counting and a Taylor remainder. N_κ gets the same bounds (score ε⁴κ^{-2}, value path ε²κ^{-1}) from submultiplicativity and three nonlinear lemmas (exp, inverse, softmax). The credit lemma then gives λ^n·N_λ in one line.
- **Scope caveat:** The formal proof does not contain prop:score-degree as a statement. If a referee wants that proposition itself formalized, it is extra work. The headline does not need it.

### Bound gradient norms through Lipschitz (two-point) estimates and `norm_fderiv_le_of_lip'`, never by computing fderiv of the network.

- **Rejected:** Track jets (value, directional derivative) through the network using HasFDerivAt chain rules, or complexify along a direction and use the Cauchy estimate (Complex.norm_deriv_le_of_forall_mem_sphere_norm_le).
- **Because:** Exact differences obey an exact Leibniz rule, ab' − ab = (a'−a)b' + a(b'−b), so no limits, chain rules or complex copies of the network are needed. Calculus is confined to the DifferentiableAt conjuncts, which follow from the pattern lemma and fun_prop.

### Group locality decompositions by support set: T : Finset (Fin n) → ((Fin n → Sym) → E).

- **Rejected:** Decompositions indexed by an arbitrary Fintype ι.
- **Because:** Avoids universe and iInf problems over types, makes products a single regrouping over S ∪ S', and loses nothing, since a sum of DependsOn-S functions is DependsOn S and its sup is at most the sum of sups.

### Use the ReLU-free network only through a pointwise rewriting lemma (pattern lemma), not as a second forward-pass definition.

- **Rejected:** Define a proof-side `logitsLin` with fixed masks and prove logits = logitsLin on the doubled ball.
- **Because:** Fewer definitions to keep in sync. If DifferentiableAt is easier against a max-free expression, define `logitsLin` proof-side; it never enters Model/.

### Prove (ii-b) by splitting off the score functional and reusing the (ii-a) owner.

- **Rejected:** A separate full analysis of z_θ.
- **Because:** Only the ζ = 0 θ-difference bound is new, and it is a subset of the step-6 lemmas. This matches the paper's 'adding the zero-order term' in lem:score-sum.

### Characters of ZMod K via Mathlib's ZMod.stdAddChar, plus a hand-proved Parseval if none exists.

- **Rejected:** Hand-rolled exp(2πi·a.val·j/K) with explicit mod-K arithmetic, or a real Wirtinger inequality on the cycle.
- **Because:** Additivity mod K comes for free (AddChar.map_add_eq_mul). Orthogonality comes from AddChar.sum_mulShift.

## Paper guide

- **Part (i): conditioning, Fourier modes, Parseval, \|ρ_j\| ≤ λ:** Paper/appendix.tex:250-293 (lem:target-contraction, proof, sharpness remark), Paper/appendix.tex:253 (K = 2 example λ = \|1−2p\|, a sanity test), Paper/body.tex:89-96 (def:lambda, eq:lambda), Paper/direct_proofs/conditional_outcome.tex:1-107 (more explicit write-up) — Writing steps 1a-1e, or when the transform of Ĝ or the cos comparison does not come out.
- **Power counting: why a score costs ε⁴ and reads two letters, and why ζ = 0 is additive:** Paper/appendix.tex:295-319 (eq:score-split, prop:score-degree), Paper/appendix.tex:325-489 (prop:network and proofs: fixed ReLU pattern, \|b_m\| ≥ (3/4)ε, the MLP offset O(ε) using ε ≤ c/N_b), Paper/body.tex:181-189 (ζ-expansion paragraph) — A step-6 layer bound has the wrong ε or κ exponent, or the MLP bias term seems to bring in Nb.
- **Summation and the role of ε² ≤ cλ, ε ≤ cλ:** Paper/appendix.tex:491-536 (lem:score-sum, its proof, the choice of c at line 528, proof of thm:main) — Choosing the explicit constants c₀, c, C in steps 7-8, and checking that C_att ε⁴ ≤ λ²/2 corresponds to locNorm ψ ≤ 1/4.
- **Forward-pass conventions and prompt layout:** Paper/appendix.tex:89-115 (forward-pass display: 1/√d_h, single W_V, MLP without output bias; prompt positions), Paper/appendix.tex:76-81 (rem:norms: Euclidean parameter norm, β excluded), Paper/appendix.tex:227-230 (start and query fixed; bounds also on the doubled ball) — Checking that a lemma about `Trace.block` or `Trace.embed` matches the paper, or justifying the doubled-ball neighbourhood.
- **Gradient = expectation of Jacobian times signal:** Paper/appendix.tex:237-248 (lem:gradient-split), Paper/body.tex:163-178 (def:credit, eq:credit-degree) — Only if a reviewer asks how the fderiv-based credit relates to E[Jᵀ(y−u)]. The Lipschitz route does not need this identity.
- **Constants' dependence:** Paper/appendix.tex:43-65 (ass:init and the constants paragraph), Paper/body.tex:117-128 — Auditing that the explicit constants use only a0, K, nH.

## Known blockers

- **Contents of `Trace.Prior` (Trace/Model/Prior.lean):** I have not seen the file. It is a hypothesis of every surface theorem and could be unused, a stated prop:score-degree, or (worst case) inconsistent. Next: Print it first (step 0). If it states prop:score-degree for Trace.logits, consider the paper's lem:score-sum summation as a shortcut for step 6. Either way, report in the handoff what the headline is conditional on.
- **Parseval / character orthogonality on ZMod K:** I am not sure whether Mathlib has a ready Parseval for ZMod.dft. The exact name and hypotheses of AddChar.sum_mulShift and of ZMod.isPrimitive_stdAddChar are medium-confidence, and their module paths may have moved. Next: `grep -rn "sum_mulShift\\|isPrimitive_stdAddChar\\|stdAddChar_apply\\|dft" Mathlib/` before writing parseval_zmod. Fallback: geometric sums with Complex.isPrimitiveRoot_exp and IsPrimitiveRoot.geom_sum_eq_zero.
- **opNorm ≤ Frobenius ≤ full parameter norm:** I do not know a confirmed Mathlib name for 'l2 operator norm ≤ Frobenius norm' that is compatible with the surface's `Trace.opNorm` (a CLM norm of Matrix.toEuclideanLin). Next: Prove it directly with ContinuousLinearMap.opNorm_le_bound and row-wise Cauchy–Schwarz (≈20 lines), plus the injective-index sum lemma of step 4. Check for `Matrix.l2_opNorm_le_frobenius*` first.
- **Passing the inverse (Neumann) series to the limit in locNorm:** locNorm is an sInf over finite decompositions. Closure under limits needs the sup-norm comparison locNorm ≤ κ^{-n}·sup and a `le_of_tendsto` argument. That is fiddly with sInf. Next: Prove the Lipschitz comparison \|locNorm f − locNorm g\| ≤ κ^{-n}‖f − g‖ once, then use continuity. Alternatively state the inverse bound with an arbitrary slack δ > 0 and absorb it into the constants.
- **`DependsOn` API names:** Not sure `DependsOn.mono`, `dependsOn_const` or the add/mul closure lemmas exist under these names. Next: Grep Mathlib/Logic/Function/DependsOn.lean. Each missing lemma is one line from the definition.
- **Explicit constant bookkeeping:** Step 6 multiplies about a dozen layer constants. Errors only show up at the final `nlinarith`. Next: Fix a₁ and a single generous M := (16·(nH+1)·(a₁+1))^8 early. State every layer lemma with M-powers and keep ε ≤ c₀ ≤ 1/M for smallness.
- **Paper discrepancies to report (not to fix in the surface):** (a) ε > 0 is missing from thm:main (repaired in the surface). (b) λ = 0 is possible but excluded by the hypotheses. (c) n ≥ 2 appears unnecessary for this route. (d) The paper's (ii) proof relies on prop:score-degree's Frobenius bounds being width-free; the locality route only needs operator norms. Next: Record these in the handoff with file:line, once the proof confirms that hn is unused.

## Initial proof owners

| # | New declaration | Discharges | File | Difficulty | Depends on |
| --- | --- | --- | --- | --- | --- |
| 1 | `Trace.Proof.unseenLettersContraction` | `Trace.unseen_letters_contraction` | `Trace/Analysis/Contraction.lean` | moderate | — |
| 2 | `Trace.Proof.unseenLettersScoreCredit` | `Trace.unseen_letters_score_credit` | `Trace/Analysis/ScoreCredit.lean` | hard | `Trace.Proof.unseenLettersContraction` |
| 3 | `Trace.Proof.unseenLettersOutcomeCredit` | `Trace.unseen_letters_outcome_credit` | `Trace/Analysis/OutcomeCredit.lean` | moderate | `Trace.Proof.unseenLettersContraction`, `Trace.Proof.unseenLettersScoreCredit` |

Rebuild green after each one. Do not batch.

## Files

| Path | Role | Purpose |
| --- | --- | --- |
| `Trace/Model/Prelude.lean` | audit-surface | The vocabulary, and the quantity the theorem is about |
| `Trace/Model/Theorem.lean` | audit-surface | The author's result, and one theorem per claim under it |

## Definition of done

- [ ] The mechanical checks pass. They are run by the extension over the whole project between rounds — the build, the `sorry` warnings, the axiom profile of every headline, a sweep for escape hatches, the closure checks, and the two layout rules — and what they find comes back to you as the next round's brief. Build the modules you created and read their errors; do not try to run the gate.
- [ ] `#print axioms` on the capstone reports exactly the three ambient axioms — propext, Classical.choice, Quot.sound — and nothing else.
- [ ] No `declaration uses sorry` warning anywhere in the build output.
- [ ] A grep sweep for `sorry`, `admit`, `native_decide` and `axiom` over the project comes back clean, and agrees with the other two checks.
- [ ] The closure checks have been observed to *fail* on a deliberately broken control, so that their passing means something.
- [ ] The audit surface still states the paper's theorem: every claim the paper makes is in it, and nothing was weakened to make a proof go through. No script decides this one — the extension's handoff check reads the paper and the surface side by side, and it runs as the last step of a run.

## Risks

- **A round closes the capstone by leaning on lemmas that are themselves still `sorry`, which looks like progress and is not.** — The open-goal list is read out of the Lean sources every round, so a lemma left `sorry` stays on it however the declaration above it was closed.
- **An agent weakens an intermediate lemma until it is provable but useless.** — Whatever depends on it stops building, and the next round re-reads the sources rather than the plan.

## Briefs


---

### Source material this was built from

- Target: `Paper/body.tex:191`
- Proof: `Paper/appendix.tex:530`
- Cross-referenced: `ass:arch`, `ass:init`, `ass:task`, `eq:credit-degree`, `lem:target-contraction`, `lem:score-sum`, `def:task`, `eq:horizon`, `prop:score-degree`, `lem:gradient-split`, `eq:thm1-full`
- Files read: `Paper/main.tex`, `Paper/body.tex`, `Paper/figures/overview.tex`, `Paper/figures/counting_chain.tex`, `Paper/figures/theory_measurement.tex`, `Paper/appendix.tex`
