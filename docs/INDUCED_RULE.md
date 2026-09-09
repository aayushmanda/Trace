# The induced rule of a trained model

The credit theorems are stated on transition matrices `P_g`. A Transformer has
none, so the bridge needs an object defined *from* the network. These are the
definitions used by `experiments/induced_rule.py` and `experiments/pullback.py`.

## Definition

At a trace-format decision position carrying source state `s` and gate `g`,

```
    P̂_g[s, s'] = P_θ(s' | s, g)
```

scored over the sixteen complete successor states and renormalised over them.
This is a **readout of the network, not a parameter**, and it depends on the
query format: change the serialisation and you change the object. The scripts
also record how much predictive mass fell outside the sixteen valid states
(`state_on_set_mass`), which is the honest measure of how well-posed the readout
is at a given checkpoint.

Reading `P̂_g` needs `K·M = 16 × 52 = 832` contexts × 16 candidates per
checkpoint. Gate strings differ in width (`x0`, `c01`, `t012`), so contexts
differ in length; absolute position embeddings make left padding unsafe and the
code buckets by length instead.

## The two scales

Split the deviation from complete mixing into its marginal and conditional
parts, `P̂_g − U = 1 ĉ_gᵀ + F̂_g`, and record

```
    γ̂        = max_g ‖ĉ_g‖₂        marginal scale
    ε̂_rule   = max_g ‖F̂_g‖_op      conditional scale
```

The depth bound is stated on a ball around complete mixing, so `ε̂_rule` says
whether a trained model is anywhere near the regime the bound describes. The
marginal-control assumption is `γ̂ ≲ ε̂_rule^(D−1)`; it is a hypothesis of the
theorem, not a conclusion, and the scripts report both scales so it can be
checked rather than assumed.

## Composition error

```
    δ_comp = E ½‖ p_θ(· | x) − e_{s₀}ᵀ P̂_{g₁} ⋯ P̂_{g_D} ‖₁
```

compares the model's own terminal answer law against the product of its induced
kernels. Both queries have to be in distribution for this to mean anything,
which is why the scripts train models that emit **a trace as well as a direct
answer** (`--condition both`). For a model trained on answers alone the
trace-format query is out of distribution and `P̂_g` is not well posed.

`δ_comp` is a function-level quantity. It does **not** bound the difference of
gradients — two functions can agree pointwise to any accuracy and have unrelated
derivatives — so it is necessary but not sufficient for the transfer corollary.

## The exponent test

Rescaling `F̂_g → λ F̂_g` leaves the model's learned rule geometry intact and
moves `ε̂_rule` exactly linearly, so a log–log fit of conditional credit against
`ε̂_rule` recovers the predicted exponent. The scripts report it two ways:

- `conditional`: the marginal part is dropped, tables stay doubly stochastic,
  which is the stated hypothesis of the bound. Predicted slope `D − 1`.
- `proportional`: marginal and conditional parts shrink together, so the
  marginal term dominates at late positions. Predicted slope falls toward 1.

The gap between the two *is* the content of the marginal-control assumption.

## The pullback

For a fixed credit field `W_g`, the pullback `Σ_g J_gᵀ vec(W_g)` with
`J_g = ∂vec P̂_g/∂θ` is exactly the parameter gradient of the scalar
`Σ_g ⟨W_g, P̂_g(θ)⟩`, so one backward pass gives it — no explicit Jacobian.
`pullback.py` takes `W_g = Rule(T_g − U)`, the true-executor direction, and
reports its cosine against `−∇_θ L_out` and `−∇_θ L_proc` on the same examples
and checkpoint, with a norm-matched random credit field in the same subspace as
a control.

Both quantities were validated by finite differences: the pullback direction
predicts the change in `Σ_g ⟨W_g, P̂_g⟩` to a ratio of 1.001 at `η = 1e-3`, and
the outcome gradient predicts the change in the loss to 0.848.
