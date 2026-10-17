# Proof source map

The paper specifies the one-block matrices and deterministic initialization
before deriving training dynamics. Source positions use `x_N`, `q=q_s-q_d`, and `\alpha(q)`;
`\bm x_{<i}` denotes the visible state prefix. Expectation uses `\mathbb E`, probability uses `\Pr`, credit
derivatives use primes, time derivatives use Leibniz fractions, and vector/matrix
gradients use `\nabla`.

| Source | Proof steps |
| --- | --- |
| `network.tex` | Attention, MLP output identity, parameter chain rule |
| `credit.tex` | Cancellation identity; matched conditional laws and total variation |
| `correlation.tex` | Initialization, balance and existence, conservation, hitting time, gate validity and prediction |
| `routing.tex` | Three supporting estimates: attention, primitive growth, barrier size; exact reliability time rescaling, routing and failure proofs |
| `cross_entropy.tex` | Exact derivatives, nonlinear bound, integrating factors, existence, escape time, continuation and complete routing |
| `trace.tex` | Causal error kernel, matched local laws, moment induction, filtration-based binomial law and concentration, shared flow, fully correct control, reliability rescaling and uniform routing bounds |

The main text has three lemmas: credit cancellation, matched distributions, and
the gradient-flow invariant. Three technical lemmas in `routing.tex` support their
use in Theorem 1. Theorem 1(i)--(ii) gives correlation routing, (iii) the failure
barrier, and (iv) complete cross-entropy routing. Theorem 2 covers the causal
construction under correlation loss.
All other calculations are numbered steps inside the proofs of the main claims.
The cross-entropy proof uses one bound constant and the local scalar `m=r^2`;
neither is an additional trainable coordinate.

The fixed gate bound is chosen before training. The correlation proof establishes
finite-time recovery of the Bayes predictor on the interval between routing and
the stopping gap `log(N-1)+1`. Trapped trajectories retain the reduction for all
time. Cross-entropy proves complete routing for fixed N and gamma by extending the
local comparison with a positive lower bound on the MLP product.
The causal result supervises all N positions after a shared seed and one warm-up
transition; its history-dependent errors create exact cancellation at every
selected position. A separate gate bound covers the variable prefix lengths.

Verification uses `one_block.check_network()` for 54 matrix-level configurations
and `first_integral.py` for routing bounds, quadrature, conservation, and failure.
`causal_credit.py` checks the finite-state causal law, 18 matrix-gradient
configurations, 36 routing trajectories against the explicit time bounds, and an eight-run
noise-rate sweep. Nine further matrix configurations check the fully correct control.
