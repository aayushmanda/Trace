# Cancelled Credit: correct reasoning from corrupted supervision

The paper specifies a normalization-free one-block Transformer, its fixed
matrix directions, the four Euclidean trainable coordinates, and their
initialization before deriving its logit and dynamics.
The source is the last visible state `x_N`; `\bm x_{<i}` denotes the prefix.
Notation uses `N`, `\eta`, `\gamma`, `\varepsilon`, `\kappa`, and `\rho`,
with `\alpha(q)`, `\phi(q)`, and `F(q)` for the three principal functions.
Expectation uses `\mathbb E`, probability uses `\Pr`, and vectors/matrices are bold.
Credit-function derivatives use primes; time derivatives use Leibniz notation
and native `\frac` / `\tfrac`;
vector/matrix gradients use `\nabla`.

Build from the repository root:

```bash
bash Paper2/build.sh
```

The output is `Paper2/build/trace.pdf` (also `main.pdf`).

The manuscript retains matrix-gradient verification, matched population flow,
conservation and quadrature, and a concise untied-training control. The legacy
GD and stale-copy scripts and raw data remain available, but their secondary
figures and tables are no longer included in the paper.

Run with PyTorch, NumPy, SciPy and Matplotlib:

```bash
python Paper2/checks/one_block.py
python Paper2/checks/first_integral.py
python Paper2/checks/causal_credit.py
python Paper2/checks/anti_law.py
python Paper2/checks/untied_ladder.py --law anti --n 16 --out Paper2/checks/results/untied_ladder_anti_n16_v2.json
python Paper2/checks/make_untied_table.py Paper2/checks/results/untied_ladder_anti_n16_v2.json
```

The main law is the anti-copy law: M anti-copies of the source (E[M] = 1 - kappa) at random
decoy positions, the other decoys iid. Under exact cancellation the multiset of displayed values
is independent of the source, so no readout of uniform attention carries rule credit.

The installed environment is `/home/hariguru/aayus/.venv/bin/python`.
`one_block.py` checks 72 matrix-level configurations and performs 96 population
flow integrations. `first_integral.py` checks routing-time bounds, conservation,
quadrature agreement, and the exact over-cancellation threshold.
`causal_credit.py` propagates the causal error law exactly, checks 18 matrix-gradient
configurations, and checks nine additional clean matrix configurations. It integrates 36
shared-parameter routing trajectories across clean, independent-error and
structured-error supervision, plus an eight-run noise-rate sweep.
`anti_law.py` integrates the cross-entropy flow of the anti-copy law.
`untied_ladder.py` trains every matrix entry from three initialization families (288 runs).
Raw results are in `checks/results/`; generated flow figures are in `figures/`.

Correlation loss gives full routing and finite-time Bayes prediction before the
specified stopping gap. Cross-entropy now proves complete routing with the same
initialization orders for fixed prefix size and positive label margin.
The causal theorem creates cancellation at every one of N supervised positions
after a supplied shared seed and one warm-up transition. Both regimes use fresh
uniform instructions and have matching local error rates; the expected incorrect
fraction of the whole trace tends to eta/2. The noisy-transition indicators are
independent Bernoulli(eta), giving a constant wrong fraction with high probability.
The audit also propagates the full error-count distribution. All selected prefixes share the four
trainable coordinates. Fixed matrix directions and relative source flags remain
part of the model; unrestricted-weight convergence is not claimed.

Two main theorems state the optimization separation and its causal realization.
Theorem 1 includes the exact correlation-loss reliability rescaling, routing
separation, the exact failure boundary, and complete
cross-entropy routing for fixed N and gamma. Three main-text lemmas give credit
cancellation, matched laws, and the gradient-flow invariant; Proposition 1 derives
the reduced model from the network.

The proof map is in [`proofs/README.md`](proofs/README.md). Three additional appendix
supporting estimates remain; the other derivations are explicit mathematical
steps within the main proofs.

The causal experiment now includes a fully correct baseline with the same seed,
instructions and supervised positions. Independent corruption rescales clean
routing time exactly by 1/(1-2 eta); structured errors add the inverse-initialization
penalty. Strict three-way ordering is proved for sufficiently small initialization,
not for every initialization. The cancelling construction does not extend to eta=0.
