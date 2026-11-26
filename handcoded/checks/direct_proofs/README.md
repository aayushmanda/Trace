# Separate direct-proof draft

Read [the PDF](build/proofs.pdf); its standalone source is [proofs.tex](proofs.tex). I have not edited the main manuscript or its PDF. The main appendix was modified concurrently after the recorded snapshot; this draft preserves the snapshot rather than incorporating those later edits.

The proof package includes all four headline theorem statements from the recorded source snapshot and complete supporting proofs, including the comparison under the same architecture, mask, initialization and trained variables.

| Result | Argument in this draft | Assessment |
| --- | --- | --- |
| Outcome interaction credit | Conditional projections, support counts, geometric summation; remainder handled by an orthogonal L2 projection | Conclusion and assumptions retained |
| Step credit | Conditional cancellation, full-prefix dilution, explicit nonzero degree-six coefficient | Conclusion and assumptions retained |
| Outcome plateau | Condition on each term's input support, bound its credit directly, then use the loss identity and flow energy | Direct route closes with the same network estimates |
| Binary joint convergence | Complete existing reduction and simultaneous-flow bootstrap, with the tanh bound corrected | Conclusion and even the existing conservative time constants retained |

The outcome proof does not use the interaction decomposition. The degreewise credit proof retains a short construction of that decomposition using commuting conditional expectations. This is a simpler presentation of the same mathematical object, not a new decomposition. The Transformer expansion and binary ODE estimates remain necessary. This is therefore a candidate replacement for the outcome proof and an independent review package for the other results, not a claim that all four theorems now have shorter or entirely new proofs.

## Corrections included

1. From `|a| >= a_sc/8`, the valid conservative bound is `tanh²(a/2) >= a_sc²/1024`; the manuscript's displayed denominator 256 is slightly too strong. The existing descent denominator 12288 already equals `3 * 4 * 1024`, so the subsequent constants 860160, 98304, and the time bounds remain valid. Theorem 4 is unchanged.
2. The rule-credit setup now distinguishes the displayed state tokens from the prompt letters. All visible tokens are not functions of the earlier letters: the prompt contains the current letter as well. The cancellation argument conditions on the other letters and uses the independence of the current match bit.
3. The interaction-credit remainder is bounded using the contraction of its orthogonal projection in L2. No unsupported supremum bound on the projected remainder is used.
4. The binary flow includes an explicit global-existence argument from its energy identity.

## Validation

- `check_direct_proofs.py`: 5,080 enumerated support/distribution cases for conditional target contraction, the tower identity, and the exact target spectrum; 48 cases for the explicit step monomial coefficient and its lower bound; corrected Phase III constants; all internal references.
- The repository's `handcoded/theory_checks/theorem4_checks.py`: all checks pass, including the initialization threshold on 250 parameter settings, derivative identities at 2,500 points, and agreement between full and reduced model flow at 200 points.
- The standalone PDF compiles with Tectonic. These computational checks support the algebra; they are not a formal proof checker or an independent peer review. The analytic proofs and architecture estimates are included in full for review.

Run from the repository root:

```sh
uv run python handcoded/checks/direct_proofs/check_direct_proofs.py
/home/hariguru/aayus/.venv/bin/python handcoded/theory_checks/theorem4_checks.py
```

Build from `handcoded/checks/direct_proofs`:

```sh
mkdir -p build
tectonic -X compile proofs.tex --outdir build --keep-logs
```

`source_hashes.txt` records the manuscript snapshot used for the preserved statements and supporting proofs. The files here do not import changing manuscript sources at build time.

## Limits retained

Joint convergence remains binary (`K=2`), excludes `p=1/2`, uses a fixed two-source gate mask and structured directions, fixes the value factor, and requires the stated imbalance for misaligned seeds. General-K convergence is not proved here. Generic nonzero step credit does not supply a uniform random-initialization margin. The plateau's probability guarantee concerns expected correct-answer probability, not greedy-decoding accuracy.

## Snapshot note

At final verification, `Paper/main.tex` and `Paper/body.tex` still matched the recorded hashes. `Paper/appendix.tex` had changed concurrently (including joint-flow material). This package does not overwrite those edits. Compare any later theorem or parameterization changes before integrating the draft; its binary joint-flow proof deliberately follows the recorded snapshot.
