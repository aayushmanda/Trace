/-
# TraceTest

Tests that are not theorems: the planted breaches that make the build-time
checkers falsifiable. Kept out of `Trace` so that the library a client
imports carries no deliberately-broken declarations.
-/

import TraceTest.CostSealBreaches
import TraceTest.CostSeal.Exchange
import TraceTest.CostSeal.Noncomputable
import TraceTest.CostSeal.CasesOn
import TraceTest.CostSeal.HandPriced
import TraceTest.CostSeal.Cost
import TraceTest.CostSeal.Transitive
import TraceTest.CostSeal.Measure
import TraceTest.CostSeal.PeakAt
