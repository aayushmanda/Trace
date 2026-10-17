import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Std

/-!
# Seal breach: observing a computation's own tally

`Charged.cost` is computable and "produces no value" — until `decide` is applied to it.
`Roster.cost_filterErase` equates a thinning pass's tally to `d.card`, so a program that may
read `cost` has a free cardinality and therefore a free `cardEq`.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def costCheat (p : Charged StdOp Cell Nat) : CostVec StdOp := p.cost

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.costCheat, Arlib.Computation.Charged.cost)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
