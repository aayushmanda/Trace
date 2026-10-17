import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Std

/-!
# Seal breach: laundering a tally through a change of currency

`exchange` is computable and takes a `Charged` to a `Charged`, so nothing the compiler checks
stands in the way; at a rate of zero it re-prices any computation at nothing.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def exchangeCheat (p : Charged StdOp Cell Nat) : Charged StdOp Cell Nat :=
  Charged.exchange (fun _ => (0 : CostVec StdOp)) p

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.exchangeCheat, Arlib.Computation.Charged.exchange)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
