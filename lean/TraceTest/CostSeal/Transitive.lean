import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Std

/-!
# Seal breach: the same read, one hop away

`Charged.steps` is deliberately absent from `forbiddenInProgram`: it is
`CostVec.steps C p.cost`, so a blacklist of *direct* mentions would miss it entirely.  This
test is what makes the transitive walk a checked fact, and it is the one to run first if
`reaches` is ever changed.  The message names the intermediary the program actually called.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def transitiveCheat (p : Charged StdOp Cell Nat) : Nat := Charged.steps (Rate.unit StdOp) p

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.transitiveCheat, Arlib.Computation.Charged.steps)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
