import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Roster
import Arlib.Computation.Std

/-!
# Seal breach: supplying your own measure

`opUpdate` is computable and takes the measure as an argument, so nothing the compiler checks
objects to a measure that reports nothing — this is `Charged.exchange (fun _ => 0)` for space.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def measureCheat (d : Roster Nat) : Charged StdOp Cell (Roster Nat) :=
  Charged.opUpdate (StdOp.roster .insert) (fun _ => Residency.ofFun fun _ => 0) id d

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.measureCheat, Arlib.Computation.Charged.opUpdate), (Trace.Program.measureCheat, Arlib.Computation.Residency.ofFun)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
