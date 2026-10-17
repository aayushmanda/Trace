import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Std

/-!
# Seal breach: reading the measured peak

`Charged.peakAt` is computable on purpose — `#guard` has to read it for execution tests — so
the compiler does not object, and the seal is what stops an algorithm from using it as a free
size query.  Same standing as `Charged.cost`.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def peakAtCheat (p : Charged StdOp Cell Nat) : Int := p.peakAt Cell.roster

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.peakAtCheat, Arlib.Computation.Charged.peakAt)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
