import Trace.Meta.CostSeal
import Arlib.Computation.Std

/-!
# Seal breach: silencing the compiler

Writing `noncomputable def` turns off the half of the seal the compiler supplies.  Nothing
else about this declaration is wrong, which is the point: the marker alone is the breach.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

noncomputable def noncomputableCheat : Real := 0

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.noncomputableCheat, noncomputableProgram)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
