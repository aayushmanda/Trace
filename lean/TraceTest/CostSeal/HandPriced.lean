import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Rand
import Arlib.Computation.Std

/-!
# Seal breach: pricing a line by hand

`Charged.op o a` is arlib's escape hatch — one operation of the currency, returning a value
the algorithm computed for itself — and a program may not call it inline.  The charge says
one coin, and the seal cannot see what the argument cost, so it forbids the hatch directly.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def handPricedCheat (level : Nat) (bits : Nat → Bool) : Charged StdOp Cell Bool :=
  Charged.op (StdOp.rand .accept) (bits level)

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.handPricedCheat, Arlib.Computation.Charged.op)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
