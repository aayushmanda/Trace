import Trace.Meta.CostSeal
import Arlib.Computation.Roster

/-!
# Seal breach: projecting a dictionary's contents through the eliminator

`casesOn` is generated public even though the constructor is private, and it is compiled,
so this is the one route the compiler genuinely leaves open.

`#programSeal` only scans the canonical namespace `Trace.Program`, so the cheat is planted
there — in a module of its own, so that the expected message is a single entry and the test
does not depend on any other breach.  This module is not imported by `Trace`.
-/

namespace Trace.Program

open Arlib.Computation

def casesOnCheat (d : Roster Nat) : Nat :=
  Roster.casesOn (motive := fun _ => Nat) d (fun elems _ => elems.length)

end Trace.Program

/--
error: Seal breach in Trace.Program: [(Trace.Program.casesOnCheat, Arlib.Computation.Roster.casesOn)].  A program that reads a dictionary without a charged operation, or that is marked noncomputable, can perform work it does not pay for.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program
