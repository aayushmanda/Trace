import Trace.Meta.CostSeal
import Arlib.Computation.Charged
import Arlib.Computation.Roster
import Arlib.Computation.Rand
import Arlib.Computation.Slot
import Arlib.Computation.Std

/-!
# The seal, made to fail

When the development has a `Model/Program.lean`, it runs `#programSeal` on every
build, and it passes.  That
is not evidence.  An audit that has never rejected anything is
not known to reject anything, and the two commands are exactly the kind of code
whose failure mode is silence — a filter that matches nothing reports a clean
build.

Each test below is a `#guard_msgs` block: it **passes when the checker produces
exactly the stated error**, so a future change that quietly opens one of these
routes turns this file red.  This is the same discipline as
`ArlibTest/Computation.lean` §3, applied to the checkers rather than to the
compiler.

`#programSeal` scans only the canonical namespace `Trace.Program`, so every cheat
it must catch is planted there, each in its **own module** under
`TraceTest/CostSeal/` — which keeps each expected message a single entry and keeps
the tests independent of the order they run in.

## What is covered

* §1 — the cheats the *compiler* rejects, so that a change making one of them
  compile is caught before either seal is reached.
* §2 — `#programSeal` refusing a retargeted or vacuous seal.  The routes past
  the compiler, inside the algorithm — including the one that only a
  *transitive* scan catches — are planted one per module in `TraceTest/CostSeal/`.
* §3 — further compiler rejections (sampling level, profile, residency).
The positive direction is covered by the main build when `Model/Program.lean`
exists: it runs the command over the real algorithm and it reports clean.
-/

namespace TraceTest

open Arlib.Computation

/-! ## 1. What the compiler rejects on its own -/

/-! Copying a computation's result to drop its charges.  This is the cheat that
would make every bound in the development vacuous, and it does not compile,
because `Charged.val` is `noncomputable`. -/
/--
error: failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Charged.val', which is 'noncomputable'
-/
#guard_msgs in
def dropTally (p : Charged StdOp Cell Nat) : Charged StdOp Cell Nat := pure p.val

/-! Reading a dictionary's size without asking it.  `Roster.card` is the
specification's view and is `noncomputable`, so a program that wants the number
must call `Roster.size` and pay. -/
/--
error: failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Roster.card', which is 'noncomputable'
-/
#guard_msgs in
def freeSize (d : Roster Nat) : Nat := d.card

/-! Forging a tally.  There is no syntax for asserting what something cost. -/
/--
error: Invalid `⟨...⟩` notation: Constructor for `Arlib.Computation.Charged` is marked as private
-/
#guard_msgs in
example : Charged StdOp Cell Nat := ⟨0, 0, 1⟩

/-! ## 2. `#programSeal` — what it refuses before it scans

The seal only scans the canonical namespace `Trace.Program`.  The breaches it must
catch inside that namespace are planted one per module under `TraceTest/CostSeal/`
(`Exchange`, `Noncomputable`, `CasesOn`, `HandPriced`, `Cost`, `Transitive`,
`Measure`, `PeakAt`), because declarations cannot be removed from the environment
and a shared namespace would make each expected message depend on the others.
This module imports none of them, so `Trace.Program` is empty here. -/

/-! **Retargeting the seal.**  A seal pointed at some other namespace would check
whatever the author chose to put there, and nothing else. -/
namespace Breach.Retarget.Program
def harmless : Nat := 0
end Breach.Retarget.Program

/--
error: Program seal must target the canonical namespace Trace.Program, not TraceTest.Breach.Retarget.Program.
-/
#guard_msgs (whitespace := lax) in
#programSeal TraceTest.Breach.Retarget.Program

/-! **A vacuous seal.**  With nothing under `Trace.Program`, the filter matches no
declaration; that must be an error, not a clean report. -/
/--
error: Program seal checked zero declarations under Trace.Program.  Put the executable       declarations from Model/Program.lean in this exact namespace; a vacuous seal is not a seal.
-/
#guard_msgs (whitespace := lax) in
#programSeal Trace.Program

/-! ## 3. More of what the compiler rejects -/

/-! **Reading the sampling level.**  The level is inside a `Sampler` and
`Sampler.levelOf` is noncomputable, so the compiler rejects this before the seal
gets to it — which is the stronger of the two rejections. -/
namespace Breach.PeekLevel.Program
/--
error: failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Sampler.levelOf', which is 'noncomputable'
-/
#guard_msgs in
def cheat (s : Sampler) (n : Nat) : Nat := n * 2 ^ s.levelOf
end Breach.PeekLevel.Program

/-! **Reading what a computation holds.**  A profile tells a program how much data
it holds — which is exactly what the data seal hides. -/
/--
error: failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Charged.space', which is 'noncomputable'
-/
#guard_msgs in
def readProfile (p : Charged StdOp Cell Nat) : Profile Cell := p.space

/-! **Measuring a dictionary directly.**  `Residency.at'` is the space analogue
of `Roster.card`, and noncomputable for the same reason. -/
/--
error: failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Residency.at'', which is 'noncomputable'
-/
#guard_msgs in
def readResidency (r : Residency Cell) : Nat := r.at' Cell.roster

end TraceTest
