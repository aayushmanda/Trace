"""Computational certificate for thm:fixed-depth-identifiability.

Computes H = <T_g T_h^-1 : g,h in the gate alphabet> for the actual
boolean_circuit gate set (reconstructed from src/data/boolean_circuit_tasks.py's
own _apply_gate, not a re-derived description of it), and its centralizer in
S_16. If C_{S_16}(H) is trivial, the theorem gives: the population outcome
objective's unique zero-loss row-stochastic executor is P_g = T_g, i.e. the
Boolean task's depth-8 outcome failure cannot be attributed to an alternative
exact factorization of the terminal map.

No model is trained; this is a pure group-theoretic computation.
"""
from __future__ import annotations

import itertools
import math

from sympy.combinatorics import Permutation, PermutationGroup

from src.data.boolean_circuit_tasks import N_BITS, _apply_gate

K = 2 ** N_BITS


def _state_to_int(state: list[int]) -> int:
    v = 0
    for b in state:
        v = (v << 1) | b
    return v


def _int_to_state(v: int) -> list[int]:
    return [(v >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)]


def all_gate_strings() -> list[str]:
    gates = [f"x{i}" for i in range(N_BITS)]
    for op in "cs":
        for first, second in itertools.permutations(range(N_BITS), 2):
            gates.append(f"{op}{first}{second}")
    for ca, cb, t in itertools.permutations(range(N_BITS), 3):
        gates.append(f"t{ca}{cb}{t}")
    return gates


def gate_to_permutation(gate: str) -> Permutation:
    images = [_state_to_int(_apply_gate(_int_to_state(v), gate)) for v in range(K)]
    return Permutation(images)


def main():
    gates = all_gate_strings()
    assert len(gates) == 52, f"expected 52 gate strings, got {len(gates)}"
    perms = {g: gate_to_permutation(g) for g in gates}

    h0 = gates[0]
    Th0_inv = perms[h0] ** -1
    diffs = [perms[g] * Th0_inv for g in gates]
    H = PermutationGroup(diffs)

    order = H.order()
    a16_order = math.factorial(K) // 2
    is_a16 = order == a16_order and all(p.is_even for p in diffs)

    s16_gens = [
        Permutation([1, 0] + list(range(2, K))),
        Permutation(list(range(1, K)) + [0]),
    ]
    S16 = PermutationGroup(s16_gens)
    centralizer_order = S16.centralizer(H).order()

    print(f"gate alphabet size: {len(gates)}")
    print(f"|H| = |<T_g T_h^-1>| = {order}")
    print(f"16!/2 = {a16_order}")
    print(f"H == A_16 (order match + all generators even): {is_a16}")
    print(f"|C_S16(H)| = {centralizer_order}")
    print(f"identifiability certificate holds (trivial centralizer): {centralizer_order == 1}")


if __name__ == "__main__":
    main()
