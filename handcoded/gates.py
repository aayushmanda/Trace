"""Reversible 4-bit gates: the exact Boolean rules that label traces."""
from dataclasses import dataclass
from itertools import permutations
import random

from handcoded import config


def bits(value, n_bits=None):
    """Integer → bit list; index 0 is the leftmost bit (paper convention)."""
    n_bits = config.N_BITS if n_bits is None else n_bits
    return [int(bit) for bit in f"{value:0{n_bits}b}"]


def state_text(state):
    """Bits → string such as `1000`."""
    return "".join(map(str, state))


def make_gate_names(n_bits=None):
    """All legal gate strings (52 at n_bits=4); `s01` and `s10` are distinct tokens."""
    n_bits = config.N_BITS if n_bits is None else n_bits
    gates = [f"x{i}" for i in range(n_bits)]
    for operation in ("c", "s"):
        gates.extend(f"{operation}{i}{j}" for i, j in permutations(range(n_bits), 2))
    gates.extend(f"t{i}{j}{k}" for i, j, k in permutations(range(n_bits), 3))
    return gates


def gate_permutation(gate, n_bits=None):
    """The gate as a tuple perm[s] = phi(s, gate): its identity as a transition."""
    n_bits = config.N_BITS if n_bits is None else n_bits
    return tuple(phi(state, gate, n_bits) for state in range(2 ** n_bits))


def canonical_gate_names(n_bits=None):
    """One representative name per distinct permutation, in a fixed order.

    At n_bits=4 the 52 strings realize only 34 distinct permutations: `s01`/`s10`
    are the same map, and Toffoli is symmetric in its two controls. Keeping the
    first name of each permutation gives an alias-free list, so a prefix of length
    M has exactly M strings AND M distinct transition functions. Matches
    `src.data.boolean_circuit_tasks.canonical_gate_names`.
    """
    n_bits = config.N_BITS if n_bits is None else n_bits
    seen, out = set(), []
    for name in make_gate_names(n_bits):
        key = gate_permutation(name, n_bits)
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def active_gate_names(n_bits=None, n_gates=None):
    """The gate set actually in use.

    `n_gates=None` (the default) keeps every string, including aliases, which is
    the historical behaviour. An integer takes that many alias-free gates from
    `canonical_gate_names`, so M is exactly what you asked for.
    """
    n_bits = config.N_BITS if n_bits is None else n_bits
    n_gates = config.N_GATES if n_gates is None else n_gates
    if n_gates is None:
        return make_gate_names(n_bits)
    canonical = canonical_gate_names(n_bits)
    if not 2 <= n_gates <= len(canonical):
        raise ValueError(
            f"n_gates={n_gates} out of range: n_bits={n_bits} has "
            f"{len(canonical)} distinct permutations (need at least 2)"
        )
    return canonical[:n_gates]


def gate_set_report(n_bits=None, n_gates=None, group_cap=500_000):
    """Sanity-check a restricted gate set before training on it.

    Shrinking M saves parameters, but a set that commutes makes the final state
    depend only on the multiset of operations, so outcome supervision stops being
    hard and the process/outcome gap disappears for a reason unrelated to credit
    placement. `noncommuting_fraction` well above 0 is what you want.
    """
    n_bits = config.N_BITS if n_bits is None else n_bits
    names = active_gate_names(n_bits, n_gates)
    perms = [gate_permutation(g, n_bits) for g in names]
    states = range(2 ** n_bits)

    pairs = noncommuting = 0
    for i, p in enumerate(perms):
        for q in perms[i + 1:]:
            pairs += 1
            if any(p[q[s]] != q[p[s]] for s in states):
                noncommuting += 1

    identity = tuple(states)
    group, frontier = {identity}, [identity]
    truncated = False
    while frontier:
        nxt = []
        for element in frontier:
            for p in perms:
                candidate = tuple(p[element[s]] for s in states)
                if candidate not in group:
                    if len(group) >= group_cap:
                        truncated = True
                        break
                    group.add(candidate)
                    nxt.append(candidate)
            if truncated:
                break
        if truncated:
            break
        frontier = nxt

    return {
        "n_bits": n_bits,
        "n_gates": len(names),
        "gates": names,
        "distinct_permutations_available": len(canonical_gate_names(n_bits)),
        "aliased": len(names) != len(set(perms)),
        "noncommuting_pairs": f"{noncommuting}/{pairs}",
        "noncommuting_fraction": round(noncommuting / pairs, 3) if pairs else 0.0,
        "group_order": f">={len(group)}" if truncated else len(group),
    }


GATE_ARITY = {"x": 1, "c": 2, "s": 2, "t": 3}


def gate_families(n_bits=None):
    """Families that fit in `n_bits` bits: Toffoli needs 3, CNOT/swap need 2."""
    n_bits = config.N_BITS if n_bits is None else n_bits
    return [op for op, arity in GATE_ARITY.items() if arity <= n_bits]


def sample_gate(rng, n_bits=None, n_gates=None):
    """Uniform family (of those that fit), then distinct bit indices.

    When `n_gates` restricts the set we draw uniformly from that list instead.
    The unrestricted path is left byte-identical so existing seeds reproduce.
    """
    n_bits = config.N_BITS if n_bits is None else n_bits
    n_gates = config.N_GATES if n_gates is None else n_gates
    if n_gates is not None:
        return rng.choice(active_gate_names(n_bits, n_gates))
    operation = rng.choice(gate_families(n_bits))
    indices = rng.sample(range(n_bits), GATE_ARITY[operation])
    return operation + "".join(map(str, indices))


def apply_gate(state, gate):
    """One reversible gate on a bit list (does not mutate `state`)."""
    next_state = state.copy()
    operation = gate[0]
    indices = list(map(int, gate[1:]))
    if operation == "x":
        (target,) = indices
        next_state[target] ^= 1
    elif operation == "c":
        control, target = indices
        next_state[target] ^= next_state[control]
    elif operation == "s":
        left, right = indices
        next_state[left], next_state[right] = next_state[right], next_state[left]
    elif operation == "t":
        control_a, control_b, target = indices
        next_state[target] ^= next_state[control_a] & next_state[control_b]
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return next_state


def phi(state_int, gate, n_bits=None):
    """Exact integer transition φ(s, g) used as the gold local rule."""
    n_bits = config.N_BITS if n_bits is None else n_bits
    return int(state_text(apply_gate(bits(state_int, n_bits), gate)), 2)


def sample_example(rng, depth=None, n_bits=None, n_gates=None):
    """Start state, gate names, and gold state after every gate."""
    depth = config.DEPTH if depth is None else depth
    n_bits = config.N_BITS if n_bits is None else n_bits
    n_states = 2 ** n_bits
    start = rng.randrange(n_states)
    gates = [sample_gate(rng, n_bits, n_gates) for _ in range(depth)]
    states = []
    current = start
    for gate in gates:
        current = phi(current, gate, n_bits)
        states.append(current)
    return start, gates, states


def __getattr__(name):
    """`GATES` follows the live config instead of freezing at import time."""
    if name == "GATES":
        return active_gate_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class Circuit:
    """One Boolean circuit example: start, gates, and intermediate states."""
    start: int
    gates: list[str]
    states: list[int]

    @property
    def answer(self):
        return self.states[-1]


def make_circuits(size, seed, depth=None, n_bits=None, n_gates=None):
    depth = config.DEPTH if depth is None else depth
    n_bits = config.N_BITS if n_bits is None else n_bits
    rng = random.Random(seed)
    return [Circuit(*sample_example(rng, depth, n_bits, n_gates)) for _ in range(size)]
