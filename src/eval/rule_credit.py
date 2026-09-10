"""Table-space rule credit: ||Rule(-grad_{P_t} l_out)||_F = ||Pi.fwd|| * ||Pi.bwd|| / p_y.

Pure post-hoc computation on an already-extracted, position-invariant
per-gate transition-probability table (Paper/body.tex, eq:conditional-credit
and eq:depth-credit-bound). Takes no model and no gradients as input; this
is the theorem's exact quantity evaluated on a saved induced-table array,
not a synthetic rescaling and not a parameter-space pullback.

Cross-checked against the independent implementation in
src/eval/induced_rule.py::credit_norms (mode="actual"): both compute the
same forward/backward recursion and the same Pi = I - 11^T/K projector.
"""
import numpy as np


def credit_terms(table, gate_index, start, gates, answer, k):
    """table: (M,K,K) per-gate transition-probability array.
    gate_index: dict gate-name -> row index into table.
    Returns the per-occurrence credit ||Pi q_{t-1}|| * ||Pi b_t|| / p_y for
    t = 1..D, one value per gate occurrence in this circuit. Empty list if
    the circuit's answer has zero probability under this table (p_y <= 0).
    """
    pi = np.eye(k) - np.full((k, k), 1.0 / k)
    fwd = [np.eye(k)[start]]
    for g in gates:
        fwd.append(fwd[-1] @ table[gate_index[g]])
    depth = len(gates)
    bwd = [None] * (depth + 1)
    bwd[depth] = np.eye(k)[answer]
    for t in range(depth - 1, -1, -1):
        bwd[t] = table[gate_index[gates[t]]] @ bwd[t + 1]
    p_y = float(fwd[depth][answer])
    if p_y <= 0:
        return []
    out = []
    for t in range(1, depth + 1):
        q, b = fwd[t - 1], bwd[t]
        out.append(float(np.linalg.norm(pi @ q) * np.linalg.norm(pi @ b) / p_y))
    return out


def credit_summary(table, gate_index, circuits, k):
    """circuits: iterable of objects with .start, .gates, .answer (e.g.
    handcoded.Circuit). Returns (credit_mean, credit_max, n_terms) pooled
    over all circuits and all gate occurrences within them. NaN mean/max
    and n_terms=0 if every circuit's answer had zero probability."""
    vals = []
    for c in circuits:
        vals.extend(credit_terms(table, gate_index, c.start, c.gates, c.answer, k))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.max(vals)), len(vals)
