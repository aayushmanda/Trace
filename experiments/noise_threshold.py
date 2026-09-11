"""Identifiability of the executor from corrupted traces, in the shared kernel.

This script answers, on separate corpora with the reliability sweep's local corruption law, a
question the sweep cannot: how much of the training trace has to be correct
before the local rule stops being recoverable at all?  It never trains a
Transformer.  It imports the released Boolean gate implementation and reproduces the local
corruption law. Gates are uniform over strings here; the Transformer sweeps
sample uniformly over gate families. The sampled corpora therefore differ.

Reported quantities, all in the shared transition-kernel substrate:

  grad      projected process gradient at complete mixing vs. (K*rho-1)/(K-1);
  recovery  rule cells and greedy answer accuracy of the empirical optimum;
  gd        the same, reached by gradient descent from random logits;
  rho0      the fully corrupted corpus decoded by least-emitted successor;
  structure symmetric corruption against one coherent wrong rule;
  counts    a fixed number of correct traces with a growing number of wrong.

Writes results/noise_threshold/summary.json and Paper/figures/noise_threshold.pdf.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.boolean_circuit_tasks import N_BITS, _apply_gate, _bits, _state_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "noise_threshold"
FIG = ROOT / "Paper" / "figures"

K = 2**N_BITS
D = 8
N_CIRCUITS = 20_000
SEEDS = (2001, 2002, 2003)

PROCESS = "#087eaa"
OUTCOME = "#c56b08"
COHERENT = "#b42318"
CHANCE = "#4a4a4a"


def gate_strings() -> list[str]:
    """The M = 52 legal gate strings of the Boolean family, enumerated."""
    gates = [f"x{i}" for i in range(N_BITS)]
    gates += [f"c{a}{b}" for a, b in itertools.permutations(range(N_BITS), 2)]
    gates += [f"s{a}{b}" for a, b in itertools.permutations(range(N_BITS), 2)]
    gates += [f"t{a}{b}{c}" for a, b, c in itertools.permutations(range(N_BITS), 3)]
    return gates


GATES = gate_strings()
M = len(GATES)
# PERM[g][i] = Phi(i, g), read off the released gate implementation.
PERM = np.array(
    [[int(_state_text(_apply_gate(_bits(i), g)), 2) for i in range(K)] for g in GATES]
)
TRUE = np.zeros((M, K, K))
TRUE[np.arange(M)[:, None], np.arange(K)[None, :], PERM] = 1.0
U = np.ones((K, K)) / K
PI = np.eye(K) - U


def corrupted_counts(rho: float, n: int, seed: int, mode: str = "symmetric", wrong=None):
    """Local (gate, displayed source, displayed target) counts from a rho-reliable pool.

    Each circuit keeps its correct answer and receives the valid trace with
    probability rho.  A corrupted trace is generated exactly as the released
    sampler does: the gate is applied to the *displayed* state and the emitted
    successor is drawn uniformly from the K-1 states that are not the valid one,
    so every displayed transition is invalid under its own prefix.
    """
    rng = random.Random(seed)
    counts = np.zeros((M, K, K))
    for _ in range(n):
        prev = rng.randrange(K)
        gates = [rng.randrange(M) for _ in range(D)]
        clean = rng.random() < rho
        for g in gates:
            valid = PERM[g][prev]
            if clean:
                nxt = valid
            elif mode == "symmetric":
                nxt = rng.choice([v for v in range(K) if v != valid])
            else:  # one fixed coherent wrong rule, same corruption *rate*
                nxt = wrong[g][prev]
            counts[g, prev, nxt] += 1
            prev = nxt
    return counts


def projected_gradient_at_mixing(counts):
    """Rule(-grad L_proc) at P = U, per gate: Pi (K * empirical law) Pi."""
    out = np.zeros((M, K, K))
    for g in range(M):
        total = counts[g].sum()
        if total:
            out[g] = PI @ (K * counts[g] / total) @ PI
    return out


def decode(counts, rule="argmax"):
    """Row-wise read-off of the empirical optimum (= the normalised counts)."""
    total = counts.sum(axis=2, keepdims=True)
    law = np.where(total > 0, counts / np.maximum(total, 1), 1.0 / K)
    return law.argmin(axis=2) if rule == "argmin" else law.argmax(axis=2)


def score(table, seed=99, n_eval=1000):
    """Rule cells recovered, and greedy answer accuracy of executing `table`."""
    rule_acc = float((table == PERM).mean())
    rng = random.Random(seed)
    ok = 0
    for _ in range(n_eval):
        s = rng.randrange(K)
        gates = [rng.randrange(M) for _ in range(D)]
        true, pred = s, s
        for g in gates:
            true, pred = PERM[g][true], table[g][pred]
        ok += true == pred
    return rule_acc, ok / n_eval


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_by_gradient_descent(counts, steps=4000, lr=5.0, seed=0):
    """Plain GD on the corrupted empirical process loss from random logits."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(M, K, K))
    row_totals = counts.sum(axis=2, keepdims=True)
    n = counts.sum()
    for _ in range(steps):
        logits -= lr * (softmax(logits) * row_totals - counts) / n
    return softmax(logits).argmax(axis=2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results: dict = {"K": K, "M": M, "D": D, "n_circuits": N_CIRCUITS, "seeds": list(SEEDS)}

    # (1) The gradient direction at complete mixing.
    grad_rows = []
    for rho in [0.0, 0.03, 1 / K, 0.10, 0.25, 0.50, 1.0]:
        counts = corrupted_counts(rho, N_CIRCUITS, seed=SEEDS[0])
        grad = projected_gradient_at_mixing(counts)
        coefs, cosines = [], []
        for g in range(M):
            direction = TRUE[g] - U
            if np.linalg.norm(grad[g]) > 1e-12:
                coefs.append((grad[g] * direction).sum() / (direction * direction).sum())
                cosines.append(
                    (grad[g] * direction).sum()
                    / (np.linalg.norm(grad[g]) * np.linalg.norm(direction))
                )
        grad_rows.append(
            {
                "rho": rho,
                "coefficient": float(np.mean(coefs)),
                "predicted": (K * rho - 1) / (K - 1),
                "cosine": float(np.mean(cosines)),
            }
        )
        print(f"grad rho={rho:6.4f} coef={grad_rows[-1]['coefficient']:+.4f} "
              f"predicted={grad_rows[-1]['predicted']:+.4f} cos={grad_rows[-1]['cosine']:+.3f}")
    results["gradient_at_mixing"] = grad_rows

    # (2) Recovery of the executor from the empirical optimum, several seeds.
    rhos = [0.02, 0.04, 1 / K, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.80, 1.00]
    recovery = []
    for rho in rhos:
        rule, ans = [], []
        for seed in SEEDS:
            r, a = score(decode(corrupted_counts(rho, N_CIRCUITS, seed)))
            rule.append(r)
            ans.append(a)
        recovery.append(
            {
                "rho": rho,
                "rule_mean": float(np.mean(rule)),
                "rule_sd": float(np.std(rule, ddof=1)),
                "answer_mean": float(np.mean(ans)),
                "answer_sd": float(np.std(ans, ddof=1)),
            }
        )
        print(f"recovery rho={rho:5.3f} rule={recovery[-1]['rule_mean']*100:6.2f}% "
              f"answer={recovery[-1]['answer_mean']*100:6.2f}%")
    results["recovery"] = recovery

    # (3) The same, but reached by gradient descent from random parameters.
    gd = []
    for rho in [0.10, 0.20, 0.50]:
        counts = corrupted_counts(rho, N_CIRCUITS, seed=SEEDS[0])
        r, a = score(fit_by_gradient_descent(counts))
        gd.append({"rho": rho, "rule": r, "answer": a})
        print(f"gd       rho={rho:5.3f} rule={r*100:6.2f}% answer={a*100:6.2f}%")
    results["gradient_descent"] = gd

    # (4) The fully corrupted corpus: the true successor is the one never emitted.
    counts0 = corrupted_counts(0.0, N_CIRCUITS, seed=SEEDS[0])
    never = float((counts0[np.arange(M)[:, None], np.arange(K)[None, :], PERM] == 0).mean())
    rule_max, ans_max = score(decode(counts0, "argmax"))
    rule_min, ans_min = score(decode(counts0, "argmin"))
    results["rho_zero"] = {
        "true_successor_never_emitted": never,
        "argmax_rule": rule_max, "argmax_answer": ans_max,
        "argmin_rule": rule_min, "argmin_answer": ans_min,
    }
    print(f"rho0     never-emitted={never*100:6.2f}%  argmax rule={rule_max*100:5.2f}%  "
          f"argmin rule={rule_min*100:6.2f}% answer={ans_min*100:6.2f}%")

    # (5) Same corruption rate, different corruption structure.
    rs = np.random.default_rng(0)
    sigma = rs.permutation(K)
    while (sigma == np.arange(K)).any():
        sigma = rs.permutation(K)
    wrong = np.array([[sigma[PERM[g][i]] for i in range(K)] for g in range(M)])
    structure = []
    for rho in [0.05, 1 / K, 0.10, 0.25, 0.40, 0.49, 0.51, 0.60, 0.80]:
        row = {"rho": rho}
        for mode in ("symmetric", "coherent"):
            r, a = score(decode(corrupted_counts(rho, N_CIRCUITS, SEEDS[0], mode, wrong)))
            row[f"{mode}_rule"], row[f"{mode}_answer"] = r, a
        structure.append(row)
        print(f"struct   rho={rho:5.3f} sym={row['symmetric_answer']*100:6.2f}% "
              f"coh={row['coherent_answer']*100:6.2f}%")
    results["structure"] = structure

    # (6) Fixed correct count, growing wrong count.
    n_plus = 2000
    counts_rows = []
    for n_minus in [0, 2000, 10_000, 20_000, 28_000, 30_000, 32_000, 60_000]:
        rng = random.Random(7)
        counts = np.zeros((M, K, K))
        for i in range(n_plus + n_minus):
            prev = rng.randrange(K)
            gates = [rng.randrange(M) for _ in range(D)]
            clean = i < n_plus
            for g in gates:
                valid = PERM[g][prev]
                nxt = valid if clean else rng.choice([v for v in range(K) if v != valid])
                counts[g, prev, nxt] += 1
                prev = nxt
        r, a = score(decode(counts))
        counts_rows.append(
            {"n_plus": n_plus, "n_minus": n_minus,
             "rho": n_plus / (n_plus + n_minus), "rule": r, "answer": a}
        )
        print(f"counts   n-={n_minus:6d} rho={counts_rows[-1]['rho']:6.4f} "
              f"rule={r*100:6.2f}% answer={a*100:6.2f}%")
    results["counts"] = counts_rows

    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    draw(results)
    print(f"\nwrote {OUT/'summary.json'} and {FIG/'noise_threshold.pdf'}")
    return 0


def draw(results: dict) -> None:
    """Rebuild the figure from a saved summary; no sweeps are rerun."""
    recovery, structure = results["recovery"], results["structure"]
    # ---- figure -----------------------------------------------------------
    # Transformer sweep for context; separate corpora and different gate sampling (Table 4).
    transformer = {0.30: 0.078, 0.50: 0.180, 0.80: 0.620, 0.85: 0.727,
                   0.90: 0.819, 0.95: 0.870, 1.00: 0.926}
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                         "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    xs = [r["rho"] for r in recovery]
    ys = [r["answer_mean"] for r in recovery]
    sd = [r["answer_sd"] for r in recovery]
    ax.errorbar(xs, ys, yerr=sd, color=PROCESS, marker="o", ms=4, lw=1.8,
                capsize=2, label="Shared kernel (empirical optimum)")
    ax.plot(sorted(transformer), [transformer[k] for k in sorted(transformer)],
            color=OUTCOME, marker="s", ms=4, lw=1.8, label="Trained Transformer (Table 4)")
    ax.axvline(1 / K, color=CHANCE, ls="--", lw=1.2)
    ax.axhline(1 / K, color=CHANCE, ls=":", lw=1.0)
    ax.annotate(r"$\rho_c=1/K$", xy=(1 / K, 0.52), xytext=(0.13, 0.52),
                fontsize=9, color=CHANCE,
                arrowprops=dict(arrowstyle="->", color=CHANCE, lw=1.0))
    ax.set_xlabel(r"Trace reliability $\rho$")
    ax.set_ylabel("Answer accuracy")
    ax.set_title("(a) Tabular recovery and trained-model accuracy", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center", bbox_to_anchor=(0.60, 0.72), frameon=False)

    ax = axes[1]
    xs = [r["rho"] for r in structure]
    ax.plot(xs, [r["symmetric_answer"] for r in structure], color=PROCESS,
            marker="o", ms=4, lw=1.8, label=r"Symmetric: $\rho_c=1/K$")
    ax.plot(xs, [r["coherent_answer"] for r in structure], color=COHERENT,
            marker="^", ms=4, lw=1.8, label=r"One coherent wrong rule: $\rho_c=1/2$")
    ax.axvline(1 / K, color=PROCESS, ls="--", lw=1.0)
    ax.axvline(0.5, color=COHERENT, ls="--", lw=1.0)
    ax.axhline(1 / K, color=CHANCE, ls=":", lw=1.0)
    ax.set_xlabel(r"Trace reliability $\rho$")
    ax.set_ylabel("Answer accuracy")
    ax.set_title("(b) Dependence on the corruption law", fontsize=11)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left", frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"noise_threshold.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        draw(json.loads((OUT / "summary.json").read_text()))
        raise SystemExit(0)
    raise SystemExit(main())
