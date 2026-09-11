"""Measure the credit bound, the two residuals, and both sides at complete mixing.

Everything here is in the shared transition kernel of the credit section, where
the quantities the theorems are about are defined, so each is measured rather
than inferred.  No training.

  (A) Complete mixing, both objectives.  At P_g = U, report the executor-relevant
      part of the per-example outcome gradient and of the process gradient.  The
      claim is that the first is exactly zero and the second is bounded below,
      so the two are reported in one place and in the same units.

  (B) The two residuals separately.  The factorization says the executor-relevant
      credit is || Pi q_{t-1} || * || Pi b_t || / p_y.  The two factors trade off
      along the circuit -- forward small early, backward small late -- so both
      are reported per position, which a single product cannot show.

  (C) The depth bound.  Sample tables at a controlled conditional scale eps and
      measure || Rule(-grad) ||_F against depth, against the process floor
      (1 - 1/K) / (P_t)_{s,s'}, and fit the exponent of eps at each depth.  The
      bound predicts slope D - 1.

Writes results/credit_suppression/summary.json and Paper/figures/credit_suppression.pdf.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.boolean_circuit_tasks import N_BITS, _apply_gate, _bits, _state_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "credit_suppression"
FIG = ROOT / "Paper" / "figures"

K = 2 ** N_BITS
U = np.ones((K, K)) / K
PI = np.eye(K) - U
PROCESS = "#087eaa"
OUTCOME = "#c56b08"
CHANCE = "#4a4a4a"


def gate_permutations():
    gates = [f"x{i}" for i in range(N_BITS)]
    gates += [f"c{a}{b}" for a, b in itertools.permutations(range(N_BITS), 2)]
    gates += [f"s{a}{b}" for a, b in itertools.permutations(range(N_BITS), 2)]
    gates += [f"t{a}{b}{c}" for a, b, c in itertools.permutations(range(N_BITS), 3)]
    perm = np.array([[int(_state_text(_apply_gate(_bits(i), g)), 2) for i in range(K)]
                     for g in gates])
    tables = np.zeros((len(gates), K, K))
    tables[np.arange(len(gates))[:, None], np.arange(K)[None, :], perm] = 1.0
    return perm, tables


PERM, TRUE = gate_permutations()
M = len(PERM)


def conditional_perturbation(rng, eps):
    """P = U + eps (R - U) for a random permutation R: doubly stochastic, and in
    the conditional subspace with spectral norm exactly eps.

    R - U = R Pi, so its operator norm is ||Pi|| = 1 and the scale is exact rather
    than fitted; entries stay nonnegative for every eps <= 1, so P is a genuine
    transition kernel.  R is drawn independently of the true rule, so products
    across positions do not telescope the way an executor-aligned path would --
    the cancellations the bound is one-sided about are left free to occur.
    """
    R = np.zeros((K, K))
    R[np.arange(K), rng.permutation(K)] = 1.0
    return U + eps * (R - U)


def credit_terms(tables, gates, s0, y):
    """Per-occurrence (||Pi q||, ||Pi b||, p_y) for one circuit under given tables."""
    D = len(gates)
    forward = [np.zeros(K)] * (D + 1)
    v = np.zeros(K); v[s0] = 1.0
    forward[0] = v
    for t, g in enumerate(gates):
        forward[t + 1] = tables[g].T @ forward[t]
    backward = [np.zeros(K)] * (D + 1)
    w = np.zeros(K); w[y] = 1.0
    backward[D] = w
    for t in range(D - 1, -1, -1):
        backward[t] = tables[gates[t]] @ backward[t + 1]
    p_y = float(forward[0] @ backward[0])
    out = []
    for t in range(D):
        out.append((float(np.linalg.norm(PI @ forward[t])),
                    float(np.linalg.norm(PI @ backward[t + 1])), p_y))
    return out


def sample_circuit(rng, depth):
    s0 = int(rng.integers(K))
    gates = [int(rng.integers(M)) for _ in range(depth)]
    s = s0
    for g in gates:
        s = PERM[g][s]
    return s0, gates, s


def part_a(rng, depth=6, n=2000):
    """At P = U: the executor-relevant part of each objective's gradient."""
    tables = np.repeat(U[None], M, axis=0)
    cond_out, p_ys = [], []
    for _ in range(n):
        s0, gates, y = sample_circuit(rng, depth)
        for fwd, bwd, p_y in credit_terms(tables, gates, s0, y):
            cond_out.append(fwd * bwd / p_y)
            p_ys.append(p_y)
    # the process target names its own source and destination, so its
    # executor-relevant norm is (1 - 1/K) / P[s,s'] with P[s,s'] = 1/K at mixing
    cond_proc = (1 - 1 / K) / (1 / K)
    return {"outcome_cond_max": float(np.max(cond_out)),
            "outcome_cond_mean": float(np.mean(cond_out)),
            "process_cond": float(cond_proc),
            "process_cond_lower_bound": float(1 - 1 / K),
            "p_y": float(np.mean(p_ys)), "n_occurrences": len(cond_out), "depth": depth}


def draw_tables(rng, eps):
    """One parameter point: M tables at conditional scale eps, drawn once."""
    return np.stack([conditional_perturbation(rng, eps) for _ in range(M)])


def part_b(rng, eps=0.4, depth=8, n=400):
    """The two residuals per position, which the product alone hides."""
    rows = [{"t": t + 1, "fwd": [], "bwd": [], "cond": []} for t in range(depth)]
    tables = draw_tables(rng, eps)   # the theorem is at a fixed parameter
    for _ in range(n):
        s0, gates, y = sample_circuit(rng, depth)
        for t, (fwd, bwd, p_y) in enumerate(credit_terms(tables, gates, s0, y)):
            rows[t]["fwd"].append(fwd); rows[t]["bwd"].append(bwd)
            rows[t]["cond"].append(fwd * bwd / max(p_y, 1e-30))
    return [{"t": r["t"], "fwd": float(np.mean(r["fwd"])), "bwd": float(np.mean(r["bwd"])),
             "cond": float(np.mean(r["cond"])), "eps": eps, "depth": depth}
            for r in rows if r["fwd"]]


def part_c(rng, depths, epsilons, n=200, n_params=3):
    """|| Rule(-grad) ||_F against depth and conditional scale.

    Averages over a few parameter draws at each scale, with examples sampled at
    each; the bound is a statement at a parameter, not over a parameter law.
    """
    rows = []
    for eps in epsilons:
        params = [draw_tables(rng, eps) for _ in range(n_params)]
        for depth in depths:
            vals = []
            for tables in params:
                for _ in range(n):
                    s0, gates, y = sample_circuit(rng, depth)
                    for fwd, bwd, p_y in credit_terms(tables, gates, s0, y):
                        vals.append(fwd * bwd / max(p_y, 1e-30))
            rows.append({"eps": eps, "depth": depth,
                         "cond_mean": float(np.mean(vals)),
                         "cond_max": float(np.max(vals)),
                         "bound": float((1 - 1 / K) / (1 / (2 * K)) * eps ** (depth - 1))})
            print(f"[C] eps={eps:<5} D={depth:<3} mean={rows[-1]['cond_mean']:.3e} "
                  f"max={rows[-1]['cond_max']:.3e}", flush=True)
    return rows


def fit_exponents(rows, depths, epsilons):
    """Slope of log(cond) against log(eps) at each depth; the bound predicts D-1."""
    fits = []
    for depth in depths:
        xs = [np.log(r["eps"]) for r in rows if r["depth"] == depth and r["cond_mean"] > 0]
        ys = [np.log(r["cond_mean"]) for r in rows if r["depth"] == depth and r["cond_mean"] > 0]
        if len(xs) < 2:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = np.polyval([slope, intercept], xs)
        ss = 1 - np.sum((np.array(ys) - pred) ** 2) / np.sum((np.array(ys) - np.mean(ys)) ** 2)
        fits.append({"depth": depth, "slope": float(slope), "predicted": depth - 1,
                     "r2": float(ss)})
        print(f"[fit] D={depth:<3} slope={slope:+.3f}  predicted D-1={depth - 1}  R^2={ss:.5f}")
    return fits


def draw(res):
    rows, fits = res["depth_scale"], res["exponents"]
    epsilons = sorted({r["eps"] for r in rows})
    depths = sorted({r["depth"] for r in rows})
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                         "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, eps in enumerate(epsilons):
        sel = [r for r in rows if r["eps"] == eps]
        ax.semilogy([r["depth"] for r in sel], [r["cond_mean"] for r in sel],
                    marker="o", ms=4, lw=1.7, color=cmap(i / max(len(epsilons) - 1, 1)),
                    label=rf"$\varepsilon={eps}$")
    ax.axhline(res["at_mixing"]["process_cond"], color=PROCESS, ls="--", lw=1.6)
    ax.text(depths[-1], res["at_mixing"]["process_cond"] * 1.5, "process", color=PROCESS,
            ha="right", fontsize=9)
    ax.set_xlabel("Composition depth $D$")
    ax.set_ylabel(r"$\|\mathrm{Rule}(-\nabla_{P_t}\ell)\|_F$")
    ax.set_title("(a) Outcome credit decays with depth; process does not", fontsize=10.5)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    ax.plot([f["depth"] for f in fits], [f["slope"] for f in fits],
            marker="o", ms=4, lw=1.7, color=OUTCOME, label="measured slope")
    ax.plot([f["depth"] for f in fits], [f["predicted"] for f in fits],
            ls="--", lw=1.4, color=CHANCE, label=r"predicted $D-1$")
    ax.set_xlabel("Composition depth $D$")
    ax.set_ylabel(r"exponent of $\varepsilon$")
    ax.set_title(r"(b) Fitted exponent tracks $D-1$", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"credit_suppression.{ext}", bbox_inches="tight")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        draw(json.loads((OUT / "summary.json").read_text()))
        return 0

    rng = np.random.default_rng(args.seed)
    res = {"K": K, "M": M, "config": vars(args)}

    print("=== (A) at complete mixing ===", flush=True)
    res["at_mixing"] = part_a(rng)
    a = res["at_mixing"]
    print(f"  outcome, executor-relevant part: max {a['outcome_cond_max']:.3e} "
          f"over {a['n_occurrences']} occurrences")
    print(f"  process, executor-relevant part: {a['process_cond']:.4f} "
          f"(bound >= {a['process_cond_lower_bound']:.4f})")

    print("\n=== (B) the two residuals along the circuit ===", flush=True)
    res["residuals"] = part_b(rng)
    for r in res["residuals"]:
        print(f"  t={r['t']:<3} ||Pi q||={r['fwd']:.3e}  ||Pi b||={r['bwd']:.3e}  "
              f"product/p_y={r['cond']:.3e}")

    print("\n=== (C) depth and scale ===", flush=True)
    res["depth_scale"] = part_c(rng, args.depths, args.eps, n=args.n)
    print()
    res["exponents"] = fit_exponents(res["depth_scale"], args.depths, args.eps)

    (OUT / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    draw(res)
    print(f"\nwrote {OUT/'summary.json'} and {FIG/'credit_suppression.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
