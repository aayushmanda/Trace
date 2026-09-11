"""Kernel closure for Cor. 3, Thm. 2, and Thm. 4 (both halves). No training.

Shared row-stochastic executor of the credit section. Every quantity plotted is
the theorem's object, not a proxy.

  Thm 4, both halves, one table. At P_g = U, the population Rule(−∇ L_out)
      is numerically zero, while process Rule(−∇ L_proc) has coefficient
      (Kρ − 1)/(K − 1) along T_g − U.

  Thm 2. On the same samples, ||q̄_{t−1}|| and ||b̄_t|| separately. Credit is
      their product / p_y; the factors trade off along the circuit.

  Cor. 3. P_g = U + ε(T_rand − U), ε ∈ {0.1, 0.3, 0.5, 0.9}, D ∈ {2,…,12}.
      Plot ||Rule(−∇_{P_t} ℓ_out)||_F vs D against the process floor
      (1 − 1/K) / (P_t)_{s,s'}. Fitted log-log slope in ε should track D − 1.

Writes results/closure_handcoded/{kernel_summary.json, thm4_mixing.csv,
residuals.csv, corollary3.csv, corollary3.pdf, thm2_residuals.pdf,
exponents.pdf}.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.boolean_circuit_tasks import N_BITS, _apply_gate, _bits, _state_text
from src.plot_style import apply_style
from src.experiments import noise_threshold as nt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "closure_handcoded"

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
    perm = np.array(
        [[int(_state_text(_apply_gate(_bits(i), g)), 2) for i in range(K)] for g in gates]
    )
    tables = np.zeros((len(gates), K, K))
    tables[np.arange(len(gates))[:, None], np.arange(K)[None, :], perm] = 1.0
    return perm, tables


PERM, TRUE = gate_permutations()
M = len(PERM)


def scaled_tables(rng, eps):
    """Doubly stochastic P_g = U + ε (T_rand − U). ||P_g − U||_2 = ε for ε ∈ [0, 1]."""
    tables = np.empty((M, K, K))
    for g in range(M):
        perm = rng.permutation(K)
        t = np.zeros((K, K))
        t[np.arange(K), perm] = 1.0
        tables[g] = U + eps * (t - U)
    return tables


def sample_circuit(rng, depth):
    s0 = int(rng.integers(K))
    gates = [int(rng.integers(M)) for _ in range(depth)]
    s = s0
    for g in gates:
        s = int(PERM[g][s])
    return s0, gates, s


def forward_backward(tables, gates, s0, y):
    """q_{0:D}, b_{0:D}, p_y. q_t is the state dist after t gates; b_t reaches y from step t."""
    d = len(gates)
    q = np.zeros((d + 1, K))
    q[0, s0] = 1.0
    for t, g in enumerate(gates):
        q[t + 1] = tables[g].T @ q[t]
    b = np.zeros((d + 1, K))
    b[d, y] = 1.0
    for t in range(d - 1, -1, -1):
        b[t] = tables[gates[t]] @ b[t + 1]
    p_y = float(q[0] @ b[0])
    return q, b, p_y


def rule_norm(q, b, p_y):
    """||Rule(−∇_{P_t} ℓ_out)||_F = ||Π q|| ||Π b|| / p_y."""
    return float(np.linalg.norm(PI @ q) * np.linalg.norm(PI @ b) / max(p_y, 1e-30))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def thm4_at_mixing(rng, depth=8, n=4000, rhos=(0.0, 1 / K, 0.25, 0.5, 1.0)):
    """Population gradients at P_g = U. Outcome Rule ≈ 0; process recovers T_g − U.

    Process half reuses Appendix I (`corrupted_counts` / `projected_gradient_at_mixing`)
    so the coefficient is on the same code path. Outcome half is the missing piece:
    average q_{t-1} b_t^⊤ / p_y over true circuits and report ||Π G Π||_F.
    """
    tables = np.repeat(U[None], M, axis=0)
    g_out = np.zeros((M, K, K))
    products = []
    for _ in range(n):
        s0, gates, y = sample_circuit(rng, depth)
        q, b, p_y = forward_backward(tables, gates, s0, y)
        for t, g in enumerate(gates):
            g_out[g] += np.outer(q[t], b[t + 1]) / max(p_y, 1e-30)
            products.append(rule_norm(q[t], b[t + 1], p_y))
    g_out /= n
    feasible = np.array([g_out[g] @ PI for g in range(M)])
    rule_out = np.array([PI @ g_out[g] @ PI for g in range(M)])
    outcome_feas_mean = float(np.mean([np.linalg.norm(feasible[g]) for g in range(M)]))
    outcome_feas_max = float(np.max([np.linalg.norm(feasible[g]) for g in range(M)]))
    outcome_rule_mean = float(np.mean([np.linalg.norm(rule_out[g]) for g in range(M)]))
    outcome_rule_max = float(np.max([np.linalg.norm(rule_out[g]) for g in range(M)]))
    outcome_product_mean = float(np.mean(products))
    outcome_product_max = float(np.max(products))

    rows = []
    for rho in rhos:
        counts = nt.corrupted_counts(float(rho), nt.N_CIRCUITS, seed=nt.SEEDS[0])
        grad = nt.projected_gradient_at_mixing(counts)
        coefs, cosines = [], []
        for g in range(M):
            direction = TRUE[g] - U
            gn, dn = np.linalg.norm(grad[g]), np.linalg.norm(direction)
            if gn > 1e-12 and dn > 1e-12:
                coefs.append(float((grad[g] * direction).sum() / (direction * direction).sum()))
                cosines.append(float((grad[g] * direction).sum() / (gn * dn)))
        predicted = (K * rho - 1) / (K - 1)
        rows.append({
            "rho": float(rho),
            "process_coefficient": float(np.mean(coefs)) if coefs else float("nan"),
            "predicted_coefficient": float(predicted),
            "process_cosine_to_T_minus_U": float(np.mean(cosines)) if cosines else float("nan"),
            "process_rule_frobenius_mean": float(np.mean([np.linalg.norm(grad[g]) for g in range(M)])),
            "outcome_feasible_frobenius_mean": outcome_feas_mean,
            "outcome_feasible_frobenius_max": outcome_feas_max,
            "outcome_rule_frobenius_mean": outcome_rule_mean,
            "outcome_rule_frobenius_max": outcome_rule_max,
            "outcome_rule_product_mean": outcome_product_mean,
            "outcome_rule_product_max": outcome_product_max,
            "n_circuits_outcome": n,
            "n_circuits_process": nt.N_CIRCUITS,
            "depth": depth,
            "theorem": "thm:complete-mixing / App I",
        })
        print(
            f"[Thm4] rho={rho:6.4f}  process coef={rows[-1]['process_coefficient']:+.4f} "
            f"(pred {predicted:+.4f})  outcome ||Rule(G)||_F="
            f"{outcome_rule_mean:.3e}  product={outcome_product_mean:.3e}",
            flush=True,
        )
    return rows


def thm2_residuals(rng, eps=0.4, depth=8, n=400):
    rows_acc = [{"t": t + 1, "fwd": [], "bwd": [], "p_y": [], "credit": [], "process_floor": []}
                for t in range(depth)]
    for _ in range(n):
        tables = scaled_tables(rng, eps)
        s0, gates, y = sample_circuit(rng, depth)
        q, b, p_y = forward_backward(tables, gates, s0, y)
        s = s0
        for t, g in enumerate(gates):
            gold = int(PERM[g][s])
            p_local = float(tables[g, s, gold])
            fwd = float(np.linalg.norm(PI @ q[t]))
            bwd = float(np.linalg.norm(PI @ b[t + 1]))
            rows_acc[t]["fwd"].append(fwd)
            rows_acc[t]["bwd"].append(bwd)
            rows_acc[t]["p_y"].append(p_y)
            rows_acc[t]["credit"].append(fwd * bwd / max(p_y, 1e-30))
            rows_acc[t]["process_floor"].append((1 - 1 / K) / max(p_local, 1e-30))
            s = gold
    out = []
    for r in rows_acc:
        out.append({
            "t": r["t"],
            "eps": eps,
            "depth": depth,
            "fwd_mean": float(np.mean(r["fwd"])),
            "bwd_mean": float(np.mean(r["bwd"])),
            "p_y_mean": float(np.mean(r["p_y"])),
            "credit_mean": float(np.mean(r["credit"])),
            "process_floor_mean": float(np.mean(r["process_floor"])),
            "theorem": "thm:forward-backward",
        })
        print(
            f"[Thm2] t={r['t']:<3} ||q̄||={out[-1]['fwd_mean']:.3e}  "
            f"||b̄||={out[-1]['bwd_mean']:.3e}  credit={out[-1]['credit_mean']:.3e}  "
            f"process floor={out[-1]['process_floor_mean']:.3f}",
            flush=True,
        )
    return out


def corollary3(rng, depths, epsilons, n=200):
    rows = []
    for eps in epsilons:
        for depth in depths:
            credits, floors, p_ys = [], [], []
            for _ in range(n):
                tables = scaled_tables(rng, eps)
                s0, gates, y = sample_circuit(rng, depth)
                q, b, p_y = forward_backward(tables, gates, s0, y)
                s = s0
                for t, g in enumerate(gates):
                    gold = int(PERM[g][s])
                    credits.append(rule_norm(q[t], b[t + 1], p_y))
                    floors.append((1 - 1 / K) / max(float(tables[g, s, gold]), 1e-30))
                    p_ys.append(p_y)
                    s = gold
            mean_p = float(np.mean(p_ys))
            bound = float((1 - 1 / K) / max(mean_p, 1e-30) * eps ** (depth - 1))
            rows.append({
                "eps": float(eps),
                "depth": int(depth),
                "credit_mean": float(np.mean(credits)),
                "credit_max": float(np.max(credits)),
                "process_floor_mean": float(np.mean(floors)),
                "p_y_mean": mean_p,
                "bound_using_mean_p_y": bound,
                "n_occurrences": len(credits),
                "theorem": "cor:near-mixing",
            })
            print(
                f"[Cor3] eps={eps:<4} D={depth:<3} credit={rows[-1]['credit_mean']:.3e}  "
                f"bound={bound:.3e}  process floor={rows[-1]['process_floor_mean']:.3f}",
                flush=True,
            )
    return rows


def fit_exponents(rows, depths):
    fits = []
    for depth in depths:
        xs = [np.log(r["eps"]) for r in rows if r["depth"] == depth and r["credit_mean"] > 0]
        ys = [np.log(r["credit_mean"]) for r in rows if r["depth"] == depth and r["credit_mean"] > 0]
        if len(xs) < 2:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = np.polyval([slope, intercept], xs)
        denom = np.sum((np.array(ys) - np.mean(ys)) ** 2)
        r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / denom if denom > 0 else float("nan")
        fits.append({
            "depth": int(depth),
            "slope": float(slope),
            "predicted_D_minus_1": depth - 1,
            "r2": float(r2),
            "theorem": "cor:near-mixing",
        })
        print(f"[fit] D={depth:<3} slope={slope:+.3f}  predicted D-1={depth - 1}  R^2={r2:.4f}",
              flush=True)
    return fits


def draw(res):
    apply_style({
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
    })
    rows = res["corollary3"]
    fits = res["exponents"]
    residuals = res["residuals"]
    epsilons = sorted({r["eps"] for r in rows})

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cmap = plt.get_cmap("viridis")
    for i, eps in enumerate(epsilons):
        sel = [r for r in rows if r["eps"] == eps]
        ax.semilogy(
            [r["depth"] for r in sel], [r["credit_mean"] for r in sel],
            marker="o", ms=4, lw=1.7, color=cmap(i / max(len(epsilons) - 1, 1)),
            label=rf"outcome, $\varepsilon={eps}$",
        )
    floor = float(np.mean([r["process_floor_mean"] for r in rows]))
    ax.axhline(floor, color=PROCESS, ls="--", lw=1.6, label=r"process floor $(1-1/K)/(P_t)_{s,s'}$")
    ax.set_xlabel("Composition depth $D$")
    ax.set_ylabel(r"$\|\mathrm{Rule}(-\nabla_{P_t}\ell_{\mathrm{out}})\|_F$")
    ax.set_title("Corollary 3: outcome rule credit decays as $\\varepsilon^{D-1}$; process does not")
    ax.legend(frameon=True, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "corollary3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "corollary3.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r["t"] for r in residuals], [r["fwd_mean"] for r in residuals],
            marker="o", color=OUTCOME, lw=1.7, label=r"$\|\bar q_{t-1}\|$ (forward)")
    ax.plot([r["t"] for r in residuals], [r["bwd_mean"] for r in residuals],
            marker="s", color=PROCESS, lw=1.7, label=r"$\|\bar b_t\|$ (backward)")
    ax.plot([r["t"] for r in residuals], [r["credit_mean"] for r in residuals],
            marker="^", color=CHANCE, lw=1.4, ls=":", label=r"product $/\ p_y$ (credit)")
    ax.set_xlabel("Occurrence $t$")
    ax.set_ylabel("Residual / credit")
    ax.set_title(
        rf"Theorem 2: factorization on the same samples ($\varepsilon={residuals[0]['eps']}$, $D={residuals[0]['depth']}$)"
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "thm2_residuals.pdf", bbox_inches="tight")
    fig.savefig(OUT / "thm2_residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot([f["depth"] for f in fits], [f["slope"] for f in fits],
            marker="o", color=OUTCOME, lw=1.7, label="measured exponent of $\\varepsilon$")
    ax.plot([f["depth"] for f in fits], [f["predicted_D_minus_1"] for f in fits],
            ls="--", color=CHANCE, lw=1.4, label=r"predicted $D-1$")
    ax.set_xlabel("Composition depth $D$")
    ax.set_ylabel(r"exponent of $\varepsilon$")
    ax.set_title("Corollary 3: fitted exponent tracks $D-1$")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "exponents.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exponents.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    mix = res["thm4_mixing"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r["rho"] for r in mix], [r["predicted_coefficient"] for r in mix],
            ls="--", color=CHANCE, lw=1.5, label=r"predicted $(K\rho-1)/(K-1)$")
    ax.plot([r["rho"] for r in mix], [r["process_coefficient"] for r in mix],
            marker="o", color=PROCESS, lw=1.7, label=r"process coefficient along $T_g-U$")
    ax.axhline(0.0, color=OUTCOME, lw=1.6, label=r"outcome $\|\mathrm{Rule}(G)\|_F\approx 0$")
    ax.set_xlabel(r"trace reliability $\rho$")
    ax.set_ylabel("projected coefficient")
    ax.set_title(r"Theorem 4 at $P_g=U$: process recovers $T_g-U$; outcome Rule is $0$")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "thm4_mixing.pdf", bbox_inches="tight")
    fig.savefig(OUT / "thm4_mixing.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(range(2, 13)))
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.9])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-mixing", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2001)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    res = {"K": K, "M": M, "config": vars(args)}

    print("=== Theorem 4, both halves, at complete mixing ===", flush=True)
    res["thm4_mixing"] = thm4_at_mixing(rng, n=args.n_mixing)
    write_csv(OUT / "thm4_mixing.csv", res["thm4_mixing"])

    print("\n=== Theorem 2, residuals separately ===", flush=True)
    res["residuals"] = thm2_residuals(rng)
    write_csv(OUT / "residuals.csv", res["residuals"])

    print("\n=== Corollary 3, credit vs depth ===", flush=True)
    res["corollary3"] = corollary3(rng, args.depths, args.eps, n=args.n)
    write_csv(OUT / "corollary3.csv", res["corollary3"])
    res["exponents"] = fit_exponents(res["corollary3"], args.depths)
    write_csv(OUT / "exponents.csv", res["exponents"])

    (OUT / "kernel_summary.json").write_text(
        json.dumps(res, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x)) + "\n"
    )
    draw(res)
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
