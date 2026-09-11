"""Direct numerical test of Proposition 8's reliability frontier.

This does not reuse the paper's own closed-form derivative. It builds the
actual K x K matrices (T_g, U, the affine path P_g(a), the noisy mixture
Q_g = rho*T_g + (1-rho)*C_g) exactly, evaluates the combined population
objective

    J_{rho,beta}(a) = L_tr^rho(a) + beta * L_out(a)

from first principles (L_tr^rho(a) as the expected process cross-entropy
under Q_g, L_out(a) as the expected outcome cross-entropy of the true final
state after D compositions of P_g(a), computed via matrix power -- no
closed-form substitution), differentiates J numerically via central finite
differences in a, and bisects in rho for the sign flip of J'_{rho,beta}(a).
That measured boundary is then compared against the paper's closed-form

    rho_c(a, D, beta) = [1+(K-1)a]/K
        - beta * D(K-1) a^{D-1} (1-a) (1+(K-1)a) / [K (1+(K-1)a^D)]

(\\cref{eq:rescue-threshold} / \\cref{prop:outcome-rescue}). Agreement here is
an independent check of the algebra, not a restatement of it.

For each D in {2,3,4,6,8} (matching the paper's own remark), beta in
{0, 1/D, 1.0}, and a on a grid in (0,1), records the measured and predicted
rho_c side by side. beta=1/D is the paper's own per-position-averaged
training weight; beta=0 recovers Corollary 6 exactly (rho_c=[1+(K-1)a]/K
for all a); beta=1.0 is an extra, more aggressive answer-weight for
generality.

Writes results/prop8_frontier/summary.json and
Paper/figures/prop8_frontier.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import reject_extra_flags
from src.plot_style import apply_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "prop8_frontier"
FIG = ROOT / "Paper" / "figures"

K = 16
D_VALUES = (2, 3, 4, 6, 8)
A_GRID = np.linspace(0.03, 0.92, 24)
EPS_A = 1e-4
BISECT_TOL = 1e-7

COLORS = {2: "#b42318", 3: "#c56b08", 4: "#8a7000", 6: "#2f7d32", 8: "#087eaa"}


def make_T(k: int = K, seed: int = 0) -> np.ndarray:
    """A genuine (fixed-point-free) permutation matrix, T[i, perm[i]] = 1."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(k)
        if not (perm == np.arange(k)).any():
            break
    return np.eye(k)[perm]


def mixture_Q(T: np.ndarray, rho: float) -> np.ndarray:
    """rho*T + (1-rho)*C, C uniform on the K-1 states T does not point to."""
    k = T.shape[0]
    target = T.argmax(axis=1)
    C = np.full((k, k), 1.0 / (k - 1))
    C[np.arange(k), target] = 0.0
    return rho * T + (1 - rho) * C


def L_tr(a: float, rho: float, T: np.ndarray, U: np.ndarray) -> float:
    P = U + a * (T - U)
    Q = mixture_Q(T, rho)
    return float(-(Q * np.log(P)).sum(axis=1).mean())


def L_out(a: float, D: int, T: np.ndarray, U: np.ndarray) -> float:
    k = T.shape[0]
    P = U + a * (T - U)
    Pd = np.linalg.matrix_power(P, D)
    Td = np.linalg.matrix_power(T, D)
    target = Td.argmax(axis=1)
    probs = Pd[np.arange(k), target]
    return float(-np.log(probs).mean())


def J(a: float, rho: float, D: int, beta: float, T: np.ndarray, U: np.ndarray) -> float:
    return L_tr(a, rho, T, U) + beta * L_out(a, D, T, U)


def Jprime_numeric(a: float, rho: float, D: int, beta: float, T: np.ndarray, U: np.ndarray,
                    eps: float = EPS_A) -> float:
    a_hi = min(a + eps, 1 - 1e-6)
    a_lo = max(a - eps, 1e-9)
    return (J(a_hi, rho, D, beta, T, U) - J(a_lo, rho, D, beta, T, U)) / (a_hi - a_lo)


def closed_form_rho_c(a: float, D: int, beta: float, K: int = K) -> float:
    num = 1 + (K - 1) * a
    correction = (beta * D * (K - 1) * a ** (D - 1) * (1 - a) * (1 + (K - 1) * a)
                  / (1 + (K - 1) * a ** D))
    return (num - correction) / K


def measured_rho_c(a: float, D: int, beta: float, T: np.ndarray, U: np.ndarray,
                    lo: float = 1e-6, hi: float = 1 - 1e-6, tol: float = BISECT_TOL):
    """Bisect for the sign flip of J'_{rho,beta}(a) in rho. J' decreases in rho
    (coefficient of rho in the process term is -K < 0), so a bracket with
    J'(lo) > 0 > J'(hi) is expected whenever the boundary lies in (0,1)."""
    flo = Jprime_numeric(a, lo, D, beta, T, U)
    fhi = Jprime_numeric(a, hi, D, beta, T, U)
    if not (flo > 0 and fhi < 0):
        return None, flo, fhi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = Jprime_numeric(a, mid, D, beta, T, U)
        if fm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), flo, fhi


def run() -> dict:
    T = make_T()
    U = np.ones((K, K)) / K
    rows = []
    for D in D_VALUES:
        for beta in (0.0, 1.0 / D, 1.0):
            for a in A_GRID:
                pred = closed_form_rho_c(float(a), D, beta)
                meas, flo, fhi = measured_rho_c(float(a), D, beta, T, U)
                row = {
                    "D": D, "beta": beta, "a": float(a),
                    "predicted_rho_c": pred,
                    "measured_rho_c": meas,
                    "bracketed": meas is not None,
                    "Jprime_at_rho0": flo, "Jprime_at_rho1": fhi,
                }
                if meas is not None:
                    row["abs_error"] = abs(meas - pred)
                rows.append(row)
    return {"K": K, "D_values": list(D_VALUES), "eps_a": EPS_A, "rows": rows}


def draw(results: dict, path: Path) -> None:
    apply_style()
    rows = results["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for D in results["D_values"]:
        beta = 1.0 / D
        sub = [r for r in rows if r["D"] == D and abs(r["beta"] - beta) < 1e-12]
        sub.sort(key=lambda r: r["a"])
        xs = [r["a"] for r in sub]
        pred = [r["predicted_rho_c"] for r in sub]
        color = COLORS.get(D, "#333333")
        ax.plot(xs, pred, "-", color=color, lw=1.8, label=f"$D={D}$ (predicted)")
        meas_x = [r["a"] for r in sub if r["bracketed"]]
        meas_y = [r["measured_rho_c"] for r in sub if r["bracketed"]]
        ax.plot(meas_x, meas_y, "o", color=color, ms=4, mfc="white", mew=1.3)
    ax.axhline(1 / K, color="#4a4a4a", ls=":", lw=1.0, label=r"$1/K$")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel(r"competence $a$")
    ax.set_ylabel(r"$\rho_c(a,D,\beta{=}1/D)$")
    ax.set_title("Proposition 8: predicted vs. measured reliability frontier")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    reject_extra_flags(argv, __doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    results = run()
    max_err = max((r["abs_error"] for r in results["rows"] if r.get("bracketed")), default=None)
    n_unbracketed = sum(1 for r in results["rows"] if not r["bracketed"])
    for D in D_VALUES:
        beta = 1.0 / D
        errs = [r["abs_error"] for r in results["rows"]
                if r["D"] == D and abs(r["beta"] - beta) < 1e-12 and r["bracketed"]]
        unb = sum(1 for r in results["rows"]
                  if r["D"] == D and abs(r["beta"] - beta) < 1e-12 and not r["bracketed"])
        print(f"D={D:2d} beta=1/D={beta:.4f}  max|err|={max(errs) if errs else float('nan'):.2e}  "
              f"unbracketed={unb}")
    print(f"\noverall: max|measured-predicted|={max_err:.2e}  "
          f"unbracketed (predicted rho_c outside (0,1)) = {n_unbracketed}/{len(results['rows'])}")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    draw(results, FIG / "prop8_frontier")
    print(f"\nwrote {OUT / 'summary.json'} and {FIG / 'prop8_frontier.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
