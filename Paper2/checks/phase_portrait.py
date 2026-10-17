"""Phase portrait of the reduced correlation-loss flow in the (q, r) plane.

Flow of Lemma (gradient-flow invariant), from theta(0) = (0, 0, eps, eps):
    r' = (gamma / 2) r phi(q),   q' = gamma r^2 phi'(q),
    alpha(q) = e^q / (e^q + N - 1),  phi(q) = (1 + rho) alpha - rho,  rho = (1 - kappa) / (N - 1).
Level sets of r^2 - F(q), F(q) = int_0^q phi/phi', are the orbits. Integrates trajectories from (0, eps), checks that
r^2 - F(q) stays at eps^2, checks the over-cancellation barrier eps^2 > -F(q_0), q_0 = log(1 - kappa), and draws three
panels: kappa = 1, 0, -1.

    uv run python Paper2/checks/phase_portrait.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
N, GAMMA = 16, 0.8                                   # eta = 0.1, as in Table 1 and Figure (kappa sweep)
KAPPAS = (1.0, 0.0, -1.0)
EPSS = (0.1, 0.3, 0.5, 0.7, 1.0)
Q1 = math.log(N - 1)                                 # alpha(q_1) = 1/2
QMAX = Q1 + 1                                        # end of the gate-valid interval
ROUTE, FAIL, SEP = "#0072B2", "#D55E00", "0.1"       # Okabe-Ito blue / vermilion


def fns(kappa):
    rho = (1 - kappa) / (N - 1)
    al = lambda q: 1 / (1 + (N - 1) * math.exp(-q))
    phi = lambda q: (1 + rho) * al(q) - rho
    dphi = lambda q: (1 + rho) * al(q) * (1 - al(q))
    return al, phi, dphi


def F(kappa, q):
    _, phi, dphi = fns(kappa)
    return quad(lambda v: phi(v) / dphi(v), 0, q, limit=200, epsabs=1e-13, epsrel=1e-12)[0]


def trajectory(kappa, eps):
    """Integrate (r, q) in time from (eps, 0) until q = q_1 + 1 (routed) or r < 1e-6 eps (failed)."""
    _, phi, dphi = fns(kappa)
    rhs = lambda t, y: [GAMMA / 2 * y[0] * phi(y[1]), GAMMA * y[0] ** 2 * dphi(y[1])]
    hit = lambda t, y: y[1] - QMAX; hit.terminal = True
    die = lambda t, y: y[0] - 1e-6 * eps; die.terminal = True
    route = lambda t, y: y[1] - Q1
    sol = solve_ivp(rhs, (0, 1e9), [eps, 0.], events=[hit, die, route], method="LSODA",
                    rtol=1e-11, atol=1e-14, dense_output=True)
    t = np.unique(np.concatenate([sol.t, np.linspace(0, sol.t[-1], 400)]))
    r, q = sol.sol(t)
    inv = r ** 2 - np.array([F(kappa, x) for x in q])
    drift = float(np.max(np.abs(inv - eps ** 2)) / eps ** 2)
    T = float(sol.t_events[2][0]) if sol.t_events[2].size else math.inf
    return q, r, T, drift


def main():
    out, worst = {}, 0.
    qg = np.linspace(0, QMAX, 600)
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.2), sharey=True)
    for ax, kappa, tag in zip(axs, KAPPAS, "abc"):
        _, phi, dphi = fns(kappa)
        Fg = np.concatenate([[0.], cumulative_trapezoid([phi(v) / dphi(v) for v in qg], qg)])
        rg = np.linspace(0, 2.9, 400)
        Q, R = np.meshgrid(qg, rg)
        C = R ** 2 - np.interp(Q, qg, Fg)
        levels = np.linspace(0.05, 1.7, 12) ** 2           # orbits r^2 - F(q) = c through (0, sqrt(c))
        ax.contour(Q, R, C, levels=levels, colors="0.72", linewidths=0.45, linestyles="solid")
        rows = []
        barrier = None
        if kappa < 0:
            q0 = math.log(1 - kappa)
            barrier = -F(kappa, q0)
            sep = np.sqrt(np.clip(barrier + Fg, 0, None))
            ax.plot(qg, sep, color=SEP, lw=1.1, ls=(0, (4, 2)), zorder=3)
            ax.plot([q0], [0], "o", ms=3.5, mfc="white", mec=SEP, mew=0.9, zorder=4, clip_on=False)
            ax.text(q0 + 0.13, 0.03, r"$q_0$", fontsize=7)
        for eps in EPSS:
            q, r, T, drift = trajectory(kappa, eps)
            routed = math.isfinite(T)
            if barrier is not None:
                assert routed == (eps ** 2 > barrier), (kappa, eps, T, barrier)
            assert drift < 1e-7, (kappa, eps, drift)
            worst = max(worst, drift)
            rows.append(dict(eps=eps, T_route=T, routed=routed, max_rel_drift=drift))
            c = ROUTE if routed else FAIL
            ax.plot(q, r, color=c, lw=1.3, ls="-" if routed else (0, (1.5, 1)), zorder=5, solid_capstyle="round")
            ax.plot([0], [eps], "o", ms=2.8, color=c, zorder=6, clip_on=False)
            k = np.searchsorted(np.cumsum(np.hypot(np.diff(q), np.diff(r))), 0.55 * np.sum(np.hypot(np.diff(q), np.diff(r))))
            k = min(k, len(q) - 2)
            ax.annotate("", xy=(q[k + 1], r[k + 1]), xytext=(q[k], r[k]), zorder=6,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=c, lw=0))
        ax.axvline(Q1, color="0.35", lw=0.7, ls=":")
        ax.text(Q1 + 0.06, 2.72, r"$\alpha=\frac{1}{2}$", fontsize=7, va="top")
        ax.set_xlim(0, QMAX); ax.set_ylim(0, 2.9)
        ax.set_xlabel(r"score gap $q$")
        ax.set_title(fr"({tag}) $\kappa={kappa:g}$", fontsize=8.5, pad=3)
        out[f"kappa={kappa:g}"] = dict(rows=rows, barrier_minus_F_q0=barrier,
                                       barrier_eps=math.sqrt(barrier) if barrier else None)
    axs[0].set_ylabel(r"rule factor $r$")
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], color=ROUTE, lw=1.3), Line2D([], [], color=FAIL, lw=1.3, ls=(0, (1.5, 1))),
                   Line2D([], [], color=SEP, lw=1.1, ls=(0, (4, 2)))],
                  ["routes", "fails", r"$r^2-F(q)=-F(q_0)$"], loc="lower right", fontsize=6.5, frameon=False,
                  handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
    for ax in axs:
        ax.tick_params(labelsize=7, length=2.5, pad=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    out.update(N=N, gamma=GAMMA, max_rel_invariant_drift=worst)
    (ROOT / "checks" / "results" / "phase_portrait.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(ROOT / "figures" / "phase_portrait.pdf")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    main()
