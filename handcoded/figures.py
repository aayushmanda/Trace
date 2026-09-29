"""Paper figures.

    python handcoded/figures.py credit
    python handcoded/figures.py overlap
    python handcoded/figures.py reliability
    python handcoded/figures.py spotlight
    python handcoded/figures.py damping
    python handcoded/figures.py spectra logs/track/*.json
"""
import sys

_COMMANDS = (
    'credit',
    'overlap',
    'reliability',
    'spotlight',
    'damping',
    'spectra',
)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"usage: python {sys.argv[0]} <command> [args...]")
        print("commands:", ", ".join(_COMMANDS))
        raise SystemExit(2)
    _cmd = sys.argv[1]
    sys.argv = [_cmd] + sys.argv[2:]
    if _cmd == 'credit':
        """Figure 2: where the credit is. Left schematic; right, measured init signals.

            python handcoded/credit_figure.py
        """
        import json
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle, FancyArrowPatch

        ROOT = Path("/home/hariguru/aayus/trace")
        DATA = ROOT / "results" / "paper" / "fig2"
        FIG = ROOT / "Paper" / "figures"

        BLUE, ORANGE, INK, MUTED = "#0a7fad", "#c8680a", "#1a1a1a", "#6a6a6a"
        NS = np.array([2, 4, 6, 8, 9, 10])


        def load_signals():
            out, cot = [], []
            for n in NS:
                o, c = [], []
                for seed in (42, 43, 44):
                    rec = json.loads((DATA / f"n{n}_s{seed}.json").read_text())
                    o.append(rec["grad_outcome"])
                    c.append(rec["grad_process"])
                out.append(o)
                cot.append(c)
            return np.array(out), np.array(cot)


        def draw():
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "stixsans",
                "axes.unicode_minus": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.size": 8,
            })
            fig = plt.figure(figsize=(7.1, 2.7))
            ax = fig.add_axes([0.01, 0.08, 0.50, 0.84])
            bx = fig.add_axes([0.58, 0.20, 0.40, 0.70])

            # --- (a) schematic ---
            n = 6
            ax.set_xlim(-0.7, n + 0.7)
            ax.set_ylim(-2.25, 1.85)
            ax.set_aspect("equal")
            ax.axis("off")
            for t in range(n + 1):
                ax.add_patch(Circle((t, 0), 0.27, fc="white", ec=INK, lw=1.2, zorder=3))
                ax.text(t, 0, f"$s_{{{t}}}$", ha="center", va="center", fontsize=9,
                        color=INK, zorder=4)
            for t in range(n):
                ax.add_patch(FancyArrowPatch((t + 0.29, 0), (t + 0.71, 0),
                                             arrowstyle="-|>", mutation_scale=8,
                                             color=MUTED, lw=1.05))
                ax.text(t + 0.5, 0.20, f"$x_{{{t + 1}}}$", ha="center", va="bottom",
                        fontsize=7, color=MUTED)
            for t in range(1, n + 1):
                ax.plot([t - 0.72, t - 0.72, t + 0.04, t + 0.04],
                        [0.52, 0.72, 0.72, 0.52], color=BLUE, lw=1.7, solid_capstyle="butt")
            ax.text(n / 2 + 0.15, 1.28,
                    r"$\mathrm{step\ target{:}\ one\ transition}$",
                    ha="center", va="center", color=BLUE, fontsize=9)
            ax.plot([0.04, 0.04, n - 0.04, n - 0.04], [-0.52, -0.82, -0.82, -0.52],
                    color=ORANGE, lw=1.7, solid_capstyle="butt")
            ax.text(n / 2, -1.22, r"$\mathrm{answer{:}\ all\ }n\mathrm{\ transitions}$",
                    ha="center", color=ORANGE, fontsize=9)
            ax.text(n / 2, -1.68, r"$\mathrm{credit\ to\ order\ }r\ \leq\ \sqrt{A_r W_r}$",
                    ha="center", color=INK, fontsize=8.5)
            ax.text(n / 2, -1.98, r"$\mathrm{reversible{:}\ \leq\,}\lambda^{n-r}$",
                    ha="center", color=MUTED, fontsize=7.5)
            ax.text(-0.65, 1.78, "(a)", fontsize=10, color=INK, fontweight="bold")

            # --- (b) three-seed measurements ---
            out, cot = load_signals()
            bx.set_facecolor("#f3f3f8")
            bx.grid(color="white", lw=1.15)
            bx.set_axisbelow(True)
            for s in bx.spines.values():
                s.set_visible(False)
            for seed_i, alpha in enumerate((0.35, 0.35, 0.35)):
                bx.plot(NS, cot[:, seed_i], "o", color=BLUE, ms=3.5, alpha=alpha, zorder=2)
                bx.plot(NS, out[:, seed_i], "s", color=ORANGE, ms=3.5, alpha=alpha, zorder=2)
            bx.plot(NS, cot[:, 0], "-", color=BLUE, lw=2.0, zorder=3)
            bx.plot(NS, out[:, 0], "-", color=ORANGE, lw=2.0, zorder=3)
            grid_n = np.linspace(2, 10, 80)
            bx.plot(grid_n, out[0, 0] * (0.5 ** (grid_n - 2)), ":", color="0.35", lw=1.45, zorder=1)
            bx.set_yscale("log")
            bx.set_xticks(list(NS))
            bx.set_xlabel("steps  $n$")
            bx.set_ylabel("gradient norm at initialization")
            bx.tick_params(labelsize=7)
            bx.set_xlim(1.4, 12.3)
            bx.text(10.2, cot[-1, 0], "step", va="center", ha="left", fontsize=8, color=BLUE)
            bx.text(10.2, out[-1, 0], "answer", va="center", ha="left", fontsize=8, color=ORANGE)
            bx.text(5.0, out[0, 0] * (0.5 ** 2.85), r"$\lambda^{n}$", fontsize=8.5, color="0.35")
            bx.text(1.45, 1.15 * bx.get_ylim()[1], "(b)", fontsize=10, color=INK,
                    fontweight="bold", ha="left", va="bottom")

            FIG.mkdir(parents=True, exist_ok=True)
            for ext in ("pdf", "png"):
                fig.savefig(FIG / f"credit_locality.{ext}", dpi=300)
            print(f"wrote {FIG / 'credit_locality.pdf'}")
            print("seed 42 answer", [f"{v:.2e}" for v in out[:, 0]])
            print("seed 42 step  ", [f"{v:.3e}" for v in cot[:, 0]])


        if __name__ == "__main__":
            draw()
    elif _cmd == 'overlap':
        """Main-text Figure: A_r vs W_r vs sqrt(A_r W_r) for counting at init.

            python handcoded/overlap_figure.py
        """
        import json
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ROOT = Path("/home/hariguru/aayus/trace")
        FIG = ROOT / "Paper" / "figures"
        REC = json.loads((ROOT / "results" / "paper" / "fig2" / "n8_s42.json").read_text())

        # A_r: exact, from the closed form (stored in the init-grad artifact).
        A = np.array(REC["A_r"], dtype=float)
        # W_r: Table 12 / app:spectrum, n=8 seed 42. D(rho) is linear, so the
        # letter-dependent energy is attributed to order 1.
        W = np.zeros_like(A)
        W[0] = 0.143
        W[1] = 4.6e-4
        overlap = np.sqrt(A * W)

        BLUE, ORANGE, INK, MUTED = "#0a7fad", "#c8680a", "#1a1a1a", "#6a6a6a"
        RS = np.arange(len(A))


        def draw():
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "stixsans",
                "axes.unicode_minus": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.size": 8,
            })
            fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
            fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.38)

            specs = [
                (axes[0], A, False, ORANGE, r"$A_r$", r"(a)  task spectrum"),
                (axes[1], W, True, BLUE, r"$W_r$", r"(b)  model at init."),
                (axes[2], overlap, True, INK, r"$\sqrt{A_r W_r}$", r"(c)  overlap"),
            ]
            for ax, vals, logy, color, ylab, title in specs:
                ax.set_facecolor("#f3f3f8")
                ax.grid(color="white", lw=1.1, axis="y")
                ax.set_axisbelow(True)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.bar(RS, np.maximum(vals, 1e-18 if logy else 0.0),
                       color=color, width=0.72, lw=0)
                ax.set_xticks(RS)
                ax.set_xlabel("order  $r$")
                ax.set_ylabel(ylab)
                ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
                ax.tick_params(labelsize=7)
                if logy:
                    ax.set_yscale("log")
                    ax.set_ylim(max(vals[vals > 0].min() / 8, 1e-8), max(vals.max() * 4, 1e-3))

            FIG.mkdir(parents=True, exist_ok=True)
            for ext in ("pdf", "png"):
                fig.savefig(FIG / f"overlap_spectra.{ext}", dpi=300)
            print(f"wrote {FIG / 'overlap_spectra.pdf'}")
            print("A_r shares", (A / A.sum()).round(4).tolist())
            print("W_r shares", (W / W.sum()).round(4).tolist())
            print("sqrt(A W)", [f"{v:.2e}" for v in overlap])


        if __name__ == "__main__":
            draw()
    elif _cmd == 'reliability':
        """Reliability figure: counting mod 3 over 16 letters, scattered errors, 32k updates.

        Data are the per-seed entries of the appendix reliability table (seeds 42-44).

            python handcoded/reliability_figure.py
        """
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        FIG = Path("/home/hariguru/aayus/trace/Paper/figures")
        BLUE, ORANGE, GREEN, INK, MUTED = "#0a7fad", "#c8680a", "#2e8b57", "#1a1a1a", "#6a6a6a"

        RHO = np.array([0.3, 0.4, 0.5, 0.7, 1.0])
        FULL_EXACT = [[17, 16, 17], [49, 92, 66], [98, 91, 93], [90, 94, 98], [100, 100, 100]]
        FULL_ANSWER = [[46, 43, 43], [66, 95, 78], [98, 94, 96], [94, 96, 98], [100, 100, 100]]
        CLEAN = [[100, 99.9, 99.9], [100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]]
        OUTCOME = [50.0, 33.5, 30.1]          # correct answers only; exact trace = answer here
        K = 3


        def panel(ax, full, title):
            ax.set_facecolor("#f3f3f8")
            ax.grid(color="white", lw=1.15)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_visible(False)
            full, clean = np.array(full, float), np.array(CLEAN, float)
            for data, color, mk, label in ((clean, GREEN, "^", "its clean traces only"),
                                           (full, BLUE, "o", "full corpus")):
                for k in range(3):
                    ax.plot(RHO, data[:, k], mk, color=color, ms=3.5, alpha=0.35, zorder=2)
                ax.plot(RHO, data.mean(1), "-", color=color, lw=2.0, zorder=3, label=label)
            ax.axhspan(min(OUTCOME), max(OUTCOME), color=ORANGE, alpha=0.18, lw=0, zorder=1)
            ax.axhline(np.mean(OUTCOME), color=ORANGE, lw=2.0, zorder=3, label="correct answers only")
            ax.axhline(100 / K, color=MUTED, lw=1.0, ls="--", zorder=1)
            ax.axvline(1 / K, color="0.35", lw=1.3, ls=":", zorder=1)
            ax.text(1 / K + 0.01, 4, r"$\rho=1/K$", fontsize=7.5, color="0.35", ha="left")
            ax.text(0.985, 100 / K - 2, "chance", fontsize=7, color=MUTED, ha="right", va="top")
            ax.set_xlim(0.26, 1.02)
            ax.set_ylim(0, 105)
            ax.set_xticks([0.3, 0.4, 0.5, 0.7, 1.0])
            ax.set_xlabel(r"fraction of clean traces  $\rho$")
            ax.set_ylabel(title)
            ax.tick_params(labelsize=7)


        def draw():
            plt.rcParams.update({
                "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "stixsans", "axes.unicode_minus": False,
                "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
            })
            fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.45))
            panel(a, FULL_EXACT, "exact-trace accuracy (%)")
            panel(b, FULL_ANSWER, "answer accuracy (%)")
            for ax, tag in ((a, "(a)"), (b, "(b)")):
                ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=10, color=INK,
                        fontweight="bold", ha="left", va="bottom")
            handles, labels = a.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
                       bbox_to_anchor=(0.55, 1.02))
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            FIG.mkdir(parents=True, exist_ok=True)
            for ext in ("pdf", "png"):
                fig.savefig(FIG / f"reliability_n16.{ext}", dpi=300)
            print(f"wrote {FIG / 'reliability_n16.pdf'}")


        if __name__ == "__main__":
            draw()
    elif _cmd == 'spotlight':
        """Plot the lambda sweep and the order intervention once their json files exist."""
        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        ROOT = Path("/home/hariguru/aayus/trace")
        LAM = ROOT / "results" / "paper" / "lambda_sweep"
        WR = ROOT / "results" / "paper" / "wr_intervention"
        FIG = ROOT / "Paper" / "figures"
        FIG.mkdir(parents=True, exist_ok=True)

        NS = (2, 4, 6, 8)


        def decay(g_short, g_long, dn):
            if g_short <= 0 or g_long <= 0:
                return float("nan")
            return float((g_long / g_short) ** (1.0 / dn))


        def lambda_figure():
            by = {}
            for path in sorted(LAM.glob("n*_p*_s*.json")):
                rec = json.loads(path.read_text())
                by.setdefault((round(rec["match_p"], 2), rec["seed"]), {})[rec["n"]] = rec
            summary = []
            for (p, seed), recs in sorted(by.items()):
                if not all(n in recs for n in NS):
                    continue
                lam = recs[2]["lambda_theory"]
                summary.append(dict(
                    p=p, seed=seed, lambda_theory=lam,
                    answer_2_to_4=decay(recs[2]["grad_outcome"], recs[4]["grad_outcome"], 2),
                    step_2_to_4=decay(recs[2]["grad_process"], recs[4]["grad_process"], 2),
                    grad_outcome={str(n): recs[n]["grad_outcome"] for n in NS},
                    grad_process={str(n): recs[n]["grad_process"] for n in NS},
                ))
            (LAM / "summary.json").write_text(json.dumps(summary, indent=1))
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, 6))
            ax = axes[0]
            for color, p in zip(colors, sorted({row["p"] for row in summary})):
                rows = [row for row in summary if row["p"] == p]
                lam = rows[0]["lambda_theory"]
                mean = np.array([np.mean([row["grad_outcome"][str(n)] for row in rows]) for n in NS])
                ax.plot(NS, mean, "o-", color=color, ms=4, label=rf"$\lambda={lam:.2f}$")
                guide = mean[0] * lam ** (np.array(NS) - NS[0])
                ax.plot(NS, guide, "--", color=color, lw=1, alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlabel("horizon $n$")
            ax.set_ylabel(r"answer gradient")
            ax.legend(frameon=False, fontsize=7, loc="lower left")
            ax.set_title("same network, six task laws")
            ax = axes[1]
            ax.plot([0, 1], [0, 1], color="0.75", lw=1, zorder=0)
            ax.scatter([row["lambda_theory"] for row in summary],
                       [row["answer_2_to_4"] for row in summary],
                       s=26, c="#c45c26", label="answer", zorder=2)
            ax.scatter([row["lambda_theory"] for row in summary],
                       [row["step_2_to_4"] for row in summary],
                       s=26, c="#3d5a80", marker="s", label="step", zorder=2)
            ax.set_xlabel(r"theoretical $\lambda$")
            ax.set_ylabel(r"measured decay, $(g_4/g_2)^{1/2}$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.35)
            ax.legend(frameon=False, fontsize=8)
            ax.set_title(r"$K=2$, seeds 42--44")
            fig.tight_layout()
            fig.savefig(FIG / "lambda_sweep.pdf")
            fig.savefig(FIG / "lambda_sweep.png", dpi=160)
            print("lambda")
            for p in sorted({row["p"] for row in summary}):
                rows = [row for row in summary if row["p"] == p]
                ans = np.mean([row["answer_2_to_4"] for row in rows])
                step = np.mean([row["step_2_to_4"] for row in rows])
                print(f"  p={p:.2f}  theory {rows[0]['lambda_theory']:.3f}  "
                      f"answer {ans:.3f}  step {step:.3f}")


        def wr_figure():
            recs = [json.loads(p.read_text()) for p in sorted(WR.glob("*.json"))]
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
            colors = {"aligned": "#c45c26", "shuffle": "#3d5a80", "off": "0.45"}
            markers = {"aligned": "o", "shuffle": "s", "off": "D"}
            ax = axes[0]
            lo, hi = 1e-6, 1.0
            ax.plot([lo, hi], [lo, hi], color="0.75", lw=1, zorder=0)
            for align in ("shuffle", "aligned", "off"):
                xs, ys = [], []
                for rec in recs:
                    if rec["align"] != align:
                        continue
                    x = abs(rec["analytic_overlap"])
                    y = abs(rec["measured_dalpha"])
                    if x <= 0 or y <= 0:
                        continue
                    xs.append(x)
                    ys.append(y)
                if xs:
                    ax.scatter(xs, ys, s=26, c=colors[align], marker=markers[align], label=align, zorder=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$|\mathbb{E}[f\cdot(y-u)]|$")
            ax.set_ylabel(r"measured $|\partial L/\partial\alpha|$")
            ax.legend(frameon=False, fontsize=8)
            ax.set_title("init gradient")
            ax = axes[1]
            for align in ("off", "shuffle", "aligned"):
                xs, ys = [], []
                for rec in recs:
                    if rec["align"] != align or rec.get("gamma", 1) not in (1, 1.0):
                        continue
                    xs.append(max(abs(rec["analytic_overlap"]), 1e-8))
                    ys.append(rec["final_answer"])
                if xs:
                    ax.scatter(xs, ys, s=26, c=colors[align], marker=markers[align], label=align, zorder=2)
            ax.axhline(0.5, color="0.85", lw=1, zorder=0)
            ax.set_xscale("log")
            ax.set_xlabel(r"answer mass in the feature")
            ax.set_ylabel("outcome accuracy at 8k")
            ax.set_ylim(0, 1.05)
            ax.legend(frameon=False, fontsize=8)
            ax.set_title(r"$n=8$, chance $1/2$")
            fig.tight_layout()
            fig.savefig(FIG / "wr_intervention.pdf")
            fig.savefig(FIG / "wr_intervention.png", dpi=160)
            print("wr", len(recs))


        if __name__ == "__main__":
            lambda_figure()
            wr_figure()
    elif _cmd == 'damping':
        import numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = np.array([2,4,6,8,9,10])
        out = np.array([8.6e-4,2.5e-4,6.5e-5,2.1e-5,7.6e-6,3.6e-6])
        cot = np.array([1.774e-2,2.984e-2,3.891e-2,3.961e-2,4.220e-2,4.269e-2])
        BLUE, ORANGE = "#0a7fad", "#c8680a"
        fig, ax = plt.subplots(figsize=(5.2,2.6))
        ax.set_facecolor("#eaeaf2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
        ax.plot(n, cot, "-o", color=BLUE, lw=2, ms=6, label="chain-of-thought")
        ax.plot(n, out, "-s", color=ORANGE, lw=2, ms=6, label="outcome (answer-dependent)")
        nn = np.linspace(2,10,50)
        ax.plot(nn, out[0]*0.5**(nn-2), ":", color="0.35", lw=1.5, label=r"$\lambda^{n}$, $\lambda=0.5$")
        ax.set_yscale("log"); ax.set_xlabel("letters $n$"); ax.set_ylabel(r"gradient norm at $\theta_0$")
        ax.set_xticks(n)
        ax.text(10.15, cot[-1], "CoT", color="0.15", va="center", fontsize=9)
        ax.text(10.15, out[-1], "outcome", color="0.15", va="center", fontsize=9)
        ax.set_xlim(1.6, 11.4)
        ax.legend(frameon=True, fontsize=8, loc="lower left")
        fig.tight_layout()
        for ext in ("pdf","png"): fig.savefig(f"Paper/figures/damping_decay.{ext}", dpi=200)
    elif _cmd == 'spectra':
        """
        Tabulate spectrum.py JSON files, e.g. one per checkpoint, in step order.

            python handcoded/spectra_table.py logs/spec_track/*.json
        """
        import json
        import re
        import sys

        rows = []
        for path in sys.argv[1:]:
            with open(path) as f:
                r = json.load(f)
            m = re.search(r"step (\d+)", r.get("weights", ""))
            step = int(m.group(1)) if m else -1
            W = r["W_r"]
            word = r["W1_fit"] + r["Whi_fit"] if "W1_fit" in r else sum(W[1:])
            hi = (r["hi_share"], r["hi_share_se"]) if "hi_share" in r else (sum(W[2:]) / word if word > 0 else 0.0, float("nan"))
            rows.append((r["site"], r.get("weights", "init").split(" ")[0], step, W[0], word, hi,
                         r.get("grad_outcome", r.get("grad_state", float("nan"))), path))

        print(f"{'site':7s} {'trained':8s} {'step':>6s}  {'W_0':>9s}  {'W_>=1':>9s}  {'order>=2 share':>16s}  {'task grad':>9s}")
        for site, mode, step, w0, word, (hi, hise), g, path in sorted(rows):
            print(f"{site:7s} {mode:8s} {step:6d}  {w0:9.3e}  {word:9.3e}  {hi:+7.2f} +- {hise:5.2f}  {g:9.3e}")
