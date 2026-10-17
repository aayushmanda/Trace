"""Family C' (order-one W_Q, W_V, W_O; eps-scale W_K, W_1, W_2) under the anti-copy law: table and report.

Reads checks/results/randinit_cprime/Cp_n*_*.json written by
    untied_ladder.py --law anti --levels Cp --cp_bias gate rand --cp_readout one eps ...
For every (MLP bias, readout, optimizer, N, kappa): runs reaching 99% clean-rule accuracy, runs leaving the chance plateau
(loss below 0.68), their median exit time, and the median final loss against the Bayes value H(0.1) = 0.325.
Then the exit time against eps per kappa, with least-squares slopes against log(1/eps) and 1/eps.
Writes figures/randinit_cprime_table.tex and results/randinit_cprime/summary.json.

    python Paper2/checks/make_randinit_cprime_table.py
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = []
for f in sorted(glob.glob(str(ROOT / "checks" / "results" / "randinit_cprime" / "Cp_n*_*.json"))):
    runs += json.load(open(f))["runs"]
BAYES = -(.9 * math.log(.9) + .1 * math.log(.1))
learned = lambda r: any(c[2] >= .99 for c in r["curve"])
exit_ = lambda r: next((c[0] for c in r["curve"] if c[1] <= .68), None)
VARS = [("rand", "one"), ("rand", "eps"), ("gate", "one"), ("gate", "eps")]
VNAME = {("rand", "one"): r"random $\bm b_1$, $\bm u\sim1$", ("rand", "eps"): r"random $\bm b_1$, $\bm u\sim\varepsilon$",
         ("gate", "one"): r"$\bm b_1=-B\bm1$, $\bm u\sim1$", ("gate", "eps"): r"$\bm b_1=-B\bm1$, $\bm u\sim\varepsilon$"}
KS = (1., 0., -1.)
sel = lambda v, opt, n, k: [r for r in runs if (r["bias"], r["readout"], r["opt"], r["n"], r["kappa"]) == (*v, opt, n, k)]
med = lambda xs: float(np.median(xs)) if xs else float("nan")
fmt = lambda x: "--" if x != x else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")

summary = {}
for v in VARS:
    for opt in ("adam", "gd"):
        for n in (8, 16, 32):
            for k in KS:
                rr = sel(v, opt, n, k)
                if not rr:
                    continue
                ex = [e for r in rr if (e := exit_(r)) is not None]
                by_eps = {}
                for eps in sorted({r["eps"] for r in rr}):
                    re_ = [r for r in rr if r["eps"] == eps]
                    exs = [e for r in re_ if (e := exit_(r)) is not None]
                    by_eps[eps] = dict(exited=len(exs), median_exit=med(exs), learned=sum(map(learned, re_)),
                                       median_final_loss=med([r["final_loss"] for r in re_]),
                                       median_final_gate=med([r["curve"][-1][6] for r in re_]),
                                       median_final_live=med([r["curve"][-1][7] for r in re_]))
                fits = {}
                pts = [(e, d["median_exit"]) for e, d in by_eps.items() if d["exited"] >= 6]
                if len(pts) >= 3:
                    x, t = np.array(pts).T
                    for name, f in (("log", np.log(1 / x)), ("inv", 1 / x)):
                        A = np.stack([f, np.ones_like(f)], 1)
                        coef, res, *_ = np.linalg.lstsq(A, t, rcond=None)
                        r2 = 1 - ((A @ coef - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-30)
                        fits[name] = dict(slope=float(coef[0]), intercept=float(coef[1]), r2=float(r2))
                summary[f"{v[0]}|{v[1]}|{opt}|{n}|{k}"] = dict(
                    runs=len(rr), learned=sum(map(learned, rr)), exited=len(ex), median_exit=med(ex),
                    median_final_loss=med([r["final_loss"] for r in rr]), min_final_loss=min(r["final_loss"] for r in rr),
                    by_eps=by_eps, exit_fits=fits)
(ROOT / "checks" / "results" / "randinit_cprime" / "summary.json").write_text(json.dumps(summary, indent=1))

# Table: per variant and N, Adam and GD (all eps): learned runs and median plateau exit, per kappa.
lines = [r"\begin{tabular}{@{}ll" + "ccc" * 4 + "@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Adam: learned} & \multicolumn{3}{c}{Adam: median exit} & "
         r"\multicolumn{3}{c}{GD: learned} & \multicolumn{3}{c}{GD: median exit}\\",
         "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(4)),
         r"Variant & $N$ & " + " & ".join([r"$\kappa=1$ & $\kappa=0$ & $\kappa=-1$"] * 4) + r"\\", r"\midrule"]
for i, v in enumerate(VARS):
    if i:
        lines.append(r"\addlinespace")
    for j, n in enumerate((8, 16, 32)):
        cells = []
        for opt in ("adam", "gd"):
            ss = [summary.get(f"{v[0]}|{v[1]}|{opt}|{n}|{k}") for k in KS]
            cells += [f"{s['learned']}/{s['runs']}" if s else "--" for s in ss]
            cells += [fmt(s["median_exit"]) if s else "--" for s in ss]
        lines.append((VNAME[v] if j == 0 else "") + f" & {n} & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "figures" / "randinit_cprime_table.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

print(f"\nBayes loss {BAYES:.4f}")
for key, s in summary.items():
    fit = " ".join(f"{nm}: slope {f['slope']:.3g} R2 {f['r2']:.2f}" for nm, f in s["exit_fits"].items())
    print(f"{key:22s} learned {s['learned']:3d}/{s['runs']} exited {s['exited']:3d} median exit {fmt(s['median_exit']):>6s} "
          f"final loss med {s['median_final_loss']:.3f} min {s['min_final_loss']:.3f} | {fit}")
    print("    exit by eps: " + "  ".join(f"{e:g}:{fmt(d['median_exit'])}({d['exited']},L{d['learned']})"
                                          for e, d in s["by_eps"].items()))
