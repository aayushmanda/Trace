"""Aggregate results/oracle_alignment/summary.json into the tables the paper would use.

Reports, per depth: mean and sample standard deviation over seeds of the cosine
between the negative population gradient and the oracle direction, on the
executor-block subspace and on the full parameter vector, each against the
chance level measured with a random direction of the same dimension.  A cosine
is only interpretable relative to that chance level, which is ~1/sqrt(dim).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.experiments import reject_extra_flags

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "oracle_alignment" / "summary.json"


def stats(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return m, sd, n


def main(argv=None) -> int:
    reject_extra_flags(argv, __doc__)
    data = json.loads(SUMMARY.read_text())
    rows = data["alignment"]
    depths = sorted({r["depth"] for r in rows})

    for space in ("blocks", "all"):
        print(f"\n=== cosine(-grad, theta* - theta_0) on the '{space}' subspace ===")
        print(f"{'D':>3} {'dim':>9} {'chance |c|':>11} "
              f"{'outcome':>20} {'process':>20} {'proc/out |c|':>13}")
        for d in depths:
            sel = [r for r in rows if r["depth"] == d]
            dim = sel[0][f"dim_{space}"]
            ch = stats([abs(r[f"cos_{space}_chance"]) for r in sel])[0]
            om, osd, n = stats([r[f"cos_{space}_outcome"] for r in sel])
            pm, psd, _ = stats([r[f"cos_{space}_process"] for r in sel])
            oa = stats([abs(r[f"cos_{space}_outcome"]) for r in sel])[0]
            pa = stats([abs(r[f"cos_{space}_process"]) for r in sel])[0]
            print(f"{d:>3} {dim:>9} {ch:>11.2e} "
                  f"{om:>+11.5f} ± {osd:<6.5f} {pm:>+11.5f} ± {psd:<6.5f} "
                  f"{pa / oa if oa else float('nan'):>13.2f}")
        print(f"  (n = {n} seeds; ± is the sample standard deviation across seeds)")

    print("\n=== |cosine| relative to chance, block subspace ===")
    print(f"{'D':>3} {'outcome/chance':>16} {'process/chance':>16}")
    for d in depths:
        sel = [r for r in rows if r["depth"] == d]
        ch = stats([abs(r["cos_blocks_chance"]) for r in sel])[0]
        oa = stats([abs(r["cos_blocks_outcome"]) for r in sel])[0]
        pa = stats([abs(r["cos_blocks_process"]) for r in sel])[0]
        print(f"{d:>3} {oa / ch:>16.1f} {pa / ch:>16.1f}")

    print("\n=== per-block alignment (mean over seeds, block subspace) ===")
    for d in depths:
        sel = [r for r in rows if r["depth"] == d]
        line_o, line_p = [], []
        for k in range(1, d + 1):
            line_o.append(stats([r[f"cos_block{k}_outcome"] for r in sel])[0])
            line_p.append(stats([r[f"cos_block{k}_process"] for r in sel])[0])
        print(f"  D={d}  outcome " + " ".join(f"{v:+.5f}" for v in line_o))
        print(f"       process " + " ".join(f"{v:+.5f}" for v in line_p))

    print("\n=== gradient norms (mean over seeds) ===")
    print(f"{'D':>3} {'|g_out|':>12} {'|g_proc|':>12} {'L_out':>9} {'L_proc':>9}")
    for d in depths:
        sel = [r for r in rows if r["depth"] == d]
        print(f"{d:>3} {stats([r['gradnorm_outcome'] for r in sel])[0]:>12.4e} "
              f"{stats([r['gradnorm_process'] for r in sel])[0]:>12.4e} "
              f"{stats([r['loss_outcome'] for r in sel])[0]:>9.4f} "
              f"{stats([r['loss_process'] for r in sel])[0]:>9.4f}")

    if "profile" in data:
        print("\n=== loss along theta(a) = theta_0 + a (theta* - theta_0) ===")
        prof = data["profile"]
        for d in depths:
            sel = sorted([r for r in prof if r["depth"] == d], key=lambda r: r["a"])
            if not sel:
                continue
            base = sel[0]["L_out"]
            # first a at which the outcome loss has fallen by 1% of its total drop
            drop = base - sel[-1]["L_out"]
            thresh = base - 0.01 * drop
            a_move = next((r["a"] for r in sel if r["L_out"] <= thresh), float("nan"))
            marks = [next(r for r in sel if abs(r["a"] - x) < 1e-9)
                     for x in (0.0, 0.25, 0.5, 0.75, 1.0) if any(abs(r["a"] - x) < 1e-9 for r in sel)]
            print(f"  D={d}  a at 1% of the drop = {a_move:.3f} | "
                  + " ".join(f"a={m['a']:.2f}:{m['L_out']:.3f}" for m in marks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
