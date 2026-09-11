"""Table 2 / Cor. 2 as a pre-registered test, not an illustration.

CPU. Writes the prediction *before* loading empirical cells.

Cor. 2 / near-mixing (`cor:near-mixing` in kernel scripts): outcome
executor-relevant credit scales as ε^{D-1} inside the mixing ball ε ≤ 1/2.
Process credit stays Θ(1). Pre-registered ε is the theorem's mixing-ball
edge 1/2 — not fitted to 97→72→12.8→6.25, not the random-init mixing radius
in init_induced.csv (that ε would wrongly predict collapse at D=2).

Predicted failure among measured depths {2,4,6,8}: first D with
(1/2)^{D-1} ≤ 1/K = 0.0625 is D=6. D=8 at chance. Process stays high at
every D. Then load results/mechanism_deep/recovered.csv and report match.

This is predictive closure on the D-block outcome net already run. It is not
mechanistic identification (probes+patches; see RUN_TRANSFORMER.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "closure_handcoded"
RECOVERED = ROOT / "results" / "mechanism_deep" / "recovered.csv"

# Pre-registered. Do not edit after seeing Table 2.
EPS = 0.5
K = 16
CHANCE = 1.0 / K
DEPTHS = (2, 4, 6, 8)
# Collapse when ε^{D-1} ≤ chance. Among DEPTHS, first such D is 6.
SUCCEED_MAX_D = 4          # D=2,4: credit_scale > chance
COLLAPSE_MIN_D = 6         # D=6,8: credit_scale ≤ chance
OUTCOME_SUCCEED_MIN = 2 * CHANCE   # 0.125: not chance
OUTCOME_COLLAPSE_MAX = 0.25        # collapse underway
OUTCOME_D8_TOL = 0.02              # |acc − 1/K|
PROCESS_MIN = 0.95


def credit_scale(depth, eps=EPS):
    return float(eps ** (depth - 1))


def predict():
    """No empirical Table 2 numbers in this object."""
    scales = {str(d): credit_scale(d) for d in DEPTHS}
    collapsed = [d for d in DEPTHS if scales[str(d)] <= CHANCE]
    return {
        "eps": EPS,
        "eps_source": (
            "theorem mixing-ball edge ε=1/2 (hypothesis of Cor. 2 / cor:near-mixing). "
            "Not fitted to Table 2. Not the init ||P̂_g−U||_2 from init_induced.csv."
        ),
        "K": K,
        "chance": CHANCE,
        "credit_scale_eps_to_D_minus_1": scales,
        "predicted_first_collapsed_depth": min(collapsed) if collapsed else None,
        "predicted": {
            "D2_outcome": "above chance (scale 0.5 >> 0.0625)",
            "D4_outcome": "above chance, below D=2 (scale 0.125 > 0.0625)",
            "D6_outcome": "collapse underway (scale 0.03125 < 0.0625)",
            "D8_outcome": "at chance 1/K",
            "process_all_D": "stays high (Θ(1) credit, independent of ε^{D-1})",
            "outcome_local": "above outcome at D≥4 if measured (local readout bypasses the product); not required for this test",
        },
        "checks": {
            "outcome_D2_D4_above_2chance": OUTCOME_SUCCEED_MIN,
            "outcome_D6_D8_at_most": OUTCOME_COLLAPSE_MAX,
            "outcome_D8_within": OUTCOME_D8_TOL,
            "process_min": PROCESS_MIN,
            "monotone_outcome": "acc(2) > acc(4) > acc(6) ≥ acc(8) - 0.02",
        },
        "not_claimed": (
            "exact 97/72/12.8/6.25, and not mechanistic use of s_t. "
            "Init mixing radius ~0.01–0.05 would predict collapse at every D including D=2; "
            "that ε is the wrong object for this test."
        ),
    }


def _frac(mean_field):
    v = float(mean_field)
    return v / 100.0 if v > 1.5 else v


def load_table(path=RECOVERED):
    path = Path(path)
    if not path.exists():
        return None
    by = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("metric") != "final_answer":
                continue
            d, cond = int(row["depth"]), row["condition"]
            by[(d, cond)] = {
                "mean": _frac(row["mean"]),
                "sd": _frac(row["sd"]) if row["sd"] not in ("", None) else float("nan"),
                "n": int(float(row["n"] or 0)),
            }
    return by


def check(pred, emp):
    def acc(d, cond):
        cell = emp.get((d, cond))
        return None if cell is None else cell["mean"]

    results = {}
    results["process_all_high"] = all(
        acc(d, "process") is not None and acc(d, "process") >= PROCESS_MIN for d in DEPTHS
        if emp.get((d, "process"))
    )
    results["outcome_D2_succeed"] = (acc(2, "outcome") or 0) >= OUTCOME_SUCCEED_MIN
    results["outcome_D4_succeed"] = (acc(4, "outcome") or 0) >= OUTCOME_SUCCEED_MIN
    results["outcome_D6_collapse"] = acc(6, "outcome") is not None and acc(6, "outcome") <= OUTCOME_COLLAPSE_MAX
    results["outcome_D8_chance"] = acc(8, "outcome") is not None and abs(acc(8, "outcome") - CHANCE) <= OUTCOME_D8_TOL
    o = [acc(d, "outcome") for d in DEPTHS]
    results["monotone"] = (
        None not in o and o[0] > o[1] > o[2] and o[2] >= o[3] - 0.02
    )
    n_ok = sum(bool(v) for v in results.values())
    results["n_passed"] = n_ok
    results["n_checks"] = len(results) - 1
    results["overall"] = "MATCH" if n_ok == results["n_checks"] else (
        "PARTIAL" if n_ok >= 4 else "MISMATCH"
    )
    results["empirical"] = {
        f"D{d}_{c}": (None if emp.get((d, c)) is None else round(emp[(d, c)]["mean"], 4))
        for d in DEPTHS for c in ("outcome", "outcome_local", "process")
    }
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovered", default=str(RECOVERED))
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pred = predict()
    pred_path = OUT / "table2_prediction.json"
    pred_path.write_text(json.dumps(pred, indent=2) + "\n")
    print("=== pre-registered prediction (ε=1/2, before Table 2) ===")
    for d in DEPTHS:
        scale = pred["credit_scale_eps_to_D_minus_1"][str(d)]
        flag = "collapse" if scale <= CHANCE else "succeed"
        print(f"  D={d}  ε^{d-1}={scale:.5f}  vs 1/K={CHANCE:.4f}  → {flag}")
    print(f"wrote {pred_path}")

    emp = load_table(args.recovered)
    if emp is None:
        print(f"no empirical file at {args.recovered}")
        print("recover with: python experiments/run.py recover-mechanism-deep")
        verdict = {"overall": "NO_DATA", "prediction": pred}
        (OUT / "table2_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        return 1
    verdict = check(pred, emp)
    payload = {"prediction": pred, "verdict": verdict, "source": str(args.recovered)}
    (OUT / "table2_verdict.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("=== check against recovered Table 2 ===")
    for k, v in verdict.items():
        if k != "empirical":
            print(f"  {k}: {v}")
    print("  empirical:", verdict["empirical"])
    print(f"wrote {OUT / 'table2_verdict.json'}")
    return 0 if verdict["overall"] == "MATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
