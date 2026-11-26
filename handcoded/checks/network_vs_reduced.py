"""Compare gradient flow on the full two-block gate network (research/gate_network/run.py) with the reduced
three-scalar ODE (reduced_ode.py) on the 27 runs of Figure 3b, and print the largest relative difference of solve times.

    python handcoded/checks/network_vs_reduced.py
"""
import json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HANDCODED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDCODED))
sys.path.insert(0, str(HANDCODED / "research" / "gate_network"))
import run as gn
import reduced_ode as ro
seeds = [("aligned", 1.3, -0.8), ("misaligned", 1.3, 0.8), ("misaligned near-balanced", 1.0, 0.98)]
jobs = [(p, e, n, xq, (1 if p < 0.5 else -1) * xk) for p in (0.1, 0.25, 0.75) for e in (0.3, 0.2, 0.1) for n, xq, xk in seeds]
if __name__ == "__main__":
    with ProcessPoolExecutor(27) as ex:
        full = list(ex.map(gn.job, jobs))
        red = list(ex.map(ro.run, [(2, p, e, 8, xq, xk, 1.0, 75.0, 601, "line") for p, e, n, xq, xk in jobs]))
    worst = 0
    for f, r, j in zip(full, red, jobs):
        err = abs(f["solved_at"] - r["solved_at"]) / r["solved_at"]; worst = max(worst, err)
        print(j[:3], f"full {f['solved_at']:.6e} reduced {r['solved_at']:.6e} rel {err:.1e}")
    print("MAX REL ERR", worst)
    json.dump(dict(full=full, reduced=red), open(Path(__file__).with_name("network_vs_reduced.json"), "w"), indent=1)
