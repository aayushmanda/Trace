"""Reprocess already-saved depth_replication diagnostics into the theorem's
exact table-space credit quantity (Paper/appendix/L_diagnostics.tex,
eq:induced-scales / eq:depth-credit-bound). No retraining and no model
reload: reads saved `both_step*.npz` induced-table arrays and each run's
`checkpoints/data.pt` probe circuits only.

Pre-registered primary output columns: credit_table_mean, credit_table_max.
These are the theorem's literal Frobenius-norm table-space credit, computed
by src.eval.rule_credit.credit_summary -- not the parameter-space
`gradient_rule_pullback_norm` (Jacobian-confounded) or
`descent_oracle_rule_cosine` (alignment, not magnitude) already saved in
metrics.csv, and not the synthetic lambda-rescaled `exponent_fit_*` columns
in results/induced_rule/*.csv. See the plan file for why.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from src.experiments import reject_extra_flags

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import handcoded as h
from src.eval.executor_comparison import display_circuits
from src.eval.rule_credit import credit_summary

IN_DIR = ROOT / "results/executor_comparison/depth_replication"


def main(argv=None):
    reject_extra_flags(argv, __doc__)
    protocol = json.loads((IN_DIR / "protocol.json").read_text())
    gate_index = {g: i for i, g in enumerate(h.GATES)}
    k = 16  # Boolean-circuit state count (N_BITS=4); matches local_conditional's (M,16,16)
    rows = []
    for depth in protocol["depths"]:
        shown = display_circuits(depth)
        for seed in protocol["seeds"]:
            run_dir = IN_DIR / f"depth{depth}_seed{seed}"
            data = torch.load(run_dir / "checkpoints/data.pt", weights_only=False)
            probes = [h.Circuit(*p) for p in data["probes"]]
            circuits = shown + probes
            for step in protocol["checkpoints"]:
                npz_path = run_dir / f"both_step{step:05d}.npz"
                if not npz_path.exists():
                    continue
                table = np.load(npz_path)["local_conditional"]
                cmean, cmax, n = credit_summary(table, gate_index, circuits, k)
                rows.append(dict(depth=depth, seed=seed, step=step,
                                  credit_table_mean=cmean, credit_table_max=cmax,
                                  n_terms=n))
    if not rows:
        raise RuntimeError(f"no rows produced from {IN_DIR}")
    out_path = IN_DIR / "rule_credit_table.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
