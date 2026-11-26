"""Table 14 (tab:J-ode): reduced gate dynamics of Theorem 4.3 from results/reduced_ode/runs.json (K = 2, d_h = 8).

Per p: runs solved, median solve time over the three seeds at eps = 0.3, 0.2, 0.1, the range over seeds of the slope of
log T against log(1/eps), and the trajectory checks of Lemma E.8 recorded by reduced_ode.py (g keeps its sign after prior
fitting, a crosses zero at most once, a g <= 0 and |a| nondecreasing once the gate is aligned with |a| >= a_sc/8).

    python handcoded/reduced_ode.py --K 2            # writes results/reduced_ode/runs.json
    python handcoded/checks/ode_table.py             # prints the table rows and the checks
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
runs = [r for r in json.loads((ROOT / 'results/reduced_ode/runs.json').read_text()) if r['K'] == 2 and r['dh'] == 8]
EPS = (0.3, 0.2, 0.1)
for p in sorted({r['p'] for r in runs}):
    rows = [r for r in runs if r['p'] == p and r['eps'] in EPS]
    solved = sum(r['solved_at'] is not None for r in rows)
    med = [statistics.median(r['solved_at'] for r in rows if r['eps'] == e) for e in EPS]
    slopes = []
    for seed in sorted({r['seed'] for r in rows}):
        pts = sorted((r['eps'], r['solved_at']) for r in rows if r['seed'] == seed)
        slopes.append(np.polyfit([math.log(1 / e) for e, _ in pts], [math.log(t) for _, t in pts], 1)[0])
    checks = all(r['G_keeps_sign'] and r['a_sign_changes'] <= 1 and r['invariant_after_alignment'] is not False
                 for r in rows)
    print(f"p={p:<5} solved {solved}/{len(rows)}  median T " + '  '.join(f'{m:.1e}' for m in med)
          + f"  slope {min(slopes):.2f}-{max(slopes):.2f}  Lemma E.8 checks {'pass' if checks else 'FAIL'}")
