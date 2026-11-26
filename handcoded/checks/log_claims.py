"""Check the counting-experiment claims read off lettertrace.py training logs (Tables 6-7: tab:gap, tab:controls):
extended 32k horizon sweeps, deeper 4-6 block models, first-occurrence controls, and counting modulo 3.

    uv run python handcoded/checks/log_claims.py        # -> results/paper/log_claims.json

Logs: logs/must_run/tab3_*.log (horizon sweep and first occurrence, two blocks), logs/remaining/depth_L{4,6}_*.log
(deeper models), logs/must_run/out32k_K3_s*.log (modulo 3). Each line `MODE step N loss L test exact E answer A` is one
evaluation.
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r'^(process|outcome)\s+step\s+(\d+)\s+loss\s+\S+\s+test exact\s+([\d.]+)', re.M)


def curves(path):
    out = {'process': {}, 'outcome': {}}
    for mode, step, exact in LINE.findall(Path(path).read_text()):
        out[mode][int(step)] = 100 * float(exact)
    return out


def at(curve, step):
    return curve[max(s for s in curve if s <= step)]


def reached(curve, budget, level=90.0):
    return any(v >= level for s, v in curve.items() if s <= budget)


def span(values):
    return [round(min(values), 1), round(max(values), 1)]


def group(pattern):
    runs = {Path(f).stem: curves(f) for f in sorted(glob.glob(str(ROOT / pattern)))}
    assert runs, pattern
    return runs


if __name__ == '__main__':
    res = {}
    for n in (8, 12, 16, 24):                                     # Table 10(a)
        runs = group(f'logs/must_run/tab3_count_n{n}_s*.log')
        res[f'horizon n={n}'] = dict(
            step_8k=span([at(c['process'], 8000) for c in runs.values()]),
            outcome_8k=span([at(c['outcome'], 8000) for c in runs.values()]),
            step_90_by_16k=sum(reached(c['process'], 16000) for c in runs.values()),
            outcome_90_by_32k=sum(reached(c['outcome'], 32000) for c in runs.values()), runs=len(runs))
    first = group('logs/must_run/tab3_first_n*_s*.log')            # Table 10(d), first occurrence
    res['first occurrence n=8-24'] = dict(
        step_8k=span([at(c['process'], 8000) for c in first.values()]),
        outcome_8k=span([at(c['outcome'], 8000) for c in first.values()]),
        step_90_by_16k=sum(reached(c['process'], 16000) for c in first.values()),
        outcome_90_by_16k=sum(reached(c['outcome'], 16000) for c in first.values()), runs=len(first))
    deep = group('logs/remaining/depth_L*_count_n*_s*.log')        # Table 10(d), deeper models
    res['count mod 2, 4 or 6 blocks'] = dict(
        step_final=span([at(c['process'], 8000) for c in deep.values()]),
        outcome_final=span([at(c['outcome'], 8000) for c in deep.values()]),
        step_above_80=sum(at(c['process'], 8000) > 80 for c in deep.values()), runs=len(deep))
    deep_first = group('logs/remaining/depth_L*_first_n*_s*.log')
    res['first occurrence, 4 or 6 blocks'] = dict(
        step_final=span([at(c['process'], 8000) for c in deep_first.values()]),
        outcome_final=span([at(c['outcome'], 8000) for c in deep_first.values()]), runs=len(deep_first))
    mod3 = group('logs/must_run/out32k_K3_s*.log')                 # counting modulo 3, n = 8
    res['mod 3, n=8, outcome at 32k'] = {k: round(at(c['outcome'], 32000), 1) for k, c in mod3.items()}
    for k, v in res.items():
        print(f'{k:34s} {v}')
    out = ROOT / 'results/paper/log_claims.json'
    out.write_text(json.dumps(res, indent=1))
