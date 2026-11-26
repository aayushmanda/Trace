"""Regenerate the pgfplots data of Figure 1 (figures/overview.tex) and Figure 4 (figures/theory_measurement.tex).

Both figures embed their coordinates inline. This script recomputes every series from the stored results, with the
loaders of main_figures.py:
  Figure 1a  median and 25th-75th seed percentiles of exact accuracy, results/paper/zero_readout_n16_8k (seeds 42-51);
  Figure 1b  target A_r (results/paper/fig2/n8_s42.json) and Jacobian J_r
             (results/paper/exact_spectra/sigma_sweep/s42_sg0.02.txt), each normalized to sum one;
  Figure 4   outcome accuracy per horizon-alphabet cell (results/paper/lambda_grid, 20 cells x 3 seeds): cell mean
             with full seed range, against lambda^n, and the same minus the prior 50 (1 + lambda^n); Spearman rho.

    python handcoded/make_inline_figures.py           # compare with the figure files, print the largest deviation
    python handcoded/make_inline_figures.py --write   # rewrite the coordinate lists in place
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_figures import ROOT, grid_cells, histories, spearman  # noqa: E402

FIG = ROOT / 'papers/silent_routing/figures'


def coords(xs, ys, fmt):
    return ' '.join(f'({x:g},{fmt(y)})' for x, y in zip(xs, ys))


def overview_series():
    steps, curves, _, _, _ = histories('k2-zero')
    x = steps / 1000
    out = {}
    for mode in ('process', 'outcome'):
        lo, med, hi = np.percentile(curves[mode], [25, 50, 75], axis=0)
        out[f'{mode}_band'] = coords(np.r_[x, x[::-1]], np.r_[lo, hi[::-1]], lambda v: f'{v:.2f}')
        out[f'{mode}_median'] = coords(x, med, lambda v: f'{v:.2f}')
    target = np.asarray(json.loads((ROOT / 'results/paper/fig2/n8_s42.json').read_text())['A_r'])
    line = next(l for l in (ROOT / 'results/paper/exact_spectra/sigma_sweep/s42_sg0.02.txt').read_text().splitlines()
                if l.startswith('W_r / W'))
    jac = np.asarray([float(v) for v in line.split(':')[1].split('(')[0].split()])
    deg = np.arange(len(target))
    out['target'] = coords(deg, target / target.sum(), lambda v: f'{v:.4e}')
    out['jacobian'] = coords(deg, jac / jac.sum(), lambda v: f'{v:.4e}')
    return [out[k] for k in ('process_band', 'outcome_band', 'process_median', 'outcome_median', 'target', 'jacobian')]


def lambda_grid_series():
    cells = {}
    for c in grid_cells():
        cells.setdefault((c['n'], c['alphabet']), []).append(c)
    keys = sorted(cells)
    bias = np.array([cells[k][0]['bias'] for k in keys])
    acc = [np.array([c['outcome'] for c in cells[k]]) for k in keys]
    mean = np.array([a.mean() for a in acc])
    up, down = np.array([a.max() for a in acc]) - mean, mean - np.array([a.min() for a in acc])
    excess = mean - 50 * (1 + bias)
    def errs(ys):
        return ' '.join(f'({b:.6g},{y:.4g}) += (0,{u:.4g}) -= (0,{d:.4g})' for b, y, u, d in zip(bias, ys, up, down))
    rho = (spearman(bias, mean), spearman(bias, excess))
    return [errs(mean), errs(excess)], rho


def numbers(text):
    return np.array([float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', text)])


def sync(path, series, write):
    """Replace, in order, the bodies of `coordinates {...}` lists that hold data (the first len(series) of them)."""
    tex = path.read_text()
    blocks = list(re.finditer(r'coordinates \{([^}]*)\}', tex))
    assert len(blocks) == len(series), (path.name, len(blocks), len(series))
    worst = 0.0
    for m, new in zip(blocks, series):
        old_v, new_v = numbers(m.group(1)), numbers(new)
        assert old_v.shape == new_v.shape, (path.name, old_v.shape, new_v.shape)
        worst = max(worst, float(np.max(np.abs(old_v - new_v) / np.maximum(np.abs(old_v), 1e-12))))
    if write:
        for m, new in reversed(list(zip(blocks, series))):
            tex = tex[:m.start(1)] + new + tex[m.end(1):]
        path.write_text(tex)
    return worst


if __name__ == '__main__':
    write = '--write' in sys.argv
    d1 = sync(FIG / 'overview.tex', overview_series(), write)
    grid, rho = lambda_grid_series()
    d4 = sync(FIG / 'theory_measurement.tex', grid, write)
    print(f'overview.tex: largest relative deviation {d1:.2e}')
    print(f'theory_measurement.tex: largest relative deviation {d4:.2e}; Spearman rho cell means {rho[0]:.2f}, '
          f'excess {rho[1]:.2f}')
    if write:
        print('rewrote both coordinate blocks')
