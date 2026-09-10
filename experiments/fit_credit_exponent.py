"""Tier 1 closure test: does the theorem's eps_rule^(D-1) suppression law
hold against real (not synthetic-rescaled) trained-model measurements?

Merges results/executor_comparison/depth_replication/rule_credit_table.csv
(this session's new table-space credit, src.eval.rule_credit) with the same
run's metrics.csv (epsilon_rule, plus the two parameter-space quantities
reported only as caveated secondary columns) on (depth, seed, step), pooling
across checkpoints for statistical power.

Pre-registered before fitting: credit_table_mean / credit_table_max are the
primary targets, because they are the theorem's literal quantity
(Paper/body.tex eq:depth-credit-bound). gradient_rule_pullback_norm
(parameter-space, Jacobian-confounded) and descent_oracle_rule_cosine
(alignment, not magnitude) are reported for transparency only and are never
selected as the headline number based on which one fits best.
"""
import csv
import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plot_style import apply_style

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / 'results/executor_comparison/depth_replication'
DATA_DIR = ROOT / 'Paper/data'
FIG_DIR = ROOT / 'Paper/figures'
COLORS = {2: '#087eaa', 4: '#c56b08', 6: '#6b4c9a'}
N_BOOT = 2000
RNG_SEED = 0


def _ols(X, y):
    """Least-squares fit; returns (coefs, r2)."""
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coefs
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return coefs, r2


def primary_fit(df, credit_col):
    """log(credit) = b0 + b1 * (D-1) * log(eps_rule). Target: b1 ~ 1."""
    x = (df['depth'] - 1) * np.log(df['epsilon_rule'])
    y = np.log(df[credit_col])
    X = np.vstack([x, np.ones_like(x)]).T
    coefs, r2 = _ols(X, y.to_numpy())
    return dict(slope=float(coefs[0]), intercept=float(coefs[1]), r2=r2)


def falsification_fit(df, credit_col):
    """log(credit) = g0 + g1*log(eps_rule) + g2*D. If g2 is far from 0 after
    controlling for log(eps_rule), the simple (D-1) form is incomplete."""
    x1 = np.log(df['epsilon_rule'])
    x2 = df['depth'].astype(float)
    y = np.log(df[credit_col])
    X = np.vstack([x1, x2, np.ones_like(x1)]).T
    coefs, r2 = _ols(X, y.to_numpy())
    return dict(eps_coef=float(coefs[0]), depth_coef=float(coefs[1]),
                intercept=float(coefs[2]), r2=r2)


def per_depth_slopes(df, credit_col):
    out = {}
    for depth, g in df.groupby('depth'):
        x = np.log(g['epsilon_rule'])
        y = np.log(g[credit_col])
        X = np.vstack([x, np.ones_like(x)]).T
        coefs, r2 = _ols(X, y.to_numpy())
        out[int(depth)] = dict(slope=float(coefs[0]), r2=r2, n=len(g))
    return out


def cluster_bootstrap_primary_ci(df, credit_col, n_boot=N_BOOT, seed=RNG_SEED):
    """Resample whole (depth,seed) run trajectories with replacement -- steps
    within one run are autocorrelated, so row-level resampling would
    understate the true uncertainty."""
    rng = np.random.default_rng(seed)
    runs = df[['depth', 'seed']].drop_duplicates().to_numpy()
    slopes = []
    for _ in range(n_boot):
        pick = runs[rng.integers(0, len(runs), size=len(runs))]
        parts = [df[(df.depth == d) & (df.seed == s)] for d, s in pick]
        boot = pd.concat(parts, ignore_index=True)
        slopes.append(primary_fit(boot, credit_col)['slope'])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def load_data():
    credit = pd.read_csv(IN_DIR / 'rule_credit_table.csv')
    metrics = pd.read_csv(IN_DIR / 'metrics.csv')
    metrics = metrics[metrics['mode'] == 'both'][
        ['depth', 'seed', 'step', 'epsilon_rule', 'gradient_rule_pullback_norm',
         'descent_oracle_rule_cosine']
    ]
    df = credit.merge(metrics, on=['depth', 'seed', 'step'], how='inner')
    before = len(df)
    df = df[(df['credit_table_mean'] > 0) & (df['credit_table_max'] > 0) &
            (df['epsilon_rule'] > 0)].reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped


def make_figure(df):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in zip(axes, ['credit_table_mean', 'credit_table_max'],
                               ['Mean table-space credit', 'Max table-space credit']):
        for depth, g in df.groupby('depth'):
            ax.scatter(g['epsilon_rule'], g[col], s=14, alpha=0.6,
                       color=COLORS.get(depth, '#333'), label=f'D={depth}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon_{\rm rule}$'); ax.set_ylabel(title)
        ax.legend(fontsize=9)
    fig.savefig(FIG_DIR / 'rule_credit_exponent.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'rule_credit_exponent.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    df, dropped = load_data()
    print(f'{len(df)} pooled rows after merge (dropped {dropped} non-positive rows)')
    if df.empty:
        raise RuntimeError('no usable rows after merge/filter')

    report = {'n_rows': len(df), 'n_runs': len(df[['depth', 'seed']].drop_duplicates())}
    macros = {}
    rows_tex = []

    for col, tag in [('credit_table_mean', 'Mean'), ('credit_table_max', 'Max')]:
        primary = primary_fit(df, col)
        fals = falsification_fit(df, col)
        per_depth = per_depth_slopes(df, col)
        ci_lo, ci_hi = cluster_bootstrap_primary_ci(df, col)

        report[col] = dict(primary=primary, falsification=fals,
                            per_depth=per_depth, ci=(ci_lo, ci_hi))

        macros[f'RuleCredit{tag}Slope'] = f"{primary['slope']:.3f}"
        macros[f'RuleCredit{tag}SlopeCILo'] = f"{ci_lo:.3f}"
        macros[f'RuleCredit{tag}SlopeCIHi'] = f"{ci_hi:.3f}"
        macros[f'RuleCredit{tag}R2'] = f"{primary['r2']:.3f}"
        macros[f'RuleCredit{tag}DepthCoef'] = f"{fals['depth_coef']:.3f}"

        print(f'\n[{col}] primary fit: slope={primary["slope"]:.3f} '
              f'(95% CI [{ci_lo:.3f}, {ci_hi:.3f}]), R2={primary["r2"]:.3f} '
              f'(target slope ~= 1.0)')
        print(f'[{col}] falsification: eps_coef={fals["eps_coef"]:.3f} '
              f'depth_coef={fals["depth_coef"]:.3f} R2={fals["r2"]:.3f} '
              f'(depth_coef far from 0 => (D-1) form incomplete)')
        for depth, d in sorted(per_depth.items()):
            target = depth - 1
            print(f'[{col}] D={depth}: slope={d["slope"]:.3f} '
                  f'(target {target}), R2={d["r2"]:.3f}, n={d["n"]}')
            rows_tex.append(
                f"{depth} & {tag} & ${d['slope']:.3f}$ & ${target}$ & ${d['r2']:.3f}$ & ${d['n']}$ \\\\"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / 'rule_credit_exponent_numbers.tex').open('w') as f:
        for name, value in macros.items():
            f.write(f'\\newcommand{{\\{name}}}{{{value}}}\n')
    with (DATA_DIR / 'rule_credit_exponent_rows.tex').open('w') as f:
        f.write('\n'.join(rows_tex) + '\n')
    with (IN_DIR / 'rule_credit_exponent_report.json').open('w') as f:
        json.dump(report, f, indent=2)

    make_figure(df)
    print(f'\nWrote {DATA_DIR/"rule_credit_exponent_numbers.tex"}, '
          f'{DATA_DIR/"rule_credit_exponent_rows.tex"}, '
          f'{IN_DIR/"rule_credit_exponent_report.json"}, '
          f'{FIG_DIR/"rule_credit_exponent.pdf"}')


if __name__ == '__main__':
    main()
