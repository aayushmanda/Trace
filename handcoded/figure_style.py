"""Shared print-scale styling for the paper's vector scientific figures.

Every figure is drawn at its final printed size (NeurIPS text width 5.5 in) and
included with width=\\linewidth, so the point sizes below are the printed sizes.
Fonts are STIX (Times-compatible, matching the body text) embedded as TrueType
(Type 42); no Type 3 fonts.

Encoding is fixed across all figures, so colour never carries meaning alone:
  step supervision     STEP    blue      square marker, solid line
  outcome supervision  OUTCOME vermillion circle marker, solid line
  theory / exact ref.  THEORY  green     dashed line
  model / Jacobian     MODEL   purple    triangle marker
  chance / baseline    MUTED   gray      short-dashed line
Colours are from the Okabe-Ito colour-blind-safe palette.
"""
import matplotlib.pyplot as plt
import numpy as np

TEXT_WIDTH = 5.5  # inches, NeurIPS \textwidth

STEP = '#0072B2'
OUTCOME = '#D55E00'
THEORY = '#009E73'
MODEL = '#7B4F9E'
INK = '#1A1A1A'
MUTED = '#6E6E6E'
STEP_LIGHT = '#E3EEF6'
OUTCOME_LIGHT = '#FBEAE0'

MARKER = {'step': 's', 'outcome': 'o', 'model': '^', 'other': 'D'}
CHANCE_LS = (0, (3, 2))
THEORY_LS = (0, (4, 2))

# Printed sizes in points.
FS_TEXT = 8      # axis labels, legends, annotations
FS_TICK = 7.5    # tick labels
FS_SMALL = 7     # minor annotations (never smaller)
FS_PANEL = 9     # bold panel letters


def style():
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': FS_TEXT, 'axes.titlesize': FS_TEXT, 'axes.labelsize': FS_TEXT,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_TEXT,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#333333', 'axes.linewidth': .7,
        'axes.labelcolor': INK, 'text.color': INK, 'axes.labelpad': 3,
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.minor.width': .45, 'ytick.minor.width': .45,
        'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'lines.linewidth': 1.3, 'lines.markersize': 3.5,
        'lines.markeredgewidth': 0,
        'errorbar.capsize': 0,
        'grid.color': '#E3E3E3', 'grid.linewidth': .45,
        'legend.frameon': False, 'legend.handlelength': 1.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
    })


def panel(ax, letter, title=None):
    """Bold 9 pt panel letter followed by a short regular-weight descriptor."""
    label = rf'$\mathbf{{({letter})}}$'
    ax.set_title(label + (f' {title}' if title else ''), loc='left', pad=5,
                 fontsize=FS_PANEL)


def clean(ax, grid=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='y')


def band(ax, x, values, color, **kwargs):
    """Pointwise median with a 25th--75th percentile seed band."""
    lo, med, hi = np.percentile(values, [25, 50, 75], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=.18, linewidth=0)
    ax.plot(x, med, color=color, **kwargs)
    return med
