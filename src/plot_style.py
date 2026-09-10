"""One matplotlib style for every figure in this repo. Call `apply_style()` before plotting."""
from __future__ import annotations

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "text.usetex": False,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.facecolor": "#EAEAF2",
    "axes.edgecolor": "#EAEAF2",
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "grid.alpha": 1.0,
    "axes.grid": True,
    "legend.framealpha": 0.82,
    "legend.facecolor": "#ECECF2",
    "legend.edgecolor": "#C8C8C8",
}


def apply_style(extra=None):
    """Set the default Trace figure style. Optional `extra` overrides/adds rcParams."""
    import matplotlib.pyplot as plt
    params = dict(STYLE)
    if extra:
        params.update(extra)
    plt.rcParams.update(params)
    return params
