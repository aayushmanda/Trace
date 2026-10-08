"""Standalone, paired horizon-versus-bias comparison from recorded grid runs.

    python3 handcoded/lambda_grid_figure.py

The main manuscript includes the same comparison in main_theory_measurement.pdf.
"""
import matplotlib.pyplot as plt
from main_figures import grid_cells, prediction_panels, save
from figure_style import TEXT_WIDTH, style

if __name__ == '__main__':
    style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.1))
    fig.subplots_adjust(left=.09, right=.985, bottom=.2, top=.87, wspace=.3)
    prediction_panels(axes, grid_cells(), letters=('a', 'b'))
    save(fig, 'lambda_grid')
    print('Wrote Paper/figures/lambda_grid.pdf')
