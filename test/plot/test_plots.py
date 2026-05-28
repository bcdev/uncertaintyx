#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.plot.plots import BernsteinBasisPlot


class BernsteinBasisPlotTest(unittest.TestCase):

    def test_plot_bernstein_basis(self):
        plot = BernsteinBasisPlot("paper")
        grid_size = 100
        x = np.linspace(0.0, 1.0, grid_size)
        y = np.linspace(0.0, 1.0, grid_size)

        fig = plot.plot(
            x, y, degree=2, cmap="viridis", savefig="bernstein_basis.png"
        )
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
