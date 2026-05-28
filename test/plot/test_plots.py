#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.plot.plots import BernsteinBasisPlot


class BernsteinBasisPlotTest(unittest.TestCase):
    """Tests plotting Bernstein basis polynomials."""

    def test_plot_bernstein_basis(self):
        plot = BernsteinBasisPlot("paper")
        grid_size = 100
        x = np.linspace(0.0, 1.0, grid_size)
        y = np.linspace(0.0, 1.0, grid_size)

        caption = (
            r"Approximation with Bernstein polynomials smoothly and "
            r"simultaneously fits both the target function and its "
            r"derivatives"
        )
        fig = plot.plot(
            x,
            y,
            degree=2,
            caption=caption,
            cmap="viridis",
            savefig="bernstein_basis.png",
        )
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
