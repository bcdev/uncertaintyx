#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.plot.plots import BernsteinBasisPlot
from uncertaintyx.plot.plots import WaterClassPlot


def read_owt_data(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Returns the optical water types (OWT) data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        rows = []
        with open(resource) as r:
            df = pd.read_csv(r, sep=";", header=None, index_col=0)
            for name, _ in df.items():
                rows.append(df[name].values)
            data = np.stack(rows, axis=-1)
    wav = data[0, :6]
    rrs = data[1:, :6]
    unc = data[1:, 6:]

    return wav, rrs, unc, rrs.shape[0], rrs.shape[1]


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


class WaterClassPlotTest(unittest.TestCase):
    """Tests plotting water classes."""

    def test_plot_water_classes(self):
        w, R, u, _, _ = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )

        fig = WaterClassPlot("paper").plot(
            w,
            R[:-2],
            u[:-2],
            xlabel=r"wavelength $\lambda$ (nm)",
            ylabel=r"remote sensing reflectance "
            r"$R_{\mathrm{rs}}(\lambda)$ (sr$^{-1}$)",
            title="Water classes (Jackson et al., 2017)",
            savefig="water_classes.png",
        )
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
