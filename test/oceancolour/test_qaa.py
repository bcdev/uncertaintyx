#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.oceancolour.qaa import Eta
from uncertaintyx.oceancolour.qaa import S
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import RegressionPlot


def read_data(package: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns resource data.

    :param package: The package name.
    :param filename: The filename.
    :returns: The x and y figure data.
    """
    with resources.path(package, filename) as resource:
        with open(resource) as r:
            df = pd.read_csv(r, sep=";")
            x = df["X"].values
            y = df["Y"].values
    return x, y


class QaaTest(unittest.TestCase):
    """
    Tests QAA model fitting.
    """

    def test_lee_2010_figure_2(self):
        """
        Tests the bootstrap method by fitting an empirical model functions
        to published data (Lee et al., 2010, Figure 2).
        """
        x, y = read_data("test.resources", "fig2.csv")

        result = Bootstrap(HomoscedasticRegression()).fit(Eta(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.96, result.popt[0], delta=0.05)
        self.assertAlmostEqual(1.69, result.popt[1], delta=0.05)
        self.assertAlmostEqual(0.11, result.punc[0], delta=0.01)
        self.assertAlmostEqual(0.13, result.punc[1], delta=0.01)
        self.assertAlmostEqual(39.5, result.cost, delta=0.5)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            title="Replica of Lee et al. (2010, Figure 2)",
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$\eta$",
            xrange=(0.5, 4.5),
            yrange=(-0.5, 2.5),
            savefig="qaa2.png",
        )
        MatrixPlot().plot(
            result.ycov_p(np.linspace(0.5, 4.5, 1000)),
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            xrange=(0.5, 4.5),
            yrange=(0.5, 4.5),
            cmap="cividis",
            cbar_label=r"variance-covariance $U_p(\eta)$",
            cbar_max=0.024,
            cbar_min=-0.024,
            savefig="qaa2-ycov.png",
        )

    def test_lee_2010_figure_3(self):
        """
        Tests the bootstrap method by fitting an empirical model function
        to published data (Lee et al., 2010, Figure 3).
        """
        x, y = read_data("test.resources", "fig3.csv")

        result = Bootstrap(HomoscedasticRegression()).fit(S(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(0.017, result.popt[0], delta=0.001)
        self.assertAlmostEqual(0.0, result.popt[1], delta=0.02)
        self.assertAlmostEqual(0.0, result.popt[2], delta=5.0)
        self.assertAlmostEqual(0.001, result.punc[0], delta=0.001)
        self.assertAlmostEqual(0.011, result.cost, delta=0.002)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            title="Replica of Lee et al. (2010, Figure 3)",
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"S ($\mathrm{nm}^{-1}$)",
            xrange=(0.050, 9.950),
            yrange=(0.005, 0.035),
            savefig="qaa3.png",
        )
        MatrixPlot().plot(
            result.ycov_p(np.linspace(0.05, 9.95, 1000)),
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            xrange=(0.05, 9.95),
            yrange=(0.05, 9.95),
            cmap="viridis",
            cbar_label=r"variance-covariance $U_p(S)$ ($\mathrm{nm}^{-2}$)",
            cbar_max=3.0e-05,
            cbar_min=0.0e-05,
            savefig="qaa3-ycov.png",
        )


if __name__ == "__main__":
    unittest.main()
