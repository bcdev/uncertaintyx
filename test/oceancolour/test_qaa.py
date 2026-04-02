#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.interface.core import M
from uncertaintyx.oceancolour.qaa import E
from uncertaintyx.oceancolour.qaa import Qaa
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

        result = Bootstrap(HomoscedasticRegression()).fit(E(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(2.0, result.popt[0], delta=0.1)
        self.assertAlmostEqual(0.7, result.popt[1], delta=0.2)
        self.assertAlmostEqual(0.2, result.punc[0], delta=0.1)
        self.assertAlmostEqual(0.1, result.punc[1], delta=0.1)
        self.assertAlmostEqual(0.3, result.rvar, delta=0.1)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)

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
            cbar_max=0.035,
            cbar_min=-0.035,
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

    def test_qaa(self):
        qaa = Qaa()
        self.assertIsInstance(qaa, M)

        W = np.array(  # noqa: N806
            [412.0, 443.0, 489.0, 510.0, 555.0, 670.0]
        )
        R = np.array(  # noqa: N806
            [0.00450, 0.00410, 0.00402, 0.00295, 0.00169, 0.00018]
        )
        aw = np.array([0.00473, 0.00635, 0.01500, 0.03250, 0.05950, 0.04390])
        bw = np.array([0.00340, 0.00250, 0.00158, 0.00133, 0.00090, 0.00000])
        x = np.expand_dims(np.stack([W, R, aw, bw]), axis=0)
        p = qaa.estimate()
        y = qaa.eval(p, x)

        print("y = ", y)


if __name__ == "__main__":
    unittest.main()
