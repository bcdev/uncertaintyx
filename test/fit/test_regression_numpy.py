#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.fit.eiv.numpy import EIV
from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.randomsampling import MonteCarlo
from uncertaintyx.fit.regression import HeteroHomoscedasticRegression
from uncertaintyx.fit.regression import HeteroscedasticRegression
from uncertaintyx.fit.regression import HomoHeteroscedasticRegression
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.m.numpy import Linear
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import RegressionPlot
from uncertaintyx.tyx import Result


def matrix(result: Result, n: int = 100) -> np.ndarray:
    """
    Returns the variance-covariance matrix of the fitted curve.
    """
    return np.squeeze(result.ycov_p(np.linspace(0.5, 99.5, n).reshape(1, n)))


class RegressionTest(unittest.TestCase):
    """
    Tests regression and random sampling methods.
    """

    def test_linear_model_homoscedastic(self):
        """
        Tests the bootstrap method by fitting a linear model to
        generated test data with unknown uncertainties.
        """
        n = 100
        x = np.linspace(0.0, 100.0, n).reshape((n, 1))
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(5489)
        y = x + rng.normal(0.0, u, (n, 1))
        x = x + rng.normal(0.0, u, (n, 1))

        result = Bootstrap(HomoscedasticRegression(EIV())).fit(Linear(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.10)
        self.assertAlmostEqual(0.0, result.popt[1], delta=5.00)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=2.00)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("zvar = ", result.zvar)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            xlabel=r"$x$",
            ylabel=r"$y$",
            xrange=(-10.0, 110.0),
            yrange=(-10.0, 110.0),
            savefig="lin1.png",
            title="Homoscedastic regression",
        )
        MatrixPlot().plot(
            matrix(result, n),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=5.0,
            cbar_min=0.0,
            cbar_label=r"variance-covariance $U_p(y)$",
            savefig="lin1-ycov.png",
            title="Homoscedastic regression",
        )

    def test_linear_model_homo_heteroscedastic(self):
        """
        Tests the bootstrap method by fitting a linear model to
        generated test data with unknown uncertainties of x and
        known uncertainties of y.
        """
        n = 100
        x = np.linspace(0.0, 100.0, n).reshape((n, 1))
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(5489)
        y = x + rng.normal(0.0, u, (n, 1))
        x = x + rng.normal(0.0, u, (n, 1))

        result = Bootstrap(HomoHeteroscedasticRegression(EIV())).fit(
            Linear(), x, y, u=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.10)
        self.assertAlmostEqual(0.0, result.popt[1], delta=2.00)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=1.00)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("zvar = ", result.zvar)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            xlabel=r"$x$",
            ylabel=r"$y$",
            xrange=(-10.0, 110.0),
            yrange=(-10.0, 110.0),
            savefig="lin2.png",
            title="Homo-heteroscedastic regression",
        )
        MatrixPlot().plot(
            matrix(result, n),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=5.0,
            cbar_min=0.0,
            cbar_label=r"variance-covariance $U_p(y)$",
            savefig="lin2-ycov.png",
            title="Homo-heteroscedastic regression",
        )

    def test_linear_model_hetero_homoscedastic(self):
        """
        Tests the bootstrap method by fitting a linear model
        to generated test data with known uncertainties of x
        and unknown uncertainties of y.
        """
        n = 100
        x = np.linspace(0.0, 100.0, n).reshape((n, 1))
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(5489)
        y = x + rng.normal(0.0, u, (n, 1))
        x = x + rng.normal(0.0, u, (n, 1))

        result = Bootstrap(HeteroHomoscedasticRegression(EIV())).fit(
            Linear(), x, y, u=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.popt[1], delta=1.50)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=1.00)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("zvar = ", result.zvar)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            xlabel=r"$x$",
            ylabel=r"$y$",
            xrange=(-10.0, 110.0),
            yrange=(-10.0, 110.0),
            savefig="lin3.png",
            title="Hetero-homoscedastic regression",
        )
        MatrixPlot().plot(
            matrix(result, n),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=5.0,
            cbar_min=0.0,
            cbar_label=r"variance-covariance $U_p(y)$",
            savefig="lin3-ycov.png",
            title="Hetero-homoscedastic regression",
        )

    def test_linear_model_heteroscedastic(self):
        """
        Tests the Monte Carlo method by fitting a linear model to
        generated test data with known uncertainties of x and y.
        """
        n = 100
        x = np.linspace(0.0, 100.0, n).reshape((n, 1))
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(5489)
        y = x + rng.normal(0.0, u, (n, 1))
        x = x + rng.normal(0.0, u, (n, 1))

        result = MonteCarlo(HeteroscedasticRegression(EIV())).fit(
            Linear(), x, y, ux=u, uy=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.popt[1], delta=1.50)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=1.00)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("zvar = ", result.zvar)
        print("cost = ", result.cost)

        RegressionPlot(result).plot(
            x,
            y,
            xlabel=r"$x$",
            ylabel=r"$y$",
            xrange=(-10.0, 110.0),
            yrange=(-10.0, 110.0),
            savefig="lin4.png",
            title="Heteroscedastic regression",
        )
        MatrixPlot().plot(
            matrix(result, n),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=5.0,
            cbar_min=0.0,
            cbar_label=r"variance-covariance $U_p(y)$",
            savefig="lin4-ycov.png",
            title="Heteroscedastic regression",
        )


if __name__ == "__main__":
    unittest.main()
