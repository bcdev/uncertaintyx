#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.randomsampling import MonteCarlo
from uncertaintyx.fit.regression import HeteroHomoscedasticRegression
from uncertaintyx.fit.regression import HeteroscedasticRegression
from uncertaintyx.fit.regression import HomoHeteroscedasticRegression
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.m.numpy import Linear
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import RegressionPlot


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
        x = np.linspace(0.0, 100.0, n)
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(42)
        y = x + rng.normal(0.0, u, n)
        x = x + rng.normal(0.0, u, n)

        result = Bootstrap(HomoscedasticRegression()).fit(Linear(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.popt[1], delta=5.0)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.punc[1], delta=5.0)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)
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
            result.ycov_p(np.linspace(0.5, 99.5, n)),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=4.0,
            cbar_min=-1.0,
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
        x = np.linspace(0.0, 100.0, n)
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(42)
        y = x + rng.normal(0.0, u, n)
        x = x + rng.normal(0.0, u, n)

        result = Bootstrap(HomoHeteroscedasticRegression()).fit(
            Linear(), x, y, u=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.popt[1], delta=5.0)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.punc[1], delta=5.0)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)
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
            result.ycov_p(np.linspace(0.5, 99.5, n)),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=4.0,
            cbar_min=-1.0,
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
        x = np.linspace(0.0, 100.0, n)
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(42)
        y = x + rng.normal(0.0, u, n)
        x = x + rng.normal(0.0, u, n)

        result = Bootstrap(HeteroHomoscedasticRegression()).fit(
            Linear(), x, y, u=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.popt[1], delta=5.0)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.1)
        self.assertAlmostEqual(0.0, result.punc[1], delta=5.0)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)
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
            result.ycov_p(np.linspace(0.5, 99.5, n)),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=4.0,
            cbar_min=-1.0,
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
        x = np.linspace(0.0, 100.0, n)
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(42)
        y = x + rng.normal(0.0, u, n)
        x = x + rng.normal(0.0, u, n)

        result = MonteCarlo(HeteroscedasticRegression()).fit(
            Linear(), x, y, ux=u, uy=u
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.popt[1], delta=2.00)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.20)
        self.assertAlmostEqual(0.0, result.punc[1], delta=2.00)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)
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
            result.ycov_p(np.linspace(0.5, 99.5, n)),
            xlabel=r"$x$",
            ylabel=r"$x$",
            xrange=(0.5, 99.5),
            yrange=(0.5, 99.5),
            cbar_max=4.0,
            cbar_min=-1.0,
            cbar_label=r"variance-covariance $U_p(y)$",
            savefig="lin4-ycov.png",
            title="Heteroscedastic regression",
        )


if __name__ == "__main__":
    unittest.main()
