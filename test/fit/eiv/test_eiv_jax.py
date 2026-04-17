#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.fit.eiv.jax import EIV
from uncertaintyx.m.jax import Linear
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import RegressionPlot


class ErrorsInVariablesTest(unittest.TestCase):
    """
    Tests EIV regression.
    """

    def test_linear_model(self):
        """
        Tests EIV regression by fitting a linear model to generated
        test data with known uncertainties of x and y.
        """
        n = 100
        x = np.linspace(0.0, 100.0, n).reshape((n, 1))
        u = 1.0 + np.sqrt(x)

        rng = np.random.default_rng(42)
        y = x + rng.normal(0.0, u, (n, 1))
        x = x + rng.normal(0.0, u, (n, 1))

        result = EIV().fit(Linear(), x, y, ux=u, uy=u)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.popt[1], delta=1.00)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=1.00)
        self.assertAlmostEqual(0.0, result.pcov[0, 0], delta=0.001)
        self.assertAlmostEqual(0.0, result.pcov[0, 1], delta=0.015)
        self.assertAlmostEqual(0.0, result.pcov[1, 0], delta=0.015)
        self.assertAlmostEqual(0.0, result.pcov[1, 1], delta=1.000)

        result = EIV().fit(Linear(), x, y, ux=u, uy=u, covar=True)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(1.0, result.popt[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.popt[1], delta=1.00)
        self.assertAlmostEqual(0.0, result.punc[0], delta=0.05)
        self.assertAlmostEqual(0.0, result.punc[1], delta=1.00)
        self.assertAlmostEqual(0.0, result.pcov[0, 0], delta=0.001)
        self.assertAlmostEqual(0.0, result.pcov[0, 1], delta=0.015)
        self.assertAlmostEqual(0.0, result.pcov[1, 0], delta=0.015)
        self.assertAlmostEqual(0.0, result.pcov[1, 1], delta=1.000)

        RegressionPlot(result).plot(
            x,
            y,
            xlabel=r"$x$",
            ylabel=r"$y$",
            xrange=(-10.0, 110.0),
            yrange=(-10.0, 110.0),
            savefig="eiv.png",
            title="Errors-in-variables regression",
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
            savefig="eiv-ycov.png",
            title="Errors-in-variables regression",
        )


if __name__ == "__main__":
    unittest.main()
