#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.fit.eiv.jax import EVM
from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.oceancolour.qaa import E
from uncertaintyx.oceancolour.qaa import QAA
from uncertaintyx.oceancolour.qaa import S
from uncertaintyx.plot.plots import MatrixPlot
from uncertaintyx.plot.plots import RegressionPlot

AW = np.array([[0.00473, 0.00635, 0.01500, 0.03250, 0.05960, 0.43900]])
"""
Test data for absorption by pure seawater (m-1).

Reference: https://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm
"""

BW = np.array([[0.00340, 0.00250, 0.00158, 0.00133, 0.00090, 0.00034]])
"""
Test data for back-scattering by pure seawater (m-1).

Reference: https://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm
"""

W = np.array([[412.0, 443.0, 489.0, 510.0, 555.0, 670.0]])
"""
Test data for precise wavelengths of spectral wavebands (nm).

Reference: https://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm
"""


def read_plot_data(
    package: str, filename: str
) -> tuple[np.ndarray, np.ndarray]:
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


def read_test_data(
    package: str, filename: str
) -> tuple[np.ndarray, int, int]:
    """
    Returns resource data table.

    :param package: The package name.
    :param filename: The filename.
    :returns: The data table.
    """
    with resources.path(package, filename) as resource:
        column = []
        with open(resource) as r:
            df = pd.read_csv(r, sep=";")
            for name, _ in df.items():
                column.append(df[name].values)
            data = np.stack(column, axis=-1)
    return data, data.shape[0], data.shape[1]


class QaaTest(unittest.TestCase):
    """
    Tests QAA model fitting.
    """

    def test_lee_2010_figure_2(self):
        """
        Tests the bootstrap method by fitting an empirical model functions
        to published data (Lee et al., 2010, Figure 2).
        """
        x, y = read_plot_data("test.resources", "fig2.csv")

        result = Bootstrap(HomoscedasticRegression(EVM())).fit(E(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(2.0, result.popt[0], delta=0.2)
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
        x, y = read_plot_data("test.resources", "fig3.csv")

        result = Bootstrap(HomoscedasticRegression(EVM())).fit(S(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(0.017, result.popt[0], delta=0.001)
        self.assertAlmostEqual(0.000, result.popt[1], delta=0.02)
        self.assertAlmostEqual(0.000, result.popt[2], delta=5.0)
        self.assertAlmostEqual(0.001, result.punc[0], delta=0.001)
        self.assertAlmostEqual(1.5e-05, result.rvar, delta=0.1e-05)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.rvar)

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

    def test_qaa_single_batch_case_1(self):
        """
        Reference: https://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm
        """
        R = np.array(  # noqa: N806
            [[0.00450, 0.00410, 0.00402, 0.00295, 0.00169, 0.00018]]
        )

        f = QAA()
        x = np.stack([W, R, AW, BW], axis=1)
        p = f.estimate(preset="case1")
        y = f.eval(p, x)

        a = y[0, 0]
        self.assertAlmostEqual(0.06198, a[0], delta=0.0001)
        self.assertAlmostEqual(0.05400, a[1], delta=0.0001)
        self.assertAlmostEqual(0.04008, a[2], delta=0.0001)
        self.assertAlmostEqual(0.04816, a[3], delta=0.0001)
        self.assertAlmostEqual(0.06491, a[4], delta=0.0001)

        bbp = y[0, 3]
        self.assertAlmostEqual(0.00237, bbp[0], delta=0.0001)
        self.assertAlmostEqual(0.00209, bbp[1], delta=0.0001)
        self.assertAlmostEqual(0.00177, bbp[2], delta=0.0001)
        self.assertAlmostEqual(0.00164, bbp[3], delta=0.0001)
        self.assertAlmostEqual(0.00142, bbp[4], delta=0.0001)

    def test_qaa_single_batch_case_2(self):
        """
        Reference: https://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm
        """
        R = np.array(  # noqa: N806
            [[0.00450, 0.00410, 0.00402, 0.00295, 0.00169, 0.00018]]
        )

        f = QAA()
        x = np.stack([W, R, AW, BW], axis=1)
        p = f.estimate(preset="case2")
        y = f.eval(p, x)

        a = y[0, 0]
        self.assertAlmostEqual(0.07072, a[0], delta=0.0002)
        self.assertAlmostEqual(0.06244, a[1], delta=0.0002)
        self.assertAlmostEqual(0.04733, a[2], delta=0.0002)
        self.assertAlmostEqual(0.05729, a[3], delta=0.0002)
        self.assertAlmostEqual(0.07853, a[4], delta=0.0002)
        self.assertAlmostEqual(0.44407, a[5], delta=0.0002)

        bbp = y[0, 3]
        self.assertAlmostEqual(0.00319, bbp[0], delta=0.0001)
        self.assertAlmostEqual(0.00281, bbp[1], delta=0.0001)
        self.assertAlmostEqual(0.00237, bbp[2], delta=0.0001)
        self.assertAlmostEqual(0.00220, bbp[3], delta=0.0001)
        self.assertAlmostEqual(0.00190, bbp[4], delta=0.0001)
        self.assertAlmostEqual(0.00137, bbp[5], delta=0.0001)

    def test_qaa_multiple_batches(self):
        """
        Test with tolerance increased because there is a logical flaw in
        the reference Excel sheet (TBC).
        """
        R, M, m = read_test_data("test.resources", "rrs.csv")  # noqa: N806
        f = QAA()
        x = np.stack(
            [
                np.broadcast_to(W, (M, m)),
                R,
                np.broadcast_to(AW, (M, m)),
                np.broadcast_to(BW, (M, m)),
            ],
            axis=1,
        )
        p = f.estimate()
        y = f.eval(p, x)

        A, _, n = read_test_data("test.resources", "a.csv")  # noqa: N806
        a = y[:, 0, :]
        for i in range(M):
            for j in range(n):
                self.assertAlmostEqual(
                    A[i, j],
                    a[i, j],
                    delta=0.015,
                    msg=f"{i}, {j}: assertion failed",
                )


if __name__ == "__main__":
    unittest.main()
