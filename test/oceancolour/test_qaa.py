#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from uncertaintyx.fit.eiv.jax import EIV
from uncertaintyx.fit.randomsampling import Bootstrap
from uncertaintyx.fit.regression import HomoscedasticRegression
from uncertaintyx.interface.core import Result
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


def matrix(result: Result, a: Any, b: Any, n: int = 1000) -> np.ndarray:
    """
    Returns the variance-covariance matrix of the fitted curve.
    """
    return np.squeeze(result.ycov_p(np.linspace(a, b, n).reshape(1, n)))


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
            x = df["X"].values.reshape((-1, 1))
            y = df["Y"].values.reshape((-1, 1))
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


def to_var(u: np.ndarray) -> np.ndarray:
    """
    Converts standard uncertainty to a diagonal uncertainty tensor.
    """
    return np.square(u)


def to_std(U: np.ndarray) -> np.ndarray:  # noqa: N806
    """
    Converts a diagonal uncertainty tensor to standard uncertainty.
    """
    return np.sqrt(U)


def diagonalize(U: np.ndarray) -> np.ndarray:  # noqa: N806
    """
    Extracts the diagonal elements from a non-diagonal uncertainty tensor.
    """
    return np.stack(
        [
            np.diag(
                U[i].reshape((-1, np.prod(U.shape[1 + U.ndim // 2 :])))
            ).reshape(U.shape[1 + U.ndim // 2 :])
            for i in range(U.shape[0])
        ]
    )


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

        result = Bootstrap(HomoscedasticRegression(EIV())).fit(
            E(), x, y, up=np.array([0.5, 0.5, 0.5])
        )

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(2.0, result.popt[0], delta=1.0)
        self.assertAlmostEqual(1.2, result.popt[1], delta=1.0)
        self.assertAlmostEqual(0.9, result.popt[1], delta=1.0)
        self.assertAlmostEqual(0.1, result.punc[0], delta=0.1)
        self.assertAlmostEqual(0.1, result.punc[1], delta=0.1)
        self.assertAlmostEqual(0.1, result.punc[1], delta=0.1)
        self.assertAlmostEqual(0.3, result.yvar_r.item(), delta=0.1)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.yvar_r)

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
            matrix(result, 0.5, 4.5, 1000),
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            xrange=(0.5, 4.5),
            yrange=(0.5, 4.5),
            cmap="viridis",
            cbar_label=r"variance-covariance $U_p(\eta)$",
            cbar_max=0.01,
            cbar_min=0.00,
            savefig="qaa2-ycov.png",
        )

    def test_lee_2010_figure_3(self):
        """
        Tests the bootstrap method by fitting an empirical model function
        to published data (Lee et al., 2010, Figure 3).
        """
        x, y = read_plot_data("test.resources", "fig3.csv")

        result = Bootstrap(HomoscedasticRegression(EIV())).fit(S(), x, y)

        self.assertEqual(0, result.info)
        self.assertAlmostEqual(0.015, result.popt[0], delta=0.005)
        self.assertAlmostEqual(0.0, result.popt[1], delta=0.1)
        self.assertAlmostEqual(0.0, result.popt[2], delta=5.0)
        self.assertAlmostEqual(0.001, result.punc[0], delta=0.001)
        self.assertAlmostEqual(0.001, result.punc[0], delta=0.001)
        self.assertAlmostEqual(5.0, result.punc[0], delta=5.0)
        self.assertAlmostEqual(1.5e-05, result.yvar_r.item(), delta=0.1e-05)

        print()
        print("popt = ", result.popt)
        print("punc = ", result.punc)
        print("pcov = ", result.pcov)
        print("rvar = ", result.yvar_r)

        RegressionPlot(result).plot(
            x,
            y,
            title="Replica of Lee et al. (2010, Figure 3)",
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$S$ ($\mathrm{nm}^{-1}$)",
            xrange=(0.050, 9.950),
            yrange=(0.005, 0.035),
            savefig="qaa3.png",
        )
        MatrixPlot().plot(
            matrix(result, 0.05, 9.95),
            xlabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            ylabel=r"$r(443~\mathrm{nm})$ / $r(555~\mathrm{nm})$",
            xrange=(0.05, 9.95),
            yrange=(0.05, 9.95),
            cmap="viridis",
            cbar_label=r"variance-covariance $U_p(S)$ ($\mathrm{nm}^{-2}$)",
            cbar_max=4.0e-06,
            cbar_min=0.0e-06,
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

    def test_apply_qaa_to_owt_classes(self):
        """
        Test QAA on optical water type (OWT) classes.
        """
        w, R, u, M, m = read_owt_data(  # noqa : N806
            "test.resources", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

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
        u = np.stack(
            [
                np.broadcast_to(0.5, (M, m)),
                u,
                np.broadcast_to(0.1 * AW, (M, m)),
                np.broadcast_to(0.1 * BW, (M, m)),
            ],
            axis=1,
        )
        p = f.estimate()
        y = f.eval(p, x)
        self.assertEqual((M, 4, m), y.shape)

        a = y[:, 0, :]
        self.assertAlmostEqual(0.014, a[0, 0], delta=0.001)
        self.assertAlmostEqual(0.015, a[0, 1], delta=0.001)
        self.assertAlmostEqual(0.019, a[0, 2], delta=0.001)
        self.assertAlmostEqual(0.034, a[0, 3], delta=0.001)
        self.assertAlmostEqual(0.060, a[0, 4], delta=0.001)
        self.assertAlmostEqual(0.291, a[0, 5], delta=0.001)

        self.assertAlmostEqual(0.515, a[13, 0], delta=0.001)
        self.assertAlmostEqual(0.391, a[13, 1], delta=0.001)
        self.assertAlmostEqual(0.249, a[13, 2], delta=0.001)
        self.assertAlmostEqual(0.224, a[13, 3], delta=0.001)
        self.assertAlmostEqual(0.184, a[13, 4], delta=0.001)
        self.assertAlmostEqual(0.514, a[13, 5], delta=0.001)

        U = to_var(u)  # noqa : N806
        U = f.lpu_x(p, x, U, True)  # noqa : N806
        self.assertEqual((M, 4, m), U.shape)

        u = to_std(U)
        ua = u[:, 0, :]
        self.assertAlmostEqual(0.013, ua[0, 0], delta=0.001)
        self.assertAlmostEqual(0.016, ua[0, 1], delta=0.001)
        self.assertAlmostEqual(0.024, ua[0, 2], delta=0.001)
        self.assertAlmostEqual(0.046, ua[0, 3], delta=0.001)
        self.assertAlmostEqual(0.006, ua[0, 4], delta=0.001)
        self.assertAlmostEqual(1.569, ua[0, 5], delta=0.001)

        self.assertAlmostEqual(0.479, ua[13, 0], delta=0.001)
        self.assertAlmostEqual(0.359, ua[13, 1], delta=0.001)
        self.assertAlmostEqual(0.235, ua[13, 2], delta=0.001)
        self.assertAlmostEqual(0.210, ua[13, 3], delta=0.001)
        self.assertAlmostEqual(0.178, ua[13, 4], delta=0.001)
        self.assertAlmostEqual(0.083, ua[13, 5], delta=0.001)

        U = to_var(0.1 * p)  # noqa: N806
        U = f.lpu_p(p, U, x)  # noqa: N806
        self.assertEqual((M, 4, m, 4, m), U.shape)

        u = to_std(diagonalize(U))
        ua = u[:, 0, :]
        self.assertAlmostEqual(0.0007, ua[0, 0], delta=0.0001)
        self.assertAlmostEqual(0.0006, ua[0, 1], delta=0.0001)
        self.assertAlmostEqual(0.0004, ua[0, 2], delta=0.0001)
        self.assertAlmostEqual(0.0005, ua[0, 3], delta=0.0001)
        self.assertAlmostEqual(0.0003, ua[0, 4], delta=0.0001)
        self.assertAlmostEqual(0.0121, ua[0, 5], delta=0.0010)

        self.assertAlmostEqual(0.042, ua[13, 0], delta=0.001)
        self.assertAlmostEqual(0.027, ua[13, 1], delta=0.001)
        self.assertAlmostEqual(0.014, ua[13, 2], delta=0.001)
        self.assertAlmostEqual(0.011, ua[13, 3], delta=0.001)
        self.assertAlmostEqual(0.007, ua[13, 4], delta=0.001)
        self.assertAlmostEqual(0.000, ua[13, 5], delta=0.001)


if __name__ == "__main__":
    unittest.main()
