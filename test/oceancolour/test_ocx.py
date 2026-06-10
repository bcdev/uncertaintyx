#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.oceancolour.ocx import CI
from uncertaintyx.oceancolour.ocx import OCx
from uncertaintyx.oceancolour.ocx import OCI


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


class OceanColourTest(unittest.TestCase):
    """Tests Ocean Colour model functions on optical water type classes."""

    def test_ci(self):
        """Tests the chlorophyll index (CI) model function."""
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = CI()
        x = np.stack([W[:, [1, 4, 5]], R[:, [1, 4, 5]]], axis=1)
        u = np.stack(
            [
                np.broadcast_to(0.0, (M, 3)),
                np.asarray([[0.05, 0.10, 0.20]] * R[:, [1, 4, 5]]),
            ],
            axis=1,
        )
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.022, y[0], places=3)
        self.assertAlmostEqual(0.033, y[1], places=3)
        self.assertAlmostEqual(0.048, y[2], places=3)
        self.assertAlmostEqual(0.074, y[3], places=3)
        self.assertAlmostEqual(0.111, y[4], places=3)
        self.assertAlmostEqual(0.152, y[5], places=3)
        self.assertAlmostEqual(0.205, y[6], places=3)
        self.assertAlmostEqual(0.259, y[7], places=3)
        self.assertAlmostEqual(0.350, y[8], places=3)
        self.assertAlmostEqual(0.426, y[9], places=3)
        self.assertAlmostEqual(0.538, y[10], places=3)
        self.assertAlmostEqual(3.897, y[11], places=3)
        # 12 and 13 are out of domain
        self.assertAlmostEqual(11.95, y[12], places=2)
        self.assertAlmostEqual(360.8, y[13], places=1)

        U = np.square(u)  # noqa : N806
        U = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), U.shape)

        u = np.sqrt(U)
        self.assertAlmostEqual(0.004, u[0], places=3)
        self.assertAlmostEqual(0.006, u[1], places=3)
        self.assertAlmostEqual(0.008, u[2], places=3)
        self.assertAlmostEqual(0.011, u[3], places=3)
        self.assertAlmostEqual(0.015, u[4], places=3)
        self.assertAlmostEqual(0.019, u[5], places=3)
        self.assertAlmostEqual(0.025, u[6], places=3)
        self.assertAlmostEqual(0.030, u[7], places=3)
        self.assertAlmostEqual(0.040, u[8], places=3)
        self.assertAlmostEqual(0.045, u[9], places=3)
        self.assertAlmostEqual(0.050, u[10], places=3)
        self.assertAlmostEqual(1.617, u[11], places=3)
        # 12 and 13 are out of domain
        self.assertAlmostEqual(8.484, u[12], places=3)
        self.assertAlmostEqual(462.8, u[13], places=1)

    def test_oc4(self):
        """Tests the OC4 model function."""
        _, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )

        f = OCx()
        x = R[:, 1:-1]
        u = np.asarray([[0.05, 0.05, 0.05, 0.10]] * R[:, 1:-1])
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.024, y[0], places=3)
        self.assertAlmostEqual(0.034, y[1], places=3)
        self.assertAlmostEqual(0.052, y[2], places=3)
        self.assertAlmostEqual(0.082, y[3], places=3)
        self.assertAlmostEqual(0.118, y[4], places=3)
        self.assertAlmostEqual(0.157, y[5], places=3)
        self.assertAlmostEqual(0.212, y[6], places=3)
        self.assertAlmostEqual(0.270, y[7], places=3)
        self.assertAlmostEqual(0.426, y[8], places=3)
        self.assertAlmostEqual(0.572, y[9], places=3)
        self.assertAlmostEqual(1.022, y[10], places=3)
        self.assertAlmostEqual(4.174, y[11], places=3)
        self.assertAlmostEqual(3.295, y[12], places=3)
        self.assertAlmostEqual(3.427, y[13], places=3)

        U = np.square(u)  # noqa : N806
        U = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), U.shape)

        u = np.sqrt(U)
        self.assertAlmostEqual(0.009, u[0], places=3)
        self.assertAlmostEqual(0.012, u[1], places=3)
        self.assertAlmostEqual(0.015, u[2], places=3)
        self.assertAlmostEqual(0.019, u[3], places=3)
        self.assertAlmostEqual(0.022, u[4], places=3)
        self.assertAlmostEqual(0.026, u[5], places=3)
        self.assertAlmostEqual(0.033, u[6], places=3)
        self.assertAlmostEqual(0.044, u[7], places=3)
        self.assertAlmostEqual(0.084, u[8], places=3)
        self.assertAlmostEqual(0.129, u[9], places=3)
        self.assertAlmostEqual(0.291, u[10], places=3)
        self.assertAlmostEqual(1.762, u[11], places=3)
        self.assertAlmostEqual(1.319, u[12], places=3)
        self.assertAlmostEqual(1.384, u[13], places=3)

    def test_oci(self):
        """Tests the OCI model function."""
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = OCI()
        x = np.stack([W[:, 1:], R[:, 1:]], axis=1)
        u = np.stack(
            [
                np.broadcast_to(0.0, (M, 5)),
                np.asarray([[0.05, 0.05, 0.05, 0.10, 0.20]] * R[:, 1:]),
            ],
            axis=1,
        )
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.022, y[0], places=3)
        self.assertAlmostEqual(0.033, y[1], places=3)
        self.assertAlmostEqual(0.048, y[2], places=3)
        self.assertAlmostEqual(0.074, y[3], places=3)
        self.assertAlmostEqual(0.111, y[4], places=3)
        self.assertAlmostEqual(0.152, y[5], places=3)
        self.assertAlmostEqual(0.205, y[6], places=3)
        self.assertAlmostEqual(0.260, y[7], places=3)  # blend
        self.assertAlmostEqual(0.426, y[8], places=3)
        self.assertAlmostEqual(0.572, y[9], places=3)
        self.assertAlmostEqual(1.022, y[10], places=3)
        self.assertAlmostEqual(4.174, y[11], places=3)
        self.assertAlmostEqual(3.295, y[12], places=3)
        self.assertAlmostEqual(3.427, y[13], places=3)

        U = np.square(u)  # noqa : N806
        U = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), U.shape)

        u = np.sqrt(U)
        self.assertAlmostEqual(0.004, u[0], places=3)
        self.assertAlmostEqual(0.006, u[1], places=3)
        self.assertAlmostEqual(0.008, u[2], places=3)
        self.assertAlmostEqual(0.011, u[3], places=3)
        self.assertAlmostEqual(0.015, u[4], places=3)
        self.assertAlmostEqual(0.019, u[5], places=3)
        self.assertAlmostEqual(0.025, u[6], places=3)
        self.assertAlmostEqual(0.034, u[7], places=3)  # blend
        self.assertAlmostEqual(0.084, u[8], places=3)
        self.assertAlmostEqual(0.129, u[9], places=3)
        self.assertAlmostEqual(0.291, u[10], places=3)
        self.assertAlmostEqual(1.762, u[11], places=3)
        self.assertAlmostEqual(1.319, u[12], places=3)
        self.assertAlmostEqual(1.384, u[13], places=3)


if __name__ == "__main__":
    unittest.main()
