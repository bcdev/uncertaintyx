#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.oceancolour.ocx import CI
from uncertaintyx.oceancolour.ocx import OCX
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


class OCxTest(unittest.TestCase):
    """Tests OCX model functions on optical water type classes."""

    def test_ci(self):
        """Tests the chlorophyll index (CI) model function."""
        w, R, u, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = CI()
        x = np.stack([W[:, [1, 4, 5]], R[:, [1, 4, 5]]], axis=1)
        u = np.stack([np.broadcast_to(0.5, (M, 3)), u[:, [1, 4, 5]]], axis=1)
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.022, y[0], delta=0.001)
        self.assertAlmostEqual(0.033, y[1], delta=0.001)
        self.assertAlmostEqual(0.048, y[2], delta=0.001)
        self.assertAlmostEqual(0.074, y[3], delta=0.001)
        self.assertAlmostEqual(0.111, y[4], delta=0.001)
        self.assertAlmostEqual(0.152, y[5], delta=0.001)
        self.assertAlmostEqual(0.205, y[6], delta=0.001)
        self.assertAlmostEqual(0.259, y[7], delta=0.001)
        self.assertAlmostEqual(0.350, y[8], delta=0.001)
        self.assertAlmostEqual(0.426, y[9], delta=0.001)
        self.assertAlmostEqual(0.538, y[10], delta=0.001)
        self.assertAlmostEqual(3.897, y[11], delta=0.001)
        # 12 and 13 are out of domain
        self.assertTrue(np.isfinite(y[12]))
        self.assertTrue(np.isfinite(y[13]))

        U = np.square(u)  # noqa : N806
        V = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), V.shape)

        v = np.sqrt(V)
        self.assertAlmostEqual(0.026, v[0], delta=0.001)
        self.assertAlmostEqual(0.024, v[1], delta=0.001)
        self.assertAlmostEqual(0.042, v[2], delta=0.001)
        self.assertAlmostEqual(0.068, v[3], delta=0.001)
        self.assertAlmostEqual(0.070, v[4], delta=0.001)
        self.assertAlmostEqual(0.074, v[5], delta=0.001)
        self.assertAlmostEqual(0.107, v[6], delta=0.001)
        self.assertAlmostEqual(0.100, v[7], delta=0.001)
        self.assertAlmostEqual(0.174, v[8], delta=0.001)
        self.assertAlmostEqual(0.204, v[9], delta=0.001)
        self.assertAlmostEqual(0.200, v[10], delta=0.001)
        self.assertAlmostEqual(5.461, v[11], delta=0.001)
        # 12 and 13 are out of domain
        self.assertTrue(np.isfinite(v[12]))
        self.assertTrue(np.isfinite(y[13]))

    def test_oc4(self):
        """Tests the OC4 model function."""
        _, R, u, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )

        f = OCX()
        x = R[:, 1:-1]
        u = u[:, 1:-1]
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.024, y[0], delta=0.001)
        self.assertAlmostEqual(0.034, y[1], delta=0.001)
        self.assertAlmostEqual(0.052, y[2], delta=0.001)
        self.assertAlmostEqual(0.082, y[3], delta=0.001)
        self.assertAlmostEqual(0.118, y[4], delta=0.001)
        self.assertAlmostEqual(0.157, y[5], delta=0.001)
        self.assertAlmostEqual(0.212, y[6], delta=0.001)
        self.assertAlmostEqual(0.270, y[7], delta=0.001)
        self.assertAlmostEqual(0.426, y[8], delta=0.001)
        self.assertAlmostEqual(0.572, y[9], delta=0.001)
        self.assertAlmostEqual(1.022, y[10], delta=0.001)
        self.assertAlmostEqual(4.174, y[11], delta=0.001)
        self.assertAlmostEqual(3.295, y[12], delta=0.001)
        self.assertAlmostEqual(3.427, y[13], delta=0.001)

        U = np.square(u)  # noqa : N806
        V = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), V.shape)

        v = np.sqrt(V)
        self.assertAlmostEqual(0.116, v[0], delta=0.001)
        self.assertAlmostEqual(0.093, v[1], delta=0.001)
        self.assertAlmostEqual(0.136, v[2], delta=0.001)
        self.assertAlmostEqual(0.171, v[3], delta=0.001)
        self.assertAlmostEqual(0.135, v[4], delta=0.001)
        self.assertAlmostEqual(0.120, v[5], delta=0.001)
        self.assertAlmostEqual(0.159, v[6], delta=0.001)
        self.assertAlmostEqual(0.159, v[7], delta=0.001)
        self.assertAlmostEqual(0.360, v[8], delta=0.001)
        self.assertAlmostEqual(0.559, v[9], delta=0.001)
        self.assertAlmostEqual(1.158, v[10], delta=0.001)
        self.assertAlmostEqual(6.411, v[11], delta=0.001)
        self.assertAlmostEqual(4.170, v[12], delta=0.001)
        self.assertAlmostEqual(4.362, v[13], delta=0.001)

    def test_oci(self):
        """Tests the OCI model function."""
        w, R, u, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = OCI()
        x = np.stack([W[:, 1:], R[:, 1:]], axis=1)
        u = np.stack([np.broadcast_to(0.5, (M, 5)), u[:, 1:]], axis=1)
        p = f.prior()
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.022, y[0], delta=0.001)
        self.assertAlmostEqual(0.033, y[1], delta=0.001)
        self.assertAlmostEqual(0.048, y[2], delta=0.001)
        self.assertAlmostEqual(0.074, y[3], delta=0.001)
        self.assertAlmostEqual(0.111, y[4], delta=0.001)
        self.assertAlmostEqual(0.152, y[5], delta=0.001)
        self.assertAlmostEqual(0.205, y[6], delta=0.001)
        self.assertAlmostEqual(0.260, y[7], delta=0.001)  # blend
        self.assertAlmostEqual(0.426, y[8], delta=0.001)
        self.assertAlmostEqual(0.572, y[9], delta=0.001)
        self.assertAlmostEqual(1.022, y[10], delta=0.001)
        self.assertAlmostEqual(4.174, y[11], delta=0.001)
        self.assertAlmostEqual(3.295, y[12], delta=0.001)
        self.assertAlmostEqual(3.427, y[13], delta=0.001)

        U = np.square(u)  # noqa : N806
        U = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), U.shape)

        v = np.sqrt(U)
        self.assertAlmostEqual(0.026, v[0], delta=0.001)
        self.assertAlmostEqual(0.024, v[1], delta=0.001)
        self.assertAlmostEqual(0.042, v[2], delta=0.001)
        self.assertAlmostEqual(0.068, v[3], delta=0.001)
        self.assertAlmostEqual(0.070, v[4], delta=0.001)
        self.assertAlmostEqual(0.074, v[5], delta=0.001)
        self.assertAlmostEqual(0.107, v[6], delta=0.001)
        self.assertAlmostEqual(0.117, v[7], delta=0.001)  # blend
        self.assertAlmostEqual(0.360, v[8], delta=0.001)
        self.assertAlmostEqual(0.559, v[9], delta=0.001)
        self.assertAlmostEqual(1.158, v[10], delta=0.001)
        self.assertAlmostEqual(6.411, v[11], delta=0.001)
        self.assertAlmostEqual(4.170, v[12], delta=0.001)
        self.assertAlmostEqual(4.362, v[13], delta=0.001)


if __name__ == "__main__":
    unittest.main()
