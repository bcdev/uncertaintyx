#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.oceancolour.ocx import OC4


def read_test_data(
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
    """Tests OC4 and OCI model functions on optical water type classes."""

    def test_oc4(self):
        _, R, u, M, m = read_test_data(  # noqa : N806
            "test.resources", "owt.csv"
        )

        f = OC4()
        x = R[:, 1:-1]
        p = f.estimate()
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

        U = np.square(u[:, 1:-1])  # noqa : N806
        V = f.propagate_x_diag(p, x, U)
        self.assertEqual((M,), V.shape)


if __name__ == "__main__":
    unittest.main()
