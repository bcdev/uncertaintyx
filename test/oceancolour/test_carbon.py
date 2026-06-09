#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from importlib import resources

import numpy as np
import pandas as pd

from uncertaintyx.oceancolour.carbon import MaranonOCI


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


class MaranonTest(unittest.TestCase):
    """
    Tests the Maranon et al. (2014) phytoplankton biomass model
    on optical water type classes.
    """

    def test_phytoplankton_carbon(self):
        """Tests the phytoplankton carbon model function."""
        w, R, _, M, m = read_owt_data(  # noqa : N806
            "test.resources.oceancolour", "owt.csv"
        )
        W = np.broadcast_to(w, (M, m))  # noqa : N806

        f = MaranonOCI(True)
        x = np.stack([W[:, 1:], R[:, 1:]], axis=1)
        u = np.stack(
            [
                np.broadcast_to(0.0, (M, 5)),
                np.asarray([[0.05, 0.05, 0.05, 0.05, 0.10]] * R[:, 1:]),
            ],
            axis=1,
        )
        p = f.prior(preset="OC4_MERIS")
        y = f.eval(p, x)

        self.assertEqual((M,), y.shape)
        self.assertAlmostEqual(0.308, y[0], places=3)
        self.assertAlmostEqual(0.473, y[1], places=3)
        self.assertAlmostEqual(0.620, y[2], places=3)
        self.assertAlmostEqual(0.781, y[3], places=3)
        self.assertAlmostEqual(0.939, y[4], places=3)
        self.assertAlmostEqual(1.062, y[5], places=3)
        self.assertAlmostEqual(1.134, y[6], places=3)
        self.assertAlmostEqual(1.277, y[7], places=3)
        self.assertAlmostEqual(1.533, y[8], places=3)
        self.assertAlmostEqual(1.651, y[9], places=3)
        self.assertAlmostEqual(1.881, y[10], places=3)
        self.assertAlmostEqual(2.426, y[11], places=3)
        self.assertAlmostEqual(2.336, y[12], places=3)
        self.assertAlmostEqual(2.351, y[13], places=3)

        U = np.square(u)  # noqa : N806
        U = f.lpu_x(p, x, U)  # noqa : N806
        self.assertEqual((M,), U.shape)

        u = np.sqrt(U)
        self.assertAlmostEqual(0.072, u[0], places=3)
        self.assertAlmostEqual(0.063, u[1], places=3)
        self.assertAlmostEqual(0.057, u[2], places=3)
        self.assertAlmostEqual(0.050, u[3], places=3)
        self.assertAlmostEqual(0.043, u[4], places=3)
        self.assertAlmostEqual(0.038, u[5], places=3)
        self.assertAlmostEqual(0.052, u[6], places=3)
        self.assertAlmostEqual(0.051, u[7], places=3)
        self.assertAlmostEqual(0.050, u[8], places=3)
        self.assertAlmostEqual(0.057, u[9], places=3)
        self.assertAlmostEqual(0.071, u[10], places=3)
        self.assertAlmostEqual(0.101, u[11], places=3)
        self.assertAlmostEqual(0.097, u[12], places=3)
        self.assertAlmostEqual(0.098, u[13], places=3)


if __name__ == "__main__":
    unittest.main()
