#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp
import numpy as np
from jax import Array

import uncertaintyx.f.jax as tyx
from test.gum import to_cor


class Impedance(tyx.ToF):
    """
    The impedance measurement model.
    """

    def __init__(self):
        def f(x: Array) -> Array:
            """The measurement model"""
            V, I, phi = x  # noqa: N806
            Z = V / I  # noqa: N806
            R = Z * jnp.cos(phi)  # noqa: N806
            X = Z * jnp.sin(phi)  # noqa: N806
            return jnp.stack(
                [
                    R,
                    X,
                    Z,
                ]
            )

        super().__init__(f)


class ExampleTest(unittest.TestCase):
    """
    Tests an impedance measurement (JCGM 102:2011, Example 9.4,
    https://doi.org/10.59161/JCGM102-2011).

    Recomputing Example 9.4 from the original measurements to machine
    precision reproduced all tabulated results to the last decimal place
    except one (off by 2), consistent with GUM guidance on numerical
    inconsistencies due to intermediate rounding [JCGM 100:2008,
    https://doi.org/10.59161/JCGM100-2008E; JCGM GUM-6:2020,
    https://doi.org/10.59161/JCGMGUM-6-2020].
    """

    def test_9_4_2_2(self):
        measured = np.array(
            [
                [5.007, 19.663e-03, 1.0456],
                [4.994, 19.639e-03, 1.0438],
                [5.005, 19.640e-03, 1.0468],
                [4.990, 19.685e-03, 1.0428],
                [4.999, 19.678e-03, 1.0433],
                [4.999, 19.661e-03, 1.0445],
            ]
        )
        X = np.broadcast_to(np.mean(measured, axis=0), (1, 3))  # noqa: N806
        U = np.broadcast_to(  # noqa: N806
            np.tensordot(measured - X, measured - X, ([0], [0])) / 30.0,
            (1, 3, 3),
        )
        f = Impedance()

        Y = f.eval(X)  # noqa: N806
        self.assertAlmostEqual(127.732, Y[0, 0], places=2)  # round-off
        self.assertAlmostEqual(219.847, Y[0, 1], places=3)
        self.assertAlmostEqual(254.260, Y[0, 2], places=3)

        G = f.jac(X)  # noqa: N806
        V, I, phi = X[0]  # noqa: N806
        self.assertAlmostEqual(np.cos(phi) / I, G[0, 0, 0])
        self.assertAlmostEqual(-V * np.cos(phi) / I**2, G[0, 0, 1])
        self.assertAlmostEqual(-V * np.sin(phi) / I, G[0, 0, 2])
        self.assertAlmostEqual(np.sin(phi) / I, G[0, 1, 0])
        self.assertAlmostEqual(-V * np.sin(phi) / I**2, G[0, 1, 1])
        self.assertAlmostEqual(V * np.cos(phi) / I, G[0, 1, 2])
        self.assertAlmostEqual(1.0 / I, G[0, 2, 0])
        self.assertAlmostEqual(-V / I**2, G[0, 2, 1])
        self.assertAlmostEqual(0.0, G[0, 2, 2])

        U = f.lpu(X, U)  # noqa: N806
        self.assertAlmostEqual(0.058, np.sqrt(U[0, 0, 0]), places=3)
        self.assertAlmostEqual(0.241, np.sqrt(U[0, 1, 1]), places=3)
        self.assertAlmostEqual(0.193, np.sqrt(U[0, 2, 2]), places=3)

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(-0.588, R[0, 0, 1], places=3)
        self.assertAlmostEqual(-0.485, R[0, 0, 2], places=3)
        self.assertAlmostEqual(-0.588, R[0, 1, 0], places=3)
        self.assertAlmostEqual(0.00749, 1.0 - R[0, 1, 2], places=5)
        self.assertAlmostEqual(-0.485, R[0, 2, 0], places=3)
        self.assertAlmostEqual(0.00749, 1.0 - R[0, 2, 1], places=5)

    def test_9_4_2_5(self):
        measured = np.array(
            [
                [5.007, 19.663e-03, 1.0456],
                [4.994, 19.639e-03, 1.0438],
                [5.005, 19.640e-03, 1.0468],
                [4.990, 19.685e-03, 1.0428],
                [4.999, 19.678e-03, 1.0433],
                [4.999, 19.661e-03, 1.0445],
            ]
        )
        X = np.broadcast_to(np.mean(measured, axis=0), (1, 3))  # noqa: N806
        V = np.broadcast_to(  # noqa: N806
            np.tensordot(measured - X, measured - X, ([0], [0])) / 6.0,
            (1, 3, 3),
        )
        f = Impedance()

        U = f.lpu(X, V)  # noqa: N806
        self.assertAlmostEqual(0.130, np.sqrt(U[0, 0, 0]), places=3)
        self.assertAlmostEqual(0.540, np.sqrt(U[0, 1, 1]), places=3)
        self.assertAlmostEqual(0.431, np.sqrt(U[0, 2, 2]), places=3)

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(-0.588, R[0, 0, 1], places=3)
        self.assertAlmostEqual(-0.485, R[0, 0, 2], places=3)
        self.assertAlmostEqual(-0.588, R[0, 1, 0], places=3)
        self.assertAlmostEqual(0.00749, 1.0 - R[0, 1, 2], places=5)
        self.assertAlmostEqual(-0.485, R[0, 2, 0], places=3)
        self.assertAlmostEqual(0.00749, 1.0 - R[0, 2, 1], places=5)


if __name__ == "__main__":
    unittest.main()
