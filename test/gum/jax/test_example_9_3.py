#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp
import numpy as np
from jax import Array

import uncertaintyx.f.jax as tyx
from test.gum import to_cor
from test.gum import to_cov


class Transformation(tyx.ToF):
    """
    The coordinate transformation.
    """

    def __init__(self):
        def f(x: Array) -> Array:
            """The additive measurement model"""
            return jnp.stack(
                [
                    jnp.sqrt(jnp.sum(jnp.square(x))),
                    jnp.arctan2(x[1], x[0]),
                ]
            )

        super().__init__(f)


class ExampleTest(unittest.TestCase):
    """
    Tests a coordinate transformation (JCGM 102:2011, Example 9.3,
    https://doi.org/10.59161/JCGM102-2011).
    """

    def test_9_3_2(self):
        X = np.array(  # noqa: N806
            [[0.001, 0.000], [0.010, 0.000], [0.100, 0.000]]
        )
        U = np.square(np.broadcast_to([0.010, 0.010], X.shape))  # noqa: N806
        f = Transformation()

        Y = f.eval(X)  # noqa: N806
        self.assertAlmostEqual(0.001, Y[0, 0])
        self.assertAlmostEqual(0.000, Y[0, 1])
        self.assertAlmostEqual(0.010, Y[1, 0])
        self.assertAlmostEqual(0.000, Y[1, 1])
        self.assertAlmostEqual(0.100, Y[2, 0])
        self.assertAlmostEqual(0.000, Y[2, 1])

        U = f.lpu(X, U)  # noqa: N806
        self.assertAlmostEqual(0.01, np.sqrt(U[0, 0, 0]))
        self.assertAlmostEqual(10.0, np.sqrt(U[0, 1, 1]))
        self.assertAlmostEqual(0.01, np.sqrt(U[1, 0, 0]))
        self.assertAlmostEqual(1.00, np.sqrt(U[1, 1, 1]))
        self.assertAlmostEqual(0.01, np.sqrt(U[2, 0, 0]))
        self.assertAlmostEqual(0.10, np.sqrt(U[2, 1, 1]))

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(0.0, R[0, 0, 1])
        self.assertAlmostEqual(0.0, R[0, 1, 0])
        self.assertAlmostEqual(0.0, R[1, 0, 1])
        self.assertAlmostEqual(0.0, R[1, 1, 0])
        self.assertAlmostEqual(0.0, R[2, 0, 1])
        self.assertAlmostEqual(0.0, R[2, 1, 0])

    def test_9_3_3(self):
        X = np.array(  # noqa: N806
            [[0.001, 0.000], [0.010, 0.000], [0.100, 0.000]]
        )
        u = np.broadcast_to([0.01, 0.01], X.shape)
        R = np.broadcast_to(  # noqa: N806
            [[1.0, 0.9], [0.9, 1.0]], X.shape + (2,)
        )
        f = Transformation()

        Y = f.eval(X)  # noqa: N806
        self.assertAlmostEqual(0.001, Y[0, 0])
        self.assertAlmostEqual(0.000, Y[0, 1])
        self.assertAlmostEqual(0.010, Y[1, 0])
        self.assertAlmostEqual(0.000, Y[1, 1])
        self.assertAlmostEqual(0.100, Y[2, 0])
        self.assertAlmostEqual(0.000, Y[2, 1])

        U = f.lpu(X, to_cov(R, u))  # noqa: N806
        self.assertAlmostEqual(0.01, np.sqrt(U[0, 0, 0]))
        self.assertAlmostEqual(10.0, np.sqrt(U[0, 1, 1]))
        self.assertAlmostEqual(0.01, np.sqrt(U[1, 0, 0]))
        self.assertAlmostEqual(1.00, np.sqrt(U[1, 1, 1]))
        self.assertAlmostEqual(0.01, np.sqrt(U[2, 0, 0]))
        self.assertAlmostEqual(0.10, np.sqrt(U[2, 1, 1]))

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(0.9, R[0, 0, 1])
        self.assertAlmostEqual(0.9, R[0, 1, 0])
        self.assertAlmostEqual(0.9, R[1, 0, 1])
        self.assertAlmostEqual(0.9, R[1, 1, 0])
        self.assertAlmostEqual(0.9, R[2, 0, 1])
        self.assertAlmostEqual(0.9, R[2, 1, 0])


if __name__ == "__main__":
    unittest.main()
