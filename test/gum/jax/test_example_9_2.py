#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp
import numpy as np
from jax import Array

import uncertaintyx.f.jax as tyx
from test.gum import to_cor

C = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
"""The sensitivity matrix of the additive measurement model."""


class AdditiveModel(tyx.ToF):
    """
    The additive measurement model.
    """

    def __init__(self):
        def f(x: Array) -> Array:
            """The measurement model"""
            return C @ x

        super().__init__(f)


class ExampleTest(unittest.TestCase):
    """
    Tests an additive measurement model (JCGM 102:2011, Example 9.2,
    https://doi.org/10.59161/JCGM102-2011).
    """

    def test_9_2_2(self):
        X = np.array([[0.0, 0.0, 0.0]])  # noqa: N806
        U = np.array([[1.0, 1.0, 1.0]])  # noqa: N806
        f = AdditiveModel()

        Y = f.eval(X)  # noqa: N806
        self.assertAlmostEqual(0.0, Y[0, 0])
        self.assertAlmostEqual(0.0, Y[0, 1])

        G = f.jac(X)  # noqa: N806
        self.assertAlmostEqual(C[0, 0], G[0, 0, 0])
        self.assertAlmostEqual(C[0, 1], G[0, 0, 1])
        self.assertAlmostEqual(C[0, 2], G[0, 0, 2])
        self.assertAlmostEqual(C[1, 0], G[0, 1, 0])
        self.assertAlmostEqual(C[1, 1], G[0, 1, 1])
        self.assertAlmostEqual(C[1, 2], G[0, 1, 2])

        U = f.lpu(X, U)  # noqa: N806
        self.assertAlmostEqual(2.0, U[0, 0, 0])
        self.assertAlmostEqual(2.0, U[0, 1, 1])

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(0.5, R[0, 0, 1])
        self.assertAlmostEqual(0.5, R[0, 1, 0])

    def test_9_2_3(self):
        r"""
        Test 9.2.3 just changes the PDF of the last component
        of :math:`X` from normal to rectangular but keeps the
        unit standard uncertainty. Since the LPU does not take
        account of the PDF, test 9.2.3 is numerically the same
        as test 9.2.2 above.
        """
        pass

    def test_9_2_4(self):
        X = np.array([[0.0, 0.0, 0.0]])  # noqa: N806
        U = np.array([[1.0, 1.0, 9.0]])  # noqa: N806
        f = AdditiveModel()

        Y = f.eval(X)  # noqa: N806
        self.assertAlmostEqual(0.0, Y[0, 0])
        self.assertAlmostEqual(0.0, Y[0, 1])

        U = f.lpu(X, U)  # noqa: N806
        self.assertAlmostEqual(10.0, U[0, 0, 0])
        self.assertAlmostEqual(10.0, U[0, 1, 1])

        R = to_cor(1, U)  # noqa: N806
        self.assertAlmostEqual(0.9, R[0, 0, 1])
        self.assertAlmostEqual(0.9, R[0, 1, 0])


if __name__ == "__main__":
    unittest.main()
