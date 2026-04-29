#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import numpy as np

from uncertaintyx.f.jax import ToF
from uncertaintyx.retrieve.oe.jax import OE


class Parabola(ToF):
    """The parabolic test function."""

    def __init__(self):
        def f(x):
            """The test function."""
            return x * x

        super().__init__(f)


class OptimalEstimationTest(unittest.TestCase):
    """
    Tests EIV regression.
    """

    def test_parabola(self):
        """
        Tests the EIV retrieval with a simple parabolic test function.
        """
        n = 100

        x = np.ones((n, 1))
        y = np.zeros((n, 1))
        result = OE().retrieve(Parabola(), x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0))
        self.assertTrue(np.allclose(result.xcov, 0.0))
        self.assertTrue(np.allclose(result.xunc, 0.0))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))


if __name__ == "__main__":
    unittest.main()
