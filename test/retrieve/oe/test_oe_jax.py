#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from typing import Literal

import numpy as np

from uncertaintyx.f.jax import Cigar
from uncertaintyx.f.jax import DifferentPowers
from uncertaintyx.f.jax import Ellipsoid
from uncertaintyx.f.jax import Line
from uncertaintyx.f.jax import Rosenbrock
from uncertaintyx.f.jax import Sphere
from uncertaintyx.f.jax import Tablet
from uncertaintyx.retrieve.oe.jax import OE

ATOL = 1.0e-06
"""The absolute tolerance for comparisons."""


class OptimalEstimationTest(unittest.TestCase):
    """
    Tests optimal estimation.
    """

    def setUp(self):
        self.rng = np.random.default_rng(5489)
        self.M = 100
        self.m = 10

    def test_line(self):
        r"""
        The line function is exemplary for linear forward models.

        The model is tested without and with prior. In the latter
        case, the expected result is the mean of prior and data
        values, and the expected uncertainty is the square root
        of :math:`1/2`.
        """
        f = Line(2.0, 1.0)

        x = self.fuzzy(1.0, "x")
        y = self.fuzzy(0.0, "x")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(f.eval(result.xopt), y, atol=ATOL))
        self.assertTrue(  # variance
            np.allclose(
                [
                    result.xcov[:, i, j]
                    for i in range(self.m)
                    for j in range(self.m)
                    if i == j
                ],
                0.25,
                atol=ATOL,
            )
        )
        self.assertTrue(  # covariance
            np.allclose(
                [
                    result.xcov[:, i, j]
                    for i in range(self.m)
                    for j in range(self.m)
                    if i != j
                ],
                0.0,
                atol=ATOL,
            )
        )
        self.assertTrue(np.allclose(result.xunc, 0.5, atol=ATOL))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

        f = Line()  # identity

        x = self.fuzzy(0.0, "x")
        y = self.fuzzy(0.0, "x")
        ux = self.sharp(1.0, "x")
        uy = self.sharp(1.0, "x")
        result = OE().retrieve(f, x, y, ux=ux, uy=uy)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(
            np.allclose(result.xopt, np.mean([x, y], axis=0), atol=ATOL)
        )
        self.assertTrue(  # variance
            np.allclose(
                [
                    result.xcov[:, i, j]
                    for i in range(self.m)
                    for j in range(self.m)
                    if i == j
                ],
                0.5,
                atol=ATOL,
            )
        )
        self.assertTrue(  # covariance
            np.allclose(
                [
                    result.xcov[:, i, j]
                    for i in range(self.m)
                    for j in range(self.m)
                    if i != j
                ],
                0.0,
                atol=ATOL,
            )
        )
        self.assertTrue(np.allclose(result.xunc, np.sqrt(0.5), atol=ATOL))
        self.assertTrue(np.allclose(result.zvar, 0.5, rtol=1.0))
        self.assertTrue(np.allclose(result.cost, 2.5, rtol=4.0))

    def test_sphere(self):
        """The sphere function has a unique minimum at zero."""
        f = Sphere()

        x = np.square(self.fuzzy(1.0, "x"))
        y = np.square(self.fuzzy(0.0, "y"))
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(f.eval(result.xopt), y, atol=ATOL))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def test_ellipsoid(self):
        """The ellipsoid function has a unique minimum at zero."""
        f = Ellipsoid()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0, atol=ATOL))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def test_cigar(self):
        """The cigar function has a unique minimum at zero."""
        f = Cigar()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0, atol=ATOL))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def test_tablet(self):
        """The tablet function has a unique minimum at zero."""
        f = Tablet()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0, atol=ATOL))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def test_rosenbrock(self):
        """
        The Rosenbrock function has a global and a local minimum. The
        minimization uses initial values biased toward the global minimum
        to establish an unambiguous test condition.
        """
        f = Rosenbrock()

        x = self.fuzzy(3.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 1.0, atol=ATOL))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0, atol=ATOL))
        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def test_different_powers(self):
        """
        The different power function has a unique minimum at zero.
        Only the value of the cost function is tested here, since
        the problem is badly scaled.
        """
        f = DifferentPowers()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.allclose(result.cost, 0.0, atol=ATOL))

    def fuzzy(self, val, shape_like: Literal["x", "y"]) -> np.ndarray:
        """Returns an array filled with fuzzy values."""
        return self.rng.normal(
            val, 1.0, (self.M, self.m) if shape_like == "x" else (self.M,)
        )

    def sharp(self, val, shape_like: Literal["x", "y"]) -> np.ndarray:
        """Returns an array filled with sharp values."""
        return np.broadcast_to(
            val, (self.M, self.m) if shape_like == "x" else (self.M,)
        )


if __name__ == "__main__":
    unittest.main()
