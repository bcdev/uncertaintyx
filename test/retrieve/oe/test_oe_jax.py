#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest
from typing import Literal

import numpy as np

from uncertaintyx.f.jax import Cigar
from uncertaintyx.f.jax import DifferentPowers
from uncertaintyx.f.jax import Ellipsoid
from uncertaintyx.f.jax import Rosenbrock
from uncertaintyx.f.jax import Sphere
from uncertaintyx.f.jax import Tablet
from uncertaintyx.retrieve.oe.jax import OE


class OptimalEstimationTest(unittest.TestCase):
    """
    Tests optimal estimation.
    """

    def setUp(self):
        self.rng = np.random.default_rng(5489)
        self.M = 50
        self.m = 10

    def test_sphere(self):
        """The sphere function has a unique minimum at zero."""
        f = Sphere()

        x = np.square(self.fuzzy(1.0, "x"))
        y = np.square(self.fuzzy(0.0, "y"))
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(f.eval(result.xopt), y))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))

    def test_ellipsoid(self):
        """The ellipsoid function has a unique minimum at zero."""
        f = Ellipsoid()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))

    def test_cigar(self):
        """The cigar function has a unique minimum at zero."""
        f = Cigar()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))

    def test_tablet(self):
        """The tablet function has a unique minimum at zero."""
        f = Tablet()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.all(result.info == 0))
        self.assertTrue(np.allclose(result.xopt, 0.0))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))

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
        self.assertTrue(np.allclose(result.xopt, 1.0))
        self.assertTrue(np.all(np.isfinite(result.xcov)))
        self.assertTrue(np.all(np.isfinite(result.xunc)))
        self.assertTrue(np.allclose(result.zvar, 0.0))
        self.assertTrue(np.allclose(result.cost, 0.0))

    def test_different_powers(self):
        """
        The different power function has a unique minimum at zero.
        Only the value of the cost function is tested, since the
        problem is badly scaled.
        """
        f = DifferentPowers()

        x = self.fuzzy(1.0, "x")
        y = self.sharp(0.0, "y")
        result = OE().retrieve(f, x, y)

        self.assertTrue(np.allclose(result.cost, 0.0))

    def fuzzy(self, val, role: Literal["x", "y"]) -> np.ndarray:
        """Returns an array filled with fuzzy values."""
        return self.rng.normal(
            val, 1.0, (self.M, self.m) if role == "x" else (self.M,)
        )

    def sharp(self, val, role: Literal["x", "y"]) -> np.ndarray:
        """Returns an array filled with sharp values."""
        return np.full((self.M, self.m) if role == "x" else (self.M,), val)


if __name__ == "__main__":
    unittest.main()
