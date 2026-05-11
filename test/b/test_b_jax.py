#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp
import numpy as np

from uncertaintyx.b.jax import b_basis


class BasisTest(unittest.TestCase):
    """Tests Bernstein basis polynomials."""

    def test_b_basis_0(self):
        m = 5
        n = 0
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(n, x)

        self.assertEqual((n + 1, m), y.shape)
        self.assertTrue(np.allclose(y, 1.0))

    def test_b_basis_1(self):
        m = 5
        n = 1
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(n, x)

        self.assertEqual((n + 1, m), y.shape)
        self.assertTrue(np.allclose(y[0, 0], 1.0))
        self.assertTrue(np.allclose(y[0, 2], 0.5))
        self.assertTrue(np.allclose(y[1, 0], 0.0))
        self.assertTrue(np.allclose(y[1, 2], 0.5))
        self.assertTrue(np.allclose(y[0, -1], 0.0))
        self.assertTrue(np.allclose(y[1, -1], 1.0))
        self.assertTrue(np.allclose(np.sum(y, axis=0), 1.0))

    def test_b_basis_2(self):
        m = 5
        n = 2
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(n, x)

        self.assertEqual((n + 1, m), y.shape)
        self.assertTrue(np.allclose(y[:1, 0], 1.0))
        self.assertTrue(np.allclose(y[1:, 0], 0.0))
        self.assertTrue(np.allclose(y[0, 2], 0.25))
        self.assertTrue(np.allclose(y[1, 2], 0.50))
        self.assertTrue(np.allclose(y[2, 2], 0.25))
        self.assertTrue(np.allclose(y[:-1, -1], 0.0))
        self.assertTrue(np.allclose(y[-1:, -1], 1.0))
        self.assertTrue(np.allclose(np.sum(y, axis=0), 1.0))

    def test_b_basis_n(self):
        m = 5
        n = 5
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(n, x)

        self.assertEqual((n + 1, m), y.shape)
        self.assertTrue(np.allclose(y[:1, 0], 1.0))
        self.assertTrue(np.allclose(y[1:, 0], 0.0))
        self.assertTrue(np.allclose(y[:-1, -1], 0.0))
        self.assertTrue(np.allclose(y[-1:, -1], 1.0))
        self.assertTrue(np.allclose(np.sum(y, axis=0), 1.0))


if __name__ == "__main__":
    unittest.main()
