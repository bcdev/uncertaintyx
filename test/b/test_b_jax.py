#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from uncertaintyx.b.jax import b_basis
from uncertaintyx.b.jax import b_poly


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


class PolyTest(unittest.TestCase):
    """Tests Bernstein polynomials."""

    def test_b_poly_0(self):
        m = 5
        n = 0
        b = jnp.ones(n + 1)
        x = jnp.linspace(0.0, 1.0, m)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(np.allclose(y, 1.0))

        b = b * 2.0
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(np.allclose(y, 2.0))

    def test_b_grad_0(self):
        def b_poly_sum(b, x):
            """To test the gradient."""
            return jnp.sum(b_poly(b, x))

        @jax.jit
        def b_poly_grad(b, x):
            """To test the gradient."""
            return jax.grad(b_poly_sum, argnums=1)(b, x)

        m = 5
        n = 0
        b = jnp.ones(n + 1)
        x = jnp.linspace(0.0, 1.0, m)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(np.allclose(g, 0.0))

        b = b * 2.0
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(np.allclose(g, 0.0))

    def test_b_poly_n(self):
        m = 5
        n = 5
        b = jnp.ones(n + 1)
        x = jnp.linspace(0.0, 1.0, m)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(np.allclose(y, 1.0))

        b = jnp.linspace(1.0, 2.0, n + 1)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(np.allclose(y, 1.0 + x))

    def test_b_grad_n(self):
        def b_poly_sum(b, x):
            """To test the gradient."""
            return jnp.sum(b_poly(b, x))

        @jax.jit
        def b_poly_grad(b, x):
            """To test the gradient."""
            return jax.grad(b_poly_sum, argnums=1)(b, x)

        m = 5
        n = 5
        b = jnp.ones(n + 1)
        x = jnp.linspace(0.0, 1.0, m)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(np.allclose(g, 0.0))

        b = jnp.linspace(1.0, 2.0, n + 1)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(np.allclose(g, 1.0))


if __name__ == "__main__":
    unittest.main()
