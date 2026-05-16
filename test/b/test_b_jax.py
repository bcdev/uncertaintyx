#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax
import jax.numpy as jnp

from uncertaintyx.b.jax import b_basis
from uncertaintyx.b.jax import b_poly
from uncertaintyx.b.jax import b_poly_grid
from uncertaintyx.b.jax import b_poly_point
from uncertaintyx.b.jax import b_poly_points


class BBasisTest(unittest.TestCase):
    """Tests the evaluation of Bernstein basis polynomials."""

    def test_b_basis_of_degree_0(self):
        m = 5
        k = 0
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(k, x)

        self.assertEqual((k + 1, m), y.shape)
        self.assertTrue(jnp.allclose(y, 1.0))

    def test_b_basis_of_degree_1(self):
        m = 5
        k = 1
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(k, x)

        self.assertEqual((k + 1, m), y.shape)
        self.assertTrue(jnp.allclose(y[0, 0], 1.0))
        self.assertTrue(jnp.allclose(y[0, 2], 0.5))
        self.assertTrue(jnp.allclose(y[1, 0], 0.0))
        self.assertTrue(jnp.allclose(y[1, 2], 0.5))
        self.assertTrue(jnp.allclose(y[0, -1], 0.0))
        self.assertTrue(jnp.allclose(y[1, -1], 1.0))
        self.assertTrue(jnp.allclose(jnp.sum(y, axis=0), 1.0))

    def test_b_basis_of_degree_2(self):
        m = 5
        k = 2
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(k, x)

        self.assertEqual((k + 1, m), y.shape)
        self.assertTrue(jnp.allclose(y[:1, 0], 1.0))
        self.assertTrue(jnp.allclose(y[1:, 0], 0.0))
        self.assertTrue(jnp.allclose(y[0, 2], 0.25))
        self.assertTrue(jnp.allclose(y[1, 2], 0.50))
        self.assertTrue(jnp.allclose(y[2, 2], 0.25))
        self.assertTrue(jnp.allclose(y[:-1, -1], 0.0))
        self.assertTrue(jnp.allclose(y[-1:, -1], 1.0))
        self.assertTrue(jnp.allclose(jnp.sum(y, axis=0), 1.0))

    def test_b_basis_of_degree_5(self):
        m = 5
        k = 5
        x = jnp.linspace(0.0, 1.0, m)
        y = b_basis(k, x)

        self.assertEqual((k + 1, m), y.shape)
        self.assertTrue(jnp.allclose(y[:1, 0], 1.0))
        self.assertTrue(jnp.allclose(y[1:, 0], 0.0))
        self.assertTrue(jnp.allclose(y[:-1, -1], 0.0))
        self.assertTrue(jnp.allclose(y[-1:, -1], 1.0))
        self.assertTrue(jnp.allclose(jnp.sum(y, axis=0), 1.0))


class BPolyTest(unittest.TestCase):
    """Tests the evaluation of Bernstein polynomials."""

    def test_b_poly_of_degree_0(self):
        m = 5
        k = 0
        b = jnp.ones(k + 1)
        x = jnp.linspace(0.0, 1.0, m)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(jnp.allclose(y, 1.0))

        b = b * 2.0
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(jnp.allclose(y, 2.0))

    def test_b_b_poly_of_degree_0_grad(self):
        def b_poly_sum(b, x):
            """To test the gradient."""
            return jnp.sum(b_poly(b, x))

        @jax.jit
        def b_poly_grad(b, x):
            """To test the gradient."""
            return jax.grad(b_poly_sum, argnums=1)(b, x)

        m = 5
        k = 0
        b = jnp.ones(k + 1)
        x = jnp.linspace(0.0, 1.0, m)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(jnp.allclose(g, 0.0))

        b = b * 2.0
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(jnp.allclose(g, 0.0))

    def test_b_poly_of_degree_5(self):
        m = 5
        k = 5
        b = jnp.ones(k + 1)
        x = jnp.linspace(0.0, 1.0, m)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(jnp.allclose(y, 1.0))

        b = jnp.linspace(1.0, 2.0, k + 1)
        y = b_poly(b, x)

        self.assertEqual((m,), y.shape)
        self.assertTrue(jnp.allclose(y, 1.0 + x))

    def test_b_poly_of_degree_5_grad(self):
        def b_poly_sum(b, x):
            """To test the gradient."""
            return jnp.sum(b_poly(b, x))

        @jax.jit
        def b_poly_grad(b, x):
            """To test the gradient."""
            return jax.grad(b_poly_sum, argnums=1)(b, x)

        m = 5
        k = 5
        b = jnp.ones(k + 1)
        x = jnp.linspace(0.0, 1.0, m)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(jnp.allclose(g, 0.0))

        b = jnp.linspace(1.0, 2.0, k + 1)
        g = b_poly_grad(b, x)

        self.assertEqual((m,), g.shape)
        self.assertTrue(jnp.allclose(g, 1.0))


class BPolyGridTest(unittest.TestCase):
    """
    Tests the evaluation of multivariate Bernstein polynomials
    against values precalculated with Mathematica.
    """

    def test_b_poly_grid(self):
        k = (4, 3, 2)
        d = tuple([k_ + 1 for k_ in k])
        b = jnp.arange(jnp.prod(jnp.asarray(d))).reshape(d) + 1.0
        x = (
            jnp.asarray([0.2718, 0.5772, 0.3141]),
            jnp.asarray([0.5772, 0.3141, 0.2718]),
            jnp.asarray([0.3141, 0.2718, 0.5772]),
        )
        y = b_poly_grid(b, x)
        precalculated = jnp.asarray(
            [
                [
                    [19.8694, 19.7848, 20.3956],
                    [17.5015, 17.4169, 18.0277],
                    [17.1208, 17.0362, 17.6470],
                ],
                [
                    [34.5286, 34.444, 35.05480],
                    [32.1607, 32.0761, 32.6869],
                    [31.7800, 31.6954, 32.3062],
                ],
                [
                    [21.8998, 21.8152, 22.4260],
                    [19.5319, 19.4473, 20.0581],
                    [19.1512, 19.0666, 19.6774],
                ],
            ]
        )
        self.assertEqual((3, 3, 3), y.shape)
        self.assertTrue(jnp.allclose(y, precalculated))


class BPolyPointsTest(unittest.TestCase):
    """
    Tests the evaluation of multivariate Bernstein polynomials
    against values precalculated with Mathematica.
    """

    def test_b_poly_point(self):
        k = (4, 3, 2)
        d = tuple([k_ + 1 for k_ in k])
        b = jnp.arange(jnp.prod(jnp.asarray(d))).reshape(d) + 1.0
        x = jnp.asarray([0.2718, 0.5772, 0.3141])
        y = b_poly_point(b, x)
        precalculated = 19.8694
        self.assertEqual((), y.shape)
        self.assertTrue(jnp.allclose(y, precalculated))

    def test_b_poly_points(self):
        k = (4, 3, 2)
        d = tuple([k_ + 1 for k_ in k])
        b = jnp.arange(jnp.prod(jnp.asarray(d))).reshape(d) + 1.0
        x = jnp.asarray(
            [
                [0.2718, 0.5772, 0.3141],
                [0.5772, 0.3141, 0.2718],
                [0.3141, 0.2718, 0.5772],
            ]
        )
        y = b_poly_points(b, x)
        precalculated = jnp.asarray([19.8694, 32.0761, 19.6774])
        self.assertEqual((3,), y.shape)
        self.assertTrue(jnp.allclose(y, precalculated))


if __name__ == "__main__":
    unittest.main()
