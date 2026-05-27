#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp
import numpy as np

from uncertaintyx.b.jax import BernsteinGrid
from uncertaintyx.b.jax import BernsteinPoly
from uncertaintyx.b.jax import b_basis
from uncertaintyx.b.jax import b_solve


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


class BernsteinGridTest(unittest.TestCase):
    """
    Tests the evaluation of multivariate Bernstein polynomials
    against values precalculated with Mathematica.
    """

    def setUp(self):
        k = (4, 3, 2)
        d = tuple([k_ + 1 for k_ in k])
        c = np.arange(np.prod(np.asarray(d))).reshape(d) + 1.0
        x = (
            np.asarray([0.2718, 0.5772, 0.3141]),
            np.asarray([0.5772, 0.3141, 0.2718]),
            np.asarray([0.3141, 0.2718, 0.5772]),
        )
        self.d = d
        self.c = c
        self.f = BernsteinGrid(x)

    def test_eval(self):
        c = self.c
        f = self.f
        y = f.eval(c)
        y_precalculated = np.asarray(
            [
                [
                    [19.8694, 19.7848, 20.3956],
                    [17.5015, 17.4169, 18.0277],
                    [17.1208, 17.0362, 17.6470],
                ],
                [
                    [34.5286, 34.4440, 35.0548],
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
        self.assertEqual(y_precalculated.shape, y.shape)
        self.assertTrue(np.allclose(y, y_precalculated))

        g = f.jac(c)
        self.assertEqual(y.shape + c.shape, g.shape)
        self.assertTrue(np.all(g > 0.0))

        u = np.square(0.1 * c)
        v = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))

        v = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))

    def test_jac(self):
        c = self.c
        f = self.f
        y = f.eval(c)

        g = f.jac(c)
        self.assertEqual(y.shape + c.shape, g.shape)
        self.assertTrue(np.all(g > 0.0))

        u = np.square(0.1 * c)
        v = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))

        v = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))

    def test_lpu(self):
        c = self.c
        f = self.f
        y = f.eval(c)

        u = np.square(0.1 * c)
        v = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))

        v = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, v.shape)
        self.assertTrue(np.all(v > 0.0))


class BernsteinPolyTest(unittest.TestCase):
    """
    Tests the evaluation of multivariate Bernstein polynomials
    against values precalculated with Mathematica.
    """

    def test_bernstein_poly(self):
        k = (4, 3, 2)
        d = tuple([k_ + 1 for k_ in k])
        c = np.arange(np.prod(np.asarray(d))).reshape(d) + 1.0
        x = np.asarray(
            [
                [0.2718, 0.5772, 0.3141],
                [0.5772, 0.3141, 0.2718],
                [0.3141, 0.2718, 0.5772],
            ]
        )
        f = BernsteinPoly(c)
        y = f.eval(c, x)
        y_precalculated = np.asarray([19.8694, 32.0761, 19.6774])
        self.assertEqual(y_precalculated.shape, y.shape)
        self.assertTrue(np.allclose(y, y_precalculated))

        g = f.jac_p(c, x)
        self.assertEqual((3,) + d, g.shape)
        self.assertTrue(np.all(g > 0.0))

        g = f.jac_x(c, x)
        self.assertEqual((3, 3), g.shape)
        self.assertTrue(np.all(g > 0.0))

    def test_from_lookup_table(self):
        k = (3, 4, 2)
        d = tuple([k_ + 1 for k_ in k])
        c = np.arange(np.prod(np.asarray(d))).reshape(d)
        x = (
            np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )
        y = BernsteinGrid(x).eval(c)

        f = BernsteinPoly.from_lookup_table(k, x, y, non_negative=True)
        b = f.prior()
        self.assertEqual(c.shape, b.shape)
        self.assertTrue(np.allclose(b, c))


class BSolveTest(unittest.TestCase):
    """
    Tests the solving function by fitting coefficients to
    Bernstein basis polynomials.
    """

    def test_b_solve_0_2(self):
        r"""Fit :math:`B_{0,2}(x)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(1.0 - x)

        c = b_solve((k,), (x,), y, non_negative=True)
        self.assertEqual((k + 1,), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(1.0, c[0].item())
        self.assertAlmostEqual(0.0, c[1].item())
        self.assertAlmostEqual(0.0, c[2].item())

    def test_b_solve_1_2(self):
        r"""Fit :math:`B_{1,2}(x)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = 2.0 * x * (1.0 - x)

        c = b_solve((k,), (x,), y, non_negative=True)
        self.assertEqual((k + 1,), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0].item())
        self.assertAlmostEqual(1.0, c[1].item())
        self.assertAlmostEqual(0.0, c[2].item())

    def test_b_solve_2_2(self):
        r"""Fit :math:`B_{2,2}(x)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(x)

        c = b_solve((k,), (x,), y, non_negative=True)
        self.assertEqual((k + 1,), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0].item())
        self.assertAlmostEqual(0.0, c[1].item())
        self.assertAlmostEqual(1.0, c[2].item())

    def test_b_solve_0_0_2_2(self):
        r"""Fit :math:`B_{(0,0),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(1.0 - x[jnp.newaxis, :]) * jnp.square(
            1.0 - x[:, jnp.newaxis]
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(1.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_1_0_2_2(self):
        r"""Fit :math:`B_{(1,0),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = (
            2.0
            * x[:, jnp.newaxis]
            * (1.0 - x[:, jnp.newaxis])
            * jnp.square(1.0 - x[jnp.newaxis, :])
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(1.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_2_0_2_2(self):
        r"""Fit :math:`B_{(2,0),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(x[:, jnp.newaxis]) * jnp.square(
            1.0 - x[jnp.newaxis, :]
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(1.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_0_1_2_2(self):
        r"""Fit :math:`B_{(0,1),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = (
            2.0
            * jnp.square(1.0 - x[:, jnp.newaxis])
            * x[jnp.newaxis, :]
            * (1.0 - x[jnp.newaxis, :])
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(1.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_1_1_2_2(self):
        r"""Fit :math:`B_{(1,1),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = (
            4.0
            * x[:, jnp.newaxis]
            * (1.0 - x[:, jnp.newaxis])
            * x[jnp.newaxis, :]
            * (1.0 - x[jnp.newaxis, :])
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(1.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_2_1_2_2(self):
        r"""Fit :math:`B_{(2,1),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = (
            2.0
            * jnp.square(x[:, jnp.newaxis])
            * x[jnp.newaxis, :]
            * (1.0 - x[jnp.newaxis, :])
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertFalse(np.any(c < 0.0))
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(1.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_0_2_2_2(self):
        r"""Fit :math:`B_{(0,2),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(1.0 - x[:, jnp.newaxis]) * jnp.square(
            x[jnp.newaxis, :]
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertFalse(np.any(c < 0.0))
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(1.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_1_2_2_2(self):
        r"""Fit :math:`B_{(1,2),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = (
            2.0
            * x[:, jnp.newaxis]
            * (1.0 - x[:, jnp.newaxis])
            * jnp.square(x[jnp.newaxis, :])
        )

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(1.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(0.0, c[2, 2].item())

    def test_b_solve_2_2_2_2(self):
        r"""Fit :math:`B_{(2,2),(2,2)}(x_0, x_1)`."""
        k = 2
        x = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = jnp.square(x[:, jnp.newaxis]) * jnp.square(x[jnp.newaxis, :])

        c = b_solve((k, k), (x, x), y, non_negative=True)
        self.assertEqual((k + 1, k + 1), c.shape)
        self.assertFalse(np.any(c < 0.0))
        self.assertAlmostEqual(0.0, c[0, 0].item())
        self.assertAlmostEqual(0.0, c[0, 1].item())
        self.assertAlmostEqual(0.0, c[0, 2].item())
        self.assertAlmostEqual(0.0, c[1, 0].item())
        self.assertAlmostEqual(0.0, c[1, 1].item())
        self.assertAlmostEqual(0.0, c[1, 2].item())
        self.assertAlmostEqual(0.0, c[2, 0].item())
        self.assertAlmostEqual(0.0, c[2, 1].item())
        self.assertAlmostEqual(1.0, c[2, 2].item())


if __name__ == "__main__":
    unittest.main()
