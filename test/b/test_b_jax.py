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
        precalculated = np.asarray(
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
        self.assertEqual((3, 3, 3), y.shape)
        self.assertTrue(np.allclose(y, precalculated))

        g = f.jac(c)
        self.assertEqual(y.shape + c.shape, g.shape)
        self.assertTrue(np.all(g > 0.0))

        u = to_var(0.1 * c)
        u = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))

        u = to_var(0.1 * c)
        u = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))

    def test_jac(self):
        c = self.c
        f = self.f
        y = f.eval(c)

        g = f.jac(c)
        self.assertEqual(y.shape + c.shape, g.shape)
        self.assertTrue(np.all(g > 0.0))

        u = to_var(0.1 * c)
        u = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))

        u = to_var(0.1 * c)
        u = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))

    def test_lpu(self):
        c = self.c
        f = self.f
        y = f.eval(c)

        u = to_var(0.1 * c)
        u = f.lpu(c, u, diag=True)
        self.assertEqual(y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))

        u = to_var(0.1 * c)
        u = f.lpu(c, u)
        self.assertEqual(y.shape + y.shape, u.shape)
        self.assertTrue(np.all(u > 0.0))


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
        precalculated = np.asarray([19.8694, 32.0761, 19.6774])
        self.assertEqual((3,), y.shape)
        self.assertTrue(jnp.allclose(y, precalculated))

        g = f.jac_p(c, x)
        self.assertEqual((3,) + d, g.shape)
        self.assertTrue(np.all(g > 0.0))

        g = f.jac_x(c, x)
        self.assertEqual((3, 3), g.shape)
        self.assertTrue(np.all(g > 0.0))

    def test_from_lookup_table(self):
        k = 5
        x = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        y = np.array(  # y = x ** 2 + 2 x + 3
            [3.00, 3.44, 3.96, 4.56, 5.24, 6.00]
        )

        f = BernsteinPoly.from_lookup_table((k,), (x,), y, non_negative=True)
        c = f.prior()
        self.assertEqual((k + 1,), c.shape)
        self.assertAlmostEqual(3.0, c[0])
        self.assertAlmostEqual(3.4, c[1])
        self.assertAlmostEqual(3.9, c[2])
        self.assertAlmostEqual(4.5, c[3])
        self.assertAlmostEqual(5.2, c[4])
        self.assertAlmostEqual(6.0, c[5])
        self.assertTrue(jnp.allclose(f.eval(c, x), y))


class BSolveTest(unittest.TestCase):
    """Tests the solving function."""

    def test_b_solve_degree_2(self):
        k = 2
        x = jnp.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        y = jnp.array(  # y = x ** 2 + 2 x + 3
            [3.00, 3.44, 3.96, 4.56, 5.24, 6.00]
        )

        c = b_solve((k,), (x,), y, non_negative=True)
        self.assertEqual((k + 1,), c.shape)
        self.assertAlmostEqual(3.0, c[0].item())
        self.assertAlmostEqual(4.0, c[1].item())
        self.assertAlmostEqual(6.0, c[2].item())

    def test_b_solve_degree_5(self):
        k = 5
        x = jnp.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        y = (
            jnp.array(  # y = x ** 2 + 2 x - 1 / 100
                [0.00, 0.44, 0.96, 1.56, 2.24, 3.00]
            )
            - 0.01
        )

        c = b_solve((k,), (x,), y)
        self.assertEqual((k + 1,), c.shape)
        self.assertAlmostEqual(-0.01, c[0].item())
        self.assertAlmostEqual(0.39, c[1].item())
        self.assertAlmostEqual(0.89, c[2].item())
        self.assertAlmostEqual(1.49, c[3].item())
        self.assertAlmostEqual(2.19, c[4].item())
        self.assertAlmostEqual(2.99, c[5].item())

        c = b_solve((k,), (x,), y, non_negative=True)
        self.assertEqual((k + 1,), c.shape)
        self.assertAlmostEqual(0.00, c[0].item())
        self.assertAlmostEqual(0.38, c[1].item(), places=2)
        self.assertAlmostEqual(0.90, c[2].item(), places=2)
        self.assertAlmostEqual(1.48, c[3].item(), places=2)
        self.assertAlmostEqual(2.19, c[4].item(), places=2)
        self.assertAlmostEqual(2.99, c[5].item(), places=2)


def to_var(u: np.ndarray) -> np.ndarray:
    """
    Converts standard uncertainty to a diagonal uncertainty tensor.
    """
    return np.square(u)


if __name__ == "__main__":
    unittest.main()
