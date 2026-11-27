#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT

import unittest

import jax.numpy as jnp

from uncertaintyx.math.jax import fuzzy_ge
from uncertaintyx.math.jax import fuzzy_gt
from uncertaintyx.math.jax import fuzzy_in
from uncertaintyx.math.jax import fuzzy_le
from uncertaintyx.math.jax import fuzzy_lt
from uncertaintyx.math.jax import fuzzy_nanmax
from uncertaintyx.math.jax import gumbel_uncertainty
from uncertaintyx.math.jax import softmax


class MathTest(unittest.TestCase):
    """Test math functions."""

    def test_softmax_properties(self):
        x = jnp.array([2.0, 3.0, 5.0])
        w = softmax(x, 1.0)
        self.assertTrue(jnp.all((w > 0.0) & (w < 1.0)))
        self.assertTrue(jnp.isclose(jnp.sum(w), 1.0))

    def test_nan_and_negative_inf_ignored(self):
        x_clean = jnp.array([1.0, 5.0])
        x_dirty = jnp.array([1.0, 5.0, jnp.nan, -jnp.inf])

        w_clean = softmax(x_clean, 1.0)
        w_dirty = softmax(x_dirty, 1.0)

        self.assertTrue(jnp.allclose(w_dirty[:2], w_clean))
        self.assertEqual(0.0, w_dirty[2])
        self.assertEqual(0.0, w_dirty[3])

        self.assertTrue(
            jnp.isclose(
                fuzzy_nanmax(x_dirty, 1.0), fuzzy_nanmax(x_clean, 1.0)
            )
        )

    def test_positive_inf_and_all_nan_produce_nan(self):
        self.assertTrue(
            jnp.all(jnp.isnan(softmax(jnp.array([1.0, jnp.inf]), 1.0)))
        )
        self.assertTrue(
            jnp.all(jnp.isnan(softmax(jnp.array([jnp.nan, jnp.nan]), 1.0)))
        )
        self.assertTrue(
            jnp.isnan(fuzzy_nanmax(jnp.array([1.0, jnp.inf]), 1.0))
        )
        self.assertTrue(
            jnp.isnan(fuzzy_nanmax(jnp.array([jnp.nan, jnp.nan]), 1.0))
        )

    def test_fuzzy_nanmax_approaches_nanmax(self):
        x = jnp.array([1.0, 5.0, 3.0, jnp.nan])
        self.assertTrue(jnp.isclose(fuzzy_nanmax(x, 0.1), 5.0))

    def test_gumbel_uncertainty(self):
        u = jnp.array([1.0, 1.0, jnp.nan])
        expected = jnp.sqrt(6.0) / jnp.pi
        self.assertTrue(jnp.isclose(gumbel_uncertainty(u), expected))

    def test_fuzzy_comparisons_unit_uncertainty(self):
        q, u = jnp.array([-1.0, 0.0, 1.0]), 1.0
        self.assertTrue(
            jnp.allclose(
                fuzzy_gt(q, 1.0, 0.0, u),
                jnp.asarray([0.16, 0.50, 0.84]),
                atol=1.0e-02,
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_ge(q, 1.0, 0.0, u),
                jnp.asarray([0.16, 0.50, 0.84]),
                atol=1.0e-02,
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_lt(q, 1.0, 0.0, u),
                jnp.asarray([0.84, 0.50, 0.16]),
                atol=1.0e-02,
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_le(q, 1.0, 0.0, u),
                jnp.asarray([0.84, 0.50, 0.16]),
                atol=1.0e-02,
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_in(q, 0.0, 1.0, 1.0, 0.0, u),
                jnp.asarray([0.14, 0.34, 0.34]),
                atol=1.0e-02,
            )
        )

    def test_fuzzy_comparisons_zero_uncertainty(self):
        q, u = jnp.array([-1.0, 0.0, 1.0]), 0.0
        self.assertTrue(
            jnp.allclose(
                fuzzy_gt(q, 1.0, 0.0, u), jnp.asarray([0.0, 0.0, 1.0])
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_ge(q, 1.0, 0.0, u), jnp.asarray([0.0, 1.0, 1.0])
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_lt(q, 1.0, 0.0, u), jnp.asarray([1.0, 0.0, 0.0])
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_le(q, 1.0, 0.0, u), jnp.asarray([1.0, 1.0, 0.0])
            )
        )
        self.assertTrue(
            jnp.allclose(
                fuzzy_in(q, 0.0, 1.0, 1.0, 0.0, u),
                jnp.asarray([0.0, 1.0, 0.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
