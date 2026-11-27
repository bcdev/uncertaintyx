#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Specific pure JAX functions.
"""

from numbers import Real

import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf


def fuzzy_ge(
    q: Array, x: Array | Real, y: Array | Real, u: Array | Real = 0.0
) -> Array:
    r"""
    The fuzzy greater-than-or-equal function.

    Select return values based on the condition :math:`q ≥ 0`.

    :param q: The test quantity.
    :param x: Returned where :math:`q ≥ 0`.
    :param y: Returned where :math:`q < 0`.
    :param u: The uncertainty of :math:`q`.
    :returns: The result values drawn from either ``x`` or ``y``.
    """
    b = jnp.isfinite(q) & (u > 0.0)
    t = jnp.where(b, q, 0.0) / jnp.where(b, u, 1.0)
    w = jnp.where(
        b,
        0.5 * (1.0 + erf(t / jnp.sqrt(2.0))),
        jnp.where(q >= 0.0, 1.0, jnp.where(q != q, jnp.nan, 0.0)),
    )
    return y + w * (x - y)


def fuzzy_gt(
    q: Array, x: Array | Real, y: Array | Real, u: Array | Real = 0.0
) -> Array:
    r"""
    The fuzzy greater-than function.

    Select return values based on the condition :math:`q > 0`.

    :param q: The test quantity.
    :param x: Returned where :math:`q > 0`.
    :param y: Returned where :math:`q ≤ 0`.
    :param u: The uncertainty of :math:`q`.
    :returns: The result values drawn from either ``x`` or ``y``.
    """
    b = jnp.isfinite(q) & (u > 0.0)
    t = jnp.where(b, q, 0.0) / jnp.where(b, u, 1.0)
    w = jnp.where(
        b,
        0.5 * (1.0 + erf(t / jnp.sqrt(2.0))),
        jnp.where(q > 0.0, 1.0, jnp.where(q != q, jnp.nan, 0.0)),
    )
    return y + w * (x - y)


def fuzzy_le(
    q: Array, x: Array | Real, y: Array | Real, u: Array | Real = 0.0
) -> Array:
    r"""
    The fuzzy less-than-or-equal function.

    Select return values based on the condition :math:`q ≤ 0`.

    :param q: The test quantity.
    :param x: Returned where :math:`q ≤ 0`.
    :param y: Returned where :math:`q > 0`.
    :param u: The uncertainty of :math:`q`.
    :returns: The result values drawn from either ``x`` or ``y``.
    """
    return fuzzy_ge(-q, x, y, u)


def fuzzy_lt(
    q: Array, x: Array | Real, y: Array | Real, u: Array | Real = 0.0
) -> Array:
    r"""
    The fuzzy less-than function.

    Select return values based on the condition :math:`q < 0`.

    :param q: The test quantity.
    :param x: Returned where :math:`q < 0`.
    :param y: Returned where :math:`q ≥ 0`.
    :param u: The uncertainty of :math:`q`.
    :returns: The result values drawn from either ``x`` or ``y``.
    """
    return fuzzy_gt(-q, x, y, u)


def fuzzy_in(
    q: Array,
    a: Array | Real,
    b: Array | Real,
    x: Array | Real = 1.0,
    y: Array | Real = 0.0,
    u: Array | Real = 0.0,
) -> Array:
    r"""
    The fuzzy in-range function.

    Select return values based on the interval condition
    :math:`a ≤ q < b`.

    :param q: The test quantity.
    :param a: The lower bound of the interval.
    :param b: The upper bound of the interval.
    :param x: Returned where :math:`a ≤ q < b`.
    :param y: Returned where :math:`q < a` or :math:`q ≥ b`.
    :param u: The standard uncertainty of :math:`q`.
    :returns: The result values drawn from either ``x`` or ``y``.
    """
    w = fuzzy_ge(q - a, 1.0, 0.0, u) - fuzzy_ge(q - b, 1.0, 0.0, u)
    return y + w * (x - y)


def fuzzy_nanmax(x: Array, u: Real) -> Array:
    r"""
    The fuzzy maximum function, ignoring NaN values.

    Let :math:`x \in \mathbb{R}^{n}` and let

    .. math::
        M_u(x) = \sum_{i=1}^n \sigma(x_i, u) x_i
        \leq \max_{i=1,\dots,n} x_i.

    denote the fuzzy maximum. If any :math:`x_i` equals infinity
    or all :math:`x_i` are NaN, the function returns NaN.


    :param x: The vector :math:`x \in \mathbb{R}^n`.
    :param u: The uncertainty parameter :math:`u > 0`.
    :returns: The fuzzy maximum :math:`M_u(x) \in \mathbb{R}`.
    """
    w = nan_softmax(x, u)
    return jnp.dot(w, jnp.where((x == x) & (w != 0.0), x, 0.0))


def gumbel_uncertainty(u: Array) -> Array:
    r"""
    Returns the Gumbel uncertainty matching heteroscedastic standard
    uncertainties.

    Approximates heteroscedastic standard uncertainties
    :math:`u \in \mathbb{R}^{n}` with a single uncertainty
    :math:`u_{\mathrm{G}} > 0` by matching the variance of the
    Gumbel distribution to the mean squared uncertainty.

    The Gumbel distribution (or type-I generalized extreme value
    distribution) models the distribution of the maximum of a sample
    of random variables. In the context of uncertainty propagation,
    adding Gumbel noise with scale :math:`u_{\mathrm{G}}` to finite
    values :math:`x` yields exact softmax probabilities
    :math:`\sigma(x, u_{\mathrm{G})` for each element being the
    maximum.

    :param u: The standard uncertainties :math:`u`.
    :returns: The Gumbel uncertainty :math:`u_{\mathrm{G}}`.
    """
    return jnp.sqrt(6.0 * jnp.nanmean(jnp.square(u))) / jnp.pi


def nan_softmax(x: Array, u) -> Array:
    r"""
    The softmax function, ignoring NaN values.

    Evaluates the normalized exponential weights

    .. math::
        \sigma(x_i, u) =
        \frac{\exp(x_i / u)}{\sum_{j=1}^n \exp(x_j / u)}.

    If any :math:`x_i` equals infinity or all :math:`x_i` are NaN,
    the function returns NaN.

    :param x: The vector :math:`x \in \mathbb{R}^n`.
    :param u: The uncertainty parameter :math:`u > 0`.
    :returns: The softmax values :math:`\sigma(x, u) \in (0, 1)^n`.
    """
    valid = x == x
    y = jnp.where(valid, x, 0.0) / u  # noqa
    z = jnn.softmax(y, where=valid)
    return z / jnp.sum(z)


softmax = nan_softmax
