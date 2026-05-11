#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def b_poly(i: int, n: int, x: Array) -> Array:
    r"""
    Evaluates the :math:`i`-th Bernstein basis polynomial
    of degree :math:`n`.

    :param i: The number of the basis polynomial
    :param n: The degree of the basis polynomial.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The basis polynomial :math:`B_{i, n}(x) \in \mathbb{R}^{m}`.
    """
    binom = jnp.exp(gammaln(n + 1) - (gammaln(i + 1) + gammaln(n - i + 1)))
    return binom * jnp.power(x, i) * jnp.power(1.0 - x, n - i)


@jax.jit(static_argnums=(0,))
def b_basis(n: int, x: Array) -> Array:
    r"""
    Evaluates the Bernstein basis of degree :math:`n`.

    :param n: The degree of the basis.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The basis :math:`B_{n}(x) \in \mathbb{R}^{(n + 1) \times m}`.
    """
    return jax.vmap(lambda i: b_poly(i, n, x))(jnp.arange(n + 1))
