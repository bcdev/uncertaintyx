#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def _binom(i, n) -> Array:
    return jnp.exp(gammaln(n + 1) - (gammaln(i + 1) + gammaln(n - i + 1)))


_b_basis = jax.custom_jvp(
    lambda n, x: (
        lambda i: (
            _binom(i, n)
            * jnp.power(x[jnp.newaxis, :], i)
            * jnp.power(1.0 - x[jnp.newaxis, :], n - i)
        )
    )(jnp.arange(n + 1)[:, jnp.newaxis]),
    nondiff_argnums=(0,),
)


def _b_basis_jvp(n, primals, tangents):
    (x,) = primals
    (t,) = tangents

    basis = _b_basis(n, x)
    if n == 0:
        return basis, jnp.zeros_like(basis)
    lower = _b_basis(n - 1, x)
    jac_x = n * (
        jnp.pad(lower, ((1, 0), (0, 0))) - jnp.pad(lower, ((0, 1), (0, 0)))
    )
    return basis, jac_x * t[jnp.newaxis, :]


_b_basis.defjvp(_b_basis_jvp)


@jax.jit(static_argnums=(0,))
def b_basis(n: int, x: Array) -> Array:
    r"""
    Evaluates the Bernstein basis of degree :math:`n`.

    :param n: The degree of the basis.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The basis :math:`B_{n}(x) \in \mathbb{R}^{(n + 1) \times m}`.
    """
    return _b_basis(n, x)


@jax.jit
def b_poly(b: Array, x: Array) -> Array:
    r"""
    Evaluates a Bernstein polynomial.

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{(n + 1)}`.
    :param x: The input :math:`x \in \mathbb{R}^{m}`.
    :returns: The polynomial :math:`B(x) \in \mathbb{R}^{(n + 1) \times m}`.
    """
    return jnp.dot(b, b_basis(b.shape[-1] - 1, x))
