#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def _binom(i, n) -> Array:
    """Returns the binomial coefficients for the whole basis."""
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


def _b_basis_jvp(n, inputs: tuple[Array], perturbations: tuple[Array]):
    r"""
    Custom forward-mode differentiation (JVP) for the Bernstein basis.

    This implementation overrides automatic differentiation using the
    mathematical identity:

    .. math::

        G_{x} B_{i,n}(x) =
        n \left(B_{i-1, n-1}(x) - B_{i, n-1}(x)\right) .

    In the context of automatic differentiation, inputs are termed
    `primals` while perturbations are termed `tangents`.

    :param n: The degree of the basis.
    :param inputs: A tuple containing the :math:`x \in \mathbb{R}^{m}`.
    :param perturbations: A tuple containing the perturbation for :math:`x`.
    :returns: A tuple containing the basis and its perturbation.
    """
    (x,) = inputs
    (d,) = perturbations

    if n == 0:
        basis = _b_basis(n, x)
        return basis, jnp.zeros_like(basis)

    basis_minus = _b_basis(n - 1, x)
    a = jnp.pad(basis_minus, ((1, 0), (0, 0)))
    b = jnp.pad(basis_minus, ((0, 1), (0, 0)))
    g = n * (a - b)
    basis = b + (a - b) * x[jnp.newaxis, :]
    basis_perturbation = g * d[jnp.newaxis, :]

    return basis, basis_perturbation


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
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{m}`.
    """
    return jnp.dot(b, b_basis(b.shape[-1] - 1, x))


@jax.jit
def b_poly_grid(b: Array, x: tuple[Array, ...]) -> Array:
    r"""
    Evaluates an N-dimensional Bernstein polynomial on a regular
    N-dimensional grid of points.

    This function is designed to fit coefficients for the efficient
    approximation of N-dimensional lookup tables.

    The implementation uses sequential tensor contraction (Tucker
    product) facilitated by the tensor product structure of the
    regular grid.

    Let :math:`N \in \mathbb{N}` be the dimension of the Bernstein
    polynomial. Let :math:`k = (k_{1}, \dots, k_{N}) \in \mathbb{N}^{N}`
    denote its degrees and let \mathbb{R}^{k + 1} denote the tensor
    space with dimensions :math:`(k_{1} + 1, \dots, k_{N} + 1)`.
    Likewise, let :math:`m = (m_{1}, \dots , m_{N}) \in \mathbb{N}^{N}`
    denote the dimensions of the grid coordinates
    :math:`x = (x_{1}, \dots, x_{N})` and let \mathbb{R}^{m} denote
    the tensor space with dimensions :math:`m`. Then:

    :param b: The Bernstein coefficients :math:`b \in \mathbb{R}^{k + 1}`.
    :param x: The grid coordinates :math:`x`.
    :returns: The polynomial values :math:`B(x) \in \mathbb{R}^{m}`.
    """
    N = b.ndim  # noqa: N806
    k = tuple(b.shape[i] - 1 for i in range(N))

    for i in range(N):  # the Tucker product
        b = jnp.tensordot(b, b_basis(k[i], x[i]), axes=(0, 0))

    return b
